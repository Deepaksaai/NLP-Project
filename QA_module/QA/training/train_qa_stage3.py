"""
QA Stage 3 — legal fine-tuning.

Loads qa_stage2_best.pt, trains on CuAD + LEDGAR + (optional COLIEE)
with QASPER and SQuAD anchors, and tracks legal-specific metrics
throughout.

Mix per batch (default batch_size=16):
    35% cuad    / 25% ledgar / 10% coliee /
    15% qasper  / 15% squad

Early-stopping metric: val_f1 on CuAD ONLY (legal_val_f1).

Usage:
    python -m QA.data.preprocess_cuad
    python -m QA.data.preprocess_ledgar
    python -m QA.data.preprocess_coliee              # no-op if no local file
    python -m QA.training.train_qa_stage3 --epochs 10 --batch_size 16
"""

import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from QA.qa_config import (
    QA_TOKENIZER_PATH, QA_DATA_ROOT,
    LEGAL_MAX_ANSWER_LEN, LEGAL_LENGTH_PENALTY,
    load_qa_special_tokens,
)
from QA.model.qa_model import build_qa_model
from QA.data.stage3_dataset import Stage3QADataset, stage3_collate
from QA.data.mixed_sampler import MixedBatchSampler
from QA.training.loss import compute_loss
from QA.training.legal_metrics import evaluate_stage3
from QA.training.unfreeze_schedule_stage3 import (
    build_stage3_optimizer, apply_stage3_schedule, STAGE3_LRS,
)
from QA.training.unfreeze_schedule import lrs_by_group


_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR = os.path.join(_ROOT, "checkpoints")
LOG_DIR  = os.path.join(_ROOT, "QA", "logs")


# Stage-3 mix. COLIEE is dropped from the mix if its pool is empty.
BASE_MIX = {
    "cuad":   0.35,
    "ledgar": 0.25,
    "coliee": 0.10,
    "qasper": 0.15,
    "squad":  0.15,
}


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.05):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.min_ratio = min_ratio
        # Base LRs are resnapshot at every epoch start because
        # apply_stage3_schedule mutates them.
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def resnapshot(self):
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            scale = self.step_num / self.warmup
        else:
            progress = min(1.0, (self.step_num - self.warmup) / (self.total - self.warmup))
            scale = self.min_ratio + 0.5 * (1 - self.min_ratio) * (1 + math.cos(math.pi * progress))
        for i, g in enumerate(self.opt.param_groups):
            g["lr"] = self.base_lrs[i] * scale


def _load_stage2_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[stage2-load] missing={len(missing)}  unexpected={len(unexpected)}")
    return ckpt


def _save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _prune_epoch_ckpts(keep: int = 3):
    files = sorted(
        [f for f in os.listdir(CKPT_DIR) if f.startswith("qa_stage3_epoch_") and f.endswith(".pt")],
        key=lambda f: int(f.replace("qa_stage3_epoch_", "").replace(".pt", "")),
    )
    while len(files) > keep:
        os.remove(os.path.join(CKPT_DIR, files.pop(0)))


def _resolve_mix(train_ds) -> dict:
    """Drop empty pools from the mix and renormalize."""
    mix = {k: v for k, v in BASE_MIX.items()
           if train_ds.indices_by_source.get(k)}
    if not mix:
        raise RuntimeError("No Stage-3 training pools available")
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip=1.0, log_every=50):
    model.train()
    running = {"total": [0.0, 0], "span": [0.0, 0], "has_answer": [0.0, 0]}
    per_source = defaultdict(lambda: [0.0, 0])
    t0 = time.time()

    for step, batch in enumerate(loader):
        input_ids   = batch["input_ids"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        attention   = batch["attention_mask"].to(device)
        start_pos   = batch["start_position"].to(device)
        end_pos     = batch["end_position"].to(device)
        has_lab     = batch["has_answer"].to(device)
        sources     = batch["sources"]

        start_logits, end_logits, has_ans = model(input_ids, segment_ids, attention)
        total, comps = compute_loss(
            start_logits, end_logits, has_ans, start_pos, end_pos, has_lab,
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = input_ids.size(0)
        running["total"][0] += comps["total"].item() * bs
        running["total"][1] += bs
        running["span"][0]  += comps["span"].item() * bs
        running["span"][1]  += bs
        running["has_answer"][0] += comps["has_answer"].item() * bs
        running["has_answer"][1] += bs

        with torch.no_grad():
            ex_start = torch.nn.functional.cross_entropy(start_logits, start_pos, reduction="none")
            ex_end   = torch.nn.functional.cross_entropy(end_logits,   end_pos,   reduction="none")
            ex_span  = 0.5 * (ex_start + ex_end)
            for b in range(bs):
                s = sources[b]
                per_source[s][0] += ex_span[b].item()
                per_source[s][1] += 1

        if step % log_every == 0:
            dt = time.time() - t0
            src_losses = {k: round(v[0] / max(1, v[1]), 3) for k, v in per_source.items()}
            lr_str = {k: f"{v:.2e}" for k, v in lrs_by_group(optimizer).items()}
            print(f"  step {step:5d}  loss={running['total'][0] / max(1, running['total'][1]):.4f}  "
                  f"span={running['span'][0] / max(1, running['span'][1]):.4f}  "
                  f"hasA={running['has_answer'][0] / max(1, running['has_answer'][1]):.4f}  "
                  f"src={src_losses}  lrs={lr_str}  {dt:.1f}s")

    return {
        "total":      running["total"][0] / max(1, running["total"][1]),
        "span":       running["span"][0]  / max(1, running["span"][1]),
        "has_answer": running["has_answer"][0] / max(1, running["has_answer"][1]),
        "by_source":  {k: v[0] / max(1, v[1]) for k, v in per_source.items()},
    }


# -------------------------------------------------------
# Validation — overall + legal-only + anchor subsets
# -------------------------------------------------------
@torch.no_grad()
def evaluate_all(model, val_ds, tokenizer, device, batch_size=16):
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=stage3_collate)
    overall = evaluate_stage3(model, val_loader, tokenizer, device)

    def _subset_metrics(indices):
        if not indices:
            return None
        sub = Subset(val_ds, indices)
        sub_loader = DataLoader(sub, batch_size=batch_size, shuffle=False, collate_fn=stage3_collate)
        return evaluate_stage3(model, sub_loader, tokenizer, device)

    cuad_metrics   = _subset_metrics(val_ds.indices_by_source.get("cuad", []))
    coliee_metrics = _subset_metrics(val_ds.indices_by_source.get("coliee", []))
    qasper_metrics = _subset_metrics(val_ds.indices_by_source.get("qasper", []))
    squad_metrics  = _subset_metrics(val_ds.indices_by_source.get("squad", []))

    return {
        "overall": overall,
        "cuad":    cuad_metrics,
        "coliee":  coliee_metrics,
        "qasper":  qasper_metrics,
        "squad":   squad_metrics,
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage3] device={device}")

    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)

    train_ds = Stage3QADataset("train")
    val_ds   = Stage3QADataset("val")
    print(f"[stage3] train={train_ds.summary()}")
    print(f"[stage3] val  ={val_ds.summary()}")

    mix = _resolve_mix(train_ds)
    print(f"[stage3] mix={mix}")

    sampler = MixedBatchSampler(
        train_ds.indices_by_source, mix=mix,
        batch_size=args.batch_size, shuffle=True,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        collate_fn=stage3_collate, num_workers=args.num_workers,
    )

    model = build_qa_model(meta, load_ckpt=None, freeze=False).to(device)
    if not os.path.exists(args.stage2_ckpt):
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {args.stage2_ckpt}")
    _load_stage2_checkpoint(model, args.stage2_ckpt, device)

    # Stage 3 uses higher dropout — rebuild dropout in place.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = args.dropout

    optimizer = build_stage3_optimizer(model)
    steps_per_epoch = max(1, len(sampler))
    total_steps = steps_per_epoch * args.epochs
    scheduler = WarmupCosine(optimizer, args.warmup_steps, total_steps)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    metrics_path = os.path.join(LOG_DIR, "stage3_metrics.json")
    history = []
    best_legal_f1 = -1.0
    patience_left = args.patience
    global_step = 0

    cfg = vars(args).copy()
    cfg["mix"] = mix

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Stage-3 Epoch {epoch}/{args.epochs} =====")
        sampler.set_epoch(epoch)

        # Apply epoch LR multiplier to encoder groups
        sched_info = apply_stage3_schedule(optimizer, epoch)
        scheduler.resnapshot()
        print(f"[lr-sched] {sched_info}")

        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip, log_every=args.log_every,
        )
        global_step += steps_per_epoch

        evald = evaluate_all(model, val_ds, tokenizer, device, batch_size=args.batch_size)
        overall  = evald["overall"]
        cuad     = evald["cuad"]
        qasper   = evald["qasper"]
        squad    = evald["squad"]

        legal_f1 = cuad["f1"] if cuad else overall["f1"]

        print(f"[val] overall f1={overall['f1']:.4f}  em={overall['em']:.4f}  "
              f"legal_fpr={overall['legal_false_positive_rate']:.4f}  "
              f"clause_bd={overall['clause_boundary_accuracy']:.4f}  "
              f"term_pres={overall['legal_term_preservation']:.4f}")
        if cuad:
            print(f"[val] cuad f1={cuad['f1']:.4f}  em={cuad['em']:.4f}")
        if qasper:
            print(f"[val] qasper anchor f1={qasper['f1']:.4f}")
        if squad:
            print(f"[val] squad anchor f1={squad['f1']:.4f}")

        row = {
            "epoch": epoch,
            "train_loss_total":   train_stats["total"],
            "train_loss_span":    train_stats["span"],
            "train_loss_has_answer": train_stats["has_answer"],
            "train_loss_by_source":  train_stats["by_source"],
            "val_f1_overall":     overall["f1"],
            "val_em_overall":     overall["em"],
            "val_has_answer_accuracy": overall["has_answer_accuracy"],
            "val_legal_false_positive_rate": overall["legal_false_positive_rate"],
            "val_clause_boundary_accuracy":  overall["clause_boundary_accuracy"],
            "val_legal_term_preservation":   overall["legal_term_preservation"],
            "val_mean_answer_length_legal":   overall["mean_answer_length_legal"],
            "val_mean_answer_length_general": overall["mean_answer_length_general"],
            "val_f1_cuad":      cuad["f1"]   if cuad   else None,
            "val_em_cuad":      cuad["em"]   if cuad   else None,
            "val_f1_qasper_anchor": qasper["f1"] if qasper else None,
            "val_f1_squad_anchor":  squad["f1"]  if squad  else None,
            "legal_val_f1":     legal_f1,
            "lrs":              lrs_by_group(optimizer),
            "epoch_lr_multiplier": sched_info["multiplier"],
            "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        history.append(row)
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        payload = {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_f1_legal": legal_f1,
            "val_em_legal": cuad["em"] if cuad else None,
            "val_has_answer_accuracy": overall["has_answer_accuracy"],
            "val_legal_false_positive_rate": overall["legal_false_positive_rate"],
            "val_clause_boundary_accuracy":  overall["clause_boundary_accuracy"],
            "anchor_val_f1_qasper": qasper["f1"] if qasper else None,
            "anchor_val_f1_squad":  squad["f1"]  if squad  else None,
            "config": cfg,
        }
        _save(payload, os.path.join(CKPT_DIR, f"qa_stage3_epoch_{epoch}.pt"))
        _prune_epoch_ckpts(keep=3)

        if legal_f1 > best_legal_f1:
            best_legal_f1 = legal_f1
            _save(payload, os.path.join(CKPT_DIR, "qa_stage3_best.pt"))
            patience_left = args.patience
            print(f"[ckpt] new best legal_f1={best_legal_f1:.4f} -> qa_stage3_best.pt")
        else:
            patience_left -= 1
            print(f"[early-stop] no improvement; patience_left={patience_left}")
            if patience_left <= 0:
                print("[early-stop] exhausted — stopping")
                break

    _save(payload, os.path.join(CKPT_DIR, "qa_stage3_final.pt"))
    print(f"[done] best_legal_f1={best_legal_f1:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--stage2_ckpt", type=str,
                   default=os.path.join(CKPT_DIR, "qa_stage2_best.pt"))
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
