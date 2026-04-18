"""
QA Stage 2 — long-document fine-tuning on QuALITY + QASPER with a
SQuAD anchor.

Loads qa_stage1_best.pt, unfreezes everything from epoch 1, runs with
Stage-2 discriminative LRs, tracks per-source losses, and adds the
chunk-retrieval metric to validation.

Usage:
    python -m QA.data.preprocess_quality
    python -m QA.data.preprocess_qasper
    python -m QA.training.train_qa_stage2 --epochs 8 --batch_size 16
"""

import os
import sys
import json
import math
import time
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from QA.qa_config import (
    QA_TOKENIZER_PATH, SUMMARIZER_CKPT_PATH, QA_DATA_ROOT,
    load_qa_special_tokens,
)
from QA.model.qa_model import build_qa_model
from QA.data.stage2_dataset import Stage2QADataset, stage2_collate
from QA.data.mixed_sampler import MixedBatchSampler
from QA.training.loss import compute_loss
from QA.training.evaluate import evaluate_model
from QA.training.chunk_retrieval import chunk_retrieval_accuracy
from QA.training.unfreeze_schedule_stage2 import (
    build_stage2_optimizer, STAGE2_LRS,
)
from QA.training.unfreeze_schedule import lrs_by_group


_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR = os.path.join(_ROOT, "checkpoints")
LOG_DIR  = os.path.join(_ROOT, "QA", "logs")


# Stage-2 mix: 40% quality / 30% qasper / 30% squad
DEFAULT_MIX = {"quality": 0.4, "qasper": 0.3, "squad": 0.3}


# -------------------------------------------------------
# Warmup + cosine (1000-step warmup per spec)
# -------------------------------------------------------
class WarmupCosine:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.05):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.min_ratio = min_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            scale = self.step_num / self.warmup
        else:
            progress = min(1.0, (self.step_num - self.warmup) / (self.total - self.warmup))
            scale = self.min_ratio + 0.5 * (1 - self.min_ratio) * (1 + math.cos(math.pi * progress))
        for i, g in enumerate(self.opt.param_groups):
            g["lr"] = self.base_lrs[i] * scale


def _load_stage1_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[stage1-load] missing={len(missing)}  unexpected={len(unexpected)}")
    return ckpt


def _save_ckpt(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def _prune_epoch_ckpts(keep: int = 3):
    files = sorted(
        [f for f in os.listdir(CKPT_DIR) if f.startswith("qa_stage2_epoch_") and f.endswith(".pt")],
        key=lambda f: int(f.replace("qa_stage2_epoch_", "").replace(".pt", "")),
    )
    while len(files) > keep:
        os.remove(os.path.join(CKPT_DIR, files.pop(0)))


# -------------------------------------------------------
# Training
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip=1.0, log_every=50):
    model.train()
    running = {
        "total":      {"sum": 0.0, "n": 0},
        "span":       {"sum": 0.0, "n": 0},
        "has_answer": {"sum": 0.0, "n": 0},
    }
    per_source = defaultdict(lambda: {"sum": 0.0, "n": 0})
    t0 = time.time()

    for step, batch in enumerate(loader):
        input_ids     = batch["input_ids"].to(device)
        segment_ids   = batch["segment_ids"].to(device)
        attention     = batch["attention_mask"].to(device)
        start_pos     = batch["start_position"].to(device)
        end_pos       = batch["end_position"].to(device)
        has_lab       = batch["has_answer"].to(device)
        sources       = batch["sources"]

        start_logits, end_logits, has_answer = model(input_ids, segment_ids, attention)

        total, comps = compute_loss(
            start_logits, end_logits, has_answer,
            start_pos, end_pos, has_lab,
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
        running["total"]["sum"]      += comps["total"].item() * bs
        running["total"]["n"]        += bs
        running["span"]["sum"]       += comps["span"].item() * bs
        running["span"]["n"]         += bs
        running["has_answer"]["sum"] += comps["has_answer"].item() * bs
        running["has_answer"]["n"]   += bs

        # Per-source loss tracking: approximate by weighting the batch
        # loss by each source's share in the batch. We recompute a
        # per-example span loss via the same CE for a clean attribution.
        with torch.no_grad():
            ex_start = torch.nn.functional.cross_entropy(
                start_logits, start_pos, reduction="none")
            ex_end   = torch.nn.functional.cross_entropy(
                end_logits, end_pos, reduction="none")
            ex_span  = 0.5 * (ex_start + ex_end)
            for b in range(bs):
                s = sources[b]
                per_source[s]["sum"] += ex_span[b].item()
                per_source[s]["n"]   += 1

        if step % log_every == 0:
            dt = time.time() - t0
            src_losses = {k: round(v["sum"] / max(1, v["n"]), 3) for k, v in per_source.items()}
            lr_str = {k: f"{v:.2e}" for k, v in lrs_by_group(optimizer).items()}
            avg_total = running["total"]["sum"] / max(1, running["total"]["n"])
            avg_span  = running["span"]["sum"]  / max(1, running["span"]["n"])
            avg_hasA  = running["has_answer"]["sum"] / max(1, running["has_answer"]["n"])
            print(f"  step {step:5d}  loss={avg_total:.4f}  span={avg_span:.4f}  "
                  f"hasA={avg_hasA:.4f}  src={src_losses}  lrs={lr_str}  {dt:.1f}s")

    def avg(rec): return rec["sum"] / max(1, rec["n"])
    return {
        "total":      avg(running["total"]),
        "span":       avg(running["span"]),
        "has_answer": avg(running["has_answer"]),
        "by_source":  {k: avg(v) for k, v in per_source.items()},
    }


# -------------------------------------------------------
# Evaluation split: overall val + SQuAD anchor subset
# -------------------------------------------------------
@torch.no_grad()
def evaluate_all(model, val_ds, tokenizer, device, batch_size=16):
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=stage2_collate)
    overall = evaluate_model(model, val_loader, tokenizer, device)

    # SQuAD anchor: evaluate only the SQuAD-sourced val examples
    anchor_indices = val_ds.indices_by_source.get("squad", [])
    anchor_f1 = None
    if anchor_indices:
        class _Sub(torch.utils.data.Dataset):
            def __init__(self, parent, idxs): self.parent = parent; self.idxs = idxs
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i): return self.parent[self.idxs[i]]
        sub = _Sub(val_ds, anchor_indices)
        sub_loader = DataLoader(sub, batch_size=batch_size, shuffle=False,
                                collate_fn=stage2_collate)
        anchor_metrics = evaluate_model(model, sub_loader, tokenizer, device)
        anchor_f1 = anchor_metrics["f1"]

    # Chunk retrieval metric over the whole val set
    retrieval = chunk_retrieval_accuracy(model, val_ds, device, batch_size=batch_size)

    return overall, anchor_f1, retrieval


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage2] device={device}")

    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)

    stage_dir = args.stage2_dir or os.path.join(QA_DATA_ROOT, "stage2")
    train_ds = Stage2QADataset("train", stage_dir=stage_dir)
    val_ds   = Stage2QADataset("val",   stage_dir=stage_dir)
    print(f"[stage2] train={train_ds.summary()}")
    print(f"[stage2] val  ={val_ds.summary()}")

    # Verify all three mix pools exist
    for k in DEFAULT_MIX:
        if not train_ds.indices_by_source.get(k):
            raise RuntimeError(
                f"[stage2] training pool {k!r} is empty; preprocess it before launching."
            )

    sampler = MixedBatchSampler(
        train_ds.indices_by_source,
        mix=DEFAULT_MIX,
        batch_size=args.batch_size,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        collate_fn=stage2_collate, num_workers=args.num_workers,
    )

    # Build fresh model, then load Stage-1 best checkpoint weights
    model = build_qa_model(meta, load_ckpt=None, freeze=False).to(device)
    if not os.path.exists(args.stage1_ckpt):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {args.stage1_ckpt}")
    _load_stage1_checkpoint(model, args.stage1_ckpt, device)

    optimizer = build_stage2_optimizer(model)
    steps_per_epoch = max(1, len(sampler))
    total_steps = steps_per_epoch * args.epochs
    scheduler = WarmupCosine(optimizer, args.warmup_steps, total_steps)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    metrics_path = os.path.join(LOG_DIR, "stage2_metrics.json")
    history = []
    best_f1 = -1.0
    patience_left = args.patience
    global_step = 0

    cfg = vars(args).copy()

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Stage-2 Epoch {epoch}/{args.epochs} =====")
        sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip, log_every=args.log_every,
        )
        global_step += steps_per_epoch

        overall, anchor_f1, retrieval = evaluate_all(
            model, val_ds, tokenizer, device, batch_size=args.batch_size,
        )

        print(f"[val] f1={overall['f1']:.4f}  em={overall['em']:.4f}  "
              f"hasA={overall['has_answer_accuracy']:.4f}  "
              f"fpr={overall['false_positive_rate']:.4f}")
        print(f"[val] per-domain={overall['per_domain']}")
        print(f"[val] anchor_squad_f1={anchor_f1}")
        print(f"[val] chunk-retrieval top1={retrieval['top1']:.4f}  "
              f"top3={retrieval['top3']:.4f}  "
              f"(n={retrieval['n_questions']})")

        epoch_row = {
            "epoch": epoch,
            "train_loss_total":      train_stats["total"],
            "train_loss_span":       train_stats["span"],
            "train_loss_has_answer": train_stats["has_answer"],
            "train_loss_by_source":  train_stats["by_source"],
            "val_f1_overall":        overall["f1"],
            "val_em_overall":        overall["em"],
            "val_has_answer_accuracy": overall["has_answer_accuracy"],
            "val_false_positive_rate": overall["false_positive_rate"],
            "val_false_negative_rate": overall["false_negative_rate"],
            "val_f1_per_domain":     {k: v["f1"] for k, v in overall["per_domain"].items()},
            "val_em_per_domain":     {k: v["em"] for k, v in overall["per_domain"].items()},
            "anchor_val_f1":         anchor_f1,
            "val_chunk_retrieval_top1": retrieval["top1"],
            "val_chunk_retrieval_top3": retrieval["top3"],
            "val_chunk_retrieval_by_source": retrieval["by_source"],
            "lrs":                   lrs_by_group(optimizer),
            "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        history.append(epoch_row)
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        # Checkpoint
        payload = {
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_f1": overall["f1"],
            "val_em": overall["em"],
            "val_has_answer_accuracy": overall["has_answer_accuracy"],
            "val_chunk_retrieval_top1": retrieval["top1"],
            "anchor_val_f1": anchor_f1,
            "config": cfg,
        }
        _save_ckpt(payload, os.path.join(CKPT_DIR, f"qa_stage2_epoch_{epoch}.pt"))
        _prune_epoch_ckpts(keep=3)

        if overall["f1"] > best_f1:
            best_f1 = overall["f1"]
            _save_ckpt(payload, os.path.join(CKPT_DIR, "qa_stage2_best.pt"))
            patience_left = args.patience
            print(f"[ckpt] new best val_f1={best_f1:.4f} -> qa_stage2_best.pt")
        else:
            patience_left -= 1
            print(f"[early-stop] no improvement; patience_left={patience_left}")
            if patience_left <= 0:
                print("[early-stop] exhausted — stopping")
                break

    _save_ckpt(payload, os.path.join(CKPT_DIR, "qa_stage2_final.pt"))
    print(f"[done] best_val_f1={best_f1:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--stage2_dir", type=str, default=None)
    p.add_argument("--stage1_ckpt", type=str,
                   default=os.path.join(CKPT_DIR, "qa_stage1_best.pt"))
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
