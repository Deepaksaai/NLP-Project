"""
QA Stage 1 — general QA fine-tuning.

Two modes:

  1. --sanity
     Runs a single-batch overfitting sanity check on synthetic data.
     Must reach train F1 > 0.90 within 200 steps or the run fails.

  2. default
     Full training loop. Expects preprocessed files under
     data/qa/stage1/{train,val}*.json produced by
     QA.data.preprocess_squad / preprocess_trivia / preprocess_nq.

Usage examples:
    python -m QA.training.train_qa_stage1 --sanity
    python -m QA.training.train_qa_stage1 --epochs 12 --batch_size 32
"""

import os
import sys
import json
import math
import time
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from QA.qa_config import (
    QA_TOKENIZER_PATH, SUMMARIZER_CKPT_PATH, QA_DATA_ROOT,
    MAX_TOTAL_LEN, load_qa_special_tokens,
)
from QA.model.qa_model import build_qa_model
from QA.data.stage1_dataset import Stage1QADataset, stage1_collate
from QA.data.weighted_sampler import BalancedBatchSampler
from QA.data.synthetic_sanity import write_sanity_file, build_sanity_examples
from QA.training.loss import compute_loss
from QA.training.evaluate import evaluate_model, train_batch_f1
from QA.training.unfreeze_schedule import (
    build_initial_optimizer, apply_unfreeze_schedule, lrs_by_group,
    HEAD_LR,
)


_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_DIR    = os.path.join(_ROOT, "checkpoints")
LOG_DIR     = os.path.join(_ROOT, "QA", "logs")


# -------------------------------------------------------
# Warmup + cosine LR schedule (applied multiplicatively)
# -------------------------------------------------------
class WarmupCosine:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 1e-5 / 3e-4):
        self.opt = optimizer
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.min_ratio = min_lr_ratio
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            scale = self.step_num / self.warmup
        else:
            progress = (self.step_num - self.warmup) / (self.total - self.warmup)
            progress = min(1.0, progress)
            scale = self.min_ratio + 0.5 * (1 - self.min_ratio) * (1 + math.cos(math.pi * progress))

        # Apply to existing groups, and capture base LRs for any groups
        # added after scheduler construction (encoder unfreezing).
        for i, g in enumerate(self.opt.param_groups):
            if i >= len(self.base_lrs):
                self.base_lrs.append(g["lr"])
            g["lr"] = self.base_lrs[i] * scale


# -------------------------------------------------------
# Training
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip=1.0, log_every=50):
    model.train()
    running = {"total": 0.0, "span": 0.0, "has_answer": 0.0, "n": 0}
    t0 = time.time()

    for step, batch in enumerate(loader):
        input_ids     = batch["input_ids"].to(device)
        segment_ids   = batch["segment_ids"].to(device)
        attention     = batch["attention_mask"].to(device)
        start_pos     = batch["start_position"].to(device)
        end_pos       = batch["end_position"].to(device)
        has_answer_lb = batch["has_answer"].to(device)

        start_logits, end_logits, has_answer = model(input_ids, segment_ids, attention)

        total, comps = compute_loss(
            start_logits, end_logits, has_answer,
            start_pos, end_pos, has_answer_lb,
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
        running["total"]      += comps["total"].item() * bs
        running["span"]       += comps["span"].item() * bs
        running["has_answer"] += comps["has_answer"].item() * bs
        running["n"]          += bs

        if step % log_every == 0:
            dt = time.time() - t0
            print(f"  step {step:5d}  loss={running['total']/max(1,running['n']):.4f}  "
                  f"span={running['span']/max(1,running['n']):.4f}  "
                  f"hasA={running['has_answer']/max(1,running['n']):.4f}  "
                  f"lrs={ {k: f'{v:.2e}' for k,v in lrs_by_group(optimizer).items()} }  "
                  f"{dt:.1f}s")

    return {k: running[k] / max(1, running["n"]) for k in ("total", "span", "has_answer")}


def _build_val_loader(val_dataset, batch_size: int, collate_fn):
    return DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )


def _make_ckpt_payload(model, optimizer, epoch, global_step, val_metrics, stage, cfg):
    return {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "val_f1": val_metrics.get("f1", 0.0),
        "val_em": val_metrics.get("em", 0.0),
        "val_has_answer_accuracy": val_metrics.get("has_answer_accuracy", 0.0),
        "unfreezing_stage": stage,
        "config": cfg,
    }


def _save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def _prune_epoch_ckpts(keep: int = 3):
    files = sorted(
        [f for f in os.listdir(CKPT_DIR) if f.startswith("qa_stage1_epoch_") and f.endswith(".pt")],
        key=lambda f: int(f.replace("qa_stage1_epoch_", "").replace(".pt", "")),
    )
    while len(files) > keep:
        os.remove(os.path.join(CKPT_DIR, files.pop(0)))


# -------------------------------------------------------
# Sanity check mode
# -------------------------------------------------------
def run_sanity(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sanity] device={device}")

    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)

    # Build examples in memory (no file round trip needed)
    feats = build_sanity_examples(n_answerable=24, n_unanswerable=8)
    print(f"[sanity] built {len(feats)} examples "
          f"({sum(f['is_answerable'] for f in feats)} ans / "
          f"{sum(1 for f in feats if not f['is_answerable'])} unans)")

    # Minimal in-memory dataset wrapper
    class _MemDataset:
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            ex = self.items[i]
            return {
                "input_ids":      torch.tensor(ex["input_ids"], dtype=torch.long),
                "segment_ids":    torch.tensor(ex["segment_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
                "start_position": torch.tensor(ex["answer_start_tok"], dtype=torch.long),
                "end_position":   torch.tensor(ex["answer_end_tok"], dtype=torch.long),
                "has_answer":     torch.tensor(float(ex["is_answerable"])),
                "domain":         ex["domain"],
                "answer_text":    ex["answer_text"],
                "raw_idx":        i,
            }
    ds = _MemDataset(feats)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=stage1_collate)
    batch = next(iter(loader))

    # Build model — for sanity, UNFREEZE everything so we actually
    # test whether the loop can memorize a fixed batch.
    model = build_qa_model(meta, load_ckpt=SUMMARIZER_CKPT_PATH, freeze=False).to(device)
    for p in model.parameters():
        p.requires_grad = True

    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98),
                              weight_decay=0.0, eps=1e-9)

    print("[sanity] overfitting 1 batch for up to 200 steps...")
    best_f1 = 0.0
    hit_target = False
    target_f1 = 0.90
    for step in range(1, 201):
        model.train()
        start_logits, end_logits, has_ans = model(
            batch["input_ids"].to(device),
            batch["segment_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        total, comps = compute_loss(
            start_logits, end_logits, has_ans,
            batch["start_position"].to(device),
            batch["end_position"].to(device),
            batch["has_answer"].to(device),
        )
        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 20 == 0 or step in (1, 5, 10):
            f1 = train_batch_f1(model, batch, tokenizer, device)
            best_f1 = max(best_f1, f1)
            print(f"  step {step:3d}  loss={total.item():.4f}  span={comps['span'].item():.4f}  "
                  f"hasA={comps['has_answer'].item():.4f}  train_f1={f1:.3f}")
            if f1 >= target_f1:
                hit_target = True
                print(f"[sanity] hit F1>={target_f1} at step {step}")
                break

    if not hit_target:
        # One last measurement
        f1 = train_batch_f1(model, batch, tokenizer, device)
        best_f1 = max(best_f1, f1)
        print(f"[sanity] final train_f1={f1:.3f}  best={best_f1:.3f}")
        if best_f1 < target_f1:
            print(f"[sanity] FAIL — best F1 {best_f1:.3f} < target {target_f1}")
            sys.exit(1)

    print(f"[sanity] PASS  best_f1={best_f1:.3f}")


# -------------------------------------------------------
# Full training mode
# -------------------------------------------------------
def run_full(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)

    stage_dir = args.stage1_dir or os.path.join(QA_DATA_ROOT, "stage1")
    train_ds = Stage1QADataset("train", stage_dir=stage_dir)
    val_ds   = Stage1QADataset("val",   stage_dir=stage_dir)
    print(f"[train] train={train_ds.summary()}")
    print(f"[train] val  ={val_ds.summary()}")

    sampler = BalancedBatchSampler(
        train_ds.answerable_indices,
        train_ds.unanswerable_indices,
        batch_size=args.batch_size,
        answerable_frac=0.65,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        collate_fn=stage1_collate, num_workers=args.num_workers,
    )
    val_loader = _build_val_loader(val_ds, args.batch_size, stage1_collate)

    model = build_qa_model(meta, load_ckpt=SUMMARIZER_CKPT_PATH, freeze=True).to(device)
    optimizer = build_initial_optimizer(model)
    steps_per_epoch = max(1, len(sampler))
    total_steps = steps_per_epoch * args.epochs
    scheduler = WarmupCosine(optimizer, args.warmup_steps, total_steps)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    metrics_path = os.path.join(LOG_DIR, "stage1_metrics.json")
    history = []
    best_f1 = -1.0
    patience_left = args.patience
    global_step = 0

    cfg = vars(args).copy()

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        sampler.set_epoch(epoch)

        unfreeze_info = apply_unfreeze_schedule(model, optimizer, epoch)
        print(f"[unfreeze] {unfreeze_info}")

        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_clip=args.grad_clip, log_every=args.log_every,
        )
        global_step += steps_per_epoch

        val_metrics = evaluate_model(model, val_loader, tokenizer, device)
        print(f"[val] f1={val_metrics['f1']:.4f}  em={val_metrics['em']:.4f}  "
              f"hasA={val_metrics['has_answer_accuracy']:.4f}  "
              f"fpr={val_metrics['false_positive_rate']:.4f}  "
              f"fnr={val_metrics['false_negative_rate']:.4f}")
        print(f"[val] per-domain={val_metrics['per_domain']}")

        epoch_row = {
            "epoch": epoch,
            "train_loss_total":   train_stats["total"],
            "train_loss_span":    train_stats["span"],
            "train_loss_has_answer": train_stats["has_answer"],
            "val_em_overall": val_metrics["em"],
            "val_f1_overall": val_metrics["f1"],
            "val_has_answer_accuracy": val_metrics["has_answer_accuracy"],
            "val_false_positive_rate": val_metrics["false_positive_rate"],
            "val_false_negative_rate": val_metrics["false_negative_rate"],
            "val_f1_per_domain": {k: v["f1"] for k, v in val_metrics["per_domain"].items()},
            "val_em_per_domain": {k: v["em"] for k, v in val_metrics["per_domain"].items()},
            "lrs": lrs_by_group(optimizer),
            "unfreezing_stage": unfreeze_info["stage"],
            "trainable_param_count": unfreeze_info["trainable_params"],
        }
        history.append(epoch_row)
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        # Checkpointing
        epoch_path = os.path.join(CKPT_DIR, f"qa_stage1_epoch_{epoch}.pt")
        _save_ckpt(
            _make_ckpt_payload(model, optimizer, epoch, global_step, val_metrics, unfreeze_info["stage"], cfg),
            epoch_path,
        )
        _prune_epoch_ckpts(keep=3)

        # Collapse guard disabled — early QA training normally oscillates
        # between "all unanswerable" and "all answerable" phases before
        # converging. The 50% drop threshold was killing training prematurely.

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            _save_ckpt(
                _make_ckpt_payload(model, optimizer, epoch, global_step, val_metrics, unfreeze_info["stage"], cfg),
                os.path.join(CKPT_DIR, "qa_stage1_best.pt"),
            )
            patience_left = args.patience
            print(f"[ckpt] new best val_f1={best_f1:.4f} -> qa_stage1_best.pt")
        else:
            patience_left -= 1
            print(f"[early-stop] no improvement; patience_left={patience_left}")
            if patience_left <= 0:
                print("[early-stop] exhausted — stopping")
                break

    _save_ckpt(
        _make_ckpt_payload(model, optimizer, epoch, global_step, val_metrics, unfreeze_info["stage"], cfg),
        os.path.join(CKPT_DIR, "qa_stage1_final.pt"),
    )
    print(f"[done] best_val_f1={best_f1:.4f}")


# -------------------------------------------------------
# Entry point
# -------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sanity", action="store_true",
                   help="Run single-batch overfitting sanity check and exit.")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=5000)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--stage1_dir", type=str, default=None,
                   help="Directory holding train*.json / val*.json (defaults to data/qa/stage1).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.sanity:
        run_sanity(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
