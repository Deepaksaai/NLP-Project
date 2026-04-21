"""
Fine-tune DeBERTa-v3-LARGE QA model on CUAD.

Changes from the base version:
  - batch_size default = 4 (large is 2x bigger per example)
  - freeze_layers default = 16 (out of 24)
  - unfreeze top 8 at epoch 3 (instead of top 4 for base)
  - mixed precision (fp16) enabled by default to save VRAM
  - gradient accumulation available via --grad_accum for effective batch sizes

Usage (from NLP-Project/ root):
    # Quick smoke test
    python -m QA_deberta.train --epochs 1 --batch_size 2

    # Full run on 16GB GPU (RTX 5060 Ti, 4080, 4090 etc.)
    python -m QA_deberta.train --epochs 5 --batch_size 4 --grad_accum 4 --lr 1e-5

    # If OOM, use smaller batch + more accumulation
    python -m QA_deberta.train --epochs 5 --batch_size 2 --grad_accum 8 --lr 1e-5

Effective batch size = batch_size * grad_accum. 
Recommended effective batch size for DeBERTa-large fine-tuning: 16-32.
"""

import os
import json
import time
import argparse
import re
import string

import torch
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

from QA_deberta.model import build_deberta_qa
from QA_deberta.loss import compute_loss, HAS_ANSWER_LOSS_WEIGHT

try:
    from transformers import get_linear_schedule_with_warmup
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


MAX_QUESTION_TOKENS = 64
MAX_CONTEXT_TOKENS  = 384


# ─── Dataset ───────────────────────────────────────────────────────────────────

class CuadDatasetDeberta(Dataset):
    """Wraps JSON format from download_cuad.py."""

    def __init__(self, json_path: str, tokenizer,
                 max_q: int = MAX_QUESTION_TOKENS,
                 max_ctx: int = MAX_CONTEXT_TOKENS):
        with open(json_path, encoding="utf-8") as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_len   = max_q + max_ctx + 3

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex       = self.examples[idx]
        question = ex["question"]
        context  = ex["context"]
        has_ans  = float(ex.get("has_answer", True))

        encoded = self.tokenizer(
            question, context,
            max_length=self.max_len,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids      = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        offset_mapping = encoded["offset_mapping"].squeeze(0)

        # Build context_mask from [SEP] positions
        sep_id = self.tokenizer.sep_token_id
        sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)[0].tolist()

        context_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if len(sep_positions) >= 2:
            context_mask[sep_positions[0] + 1: sep_positions[1]] = True

        start_pos = 0
        end_pos   = 0

        if has_ans and "answer_start" in ex and ex["answer_start"] >= 0:
            char_start = ex["answer_start"]
            char_end   = char_start + len(ex.get("answer_text", ""))

            for i, (tok_s, tok_e) in enumerate(offset_mapping.tolist()):
                if context_mask[i]:
                    if tok_s <= char_start < tok_e:
                        start_pos = i
                    if tok_s < char_end <= tok_e:
                        end_pos = i
                        break

            if end_pos < start_pos:
                end_pos = start_pos

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "context_mask":   context_mask,
            "start_pos":      torch.tensor(start_pos, dtype=torch.long),
            "end_pos":        torch.tensor(end_pos,   dtype=torch.long),
            "has_answer":     torch.tensor(has_ans,   dtype=torch.float),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ─── Metrics ───────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def quick_eval(model, loader, device, tokenizer):
    model.eval()
    total_f1 = 0.0
    total_em = 0
    n        = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        context_mask   = batch["context_mask"].to(device)
        start_pos      = batch["start_pos"]
        end_pos        = batch["end_pos"]

        s_logits, e_logits, _ = model(input_ids, attention_mask, context_mask)

        # Disallow position 0 (positives-only training)
        s_logits_ncls = s_logits.clone()
        e_logits_ncls = e_logits.clone()
        s_logits_ncls[:, 0] = float("-inf")
        e_logits_ncls[:, 0] = float("-inf")
        pred_starts = s_logits_ncls.argmax(dim=-1).cpu()
        pred_ends   = e_logits_ncls.argmax(dim=-1).cpu()

        for i in range(input_ids.size(0)):
            ps = pred_starts[i].item()
            pe = max(pred_ends[i].item(), ps)
            gs = start_pos[i].item()
            ge = end_pos[i].item()

            pred_ids = input_ids[i, ps:pe + 1].cpu().tolist()
            gold_ids = input_ids[i, gs:ge + 1].cpu().tolist()

            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            gold_text = tokenizer.decode(gold_ids, skip_special_tokens=True)

            total_f1 += token_f1(pred_text, gold_text)
            total_em += int(normalize(pred_text) == normalize(gold_text))
            n        += 1

    model.train()
    return {"f1": total_f1 / max(n, 1), "em": total_em / max(n, 1), "n": n}


# ─── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DeBERTa-LARGE-QA] device={device}")
    if device.type == "cuda":
        print(f"[DeBERTa-LARGE-QA] GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[DeBERTa-LARGE-QA] VRAM: {vram:.1f} GB")

    model, tokenizer = build_deberta_qa(
        freeze_layers=args.freeze_layers,
        dropout=args.dropout,
    )
    model.to(device)
    model.deberta.gradient_checkpointing_enable()
    print(f"[DeBERTa-LARGE-QA] gradient checkpointing: ON")
    print(f"[DeBERTa-LARGE-QA] mixed precision (fp16): {'ON' if args.fp16 else 'OFF'}")
    print(f"[DeBERTa-LARGE-QA] frozen layers: {args.freeze_layers}/24")
    print(f"[DeBERTa-LARGE-QA] batch={args.batch_size} × grad_accum={args.grad_accum} "
          f"= effective batch {args.batch_size * args.grad_accum}")

    train_f = os.path.join(args.data_dir, "train.json")
    val_f   = os.path.join(args.data_dir, "val.json")

    if not os.path.exists(train_f):
        raise FileNotFoundError(
            f"Train file not found: {train_f}\n"
            f"Run: python -m QA_deberta.download_cuad"
        )

    train_ds = CuadDatasetDeberta(train_f, tokenizer)
    val_ds   = CuadDatasetDeberta(val_f, tokenizer) if os.path.exists(val_f) else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)
    val_loader   = (DataLoader(val_ds, batch_size=args.batch_size * 2,
                               collate_fn=collate_fn) if val_ds else None)

    head_params = (list(model.start_head.parameters()) +
                   list(model.end_head.parameters()) +
                   list(model.has_answer_head.parameters()))
    deberta_params = [p for p in model.deberta.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": head_params,    "lr": args.lr * 5},
        {"params": deberta_params, "lr": args.lr},
    ], weight_decay=0.01)

    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * 0.06)   # 6% warmup — DeBERTa paper recommendation
    scheduler = (get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
                 if _HF_AVAILABLE else None)

    # Mixed precision scaler
    scaler = GradScaler("cuda", enabled=args.fp16)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "deberta_large_train_log.json")
    log      = []
    best_f1  = 0.0

    print(f"[DeBERTa-LARGE-QA] {len(train_ds)} train examples, {args.epochs} epochs")
    print(f"[DeBERTa-LARGE-QA] total steps: {total_steps}, warmup: {warmup_steps}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Progressive unfreezing: at epoch 3, unfreeze top 8 layers
        # (out of 24 total, keep bottom 8 frozen)
        if epoch == 3:
            print(f"[DeBERTa-LARGE-QA] Unfreezing top 8 encoder layers...")
            model.unfreeze_top_layers(keep_frozen=max(args.freeze_layers - 8, 0))
            deberta_params = [p for p in model.deberta.parameters() if p.requires_grad]
            optimizer.param_groups[1]["params"] = deberta_params
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[DeBERTa-LARGE-QA] trainable params now: {trainable/1e6:.1f}M")

        epoch_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            context_mask   = batch["context_mask"].to(device)
            start_pos      = batch["start_pos"].to(device)
            end_pos        = batch["end_pos"].to(device)
            has_answer     = batch["has_answer"].to(device)

            with autocast("cuda", enabled=args.fp16, dtype=torch.float16):
                s_logits, e_logits, ha_logit = model(input_ids, attention_mask, context_mask)
                loss, components = compute_loss(
                    s_logits, e_logits, ha_logit,
                    start_pos, end_pos, has_answer,
                    has_answer_weight=HAS_ANSWER_LOSS_WEIGHT,
                )
                loss = loss / args.grad_accum   # scale for gradient accumulation

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            # Optimizer step every grad_accum batches
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                avg = epoch_loss / (step + 1)
                print(f"  Epoch {epoch} step {step+1}/{len(train_loader)} "
                      f"loss={avg:.4f} "
                      f"span={components['span'].item():.4f} "
                      f"has_ans={components['has_answer'].item():.4f}")

        val_metrics = {}
        if val_loader:
            val_metrics = quick_eval(model, val_loader, device, tokenizer)
            print(f"[Epoch {epoch}] val F1={val_metrics['f1']:.4f}  "
                  f"EM={val_metrics['em']:.4f}  time={time.time()-t0:.0f}s")

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "val_f1":      best_f1,
                    "val_em":      val_metrics["em"],
                }, os.path.join(args.output_dir, "deberta_large_best.pt"))
                print(f"  ✓ Saved best (F1={best_f1:.4f})")

        log.append({
            "epoch":      epoch,
            "train_loss": epoch_loss / len(train_loader),
            "val_f1":     val_metrics.get("f1"),
            "val_em":     val_metrics.get("em"),
        })
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    torch.save({"epoch": args.epochs, "model_state": model.state_dict()},
               os.path.join(args.output_dir, "deberta_large_final.pt"))
    print(f"[DeBERTa-LARGE-QA] Done. Best val F1={best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--batch_size",    type=int,   default=4,
                        help="Per-step batch size. Drop to 2 if OOM.")
    parser.add_argument("--grad_accum",    type=int,   default=4,
                        help="Gradient accumulation steps. Effective batch = batch_size * grad_accum")
    parser.add_argument("--lr",            type=float, default=1e-5,
                        help="Learning rate. Lower than base (2e-5) because large is more sensitive.")
    parser.add_argument("--freeze_layers", type=int,   default=16,
                        help="Number of encoder layers to freeze (out of 24).")
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--fp16",          action="store_true", default=True,
                        help="Use mixed precision training (saves VRAM).")
    parser.add_argument("--no_fp16",       action="store_false", dest="fp16")
    parser.add_argument("--data_dir",      type=str,   default="QA_deberta/data")
    parser.add_argument("--output_dir",    type=str,   default="QA_deberta/checkpoints")
    args = parser.parse_args()
    train(args)
