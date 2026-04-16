"""
Stage 2 — Intermediate Domain Training (Long Formal Documents).

Continues from Stage 1 checkpoint on arXiv + PubMed aligned chunks,
with CNN/DailyMail anchor samples (20%) and second-pass examples (24%).

Key differences from Stage 1:
  - LR: 1e-4 (5x lower — preserve Stage 1 knowledge)
  - Dropout: 0.15 (slightly higher)
  - Warmup: 2000 steps (shorter — model already trained)
  - Mixed batches: 56% chunk + 24% second-pass + 20% anchor
  - Per-type loss tracking

Usage:
    python -m training.train_stage2

Prerequisites:
    1. checkpoints/stage1_best.pt must exist
    2. Run data preprocessing first:
       python -m data.preprocess_stage2
"""

import os
import sys
import math
import time
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import build_model
from data.preprocess import load_tokenizer, PAD_ID
from data.stage2_dataset import build_stage2_data
from training.loss import LabelSmoothedCrossEntropy
from training.schedule import WarmupCosineSchedule


# -------------------------------------------------------
# Config
# -------------------------------------------------------
STAGE2_CONFIG = {
    # Optimizer
    "lr": 1e-4,
    "betas": (0.9, 0.98),
    "weight_decay": 0.01,
    "eps": 1e-9,

    # Schedule
    "warmup_steps": 4000,
    "min_lr": 1e-6,

    # Training
    "epochs": 8,
    "batch_size": 16,
    "grad_clip": 1.0,
    "dropout": 0.15,

    # Loss
    "label_smoothing": 0.1,
    "ignore_index": 0,

    # Data
    "max_src": 400,
    "max_tgt": 128,

    # Early stopping
    "patience": 4,

    # Checkpointing
    "save_dir": "checkpoints",
    "log_dir": "logs",
    "log_every": 200,
    "keep_last_n": 3,

    # Model (must match Stage 0/1)
    "vocab_size": 32000,
    "d_model": 384,
    "n_heads": 6,
    "n_encoder_layers": 6,
    "n_decoder_layers": 4,
    "d_ff": 1536,
    "max_seq_len": 512,
}


# -------------------------------------------------------
# Per-type loss computation
# -------------------------------------------------------
def compute_per_type_loss(logits, tgt_out, types, criterion):
    """
    Compute loss per example type (chunk, second_pass, anchor).
    Returns total loss and per-type loss dict.
    """
    total_loss = criterion(logits, tgt_out)

    # Per-type tracking
    type_losses = {}
    batch_size = logits.size(0)
    seq_len = logits.size(1)
    vocab_size = logits.size(2)

    for dtype in ["chunk", "second_pass", "anchor"]:
        indices = [i for i, t in enumerate(types) if t == dtype]
        if indices:
            idx = torch.tensor(indices, device=logits.device)
            type_logits = logits[idx]
            type_targets = tgt_out[idx]
            type_loss = criterion(type_logits, type_targets)
            type_losses[dtype] = type_loss.item()

    return total_loss, type_losses


# -------------------------------------------------------
# Training
# -------------------------------------------------------
def train_stage2():
    config = STAGE2_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = load_tokenizer("tokenizer/tokenizer.json")

    # ---- Data ----
    train_loader, val_loader = build_stage2_data(
        tokenizer,
        max_src=config["max_src"],
        max_tgt=config["max_tgt"],
        batch_size=config["batch_size"],
    )

    # ---- Model ----
    model = build_model(
        device=device,
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_encoder_layers=config["n_encoder_layers"],
        n_decoder_layers=config["n_decoder_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        pad_idx=PAD_ID,
    )

    # ---- Load Stage 1 checkpoint ----
    stage1_path = os.path.join(config["save_dir"], "stage1_best.pt")
    if not os.path.exists(stage1_path):
        print(f"ERROR: {stage1_path} not found. Run Stage 1 first.")
        sys.exit(1)

    print(f"\nLoading Stage 1 checkpoint: {stage1_path}")
    checkpoint = torch.load(stage1_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    stage1_val_loss = checkpoint.get("val_loss", float("inf"))
    print(f"  Stage 1 val loss: {stage1_val_loss:.4f}")
    print(f"  Stage 1 epoch:    {checkpoint.get('epoch', '?')}")

    # ---- Update dropout ----
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = config["dropout"]
    print(f"  Dropout updated to {config['dropout']}")

    # ---- Loss ----
    criterion = LabelSmoothedCrossEntropy(
        vocab_size=config["vocab_size"],
        smoothing=config["label_smoothing"],
        ignore_index=config["ignore_index"],
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        betas=config["betas"],
        weight_decay=config["weight_decay"],
        eps=config["eps"],
    )

    total_steps = len(train_loader) * config["epochs"]
    scheduler = WarmupCosineSchedule(
        optimizer,
        total_steps=total_steps,
        warmup_steps=config["warmup_steps"],
        max_lr=config["lr"],
        min_lr=config["min_lr"],
    )
    print(f"\nTotal steps: {total_steps}, Warmup: {config['warmup_steps']}")

    # ---- Sample for inspection ----
    val_iter = iter(val_loader)
    sample_batch = next(val_iter)
    sample_src = sample_batch["src"][0].to(device)
    sample_ref_ids = sample_batch["tgt_out"][0].tolist()
    sample_ref = tokenizer.decode([i for i in sample_ref_ids if i != PAD_ID])
    print(f"\nReference: {sample_ref[:200]}")
    print("=" * 60)

    # ---- Training loop ----
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    history = []
    saved_checkpoints = []

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        type_loss_sums = {"chunk": 0.0, "second_pass": 0.0, "anchor": 0.0}
        type_counts = {"chunk": 0, "second_pass": 0, "anchor": 0}
        n_train = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            types = batch["types"]

            logits = model(src, tgt_in)
            loss, per_type = compute_per_type_loss(
                logits, tgt_out, types, criterion
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            ).item()
            optimizer.step()
            lr = scheduler.step()

            train_loss_sum += loss.item()
            for dtype, l in per_type.items():
                type_loss_sums[dtype] += l
                type_counts[dtype] += 1
            global_step += 1

            if (batch_idx + 1) % config["log_every"] == 0:
                print(f"  batch {batch_idx+1:5d}/{n_train} | "
                      f"loss {loss.item():.4f} | "
                      f"grad {grad_norm:.3f} | "
                      f"lr {lr:.2e}")

        avg_train_loss = train_loss_sum / n_train
        avg_type_losses = {
            k: type_loss_sums[k] / max(type_counts[k], 1)
            for k in type_loss_sums
        }

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        n_val = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device)
                tgt_in = batch["tgt_in"].to(device)
                tgt_out = batch["tgt_out"].to(device)
                logits = model(src, tgt_in)
                loss = criterion(logits, tgt_out)
                val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / n_val
        perplexity = math.exp(min(avg_val_loss, 20))
        elapsed = time.time() - t0

        # ── Sample generation ──
        with torch.no_grad():
            gen_ids = model.greedy_generate(
                sample_src.unsqueeze(0), max_len=80, sos_idx=1, eos_idx=2
            )
            sample_out = tokenizer.decode(gen_ids.squeeze(0).tolist())

        # ── Logging ──
        print(f"\nEpoch {epoch}/{config['epochs']} | time: {elapsed/60:.1f}min")
        print(f"  train loss   : {avg_train_loss:.4f}")
        print(f"    chunk      : {avg_type_losses['chunk']:.4f}")
        print(f"    second_pass: {avg_type_losses['second_pass']:.4f}")
        print(f"    anchor     : {avg_type_losses['anchor']:.4f}")
        print(f"  val loss     : {avg_val_loss:.4f}")
        print(f"  perplexity   : {perplexity:.2f}")
        print(f"  lr           : {scheduler.current_lr():.2e}")
        print(f"  sample       : {sample_out[:200]}")
        print(f"  reference    : {sample_ref[:200]}")

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "perplexity": perplexity,
            "lr": scheduler.current_lr(),
            "grad_norm": grad_norm,
            "patience": patience_counter,
            "chunk_loss": avg_type_losses["chunk"],
            "second_pass_loss": avg_type_losses["second_pass"],
            "anchor_loss": avg_type_losses["anchor"],
            "sample": sample_out[:300],
        }
        history.append(epoch_metrics)

        with open(os.path.join(config["log_dir"], "stage2_metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        # ── Checkpointing ──
        # Save every epoch (keep last N)
        epoch_path = os.path.join(
            config["save_dir"], f"stage2_epoch_{epoch}.pt"
        )
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "global_step": global_step,
            "config": config,
        }, epoch_path)
        saved_checkpoints.append(epoch_path)

        # Clean old checkpoints
        while len(saved_checkpoints) > config["keep_last_n"]:
            old = saved_checkpoints.pop(0)
            if os.path.exists(old):
                os.remove(old)

        # Best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "global_step": global_step,
                "config": config,
            }, os.path.join(config["save_dir"], "stage2_best.pt"))
            print(f"  >> new best val loss — saved stage2_best.pt")

            if best_val_loss < stage1_val_loss:
                print(f"  >> BELOW Stage 1 val loss ({stage1_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  >> no improvement ({patience_counter}/{config['patience']}) "
                  f"best={best_val_loss:.4f}")

            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save final
    torch.save(model.state_dict(),
               os.path.join(config["save_dir"], "stage2_final.pt"))

    print("\n" + "=" * 60)
    print("Stage 2 COMPLETE")
    print(f"  Best val loss   : {best_val_loss:.4f}")
    print(f"  Stage 1 val loss: {stage1_val_loss:.4f}")
    print(f"  Improvement     : {stage1_val_loss - best_val_loss:.4f}")
    print(f"  Best model      : {config['save_dir']}/stage2_best.pt")
    print(f"  Final model     : {config['save_dir']}/stage2_final.pt")
    print(f"  Metrics         : {config['log_dir']}/stage2_metrics.json")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    train_stage2()
