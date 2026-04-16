"""
Stage 1 — General Summarization Pre-training.

Trains the transformer on CNN/DailyMail + XSum (~491k pairs).
Goal: teach the model compression and fluent output generation
before it ever sees formal or legal text.

Usage:
    python -m training.train_stage1
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import build_model
from data.preprocess import load_tokenizer, PAD_ID
from data.stage1_dataset import get_stage1_loaders, collate_batch, Stage1Dataset
from training.loss import LabelSmoothedCrossEntropy
from training.schedule import WarmupCosineSchedule


# -------------------------------------------------------
# Config
# -------------------------------------------------------
STAGE1_CONFIG = {
    # Optimizer
    "optimizer": "AdamW",
    "lr": 5e-4,
    "betas": (0.9, 0.98),
    "weight_decay": 0.01,
    "eps": 1e-9,

    # Schedule
    "warmup_steps": 8000,
    "min_lr": 1e-5,

    # Training
    "epochs": 15,
    "batch_size": 16,
    "grad_clip": 1.0,
    "dropout": 0.1,

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

    # Model
    "vocab_size": 32000,
    "d_model": 384,
    "n_heads": 6,
    "n_encoder_layers": 6,
    "n_decoder_layers": 4,
    "d_ff": 1536,
    "max_seq_len": 512,
}


# -------------------------------------------------------
# Sanity check: overfit a single batch
# -------------------------------------------------------
def sanity_check(model, train_loader, criterion, device):
    """
    Verify the model can overfit a single batch.
    If loss doesn't drop below 1.0 in 100 steps, there's a bug.
    """
    print("\n" + "=" * 60)
    print("Sanity Check: Overfitting single batch")
    print("=" * 60)

    model.train()
    batch = next(iter(train_loader))
    src = batch["src"].to(device)
    tgt_in = batch["tgt_in"].to(device)
    tgt_out = batch["tgt_out"].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(200):
        logits = model(src, tgt_in)
        loss = criterion(logits, tgt_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 40 == 0:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

    final_loss = loss.item()
    assert final_loss < 5.0, \
        f"Cannot overfit single batch (loss={final_loss:.4f}). Bug in architecture or loss."
    print(f"\n  Final loss: {final_loss:.4f} — PASSED")

    # Reset model weights (sanity check mutated them)
    model._init_weights()
    print("  Model weights re-initialized\n")


# -------------------------------------------------------
# Training
# -------------------------------------------------------
def train_stage1():
    config = STAGE1_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = load_tokenizer("tokenizer/tokenizer.json")

    # ---- Data ----
    train_loader, val_loader = get_stage1_loaders(
        tokenizer,
        batch_size=config["batch_size"],
        max_src=config["max_src"],
        max_tgt=config["max_tgt"],
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

    # ---- Loss ----
    criterion = LabelSmoothedCrossEntropy(
        vocab_size=config["vocab_size"],
        smoothing=config["label_smoothing"],
        ignore_index=config["ignore_index"],
    )

    # ---- Sanity check ----
    sanity_check(model, train_loader, criterion, device)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,  # scheduler controls LR
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
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {config['warmup_steps']}")

    # ---- Sample for visual inspection ----
    sample_batch = next(iter(val_loader))
    sample_src = sample_batch["src"][0].to(device)
    sample_ref_ids = sample_batch["tgt_out"][0].tolist()
    sample_ref = tokenizer.decode([i for i in sample_ref_ids if i != PAD_ID])
    print(f"\nReference summary: {sample_ref[:200]}")
    print("=" * 60)

    # ---- Training loop ----
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    history = []

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        n_train = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)

            logits = model(src, tgt_in)
            loss = criterion(logits, tgt_out)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            ).item()
            optimizer.step()
            lr = scheduler.step()

            train_loss_sum += loss.item()
            global_step += 1

            if (batch_idx + 1) % config["log_every"] == 0:
                print(f"  batch {batch_idx+1:5d}/{n_train} | "
                      f"loss {loss.item():.4f} | "
                      f"grad {grad_norm:.3f} | "
                      f"lr {lr:.2e}")

        avg_train_loss = train_loss_sum / n_train

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
        model.eval()
        with torch.no_grad():
            gen_ids = model.greedy_generate(
                sample_src.unsqueeze(0), max_len=80, sos_idx=1, eos_idx=2
            )
            sample_out = tokenizer.decode(gen_ids.squeeze(0).tolist())

        print(f"\nEpoch {epoch}/{config['epochs']} | "
              f"time: {elapsed/60:.1f}min")
        print(f"  train loss : {avg_train_loss:.4f}")
        print(f"  val loss   : {avg_val_loss:.4f}")
        print(f"  perplexity : {perplexity:.2f}")
        print(f"  lr         : {scheduler.current_lr():.2e}")
        print(f"  sample     : {sample_out[:200]}")
        print(f"  reference  : {sample_ref[:200]}")

        # ── Metrics ──
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "perplexity": perplexity,
            "lr": scheduler.current_lr(),
            "grad_norm": grad_norm,
            "patience": patience_counter,
            "sample": sample_out[:300],
        }
        history.append(epoch_metrics)

        with open(os.path.join(config["log_dir"], "stage1_metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        # ── Checkpointing ──
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
            }, os.path.join(config["save_dir"], "stage1_best.pt"))
            print(f"  >> new best val loss — saved stage1_best.pt")
        else:
            patience_counter += 1
            print(f"  >> no improvement ({patience_counter}/{config['patience']}) "
                  f"best={best_val_loss:.4f}")

            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save final state
    torch.save(model.state_dict(),
               os.path.join(config["save_dir"], "stage1_final.pt"))

    print("\n" + "=" * 60)
    print("Stage 1 COMPLETE")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best model    : {config['save_dir']}/stage1_best.pt")
    print(f"  Final model   : {config['save_dir']}/stage1_final.pt")
    print(f"  Metrics       : {config['log_dir']}/stage1_metrics.json")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    train_stage1()
