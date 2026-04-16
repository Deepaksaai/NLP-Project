"""
Stage 3 — Legal Fine-tuning.

Loads stage2_best.pt and specializes it for legal summarization.

Key features:
  - Gradual unfreezing schedule (epoch 1: top decoder only, epoch 2: + top encoder
    + cross-attn, epoch 3+: everything)
  - Discriminative LR (lower layers get smaller LR after full unfreeze)
  - Mixed batches: 9 legal + 4 second-pass + 2 arXiv anchor + 1 CNN/DM anchor
  - <legal> token prepended to legal inputs, NOT to anchors
  - No sentence shuffle (legal order matters)
  - Per-type loss tracking for catastrophic forgetting detection
  - Tight early stopping (patience=3) — small dataset, easy to overfit

Usage:
    python -m training.train_stage3

Prerequisites:
    1. checkpoints/stage2_best.pt
    2. python -m data.preprocess_stage3
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import build_model
from data.preprocess import load_tokenizer, PAD_ID, LEGAL_ID, SOS_ID, EOS_ID
from data.stage3_dataset import build_stage3_data
from training.loss import LabelSmoothedCrossEntropy
from training.schedule import WarmupCosineSchedule


# -------------------------------------------------------
# Config
# -------------------------------------------------------
STAGE3_CONFIG = {
    # Optimizer
    "lr": 5e-5,
    "betas": (0.9, 0.98),
    "weight_decay": 0.01,
    "eps": 1e-9,

    # Schedule
    "warmup_steps": 500,
    "min_lr": 1e-6,

    # Training
    "epochs": 10,
    "batch_size": 16,
    "grad_clip": 1.0,
    "dropout": 0.3,

    # Loss
    "label_smoothing": 0.1,
    "ignore_index": 0,

    # Data
    "max_src": 400,
    "max_tgt": 128,
    "val_subsample": 2000,
    "arxiv_anchor_limit": 20000,
    "cnn_anchor_limit": 10000,
    "steps_per_epoch": None,    # None = use full chunk pass

    # Early stopping
    "patience": 3,

    # Checkpointing
    "save_dir": "checkpoints",
    "log_dir": "logs",
    "log_every": 100,
    "keep_last_n": 3,

    # Model (must match Stage 0/1/2)
    "vocab_size": 32000,
    "d_model": 384,
    "n_heads": 6,
    "n_encoder_layers": 6,
    "n_decoder_layers": 4,
    "d_ff": 1536,
    "max_seq_len": 512,
}


# -------------------------------------------------------
# Gradual unfreezing
# -------------------------------------------------------
def freeze_for_epoch(model, epoch):
    """
    Apply freezing schedule:
      Epoch 1: only top 2 decoder layers + decoder norm + output bias trainable
               (embedding tied with output, so embedding must also be unfrozen)
      Epoch 2: + top 2 encoder layers + bottom 2 decoder layers
      Epoch 3+: everything trainable

    Returns the unfreezing stage number (1, 2, or 3).
    """
    if epoch == 1:
        stage = 1
        # Freeze everything
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze top 2 decoder layers (last 2 in the list)
        for layer in model.decoder.layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        # Unfreeze decoder final norm
        for p in model.decoder.norm.parameters():
            p.requires_grad = True
        # Unfreeze output bias
        model.out_bias.requires_grad = True
        # Note: output projection is tied with embedding.weight, so we need
        # to allow the embedding to be updated too — otherwise the output
        # layer can't actually adapt.
        for p in model.embedding.parameters():
            p.requires_grad = True

    elif epoch == 2:
        stage = 2
        # Keep what was unfrozen in epoch 1, plus:
        # Top 2 encoder layers
        for layer in model.encoder.layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        # Bottom 2 decoder layers (these contain cross-attention as well)
        for layer in model.decoder.layers[:2]:
            for p in layer.parameters():
                p.requires_grad = True

    else:
        stage = 3
        # Unfreeze everything
        for p in model.parameters():
            p.requires_grad = True

    return stage


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------------------------------
# Discriminative LR optimizer
# -------------------------------------------------------
def build_discriminative_optimizer(model, base_lr, weight_decay=0.01,
                                    betas=(0.9, 0.98), eps=1e-9):
    """
    Build AdamW with per-layer LR multipliers:
      Encoder layers 1-2 → 0.1x
      Encoder layers 3-4 → 0.3x
      Encoder layers 5-6 → 0.6x
      Decoder layers 1-2 → 0.6x
      Decoder layers 3-4 → 1.0x
      Output (embedding + bias + norms) → 1.0x

    Each param group stores a `base_lr_mult` so the scheduler can
    multiply by it after computing the global cosine LR.
    """
    enc_layers = model.encoder.layers
    dec_layers = model.decoder.layers
    n_enc = len(enc_layers)  # 6
    n_dec = len(dec_layers)  # 4

    # Helper: multiplier for encoder layer index
    def enc_mult(i):
        if i < 2:
            return 0.1
        elif i < 4:
            return 0.3
        else:
            return 0.6

    def dec_mult(i):
        if i < 2:
            return 0.6
        else:
            return 1.0

    groups = []

    # Encoder layers grouped by depth
    for i, layer in enumerate(enc_layers):
        groups.append({
            "params": list(layer.parameters()),
            "lr": base_lr * enc_mult(i),
            "base_lr_mult": enc_mult(i),
            "name": f"encoder_layer_{i}",
        })

    # Decoder layers grouped by depth
    for i, layer in enumerate(dec_layers):
        groups.append({
            "params": list(layer.parameters()),
            "lr": base_lr * dec_mult(i),
            "base_lr_mult": dec_mult(i),
            "name": f"decoder_layer_{i}",
        })

    # Embedding + position encoding + final norms + output bias
    other_params = (
        list(model.embedding.parameters())
        + list(model.encoder.norm.parameters())
        + list(model.decoder.norm.parameters())
        + [model.out_bias]
    )
    groups.append({
        "params": other_params,
        "lr": base_lr * 1.0,
        "base_lr_mult": 1.0,
        "name": "embedding_and_output",
    })

    optimizer = torch.optim.AdamW(
        groups, betas=betas, weight_decay=weight_decay, eps=eps
    )
    return optimizer


class DiscriminativeWarmupCosine:
    """
    Like WarmupCosineSchedule but multiplies each group's LR by its
    stored base_lr_mult.
    """

    def __init__(self, optimizer, total_steps, warmup_steps=500,
                 max_lr=5e-5, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            base = self.max_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            base = self.min_lr + (self.max_lr - self.min_lr) * cosine

        for pg in self.optimizer.param_groups:
            mult = pg.get("base_lr_mult", 1.0)
            pg["lr"] = base * mult
        return base

    def current_lr(self):
        return self.optimizer.param_groups[-1]["lr"]


# -------------------------------------------------------
# Per-type loss helper
# -------------------------------------------------------
def compute_per_type_loss(logits, tgt_out, types, criterion):
    total_loss = criterion(logits, tgt_out)
    type_losses = {}
    for dtype in ["legal_chunk", "second_pass", "arxiv_anchor", "cnn_anchor"]:
        indices = [i for i, t in enumerate(types) if t == dtype]
        if indices:
            idx = torch.tensor(indices, device=logits.device)
            type_loss = criterion(logits[idx], tgt_out[idx])
            type_losses[dtype] = type_loss.item()
    return total_loss, type_losses


# -------------------------------------------------------
# Training
# -------------------------------------------------------
def train_stage3():
    config = STAGE3_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(config["save_dir"], exist_ok=True)
    os.makedirs(config["log_dir"], exist_ok=True)

    # ---- Tokenizer ----
    tokenizer = load_tokenizer("tokenizer/tokenizer.json")

    # ---- Data ----
    train_loader, val_loader = build_stage3_data(
        tokenizer,
        max_src=config["max_src"],
        max_tgt=config["max_tgt"],
        batch_size=config["batch_size"],
        steps_per_epoch=config["steps_per_epoch"],
        val_subsample=config["val_subsample"],
        arxiv_anchor_limit=config["arxiv_anchor_limit"],
        cnn_anchor_limit=config["cnn_anchor_limit"],
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

    # ---- Load Stage 2 checkpoint ----
    stage2_path = os.path.join(config["save_dir"], "stage2_best.pt")
    if not os.path.exists(stage2_path):
        print(f"ERROR: {stage2_path} not found. Run Stage 2 first.")
        sys.exit(1)
    print(f"\nLoading Stage 2 checkpoint: {stage2_path}")
    checkpoint = torch.load(stage2_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    stage2_val_loss = checkpoint.get("val_loss", float("inf"))
    print(f"  Stage 2 val loss: {stage2_val_loss:.4f}")
    print(f"  Stage 2 epoch:    {checkpoint.get('epoch', '?')}")

    # ---- Update dropout ----
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = config["dropout"]
    print(f"  Dropout updated to {config['dropout']}")

    # ---- Loss ----
    criterion = LabelSmoothedCrossEntropy(
        vocab_size=config["vocab_size"],
        smoothing=config["label_smoothing"],
        ignore_index=config["ignore_index"],
    )

    # ---- Optimizer (will be rebuilt at epoch 3 for discriminative LR) ----
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
    current_unfreeze_stage = 0
    is_discriminative = False

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        # ── Apply unfreezing schedule ──
        new_stage = freeze_for_epoch(model, epoch)
        if new_stage != current_unfreeze_stage:
            current_unfreeze_stage = new_stage
            n_train = count_trainable(model)
            print(f"\n>>> Unfreezing stage {new_stage}: "
                  f"{n_train:,} trainable parameters")

            # At epoch 3, switch to discriminative LR optimizer
            if new_stage == 3 and not is_discriminative:
                print(">>> Switching to discriminative LR optimizer")
                optimizer = build_discriminative_optimizer(
                    model,
                    base_lr=config["lr"],
                    weight_decay=config["weight_decay"],
                    betas=config["betas"],
                    eps=config["eps"],
                )
                # Recompute remaining steps for the new scheduler
                remaining_steps = (config["epochs"] - epoch + 1) * len(train_loader)
                scheduler = DiscriminativeWarmupCosine(
                    optimizer,
                    total_steps=remaining_steps,
                    warmup_steps=200,  # tiny re-warmup after optimizer swap
                    max_lr=config["lr"],
                    min_lr=config["min_lr"],
                )
                is_discriminative = True

        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        type_loss_sums = {
            "legal_chunk": 0.0, "second_pass": 0.0,
            "arxiv_anchor": 0.0, "cnn_anchor": 0.0
        }
        type_counts = {k: 0 for k in type_loss_sums}
        n_batches = len(train_loader)

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
                [p for p in model.parameters() if p.requires_grad],
                config["grad_clip"]
            ).item()
            optimizer.step()
            lr = scheduler.step()

            train_loss_sum += loss.item()
            for dtype, l in per_type.items():
                type_loss_sums[dtype] += l
                type_counts[dtype] += 1
            global_step += 1

            if (batch_idx + 1) % config["log_every"] == 0:
                print(f"  batch {batch_idx+1:5d}/{n_batches} | "
                      f"loss {loss.item():.4f} | "
                      f"grad {grad_norm:.3f} | "
                      f"lr {lr:.2e}")

        avg_train_loss = train_loss_sum / n_batches
        avg_type_losses = {
            k: type_loss_sums[k] / max(type_counts[k], 1)
            for k in type_loss_sums
        }

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device)
                tgt_in = batch["tgt_in"].to(device)
                tgt_out = batch["tgt_out"].to(device)
                logits = model(src, tgt_in)
                loss = criterion(logits, tgt_out)
                val_loss_sum += loss.item()
        avg_val_loss = val_loss_sum / len(val_loader)
        perplexity = math.exp(min(avg_val_loss, 20))
        elapsed = time.time() - t0

        # ── Sample generation ──
        with torch.no_grad():
            gen_ids = model.greedy_generate(
                sample_src.unsqueeze(0), max_len=80, sos_idx=SOS_ID, eos_idx=EOS_ID
            )
            sample_out = tokenizer.decode(gen_ids.squeeze(0).tolist())

        n_train = count_trainable(model)

        print(f"\nEpoch {epoch}/{config['epochs']} | time: {elapsed/60:.1f}min | "
              f"unfreeze stage {current_unfreeze_stage} | "
              f"trainable {n_train:,}")
        print(f"  train loss   : {avg_train_loss:.4f}")
        print(f"    legal_chunk  : {avg_type_losses['legal_chunk']:.4f}")
        print(f"    second_pass  : {avg_type_losses['second_pass']:.4f}")
        print(f"    arxiv_anchor : {avg_type_losses['arxiv_anchor']:.4f}")
        print(f"    cnn_anchor   : {avg_type_losses['cnn_anchor']:.4f}")
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
            "unfreezing_stage": current_unfreeze_stage,
            "trainable_params": n_train,
            "legal_chunk_loss": avg_type_losses["legal_chunk"],
            "second_pass_loss": avg_type_losses["second_pass"],
            "arxiv_anchor_loss": avg_type_losses["arxiv_anchor"],
            "cnn_anchor_loss": avg_type_losses["cnn_anchor"],
            "sample": sample_out[:300],
        }
        history.append(epoch_metrics)
        with open(os.path.join(config["log_dir"], "stage3_metrics.json"), "w") as f:
            json.dump(history, f, indent=2)

        # ── Checkpoint per epoch ──
        epoch_path = os.path.join(
            config["save_dir"], f"stage3_epoch_{epoch}.pt"
        )
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
            "global_step": global_step,
            "unfreezing_stage": current_unfreeze_stage,
            "config": config,
        }, epoch_path)
        saved_checkpoints.append(epoch_path)

        while len(saved_checkpoints) > config["keep_last_n"]:
            old = saved_checkpoints.pop(0)
            if os.path.exists(old):
                os.remove(old)

        # ── Best checkpoint ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "train_loss": avg_train_loss,
                "global_step": global_step,
                "unfreezing_stage": current_unfreeze_stage,
                "config": config,
            }, os.path.join(config["save_dir"], "stage3_best.pt"))
            print(f"  >> new best val loss — saved stage3_best.pt")
            if best_val_loss < stage2_val_loss:
                print(f"  >> BELOW Stage 2 val loss ({stage2_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  >> no improvement ({patience_counter}/{config['patience']}) "
                  f"best={best_val_loss:.4f}")
            if patience_counter >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final save
    torch.save(model.state_dict(),
               os.path.join(config["save_dir"], "stage3_final.pt"))

    print("\n" + "=" * 60)
    print("Stage 3 COMPLETE")
    print(f"  Best val loss   : {best_val_loss:.4f}")
    print(f"  Stage 2 val loss: {stage2_val_loss:.4f}")
    print(f"  Best model      : {config['save_dir']}/stage3_best.pt")
    print(f"  Final model     : {config['save_dir']}/stage3_final.pt")
    print(f"  Metrics         : {config['log_dir']}/stage3_metrics.json")
    print("=" * 60)

    return model, history


if __name__ == "__main__":
    train_stage3()
