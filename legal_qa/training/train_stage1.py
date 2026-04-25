"""
Stage 1 Training Script: General QA Pretraining.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

# Add root project dir to path so we can import legal_qa
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from legal_qa.data.stage1_dataset import GeneralQADataset, WeightedBalancedSampler, collate_fn
from legal_qa.model.qa_model import LegalQAModel
from legal_qa.training.loss import QALoss
from legal_qa.training.schedule import get_linear_schedule_with_warmup
from legal_qa.training.discriminative_optimizer import get_discriminative_optimizer
from legal_qa.training.trainer import QATrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 General QA")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint instead of starting fresh")
    parser.add_argument("--dry_run", action="store_true", help="Run only 10 batches per epoch for testing")
    return parser.parse_args()

def main():
    args = get_args()

    # ---- Config ----
    config: Dict[str, Any] = {
        "model_name": "microsoft/deberta-v3-large",
        "max_seq_len": 512,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "num_epochs": 3,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "fp16": False,
        "stage": "stage1",
        "checkpoint_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"),
        "log_dir": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"),
        "early_stopping_patience": 2,
        "has_answer_weight": 0.5,
    }

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    best_checkpoint_path = os.path.join(config["checkpoint_dir"], "qa_stage1_best.pt")
    emergency_checkpoint_path = os.path.join(config["checkpoint_dir"], "qa_stage1_emergency.pt")
    latest_checkpoint_path = os.path.join(config["checkpoint_dir"], "qa_stage1_latest.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ---- Data loaders ----
    logger.info("Initializing Stage 1 Datasets...")
    # NOTE: To make the dry run truly fast, limit the dataset loading too
    max_ex = 50 if args.dry_run else None

    train_dataset = GeneralQADataset(split="train", max_squad=max_ex or 30000, max_triviaqa=max_ex or 20000, max_nq=max_ex)
    val_dataset = GeneralQADataset(split="validation", max_squad=max_ex or 5000, max_triviaqa=max_ex or 3000, max_nq=max_ex)

    train_sampler = WeightedBalancedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # ---- Model ----
    logger.info("Initializing LegalQAModel...")
    model = LegalQAModel(model_name=config["model_name"])
    model.to(device)

    # ---- Optimizer & Scheduler ----
    optimizer = get_discriminative_optimizer(model, stage=1)
    loss_fn = QALoss(has_answer_weight=config["has_answer_weight"])

    total_steps = (len(train_dataloader) // config["gradient_accumulation_steps"]) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    
    if args.dry_run:
        total_steps = 10 * config["num_epochs"]
        warmup_steps = 1
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler() if config["fp16"] and device == "cuda" else None

    # ---- Trainer ----
    trainer = QATrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config=config,
        stage_name=config["stage"],
        scaler=scaler,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        dry_run=args.dry_run,
    )

    start_epoch = 1
    best_f1 = 0.0
    patience_counter = 0

    if args.resume and os.path.exists(latest_checkpoint_path):
        logger.info("Resuming from latest checkpoint...")
        ckpt = trainer.load_checkpoint(latest_checkpoint_path)
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_val_f1"]

    # ---- Sanity Check ----
    logger.info("Running pre-training sanity check on one batch...")
    try:
        model.train()
        sanity_batch = next(iter(train_dataloader))
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(
                input_ids=sanity_batch["input_ids"].to(device),
                attention_mask=sanity_batch["attention_mask"].to(device),
                token_type_ids=sanity_batch["token_type_ids"].to(device),
                training=True,
            )
            sc_loss = loss_fn(
                start_logits=outputs["start_logits"],
                end_logits=outputs["end_logits"],
                has_answer_logit=outputs["has_answer_logit"],
                start_positions=sanity_batch["start_positions"].to(device),
                end_positions=sanity_batch["end_positions"].to(device),
                is_answerable=sanity_batch["is_answerable"].to(device),
            )["total_loss"]
        
        if torch.isnan(sc_loss):
            logger.error("Sanity Check: Loss is NaN — aborting.")
            sys.exit(1)
        if sc_loss.item() > 8.0:
            logger.warning(f"Sanity Check: Loss is high ({sc_loss.item():.2f}) — continuing anyway.")
        sc_loss.backward()
        optimizer.zero_grad()
        logger.info(f"Sanity check done. Loss={sc_loss.item():.4f}")
    except Exception as e:
        logger.error(f"Sanity Check exception: {e}")
        sys.exit(1)

    # ---- Training Loop ----
    logger.info(f"Starting Training for {config['num_epochs']} epochs...")
    try:
        for epoch in range(start_epoch, config["num_epochs"] + 1):
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate()

            trainer.log_epoch(epoch, train_metrics, val_metrics)
            logger.info(f"Epoch {epoch} | Train Loss: {train_metrics['train_loss_total']:.4f} | Val F1: {val_metrics['val_f1']:.4f} | Val EM: {val_metrics['val_em']:.4f}")

            # Save latest checkpoint
            trainer.save_checkpoint(epoch, best_f1, latest_checkpoint_path)

            stop, best_f1, patience_counter = trainer.early_stopping(val_metrics["val_f1"], best_f1, patience_counter, config["early_stopping_patience"])

            if patience_counter == 0:
                trainer.save_checkpoint(epoch, best_f1, best_checkpoint_path)

            if stop:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        trainer.save_checkpoint(epoch if 'epoch' in locals() else 0, best_f1, emergency_checkpoint_path)
        logger.info("Emergency checkpoint saved.")
        sys.exit(1)
        
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
