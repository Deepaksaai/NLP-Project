import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import os
import json

from model import build_model
from dataset import load_billsum, collate_fn


def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    """
    Runs one epoch of training.
    Returns average loss for the epoch.
    """

    model.train()
    epoch_loss = 0
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):

        src = batch['src'].to(device)
        trg = batch['trg'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # output: (trg_len, batch_size, vocab_size)
        output = model(src, trg, teacher_forcing_ratio=0.5)

        # Reshape for loss calculation
        # output: (trg_len * batch_size, vocab_size)
        # trg: (batch_size * trg_len)
        output_dim = output.shape[-1]

        # Skip first token (SOS) when calculating loss
        output = output[1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        # Calculate loss
        loss = criterion(output, trg)

        # Backward pass
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{total_batches} | "
                  f"Loss: {loss.item():.4f}")

    return epoch_loss / total_batches


def evaluate_epoch(model, dataloader, criterion, device):
    """
    Runs evaluation on validation/test set.
    Returns average loss.
    """

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:

            src = batch['src'].to(device)
            trg = batch['trg'].to(device)

            # No teacher forcing during evaluation
            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - (mins * 60))
    return mins, secs


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss,
                    vocab, path='checkpoint.pt'):
    """
    Saves model checkpoint so training can be resumed if interrupted.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'vocab_size': len(vocab)
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path='checkpoint.pt'):
    """
    Loads a saved checkpoint to resume training.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded — resuming from epoch {epoch + 1}")
    return model, optimizer, epoch


def train(
    num_epochs=10,
    batch_size=32,
    learning_rate=0.001,
    clip=1.0,
    checkpoint_path='checkpoint.pt',
    best_model_path='best_model.pt',
    resume=False
):
    """
    Main training function.
    """

    # -------------------------------------------------------
    # Setup
    # -------------------------------------------------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Load dataset and vocabulary
    print("\nLoading dataset...")
    train_dataset, test_dataset, vocab = load_billsum()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # -------------------------------------------------------
    # Build model
    # -------------------------------------------------------

    print("\nBuilding model...")
    model = build_model(
        vocab_size=len(vocab),
        device=device,
        embed_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3
    )
    model = model.to(device)

    # -------------------------------------------------------
    # Loss function and optimizer
    # -------------------------------------------------------

    # Ignore padding index in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Learning rate scheduler
    # Reduces LR by half if validation loss doesn't improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    # -------------------------------------------------------
    # Resume from checkpoint if requested
    # -------------------------------------------------------

    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(
            model, optimizer, checkpoint_path
        )
        start_epoch += 1

    # -------------------------------------------------------
    # Training loop
    # -------------------------------------------------------

    best_val_loss = float('inf')
    history = []

    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):

        start_time = time.time()

        # Train
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, clip, device
        )

        # Evaluate
        val_loss = evaluate_epoch(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        end_time = time.time()
        mins, secs = epoch_time(start_time, end_time)

        # Calculate perplexity
        # Perplexity = e^loss — lower is better
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Time: {mins}m {secs}s")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_ppl': train_ppl,
            'val_ppl': val_ppl
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved — Val Loss: {val_loss:.4f}")

        # Save checkpoint after every epoch
        save_checkpoint(
            model, optimizer, epoch,
            train_loss, val_loss, vocab,
            checkpoint_path
        )

        # Save training history
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")

    return model, history


# Run training
if __name__ == "__main__":

    model, history = train(
        num_epochs=10,
        batch_size=32,
        learning_rate=0.001,
        clip=1.0,
        resume=False
    )


