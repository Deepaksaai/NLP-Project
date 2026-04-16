"""
Stage 1 dataset: CNN/DailyMail + XSum combined.

No chunking needed — both datasets have short-enough documents
to process whole within the 400-token source limit.

CNN/DailyMail teaches structured multi-sentence summarization.
XSum teaches aggressive single-sentence compression.
Together they give the model a broad foundation in summarization.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

from data.preprocess import tokenize_pair, load_tokenizer, PAD_ID


class Stage1Dataset(Dataset):
    """
    Combined CNN/DailyMail + XSum dataset.
    Each item is a pre-tokenized (input_ids, target_ids) pair.
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "target_ids": torch.tensor(ex["target_ids"], dtype=torch.long),
        }


def collate_batch(batch, pad_id=PAD_ID):
    """
    Pad src and tgt to the longest sequence in the batch (not global max).

    Returns:
        src:     (batch, src_len)  — encoder input
        tgt_in:  (batch, tgt_len)  — decoder input  (drop last token)
        tgt_out: (batch, tgt_len)  — decoder target (drop first token)
    """
    src_ids = [b["input_ids"] for b in batch]
    tgt_ids = [b["target_ids"] for b in batch]

    src_padded = pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)

    return {
        "src": src_padded,
        "tgt_in": tgt_padded[:, :-1],    # decoder input (drop last)
        "tgt_out": tgt_padded[:, 1:],     # decoder target (drop first)
    }


def build_stage1_data(tokenizer, max_src=400, max_tgt=128):
    """
    Load CNN/DailyMail + XSum, tokenize everything, return train/val datasets.

    This tokenizes upfront (not on-the-fly) so training epochs are fast.
    Memory cost: ~2-3 GB for ~491k tokenized pairs (just integer lists).
    """
    print("=" * 60)
    print("Stage 1: Building dataset (CNN/DailyMail + XSum)")
    print("=" * 60)

    # ---- CNN/DailyMail ----
    print("\nLoading CNN/DailyMail...")
    cnn_dm = load_dataset("cnn_dailymail", "3.0.0")

    print("  Tokenizing train split...")
    cnn_train = [
        tokenize_pair(ex["article"], ex["highlights"], tokenizer, max_src, max_tgt)
        for ex in cnn_dm["train"]
    ]
    print(f"  CNN/DM train: {len(cnn_train)} pairs")

    print("  Tokenizing val split...")
    cnn_val = [
        tokenize_pair(ex["article"], ex["highlights"], tokenizer, max_src, max_tgt)
        for ex in cnn_dm["validation"]
    ]
    print(f"  CNN/DM val: {len(cnn_val)} pairs")

    # Free memory
    del cnn_dm

    # ---- XSum ----
    print("\nLoading XSum...")
    xsum = load_dataset("EdinburghNLP/xsum")

    print("  Tokenizing train split...")
    xsum_train = [
        tokenize_pair(ex["document"], ex["summary"], tokenizer, max_src, max_tgt)
        for ex in xsum["train"]
    ]
    print(f"  XSum train: {len(xsum_train)} pairs")

    print("  Tokenizing val split...")
    xsum_val = [
        tokenize_pair(ex["document"], ex["summary"], tokenizer, max_src, max_tgt)
        for ex in xsum["validation"]
    ]
    print(f"  XSum val: {len(xsum_val)} pairs")

    del xsum

    # ---- Combine ----
    all_train = cnn_train + xsum_train
    all_val = cnn_val + xsum_val

    print(f"\nCombined — Train: {len(all_train)} | Val: {len(all_val)}")

    train_ds = Stage1Dataset(all_train)
    val_ds = Stage1Dataset(all_val)

    return train_ds, val_ds


def get_stage1_loaders(tokenizer, batch_size=32, max_src=400, max_tgt=128):
    """Convenience: build datasets + wrap in DataLoaders."""
    train_ds, val_ds = build_stage1_data(tokenizer, max_src, max_tgt)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader
