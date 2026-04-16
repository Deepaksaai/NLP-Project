"""
Stage 2 dataset: Mixed batches from arXiv/PubMed aligned chunks,
second-pass examples, and CNN/DailyMail anchor samples.

Batch composition per step:
  56% arXiv/PubMed aligned chunks
  24% second-pass examples (summaries of summaries)
  20% CNN/DailyMail anchor samples (prevent forgetting)

Sentence shuffle augmentation applied to arXiv/PubMed chunks only
at 20% probability (reduced from Stage 1's 30% because scientific
papers have stronger logical flow).
"""

import os
import re
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

from data.preprocess import tokenize_pair, load_tokenizer, PAD_ID


# -------------------------------------------------------
# Sentence shuffle augmentation
# -------------------------------------------------------
def shuffle_sentences(text, prob=0.2):
    """Randomly shuffle sentences with given probability."""
    if random.random() > prob:
        return text
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    if len(sentences) < 3:
        return text
    random.shuffle(sentences)
    return " ".join(sentences)


# -------------------------------------------------------
# Load aligned data from preprocessed JSONL files
# -------------------------------------------------------
def load_aligned_jsonl(path):
    """Load JSONL file, return list of dicts."""
    examples = []
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping")
        return examples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} examples from {os.path.basename(path)}")
    return examples


# -------------------------------------------------------
# Dataset classes
# -------------------------------------------------------
class AlignedChunkDataset(Dataset):
    """arXiv/PubMed aligned chunk-target pairs with sentence shuffle."""

    def __init__(self, examples, tokenizer, max_src=400, max_tgt=128,
                 augment=True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt
        self.augment = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["input"]
        if self.augment:
            text = shuffle_sentences(text, prob=0.2)
        pair = tokenize_pair(text, ex["target"], self.tokenizer,
                             self.max_src, self.max_tgt)
        return {
            "input_ids": torch.tensor(pair["input_ids"], dtype=torch.long),
            "target_ids": torch.tensor(pair["target_ids"], dtype=torch.long),
            "type": "chunk",
        }


class SecondPassDataset(Dataset):
    """Second-pass examples: concatenated chunk summaries → full abstract."""

    def __init__(self, examples, tokenizer, max_src=400, max_tgt=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        pair = tokenize_pair(ex["input"], ex["target"], self.tokenizer,
                             self.max_src, self.max_tgt)
        return {
            "input_ids": torch.tensor(pair["input_ids"], dtype=torch.long),
            "target_ids": torch.tensor(pair["target_ids"], dtype=torch.long),
            "type": "second_pass",
        }


class AnchorDataset(Dataset):
    """CNN/DailyMail anchor samples to prevent forgetting."""

    def __init__(self, examples, tokenizer, max_src=400, max_tgt=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        pair = tokenize_pair(ex["text"], ex["summary"], self.tokenizer,
                             self.max_src, self.max_tgt)
        return {
            "input_ids": torch.tensor(pair["input_ids"], dtype=torch.long),
            "target_ids": torch.tensor(pair["target_ids"], dtype=torch.long),
            "type": "anchor",
        }


# -------------------------------------------------------
# Mixed batch sampler
# -------------------------------------------------------
class MixedBatchSampler(Sampler):
    """
    Yields batches with the target composition:
      56% chunk, 24% second_pass, 20% anchor

    Each batch samples independently from each source dataset
    with replacement when a source is exhausted.
    """

    def __init__(self, chunk_size, second_pass_size, anchor_size,
                 batch_size=16, steps_per_epoch=None):
        self.chunk_size = chunk_size
        self.second_pass_size = second_pass_size
        self.anchor_size = anchor_size
        self.batch_size = batch_size

        # Compute per-batch counts
        self.n_chunk = max(1, round(batch_size * 0.56))
        self.n_second = max(1, round(batch_size * 0.24))
        self.n_anchor = batch_size - self.n_chunk - self.n_second

        if steps_per_epoch is None:
            # Enough steps to see all chunk data once
            self.steps_per_epoch = chunk_size // self.n_chunk
        else:
            self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        chunk_indices = list(range(self.chunk_size))
        sp_indices = list(range(self.second_pass_size))
        anchor_indices = list(range(self.anchor_size))

        random.shuffle(chunk_indices)
        random.shuffle(sp_indices)
        random.shuffle(anchor_indices)

        ci, si, ai = 0, 0, 0

        for _ in range(self.steps_per_epoch):
            batch = []

            # Chunk samples
            for _ in range(self.n_chunk):
                if ci >= len(chunk_indices):
                    random.shuffle(chunk_indices)
                    ci = 0
                batch.append(("chunk", chunk_indices[ci]))
                ci += 1

            # Second-pass samples
            for _ in range(self.n_second):
                if si >= len(sp_indices):
                    random.shuffle(sp_indices)
                    si = 0
                batch.append(("second_pass", sp_indices[si]))
                si += 1

            # Anchor samples
            for _ in range(self.n_anchor):
                if ai >= len(anchor_indices):
                    random.shuffle(anchor_indices)
                    ai = 0
                batch.append(("anchor", anchor_indices[ai]))
                ai += 1

            yield batch

    def __len__(self):
        return self.steps_per_epoch


# -------------------------------------------------------
# Mixed dataset wrapper
# -------------------------------------------------------
class MixedStage2Dataset(Dataset):
    """
    PyTorch Dataset wrapping three sub-datasets, indexed by (type, idx) tuples
    from MixedBatchSampler. The default DataLoader fetcher will call
    self[("chunk", 5)] which returns the corresponding example.
    """

    def __init__(self, chunk_ds, second_pass_ds, anchor_ds):
        self.datasets = {
            "chunk": chunk_ds,
            "second_pass": second_pass_ds,
            "anchor": anchor_ds,
        }

    def __len__(self):
        # Not actually used because we provide batch_sampler explicitly
        return (
            len(self.datasets["chunk"])
            + len(self.datasets["second_pass"])
            + len(self.datasets["anchor"])
        )

    def __getitem__(self, key):
        """Key is a (type, index) tuple from MixedBatchSampler."""
        dtype, idx = key
        return self.datasets[dtype][idx]


def mixed_collate_fn(items, pad_id=PAD_ID):
    """Collate already-fetched items into a padded batch."""
    src_ids = [item["input_ids"] for item in items]
    tgt_ids = [item["target_ids"] for item in items]

    src_padded = pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)

    types = [item["type"] for item in items]

    return {
        "src": src_padded,
        "tgt_in": tgt_padded[:, :-1],
        "tgt_out": tgt_padded[:, 1:],
        "types": types,
    }


# -------------------------------------------------------
# Build everything
# -------------------------------------------------------
def build_stage2_data(tokenizer, max_src=400, max_tgt=128, batch_size=16,
                      steps_per_epoch=60000, val_subsample=5000):
    """
    Build Stage 2 training data:
      1. Load aligned arXiv/PubMed chunks from disk
      2. Load second-pass examples from disk
      3. Load CNN/DM anchor data from HuggingFace
      4. Create mixed batch sampler with capped steps per epoch
      5. Return DataLoader + validation DataLoader (subsampled)

    The aligned data must be preprocessed first via:
        python -m data.preprocess_stage2

    Args:
        steps_per_epoch: cap on training steps per epoch.
            With 60k steps × 9 chunks/batch = 540k chunks sampled per epoch
            (~24% of 2.28M total). Across 8 epochs with reshuffling,
            the model effectively sees nearly all available data.
        val_subsample: limit validation set to this many examples.
            The full first-chunk val set has 318k entries which would
            take hours per epoch to evaluate. 5k gives stable loss
            tracking with negligible time overhead.
    """
    print("=" * 60)
    print("Stage 2: Building dataset")
    print("=" * 60)

    aligned_dir = "data/stage2_aligned"

    # ---- Load aligned chunks ----
    print("\nLoading aligned chunk data...")
    arxiv_chunks = load_aligned_jsonl(os.path.join(aligned_dir, "arxiv_aligned.jsonl"))
    pubmed_chunks = load_aligned_jsonl(os.path.join(aligned_dir, "pubmed_aligned.jsonl"))
    all_chunks = arxiv_chunks + pubmed_chunks
    print(f"  Total aligned chunks: {len(all_chunks)}")

    # ---- Load second-pass ----
    print("\nLoading second-pass examples...")
    second_pass = load_aligned_jsonl(os.path.join(aligned_dir, "second_pass.jsonl"))

    # ---- Load CNN/DM anchor ----
    print("\nLoading CNN/DailyMail anchor data...")
    cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="train")
    # Subsample to ~20% of chunk count for balanced mixing
    anchor_target = len(all_chunks) // 4
    anchor_data = [
        {"text": ex["article"], "summary": ex["highlights"]}
        for i, ex in enumerate(cnn_dm)
        if i < anchor_target
    ]
    del cnn_dm
    print(f"  CNN/DM anchor samples: {len(anchor_data)}")

    # ---- Build datasets ----
    chunk_ds = AlignedChunkDataset(all_chunks, tokenizer, max_src, max_tgt,
                                   augment=True)
    sp_ds = SecondPassDataset(second_pass, tokenizer, max_src, max_tgt)
    anchor_ds = AnchorDataset(anchor_data, tokenizer, max_src, max_tgt)

    mixed = MixedStage2Dataset(chunk_ds, sp_ds, anchor_ds)

    # ---- Mixed batch sampler ----
    sampler = MixedBatchSampler(
        chunk_size=len(chunk_ds),
        second_pass_size=len(sp_ds),
        anchor_size=len(anchor_ds),
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    train_loader = DataLoader(
        mixed,
        batch_sampler=sampler,
        collate_fn=mixed_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # ---- Validation: first chunk only, no augmentation, subsampled ----
    print("\nBuilding validation set (first chunk per paper, no augment)...")
    seen_docs = set()
    val_chunks = []
    for ex in all_chunks:
        doc_id = ex.get("doc_id", "")
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            val_chunks.append(ex)
            if len(val_chunks) >= val_subsample:
                break

    val_ds = AlignedChunkDataset(val_chunks, tokenizer, max_src, max_tgt,
                                  augment=False)

    def val_collate(batch):
        src_ids = [b["input_ids"] for b in batch]
        tgt_ids = [b["target_ids"] for b in batch]
        src_padded = pad_sequence(src_ids, batch_first=True, padding_value=PAD_ID)
        tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=PAD_ID)
        return {
            "src": src_padded,
            "tgt_in": tgt_padded[:, :-1],
            "tgt_out": tgt_padded[:, 1:],
        }

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collate,
        num_workers=0,
        pin_memory=True,
    )

    print(f"\n  Chunk examples    : {len(chunk_ds)}")
    print(f"  Second-pass       : {len(sp_ds)}")
    print(f"  Anchor (CNN/DM)   : {len(anchor_ds)}")
    print(f"  Val (first chunk) : {len(val_ds)}")
    print(f"  Train steps/epoch : {len(sampler)}")
    print(f"  Val batches       : {len(val_loader)}")
    print(f"  Batch composition : {sampler.n_chunk} chunk + "
          f"{sampler.n_second} second-pass + {sampler.n_anchor} anchor "
          f"= {batch_size}")

    return train_loader, val_loader
