"""
Stage 3 dataset: Legal fine-tuning with mixed batches.

Batch composition per step (16 total):
   9  MultiLexSum/BillSum chunks  (~56%)
   4  Second-pass legal examples  (~25%)
   2  arXiv anchor (no <legal>)   (~12%)
   1  CNN/DM anchor (no <legal>)  (~6%)

The <legal> token is prepended to legal chunk inputs and second-pass
inputs at tokenization time, NOT to anchor samples.

Sentence shuffle is DISABLED entirely — legal document order matters.
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

from data.preprocess import tokenize_pair, load_tokenizer, PAD_ID, LEGAL_ID, SOS_ID, EOS_ID


def load_jsonl(path, limit=None):
    examples = []
    if not os.path.exists(path):
        print(f"  Warning: {path} not found")
        return examples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
                if limit and len(examples) >= limit:
                    break
    print(f"  Loaded {len(examples)} from {os.path.basename(path)}")
    return examples


# -------------------------------------------------------
# Tokenization with optional <legal> prefix
# -------------------------------------------------------
def encode_with_legal(text, target, tokenizer, max_src=400, max_tgt=128,
                     prepend_legal=True):
    """
    Encode a (text, target) pair, optionally prepending <legal> to source.
    Returns dict with input_ids, target_ids tensors.
    """
    src_ids = tokenizer.encode(text.strip()).ids[:max_src - 1]
    if prepend_legal:
        src_ids = [LEGAL_ID] + src_ids
        src_ids = src_ids[:max_src]

    tgt_tokens = tokenizer.encode(target.strip()).ids[:max_tgt - 2]
    tgt_ids = [SOS_ID] + tgt_tokens + [EOS_ID]

    return {
        "input_ids": torch.tensor(src_ids, dtype=torch.long),
        "target_ids": torch.tensor(tgt_ids, dtype=torch.long),
    }


# -------------------------------------------------------
# Dataset classes (one per example type)
# -------------------------------------------------------
class LegalChunkDataset(Dataset):
    """MultiLexSum + BillSum aligned chunks. Always prepends <legal>."""

    def __init__(self, examples, tokenizer, max_src=400, max_tgt=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        item = encode_with_legal(
            ex["input"], ex["target"], self.tokenizer,
            self.max_src, self.max_tgt, prepend_legal=True
        )
        item["type"] = "legal_chunk"
        return item


class LegalSecondPassDataset(Dataset):
    """Concatenated chunk targets → full case summary. Prepends <legal>."""

    def __init__(self, examples, tokenizer, max_src=400, max_tgt=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        item = encode_with_legal(
            ex["input"], ex["target"], self.tokenizer,
            self.max_src, self.max_tgt, prepend_legal=True
        )
        item["type"] = "second_pass"
        return item


class ArxivAnchorDataset(Dataset):
    """arXiv/PubMed chunks from Stage 2 alignment. NO <legal> token."""

    def __init__(self, examples, tokenizer, max_src=400, max_tgt=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        item = encode_with_legal(
            ex["input"], ex["target"], self.tokenizer,
            self.max_src, self.max_tgt, prepend_legal=False
        )
        item["type"] = "arxiv_anchor"
        return item


class CnnAnchorDataset(Dataset):
    """CNN/DM articles + highlights. NO <legal> token."""

    def __init__(self, examples, tokenizer, max_src=400, max_tgt=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        item = encode_with_legal(
            ex["text"], ex["summary"], self.tokenizer,
            self.max_src, self.max_tgt, prepend_legal=False
        )
        item["type"] = "cnn_anchor"
        return item


# -------------------------------------------------------
# Mixed sampler (4 sources)
# -------------------------------------------------------
class Stage3MixedSampler(Sampler):
    """
    Yields batches with:
      9 legal_chunk + 4 second_pass + 2 arxiv_anchor + 1 cnn_anchor = 16
    """

    def __init__(self, chunk_size, sp_size, arxiv_size, cnn_size,
                 batch_size=16, steps_per_epoch=None):
        self.sizes = {
            "legal_chunk": chunk_size,
            "second_pass": sp_size,
            "arxiv_anchor": arxiv_size,
            "cnn_anchor": cnn_size,
        }
        self.batch_size = batch_size

        # Composition for batch_size=16: 9/4/2/1
        self.counts = {
            "legal_chunk": max(1, round(batch_size * 0.5625)),  # 9
            "second_pass": max(1, round(batch_size * 0.25)),     # 4
            "arxiv_anchor": max(1, round(batch_size * 0.125)),   # 2
            "cnn_anchor": max(1, batch_size - 9 - 4 - 2),        # 1
        }

        if steps_per_epoch is None:
            # One epoch = enough steps to cover legal chunks once
            self.steps_per_epoch = chunk_size // self.counts["legal_chunk"]
        else:
            self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        indices = {
            k: list(range(v)) for k, v in self.sizes.items()
        }
        for k in indices:
            random.shuffle(indices[k])

        cursors = {k: 0 for k in self.sizes}

        for _ in range(self.steps_per_epoch):
            batch = []
            for dtype, n in self.counts.items():
                for _ in range(n):
                    if cursors[dtype] >= len(indices[dtype]):
                        random.shuffle(indices[dtype])
                        cursors[dtype] = 0
                    batch.append((dtype, indices[dtype][cursors[dtype]]))
                    cursors[dtype] += 1
            yield batch

    def __len__(self):
        return self.steps_per_epoch


# -------------------------------------------------------
# Wrapper dataset for tuple indexing
# -------------------------------------------------------
class Stage3MixedDataset(Dataset):
    def __init__(self, chunk_ds, sp_ds, arxiv_ds, cnn_ds):
        self.datasets = {
            "legal_chunk": chunk_ds,
            "second_pass": sp_ds,
            "arxiv_anchor": arxiv_ds,
            "cnn_anchor": cnn_ds,
        }

    def __len__(self):
        return sum(len(d) for d in self.datasets.values())

    def __getitem__(self, key):
        dtype, idx = key
        return self.datasets[dtype][idx]


def stage3_collate_fn(items, pad_id=PAD_ID):
    src_ids = [it["input_ids"] for it in items]
    tgt_ids = [it["target_ids"] for it in items]
    src_padded = pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)
    types = [it["type"] for it in items]
    return {
        "src": src_padded,
        "tgt_in": tgt_padded[:, :-1],
        "tgt_out": tgt_padded[:, 1:],
        "types": types,
    }


def val_collate_fn(items, pad_id=PAD_ID):
    src_ids = [it["input_ids"] for it in items]
    tgt_ids = [it["target_ids"] for it in items]
    src_padded = pad_sequence(src_ids, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_id)
    return {
        "src": src_padded,
        "tgt_in": tgt_padded[:, :-1],
        "tgt_out": tgt_padded[:, 1:],
    }


# -------------------------------------------------------
# Anchor data loaders
# -------------------------------------------------------
def load_arxiv_anchor(limit=20000):
    """Subsample from Stage 2 aligned data."""
    print(f"Loading arXiv anchor (limit={limit})...")
    examples = []
    arxiv_path = "data/stage2_aligned/arxiv_aligned.jsonl"
    pubmed_path = "data/stage2_aligned/pubmed_aligned.jsonl"

    half = limit // 2
    examples.extend(load_jsonl(arxiv_path, limit=half))
    examples.extend(load_jsonl(pubmed_path, limit=half))
    random.shuffle(examples)
    return examples[:limit]


def load_cnn_anchor(limit=10000):
    """Subsample CNN/DM from HuggingFace."""
    print(f"Loading CNN/DM anchor (limit={limit})...")
    from datasets import load_dataset
    cnn_dm = load_dataset("cnn_dailymail", "3.0.0", split="train")
    examples = []
    for i, ex in enumerate(cnn_dm):
        if i >= limit:
            break
        examples.append({"text": ex["article"], "summary": ex["highlights"]})
    print(f"  Loaded {len(examples)} CNN/DM samples")
    return examples


# -------------------------------------------------------
# Validation set: first chunk per case, no anchors, no second-pass
# -------------------------------------------------------
def build_val_set(val_examples, val_subsample=2000):
    """First chunk only of each case, with <legal> token."""
    seen_cases = set()
    val_chunks = []
    for ex in val_examples:
        cid = ex.get("case_id", "")
        if cid not in seen_cases:
            seen_cases.add(cid)
            val_chunks.append(ex)
            if len(val_chunks) >= val_subsample:
                break
    return val_chunks


# -------------------------------------------------------
# Build everything
# -------------------------------------------------------
def build_stage3_data(tokenizer, max_src=400, max_tgt=128, batch_size=16,
                      steps_per_epoch=None, val_subsample=2000,
                      arxiv_anchor_limit=20000, cnn_anchor_limit=10000):
    """
    Build Stage 3 training data.
    Run preprocessing first: python -m data.preprocess_stage3
    """
    print("=" * 60)
    print("Stage 3: Building dataset")
    print("=" * 60)

    aligned_dir = "data/stage3_aligned"

    # ---- Load legal chunks ----
    print("\nLoading legal chunks...")
    train_legal = load_jsonl(os.path.join(aligned_dir, "legal_train.jsonl"))
    val_legal = load_jsonl(os.path.join(aligned_dir, "legal_val.jsonl"))
    print(f"  Total legal train: {len(train_legal)}")
    print(f"  Total legal val:   {len(val_legal)}")

    # ---- Load second-pass ----
    print("\nLoading second-pass legal examples...")
    second_pass = load_jsonl(os.path.join(aligned_dir, "legal_second_pass.jsonl"))
    print(f"  Total second-pass: {len(second_pass)}")

    # ---- Load anchors ----
    print()
    arxiv_anchor = load_arxiv_anchor(limit=arxiv_anchor_limit)
    cnn_anchor = load_cnn_anchor(limit=cnn_anchor_limit)

    # ---- Build datasets ----
    chunk_ds = LegalChunkDataset(train_legal, tokenizer, max_src, max_tgt)
    sp_ds = LegalSecondPassDataset(second_pass, tokenizer, max_src, max_tgt)
    arxiv_ds = ArxivAnchorDataset(arxiv_anchor, tokenizer, max_src, max_tgt)
    cnn_ds = CnnAnchorDataset(cnn_anchor, tokenizer, max_src, max_tgt)

    mixed = Stage3MixedDataset(chunk_ds, sp_ds, arxiv_ds, cnn_ds)

    # ---- Mixed sampler ----
    sampler = Stage3MixedSampler(
        chunk_size=len(chunk_ds),
        sp_size=len(sp_ds),
        arxiv_size=len(arxiv_ds),
        cnn_size=len(cnn_ds),
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    train_loader = DataLoader(
        mixed,
        batch_sampler=sampler,
        collate_fn=stage3_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # ---- Validation set ----
    val_chunks = build_val_set(val_legal, val_subsample=val_subsample)
    val_ds = LegalChunkDataset(val_chunks, tokenizer, max_src, max_tgt)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    print(f"\n  Legal chunks      : {len(chunk_ds)}")
    print(f"  Second-pass       : {len(sp_ds)}")
    print(f"  arXiv anchor      : {len(arxiv_ds)}")
    print(f"  CNN/DM anchor     : {len(cnn_ds)}")
    print(f"  Val set           : {len(val_ds)}")
    print(f"  Train steps/epoch : {len(sampler)}")
    print(f"  Val batches       : {len(val_loader)}")
    print(f"  Batch composition : {sampler.counts['legal_chunk']} legal + "
          f"{sampler.counts['second_pass']} 2nd-pass + "
          f"{sampler.counts['arxiv_anchor']} arXiv + "
          f"{sampler.counts['cnn_anchor']} cnn = {batch_size}")

    return train_loader, val_loader
