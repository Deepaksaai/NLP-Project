"""
Stage 2 dataset + mixed-source batch sampler.

Loads three preprocessed sources from data/qa/stage2/:
    train_quality.json
    train_qasper.json
    train_squad.json   <- SQuAD anchor carried over from Stage 1

Each example still conforms to the unified contract (input_ids /
segment_ids / attention_mask / answer_start_tok / answer_end_tok /
is_answerable / domain) plus Stage 2 extras:
    source      : "quality" | "qasper" | "squad"
    doc_id      : grouping key for the chunk-retrieval metric
    question_id : unique per (doc, question) — all chunks of the
                  same question share this id
    chunk_idx   : chunk index within the document
    n_chunks    : total number of chunks in that document

The dataset also exposes three index lists so the MixedBatchSampler
can compose batches with the target 40 / 30 / 30 ratio.
"""

import os
import json
import glob
import random
from collections import defaultdict
from typing import List

import torch
from torch.utils.data import Dataset, Sampler

from QA.qa_config import QA_DATA_ROOT


class Stage2QADataset(Dataset):

    KNOWN_SOURCES = ("quality", "qasper", "squad")

    def __init__(self, split: str, stage_dir: str = None, files=None):
        if files is None:
            if stage_dir is None:
                stage_dir = os.path.join(QA_DATA_ROOT, "stage2")
            # Pull stage-2 files AND the Stage-1 SQuAD file (anchor)
            files = sorted(glob.glob(os.path.join(stage_dir, f"{split}*.json")))
            squad_anchor = os.path.join(
                QA_DATA_ROOT, "stage1", f"{split}_squad.json",
            )
            if os.path.exists(squad_anchor):
                files.append(squad_anchor)

        if not files:
            raise FileNotFoundError(
                f"Stage2QADataset: no files found for split={split!r}"
            )

        self.files = files
        self.examples: List[dict] = []
        self.indices_by_source = {s: [] for s in self.KNOWN_SOURCES}
        per_file_counts = {}

        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            per_file_counts[os.path.basename(fp)] = len(data)
            for ex in data:
                src = ex.get("source")
                if src is None:
                    # SQuAD stage-1 files don't carry a 'source' field;
                    # infer from the filename.
                    src = "squad" if "squad" in os.path.basename(fp).lower() else "unknown"
                    ex["source"] = src
                ex.setdefault("doc_id",      f"{src}_{len(self.examples)}")
                ex.setdefault("question_id", ex["doc_id"])
                ex.setdefault("chunk_idx",   0)
                ex.setdefault("n_chunks",    1)
                idx = len(self.examples)
                self.examples.append(ex)
                if src in self.indices_by_source:
                    self.indices_by_source[src].append(idx)

        self.per_file_counts = per_file_counts
        self.answerable_indices   = [i for i, e in enumerate(self.examples) if e["is_answerable"]]
        self.unanswerable_indices = [i for i, e in enumerate(self.examples) if not e["is_answerable"]]

    def summary(self) -> dict:
        n = len(self.examples)
        return {
            "total": n,
            "by_source": {s: len(v) for s, v in self.indices_by_source.items()},
            "answerable": len(self.answerable_indices),
            "unanswerable": len(self.unanswerable_indices),
            "per_file": self.per_file_counts,
        }

    # Grouping for chunk-retrieval eval: returns dict question_id -> [example_idx, ...]
    def grouped_by_question(self) -> dict:
        g = defaultdict(list)
        for i, e in enumerate(self.examples):
            g[e["question_id"]].append(i)
        return dict(g)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        return {
            "input_ids":      torch.tensor(ex["input_ids"],      dtype=torch.long),
            "segment_ids":    torch.tensor(ex["segment_ids"],    dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "start_position": torch.tensor(ex["answer_start_tok"], dtype=torch.long),
            "end_position":   torch.tensor(ex["answer_end_tok"],   dtype=torch.long),
            "has_answer":     torch.tensor(float(ex["is_answerable"])),
            "domain":         ex["domain"],
            "source":         ex["source"],
            "doc_id":         ex["doc_id"],
            "question_id":    ex["question_id"],
            "chunk_idx":      ex["chunk_idx"],
            "answer_text":    ex["answer_text"],
            "raw_idx":        i,
        }


def stage2_collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "segment_ids":    torch.stack([b["segment_ids"]    for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "start_position": torch.stack([b["start_position"] for b in batch]),
        "end_position":   torch.stack([b["end_position"]   for b in batch]),
        "has_answer":     torch.stack([b["has_answer"]     for b in batch]),
        "sources":        [b["source"]      for b in batch],
        "domains":        [b["domain"]      for b in batch],
        "doc_ids":        [b["doc_id"]      for b in batch],
        "question_ids":   [b["question_id"] for b in batch],
        "answer_texts":   [b["answer_text"] for b in batch],
        "raw_indices":    [b["raw_idx"]     for b in batch],
    }
