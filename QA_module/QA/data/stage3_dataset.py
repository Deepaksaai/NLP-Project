"""
Stage 3 dataset — merges CuAD / LEDGAR / COLIEE preprocessed files
with QASPER and SQuAD anchor files carried forward from earlier
stages.

Expected layout on disk:
    data/qa/stage3/{train,val}_cuad.json
    data/qa/stage3/{train,val}_ledgar.json
    data/qa/stage3/{train,val}_coliee.json          (optional)
    data/qa/stage2/{train,val}_qasper.json          (anchor)
    data/qa/stage1/{train,val}_squad.json           (anchor)

MixedBatchSampler can be fed this dataset's `indices_by_source`
directly with the Stage-3 mix: 35/25/10/15/15.
"""

import os
import json
import glob
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from QA.qa_config import QA_DATA_ROOT


class Stage3QADataset(Dataset):

    KNOWN_SOURCES = ("cuad", "ledgar", "coliee", "qasper", "squad")

    def __init__(self, split: str):
        stage3_dir = os.path.join(QA_DATA_ROOT, "stage3")
        stage2_dir = os.path.join(QA_DATA_ROOT, "stage2")
        stage1_dir = os.path.join(QA_DATA_ROOT, "stage1")

        files = []
        files += sorted(glob.glob(os.path.join(stage3_dir, f"{split}_cuad*.json")))
        files += sorted(glob.glob(os.path.join(stage3_dir, f"{split}_ledgar*.json")))
        files += sorted(glob.glob(os.path.join(stage3_dir, f"{split}_coliee*.json")))
        files += sorted(glob.glob(os.path.join(stage2_dir, f"{split}_qasper*.json")))
        files += sorted(glob.glob(os.path.join(stage1_dir, f"{split}_squad*.json")))

        if not files:
            raise FileNotFoundError(
                f"Stage3QADataset: no files found for split={split!r}"
            )

        self.files = files
        self.examples = []
        self.indices_by_source = {s: [] for s in self.KNOWN_SOURCES}
        per_file_counts = {}

        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            per_file_counts[os.path.basename(fp)] = len(data)

            fname = os.path.basename(fp).lower()
            # Source inference falls back on filename for older files
            default_src = next(
                (s for s in self.KNOWN_SOURCES if s in fname),
                "unknown",
            )
            for ex in data:
                src = ex.get("source") or default_src
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
        self.legal_indices        = [
            i for i, e in enumerate(self.examples)
            if e["source"] in ("cuad", "ledgar", "coliee")
        ]

    def summary(self) -> dict:
        return {
            "total": len(self.examples),
            "by_source": {s: len(v) for s, v in self.indices_by_source.items()},
            "answerable":   len(self.answerable_indices),
            "unanswerable": len(self.unanswerable_indices),
            "legal":        len(self.legal_indices),
            "per_file":     self.per_file_counts,
        }

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
            "answer_text":    ex["answer_text"],
            "raw_idx":        i,
        }


def stage3_collate(batch):
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
