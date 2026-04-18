"""
Stage 1 dataset — merges SQuAD 2.0 / TriviaQA / Natural Questions
preprocessed JSON files into a single PyTorch Dataset.

All input files must have been produced by
QA.data.qa_dataset.build_features() so every example already carries:
    input_ids, segment_ids, attention_mask,
    answer_start_tok, answer_end_tok, is_answerable, domain.

The dataset also exposes `answerable_indices` / `unanswerable_indices`
so the BalancedBatchSampler can compose balanced batches without
rescanning the data.
"""

import os
import json
import glob
import torch
from torch.utils.data import Dataset

from QA.qa_config import QA_DATA_ROOT
from QA.data.qa_dataset import REQUIRED_RAW_FIELDS, REQUIRED_TOKENIZED_FIELDS


class Stage1QADataset(Dataset):

    def __init__(self, split: str, stage_dir: str = None, files=None):
        """
        split     : "train" or "val"
        stage_dir : directory containing <split>*.json files
                    (defaults to data/qa/stage1)
        files     : optional explicit list of files (overrides stage_dir)
        """
        if files is None:
            if stage_dir is None:
                stage_dir = os.path.join(QA_DATA_ROOT, "stage1")
            # Any file that starts with the split name (train.json,
            # train_squad.json, train_trivia.json, val.json, ...)
            files = sorted(glob.glob(os.path.join(stage_dir, f"{split}*.json")))

        if not files:
            raise FileNotFoundError(
                f"Stage1QADataset: no files found for split={split!r}"
            )

        self.files = files
        self.examples = []
        self.per_file_counts = {}

        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._validate(data, f)
            self.examples.extend(data)
            self.per_file_counts[os.path.basename(f)] = len(data)

        self.answerable_indices   = [i for i, e in enumerate(self.examples) if e["is_answerable"]]
        self.unanswerable_indices = [i for i, e in enumerate(self.examples) if not e["is_answerable"]]

    @staticmethod
    def _validate(data, path):
        if not data:
            return
        ex = data[0]
        for f in REQUIRED_RAW_FIELDS + REQUIRED_TOKENIZED_FIELDS:
            if f not in ex:
                raise ValueError(f"{path}: example missing required field {f!r}")

    # -------------------------------------------------------
    def summary(self) -> dict:
        total = len(self.examples)
        n_ans = len(self.answerable_indices)
        n_unans = len(self.unanswerable_indices)
        by_domain = {}
        for e in self.examples:
            by_domain[e["domain"]] = by_domain.get(e["domain"], 0) + 1
        return {
            "total": total,
            "answerable": n_ans,
            "unanswerable": n_unans,
            "answerable_pct": n_ans / total if total else 0.0,
            "unanswerable_pct": n_unans / total if total else 0.0,
            "by_domain": by_domain,
            "per_file": self.per_file_counts,
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
            "answer_text":    ex["answer_text"],
            "raw_idx":        i,
        }


def stage1_collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "segment_ids":    torch.stack([b["segment_ids"]    for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "start_position": torch.stack([b["start_position"] for b in batch]),
        "end_position":   torch.stack([b["end_position"]   for b in batch]),
        "has_answer":     torch.stack([b["has_answer"]     for b in batch]),
        "domains":        [b["domain"]      for b in batch],
        "answer_texts":   [b["answer_text"] for b in batch],
        "raw_indices":    [b["raw_idx"]     for b in batch],
    }
