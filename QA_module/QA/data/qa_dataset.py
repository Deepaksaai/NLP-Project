"""
Unified QA dataset contract — shared by Stage 1 (general), Stage 2
(scientific), and Stage 3 (legal) training.

Preprocessed example on disk is a dict with fields:
    question         : str
    context          : str
    answer_text      : str         (empty string if is_answerable = False)
    answer_start     : int         (char offset in context; -1 if unanswerable)
    is_answerable    : bool
    domain           : str         ("general" | "scientific" | "legal")

    # Tokenized fields filled in by build_features():
    input_ids        : list[int]   len = MAX_TOTAL_LEN
    segment_ids      : list[int]   len = MAX_TOTAL_LEN
    attention_mask   : list[int]   len = MAX_TOTAL_LEN
    answer_start_tok : int         token index inside input_ids (0 if unanswerable)
    answer_end_tok   : int         token index inside input_ids (0 if unanswerable)
"""

import json
import os
import torch
from torch.utils.data import Dataset

from QA.qa_config import (
    MAX_QUESTION_LEN, MAX_CONTEXT_LEN, MAX_TOTAL_LEN, PAD_ID,
)


REQUIRED_RAW_FIELDS = [
    "question", "context", "answer_text",
    "answer_start", "is_answerable", "domain",
]

REQUIRED_TOKENIZED_FIELDS = [
    "input_ids", "segment_ids", "attention_mask",
    "answer_start_tok", "answer_end_tok",
]


def validate_raw_example(ex: dict):
    for f in REQUIRED_RAW_FIELDS:
        assert f in ex, f"missing field {f!r} in example"
    assert ex["domain"] in ("general", "scientific", "legal")
    assert isinstance(ex["is_answerable"], bool)


def build_features(
    raw: dict,
    tokenizer,
    cls_id: int,
    sep_id: int,
    pad_id: int = PAD_ID,
    max_q: int = MAX_QUESTION_LEN,
    max_ctx: int = MAX_CONTEXT_LEN,
    max_total: int = MAX_TOTAL_LEN,
    prepend_legal: bool = False,
    legal_id: int = None,
):
    """
    Convert a raw example into the tokenized feature dict. Returns
    None if the answer span cannot be recovered after tokenization
    (caller should discard such examples).

    If prepend_legal=True the input layout becomes:
        [CLS] q_ids [SEP] <legal> ctx_ids [SEP]
    with the <legal> token placed in segment 1 (context segment).
    """
    validate_raw_example(raw)

    q_ids = tokenizer.encode(raw["question"]).ids[:max_q]
    ctx_enc = tokenizer.encode(raw["context"])
    ctx_ids = ctx_enc.ids[:max_ctx]
    ctx_offsets = ctx_enc.offsets[:max_ctx]   # list of (char_start, char_end)

    # Assemble the input, optionally with the <legal> signal token
    if prepend_legal:
        if legal_id is None:
            raise ValueError("prepend_legal=True requires legal_id")
        input_ids   = [cls_id] + q_ids + [sep_id] + [legal_id] + ctx_ids + [sep_id]
        segment_ids = [0] * (1 + len(q_ids) + 1) + [1] * (1 + len(ctx_ids) + 1)
        ctx_base = 1 + len(q_ids) + 1 + 1   # +1 for [legal]
    else:
        input_ids   = [cls_id] + q_ids + [sep_id] + ctx_ids + [sep_id]
        segment_ids = [0] * (1 + len(q_ids) + 1) + [1] * (len(ctx_ids) + 1)
        ctx_base = 1 + len(q_ids) + 1
    attention   = [1] * len(input_ids)

    # Padding
    pad_len = max_total - len(input_ids)
    if pad_len < 0:
        # shouldn't happen given the MAX_* budget, but truncate defensively
        input_ids   = input_ids[:max_total]
        segment_ids = segment_ids[:max_total]
        attention   = attention[:max_total]
    else:
        input_ids   += [pad_id] * pad_len
        segment_ids += [1]      * pad_len   # padding sits in context segment
        attention   += [0]      * pad_len

    ctx_start_in_seq = ctx_base   # first real context token index

    if not raw["is_answerable"]:
        answer_start_tok = 0   # [CLS]
        answer_end_tok   = 0
    else:
        ans_char_start = raw["answer_start"]
        ans_char_end   = ans_char_start + len(raw["answer_text"])

        # Map char offsets -> token positions inside ctx_ids
        tok_start = None
        tok_end   = None
        for i, (cs, ce) in enumerate(ctx_offsets):
            if tok_start is None and cs <= ans_char_start < ce:
                tok_start = i
            if cs < ans_char_end <= ce:
                tok_end = i
                break

        if tok_start is None or tok_end is None or tok_end < tok_start:
            return None  # answer got truncated out; caller discards

        answer_start_tok = ctx_start_in_seq + tok_start
        answer_end_tok   = ctx_start_in_seq + tok_end

        # Sanity check: decoded span must contain the answer text
        span_ids = input_ids[answer_start_tok: answer_end_tok + 1]
        decoded = tokenizer.decode(span_ids)
        if raw["answer_text"].strip().lower() not in decoded.lower():
            return None

    feat = dict(raw)
    feat.update({
        "input_ids":        input_ids,
        "segment_ids":      segment_ids,
        "attention_mask":   attention,
        "answer_start_tok": answer_start_tok,
        "answer_end_tok":   answer_end_tok,
    })
    return feat


class QADataset(Dataset):
    """Loads a preprocessed JSON lines or JSON list file from disk."""

    def __init__(self, path: str):
        self.examples = self._load(path)

    @staticmethod
    def _load(path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ex in data:
            for f_ in REQUIRED_RAW_FIELDS + REQUIRED_TOKENIZED_FIELDS:
                assert f_ in ex, f"{path}: example missing {f_!r}"
        return data

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        return {
            "input_ids":      torch.tensor(ex["input_ids"],    dtype=torch.long),
            "segment_ids":    torch.tensor(ex["segment_ids"],  dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "start_position": torch.tensor(ex["answer_start_tok"], dtype=torch.long),
            "end_position":   torch.tensor(ex["answer_end_tok"],   dtype=torch.long),
            "has_answer":     torch.tensor(float(ex["is_answerable"])),
            "domain":         ex["domain"],
        }


def qa_collate(batch):
    out = {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "segment_ids":    torch.stack([b["segment_ids"]    for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "start_position": torch.stack([b["start_position"] for b in batch]),
        "end_position":   torch.stack([b["end_position"]   for b in batch]),
        "has_answer":     torch.stack([b["has_answer"]     for b in batch]),
        "domains":        [b["domain"] for b in batch],
    }
    return out
