"""
Legal-specific evaluation metrics for Stage 3.

Implements three metrics on top of the Stage-1/2 EM/F1 evaluator:

    1. Clause boundary accuracy — 1 if predicted span starts and ends
       at clause boundaries (sentence-end punctuation, newline, or a
       section marker like "(a)" / "1." / "Article N").

    2. Legal term preservation — how many "critical" legal tokens in
       the gold span are reproduced in the prediction:
         - defined terms (Title-Case multi-word phrases)
         - monetary amounts ($X, dollars, etc.)
         - time periods (N days/months/years)
         - party references (Licensor, Licensee, ...)

    3. Legal false positive rate — fraction of LEGAL unanswerable
       examples where the model predicts a non-empty span. Computed
       over {cuad, ledgar, coliee} only.

Also reports the mean predicted answer length (in whitespace tokens)
separately for legal vs non-legal examples.
"""

import re
from collections import defaultdict
from typing import Dict

import torch

from QA.training.evaluate import _f1_one, _em_one, decode_span
from QA.training.span_select import joint_span_select
from QA.qa_config import LEGAL_MAX_ANSWER_LEN, LEGAL_LENGTH_PENALTY


# -------------------------------------------------------
# Text helpers
# -------------------------------------------------------
_SECTION_MARKER_RE = re.compile(
    r"^\s*(\([a-zA-Z0-9]+\)|\d+[\.\)]|Article\s+\w+|Section\s+\w+|§)",
    re.IGNORECASE,
)
_BOUNDARY_CHARS = set(".;:\n")

_DEFINED_TERM_RE = re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)+\b")
_MONEY_RE        = re.compile(r"\$[0-9][\d,]*(?:\.\d+)?|\b\d[\d,]*\s*(?:dollars?|USD|cents?)\b", re.IGNORECASE)
_TIME_RE         = re.compile(r"\b\d+\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\b", re.IGNORECASE)
_PARTY_WORDS     = {
    "licensor", "licensee", "company", "customer", "supplier",
    "seller", "buyer", "lessor", "lessee", "party", "parties",
    "counterparty", "contractor", "subcontractor", "employer", "employee",
    "agent", "principal", "guarantor",
}


def _starts_at_boundary(context: str, char_start: int) -> bool:
    if char_start <= 0:
        return True
    prev = context[char_start - 1]
    if prev in _BOUNDARY_CHARS:
        return True
    # Section marker on the line containing the span start
    line_start = context.rfind("\n", 0, char_start) + 1
    line_head = context[line_start: char_start + 1]
    return bool(_SECTION_MARKER_RE.match(line_head))


def _ends_at_boundary(context: str, char_end: int) -> bool:
    if char_end >= len(context):
        return True
    ch = context[char_end - 1] if char_end > 0 else ""
    if ch in _BOUNDARY_CHARS:
        return True
    nxt = context[char_end] if char_end < len(context) else ""
    return nxt in _BOUNDARY_CHARS or nxt == "\n"


def _extract_terms(text: str):
    terms = set(m.group() for m in _DEFINED_TERM_RE.finditer(text))
    terms.update(m.group().lower() for m in _MONEY_RE.finditer(text))
    terms.update(m.group().lower() for m in _TIME_RE.finditer(text))
    for w in re.findall(r"\b\w+\b", text.lower()):
        if w in _PARTY_WORDS:
            terms.add(w)
    return terms


def legal_term_preservation_one(pred: str, gold: str) -> float:
    gold_terms = _extract_terms(gold)
    if not gold_terms:
        return 1.0  # nothing to preserve -> trivially satisfied
    pred_terms = _extract_terms(pred)
    overlap = len(gold_terms & pred_terms)
    return overlap / len(gold_terms)


# -------------------------------------------------------
# Batched evaluator — legal-aware
# -------------------------------------------------------
@torch.no_grad()
def evaluate_stage3(model, data_loader, tokenizer, device,
                    max_answer_len: int = LEGAL_MAX_ANSWER_LEN,
                    length_penalty: float = LEGAL_LENGTH_PENALTY) -> Dict:
    """
    Drop-in replacement for Stage 1/2 evaluate_model that also
    produces legal-specific metrics. Iterates over batches produced
    by `stage3_collate`.
    """
    model.eval()

    # Accumulators
    n = 0
    em_sum = f1_sum = 0.0
    tp = tn = fp = fn = 0

    # Legal-specific
    legal_n = 0
    legal_fp = 0
    legal_unans = 0
    clause_ok_sum = 0.0
    term_pres_sum = 0.0

    ans_len_sum_legal = 0.0; ans_len_n_legal = 0
    ans_len_sum_gen   = 0.0; ans_len_n_gen   = 0

    per_source = defaultdict(lambda: {"n": 0, "em": 0.0, "f1": 0.0})

    for batch in data_loader:
        input_ids   = batch["input_ids"].to(device)
        segment_ids = batch["segment_ids"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        gold_has_ans = batch["has_answer"].to(device)
        gold_texts   = batch["answer_texts"]
        sources      = batch["sources"]

        start_logits, end_logits, has_ans_logits = model(input_ids, segment_ids, attn_mask)
        start_idx, end_idx = joint_span_select(
            start_logits, end_logits,
            max_answer_len=max_answer_len,
            length_penalty=length_penalty,
        )
        has_ans_prob = torch.sigmoid(has_ans_logits)
        pred_has_ans = has_ans_prob > 0.5

        for b in range(input_ids.size(0)):
            gold_ans = bool(gold_has_ans[b].item() > 0.5)
            pred_ans = bool(pred_has_ans[b].item())
            src = sources[b]
            is_legal = src in ("cuad", "ledgar", "coliee")

            if pred_ans:
                pred_text = decode_span(
                    tokenizer, input_ids[b],
                    int(start_idx[b].item()), int(end_idx[b].item()),
                )
            else:
                pred_text = ""

            gold_text = gold_texts[b] if gold_ans else ""

            em = _em_one(pred_text, gold_text)
            f1 = _f1_one(pred_text, gold_text)
            em_sum += em
            f1_sum += f1
            n += 1

            d = per_source[src]
            d["n"] += 1; d["em"] += em; d["f1"] += f1

            if   gold_ans and pred_ans:         tp += 1
            elif (not gold_ans) and (not pred_ans): tn += 1
            elif (not gold_ans) and pred_ans:   fp += 1
            elif gold_ans and (not pred_ans):   fn += 1

            # answer length (whitespace tokens)
            pred_len = len(pred_text.split())
            if is_legal:
                ans_len_sum_legal += pred_len
                ans_len_n_legal += 1
            else:
                ans_len_sum_gen += pred_len
                ans_len_n_gen += 1

            # Legal-specific accumulators
            if is_legal:
                legal_n += 1
                if (not gold_ans) and pred_ans:
                    legal_fp += 1
                if not gold_ans:
                    legal_unans += 1

                # Clause boundary + term preservation only meaningful
                # when BOTH a gold answer and a predicted span exist.
                if gold_ans and pred_ans and pred_text and gold_text:
                    # Work in the original gold context, if we can find the pred text inside it
                    start_char = gold_text.find(pred_text)
                    if start_char >= 0:
                        end_char = start_char + len(pred_text)
                        starts_ok = _starts_at_boundary(gold_text, start_char)
                        ends_ok = _ends_at_boundary(gold_text, end_char)
                    else:
                        # fall back: prediction diverged from gold — no boundary
                        starts_ok = ends_ok = False
                    clause_ok_sum += 1.0 if (starts_ok and ends_ok) else 0.0

                    term_pres_sum += legal_term_preservation_one(pred_text, gold_text)

    def sd(a, b): return a / b if b > 0 else 0.0

    return {
        "n": n,
        "em": sd(em_sum, n),
        "f1": sd(f1_sum, n),
        "has_answer_accuracy": sd(tp + tn, n),
        "false_positive_rate": sd(fp, fp + tn),
        "false_negative_rate": sd(fn, fn + tp),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "per_source": {
            k: {"n": v["n"], "em": sd(v["em"], v["n"]), "f1": sd(v["f1"], v["n"])}
            for k, v in per_source.items()
        },
        # Legal-specific:
        "legal_n": legal_n,
        "legal_false_positive_rate": sd(legal_fp, legal_unans) if legal_unans else 0.0,
        "clause_boundary_accuracy":  sd(clause_ok_sum, max(1, legal_n - legal_unans)),
        "legal_term_preservation":   sd(term_pres_sum, max(1, legal_n - legal_unans)),
        "mean_answer_length_legal":   sd(ans_len_sum_legal, ans_len_n_legal),
        "mean_answer_length_general": sd(ans_len_sum_gen,   ans_len_n_gen),
    }
