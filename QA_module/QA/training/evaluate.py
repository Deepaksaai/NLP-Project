"""
SQuAD-style evaluation metrics for the QA model.

Implements:
  - Normalization (lowercase, strip punct/articles/whitespace)
  - Exact Match (EM)
  - Token-level F1
  - Has-answer accuracy, false-positive rate, false-negative rate
  - Per-domain aggregation

Decision rule for 'is this question answerable?':
    model_says_answerable = sigmoid(has_answer_logit) > 0.5

The predicted text span is also considered 'no answer' if the joint
span selector picks (0, 0) (the [CLS] position). Either condition
can flip the final decision — see SquadV2 scoring convention.
"""

import re
import string
import collections
from typing import List, Dict

import torch

from QA.training.span_select import joint_span_select
from QA.qa_config import MAX_ANSWER_LEN


# -------------------------------------------------------
# Text normalization
# -------------------------------------------------------
def _normalize(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)
    def white_space(t):
        return " ".join(t.split())
    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))
    def lower(t):
        return t.lower()
    return white_space(remove_articles(remove_punc(lower(s))))


def _f1_one(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _em_one(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


# -------------------------------------------------------
# Prediction decoding — pull text out of (start_idx, end_idx)
# -------------------------------------------------------
def decode_span(tokenizer, input_ids_row, start_idx: int, end_idx: int) -> str:
    if end_idx < start_idx:
        return ""
    ids = input_ids_row[start_idx: end_idx + 1].tolist()
    return tokenizer.decode(ids).strip()


# -------------------------------------------------------
# Evaluator
# -------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, data_loader, tokenizer, device, max_answer_len: int = MAX_ANSWER_LEN) -> Dict:
    model.eval()

    n_total = 0
    em_sum = 0.0
    f1_sum = 0.0

    # has-answer confusion matrix
    tp = tn = fp = fn = 0

    per_domain = collections.defaultdict(lambda: {"n": 0, "em": 0.0, "f1": 0.0})

    for batch in data_loader:
        input_ids    = batch["input_ids"].to(device)
        segment_ids  = batch["segment_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        gold_has_ans = batch["has_answer"].to(device)
        gold_texts   = batch["answer_texts"]
        domains      = batch["domains"]

        start_logits, end_logits, has_answer_logits = model(input_ids, segment_ids, attn_mask)

        start_idx, end_idx = joint_span_select(start_logits, end_logits, max_answer_len)

        has_ans_prob = torch.sigmoid(has_answer_logits)
        pred_has_ans = (has_ans_prob > 0.5)

        for b in range(input_ids.size(0)):
            gold_ans = bool(gold_has_ans[b].item() > 0.5)
            pred_ans = bool(pred_has_ans[b].item())

            if pred_ans:
                pred_text = decode_span(tokenizer, input_ids[b], int(start_idx[b].item()), int(end_idx[b].item()))
            else:
                pred_text = ""

            gold_text = gold_texts[b] if gold_ans else ""

            em = _em_one(pred_text, gold_text)
            f1 = _f1_one(pred_text, gold_text)

            n_total += 1
            em_sum  += em
            f1_sum  += f1

            d = per_domain[domains[b]]
            d["n"]  += 1
            d["em"] += em
            d["f1"] += f1

            if   gold_ans and pred_ans:     tp += 1
            elif not gold_ans and not pred_ans: tn += 1
            elif not gold_ans and pred_ans: fp += 1
            elif gold_ans and not pred_ans: fn += 1

    def safe_div(a, b): return a / b if b > 0 else 0.0

    out = {
        "n": n_total,
        "em": safe_div(em_sum, n_total),
        "f1": safe_div(f1_sum, n_total),
        "has_answer_accuracy": safe_div(tp + tn, n_total),
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "per_domain": {
            k: {"n": v["n"], "em": safe_div(v["em"], v["n"]), "f1": safe_div(v["f1"], v["n"])}
            for k, v in per_domain.items()
        },
    }
    return out


# -------------------------------------------------------
# Training-time F1 (used by the sanity-check loop)
# -------------------------------------------------------
@torch.no_grad()
def train_batch_f1(model, batch, tokenizer, device) -> float:
    model.eval()
    input_ids   = batch["input_ids"].to(device)
    segment_ids = batch["segment_ids"].to(device)
    attn_mask   = batch["attention_mask"].to(device)
    gold_texts  = batch["answer_texts"]
    gold_has    = batch["has_answer"]

    start_logits, end_logits, has_answer_logits = model(input_ids, segment_ids, attn_mask)
    start_idx, end_idx = joint_span_select(start_logits, end_logits)
    pred_has = torch.sigmoid(has_answer_logits) > 0.5

    f1_sum = 0.0
    for b in range(input_ids.size(0)):
        gold_ans_flag = bool(gold_has[b].item() > 0.5)
        if pred_has[b].item():
            pred_text = decode_span(tokenizer, input_ids[b], int(start_idx[b].item()), int(end_idx[b].item()))
        else:
            pred_text = ""
        gold_text = gold_texts[b] if gold_ans_flag else ""
        f1_sum += _f1_one(pred_text, gold_text)

    model.train()
    return f1_sum / input_ids.size(0)
