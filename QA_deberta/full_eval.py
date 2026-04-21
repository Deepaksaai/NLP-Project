"""
Comprehensive evaluation of the fine-tuned DeBERTa-v3-LARGE QA model on CUAD val.

Produces:
  - Overall EM / F1
  - Separate metrics for ANSWERABLE vs UNANSWERABLE
  - HasAnswer classifier confusion matrix
  - Per-question-type breakdown
  - Sample of wrong predictions

Usage (from NLP-Project/ root):
    python -m QA_deberta.full_eval
    python -m QA_deberta.full_eval --n_examples 0    # 0 = all examples
"""

import os
import json
import re
import string
import argparse
import random
import time
from collections import defaultdict

import torch

from QA_deberta.model import build_deberta_qa


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def extract_category(question: str) -> str:
    m = re.search(r'related to "([^"]+)"', question)
    return m.group(1) if m else "unknown"


@torch.no_grad()
def predict_one(model, tokenizer, question: str, context: str, device,
                max_len: int = 451, max_answer_len: int = 150):
    enc = tokenizer(
        question, context,
        max_length=max_len,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    ).to(device)

    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    offsets        = enc["offset_mapping"][0].cpu().tolist()

    sep_id = tokenizer.sep_token_id
    seps = (input_ids[0] == sep_id).nonzero(as_tuple=True)[0].tolist()
    context_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if len(seps) >= 2:
        context_mask[0, seps[0] + 1: seps[1]] = True

    s_logits, e_logits, ha_logit = model(input_ids, attention_mask, context_mask)
    s_logits = s_logits[0]
    e_logits = e_logits[0]
    L = s_logits.size(0)

    span_scores = s_logits.unsqueeze(1) + e_logits.unsqueeze(0)
    idx = torch.arange(L, device=device)
    valid = (idx.unsqueeze(0) >= idx.unsqueeze(1)) & \
            (idx.unsqueeze(0) - idx.unsqueeze(1) < max_answer_len)
    span_scores = span_scores.masked_fill(~valid, float("-inf"))

    # Disallow [CLS] — training was positives-only
    span_scores[0, :] = float("-inf")
    span_scores[:, 0] = float("-inf")

    flat_idx = span_scores.view(-1).argmax().item()
    best_s = flat_idx // L
    best_e = flat_idx % L

    if best_s == 0 and best_e == 0:
        pred_text = ""
    elif offsets and best_s < len(offsets) and offsets[best_s][1] > 0:
        char_s = offsets[best_s][0]
        char_e = offsets[best_e][1]
        pred_text = context[char_s:char_e].strip()
    else:
        pred_ids = input_ids[0, best_s: best_e + 1].cpu().tolist()
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

    has_ans_prob = torch.sigmoid(ha_logit[0]).item()
    return pred_text, has_ans_prob, best_s, best_e


def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[full_eval] device={device}")

    model, tokenizer = build_deberta_qa(freeze_layers=0)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state") or ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"[full_eval] Loaded checkpoint: {args.ckpt}")

    with open(args.data_file, encoding="utf-8") as f:
        val = json.load(f)
    if args.n_examples > 0 and args.n_examples < len(val):
        random.seed(42)
        val = random.sample(val, args.n_examples)
    print(f"[full_eval] Evaluating on {len(val)} examples")

    answerable_em,   answerable_f1,   answerable_n   = 0, 0.0, 0
    unanswerable_em, unanswerable_f1, unanswerable_n = 0, 0.0, 0
    ha_tp = ha_fp = ha_tn = ha_fn = 0
    per_cat = defaultdict(lambda: {"em": 0, "f1": 0.0, "n": 0, "has_ans": 0})
    errors = []

    t0 = time.time()
    for i, ex in enumerate(val):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(val) - i - 1) / rate
            print(f"  [{i+1}/{len(val)}]  {rate:.1f} ex/s, ETA {eta:.0f}s")

        pred_text, has_ans_prob, pred_s, pred_e = predict_one(
            model, tokenizer, ex["question"], ex["context"], device
        )

        gold_text = ex.get("answer_text", "")
        gold_has  = bool(ex.get("has_answer", bool(gold_text)))
        category  = extract_category(ex["question"])
        pred_has  = (pred_s != 0 or pred_e != 0)

        f1 = token_f1(pred_text, gold_text)
        em = exact_match(pred_text, gold_text)

        if gold_has:
            answerable_em += em
            answerable_f1 += f1
            answerable_n  += 1
            if pred_has: ha_tp += 1
            else:        ha_fn += 1
        else:
            unanswerable_em += em
            unanswerable_f1 += f1
            unanswerable_n  += 1
            if pred_has: ha_fp += 1
            else:        ha_tn += 1

        per_cat[category]["n"]       += 1
        per_cat[category]["em"]      += em
        per_cat[category]["f1"]      += f1
        per_cat[category]["has_ans"] += int(gold_has)

        if f1 < 0.5 and len(errors) < 20:
            errors.append({
                "category":     category,
                "question":     ex["question"][:100],
                "gold":         gold_text[:150],
                "pred":         pred_text[:150],
                "has_ans_prob": round(has_ans_prob, 3),
                "f1":           round(f1, 3),
            })

    n_total = answerable_n + unanswerable_n
    overall_em = (answerable_em + unanswerable_em) / max(n_total, 1)
    overall_f1 = (answerable_f1 + unanswerable_f1) / max(n_total, 1)

    print("\n" + "=" * 70)
    print("OVERALL RESULTS (DeBERTa-v3-LARGE)")
    print("=" * 70)
    print(f"  Total examples      : {n_total}")
    print(f"  Overall F1          : {overall_f1:.4f}")
    print(f"  Overall EM          : {overall_em:.4f}")
    print(f"  Total time          : {time.time()-t0:.0f}s")

    print("\n" + "─" * 70)
    print("BREAKDOWN BY ANSWERABILITY")
    print("─" * 70)
    if answerable_n > 0:
        print(f"  Answerable   ({answerable_n} ex):")
        print(f"    F1  = {answerable_f1/answerable_n:.4f}")
        print(f"    EM  = {answerable_em/answerable_n:.4f}")
    if unanswerable_n > 0:
        print(f"  Unanswerable ({unanswerable_n} ex):")
        print(f"    F1  = {unanswerable_f1/unanswerable_n:.4f}")
        print(f"    EM  = {unanswerable_em/unanswerable_n:.4f}")

    print("\n" + "─" * 70)
    print("HASANSWER CLASSIFIER")
    print("─" * 70)
    print(f"  True Positives  : {ha_tp}")
    print(f"  False Negatives : {ha_fn}  <- missed real answers")
    print(f"  False Positives : {ha_fp}  <- hallucinated answers")
    print(f"  True Negatives  : {ha_tn}")
    if (ha_tp + ha_fp) > 0:
        print(f"  Precision = {ha_tp/(ha_tp+ha_fp):.4f}")
    if (ha_tp + ha_fn) > 0:
        print(f"  Recall    = {ha_tp/(ha_tp+ha_fn):.4f}")

    print("\n" + "─" * 70)
    print("PER-CATEGORY BREAKDOWN (top 15 most common)")
    print("─" * 70)
    print(f"  {'Category':<40} {'N':>5} {'F1':>8} {'EM':>8} {'%HasA':>7}")
    print(f"  {'-'*40} {'-'*5} {'-'*8} {'-'*8} {'-'*7}")
    for cat, stats in sorted(per_cat.items(), key=lambda kv: -kv[1]["n"])[:15]:
        n = stats["n"]
        f1 = stats["f1"] / max(n, 1)
        em = stats["em"] / max(n, 1)
        ha = stats["has_ans"] / max(n, 1)
        print(f"  {cat[:40]:<40} {n:>5} {f1:>8.4f} {em:>8.4f} {ha:>7.2%}")

    print("\n" + "─" * 70)
    print("SAMPLE OF ERRORS (F1 < 0.5, up to 10)")
    print("─" * 70)
    for i, err in enumerate(errors[:10]):
        print(f"\n  #{i+1}  [{err['category']}]  F1={err['f1']}  has_ans_prob={err['has_ans_prob']}")
        print(f"    GOLD: {err['gold']}")
        print(f"    PRED: {err['pred']}")

    out_path = os.path.join(os.path.dirname(args.ckpt), "full_eval_report_large.json")
    report = {
        "n_total":      n_total,
        "overall_f1":   overall_f1,
        "overall_em":   overall_em,
        "answerable":   {"n": answerable_n,
                         "f1": answerable_f1/max(answerable_n,1),
                         "em": answerable_em/max(answerable_n,1)},
        "unanswerable": {"n": unanswerable_n,
                         "f1": unanswerable_f1/max(unanswerable_n,1),
                         "em": unanswerable_em/max(unanswerable_n,1)},
        "has_answer":   {"tp": ha_tp, "fp": ha_fp, "tn": ha_tn, "fn": ha_fn},
        "per_category": {c: {"n": s["n"],
                             "f1": s["f1"]/max(s["n"],1),
                             "em": s["em"]/max(s["n"],1)}
                         for c, s in per_cat.items()},
        "error_samples": errors,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[full_eval] Full report saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       type=str, default="QA_deberta/checkpoints/deberta_large_best.pt")
    parser.add_argument("--data_file",  type=str, default="QA_deberta/data/val.json")
    parser.add_argument("--n_examples", type=int, default=500)
    args = parser.parse_args()
    run_eval(args)
