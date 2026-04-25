"""
Metric 3 — Has-answer performance + calibration curve.

Reads evaluation/qa/generated_answers.json and writes:
    evaluation/qa/report/has_answer_analysis.json
    evaluation/qa/report/calibration_curve.json

Reports:
  * TP/TN/FP/FN at the default 0.5 threshold
  * Per-split breakdown: legal (cuad+coliee+ledgar) vs general (squad)
  * Cross-document vs wrong-clause negative subsets (if tagged)
  * Calibration curve: FPR/TPR/accuracy across thresholds 0.30..0.90
  * Recommended deployment threshold — the lowest threshold that keeps
    legal false positive rate < 0.08 while holding TPR > 0.75.
"""

import os
import json
import collections
from typing import List, Dict


LEGAL_SOURCES = ("cuad", "coliee", "ledgar")


def _decide(prob: float, threshold: float, extracted: str) -> bool:
    """Final decision: answerable iff has_answer_prob above threshold AND non-empty span."""
    return prob >= threshold and bool(extracted.strip())


def _confusion(records: List[Dict], threshold: float, source_filter=None):
    tp = tn = fp = fn = 0
    for r in records:
        if source_filter is not None and r["source"] not in source_filter:
            continue
        gold = bool(r["is_answerable"])
        pred = _decide(float(r.get("has_answer_prob", 0.0)), threshold,
                       r.get("extracted_span", ""))
        if   gold and pred:         tp += 1
        elif (not gold) and (not pred): tn += 1
        elif (not gold) and pred:   fp += 1
        elif gold and (not pred):   fn += 1
    n = tp + tn + fp + fn
    def sd(a, b): return a / b if b > 0 else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
        "accuracy":            sd(tp + tn, n),
        "true_positive_rate":  sd(tp, tp + fn),
        "true_negative_rate":  sd(tn, tn + fp),
        "false_positive_rate": sd(fp, fp + tn),
        "false_negative_rate": sd(fn, fn + tp),
    }


def compute(records: List[Dict]):
    overall = _confusion(records, 0.5)
    legal = _confusion(records, 0.5, source_filter=set(LEGAL_SOURCES))
    general = _confusion(records, 0.5, source_filter={"squad"})

    # Subset-specific FP analysis: need neg_kind tag on the records.
    # We inspect example_id suffixes / source extras — if not present,
    # these fields are empty.
    xdoc = wrong_clause = None
    # Records from preprocessed files don't flow through here — we're
    # scoring the raw test splits, which don't have synthetic negatives.
    # Left intentionally null to be filled in if/when synthetic negatives
    # are added to the test set.

    # Calibration sweep
    sweep = []
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        row = {
            "threshold": t,
            "overall":   _confusion(records, t),
            "legal":     _confusion(records, t, source_filter=set(LEGAL_SOURCES)),
            "general":   _confusion(records, t, source_filter={"squad"}),
        }
        sweep.append(row)

    # Recommended threshold: lowest t with legal FPR < 0.08 and legal TPR > 0.75
    recommended = None
    for row in sweep:
        legal_row = row["legal"]
        if legal_row["false_positive_rate"] < 0.08 and legal_row["true_positive_rate"] > 0.75:
            recommended = row["threshold"]
            break
    if recommended is None:
        # Fallback: whichever threshold minimizes legal FPR while TPR > 0.6
        feasible = [r for r in sweep
                    if r["legal"]["true_positive_rate"] > 0.60]
        if feasible:
            best = min(feasible, key=lambda r: r["legal"]["false_positive_rate"])
            recommended = best["threshold"]

    return {
        "overall":                overall,
        "legal":                  legal,
        "general":                general,
        "cross_document_negatives": xdoc,
        "wrong_clause_negatives":   wrong_clause,
        "calibration_curve":      sweep,
        "recommended_threshold":  recommended,
    }


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "evaluation", "qa")
    with open(os.path.join(root, "generated_answers.json"), "r", encoding="utf-8") as f:
        records = json.load(f)
    result = compute(records)
    os.makedirs(os.path.join(root, "report"), exist_ok=True)
    with open(os.path.join(root, "report", "has_answer_analysis.json"), "w") as f:
        json.dump({k: v for k, v in result.items() if k != "calibration_curve"}, f, indent=2)
    with open(os.path.join(root, "report", "calibration_curve.json"), "w") as f:
        json.dump({"sweep": result["calibration_curve"],
                   "recommended_threshold": result["recommended_threshold"]}, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "calibration_curve"}, indent=2))


if __name__ == "__main__":
    main()
