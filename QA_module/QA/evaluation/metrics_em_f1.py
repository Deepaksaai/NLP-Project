"""
Metric 1 + Metric 2 — Exact Match and token-level F1.

Reads evaluation/qa/generated_answers.json and writes:
    evaluation/qa/report/exact_match_scores.json
    evaluation/qa/report/f1_scores.json

Scoring rules:
  * SQuAD-style normalization (lowercase, strip punct/articles/whitespace).
  * For examples with multiple gold answers, take the max EM/F1.
  * For UNANSWERABLE gold: EM=F1=1 iff model returned empty prediction.
  * Per-split breakdown (cuad / coliee / ledgar / squad) plus
    answerable-only / unanswerable-only splits.
"""

import os
import json
import re
import string
import collections
from typing import List, Dict

from QA.training.evaluate import _normalize, _f1_one, _em_one


def _em_max(pred: str, golds: List[str]) -> float:
    if not golds:
        return 1.0 if not pred.strip() else 0.0
    return max(_em_one(pred, g) for g in golds)


def _f1_max(pred: str, golds: List[str]) -> float:
    if not golds:
        return 1.0 if not pred.strip() else 0.0
    return max(_f1_one(pred, g) for g in golds)


def compute(records: List[Dict]) -> Dict:
    by_source = collections.defaultdict(lambda: {"em": 0.0, "f1": 0.0, "n": 0})
    ans   = {"em": 0.0, "f1": 0.0, "n": 0}
    unans = {"em": 0.0, "f1": 0.0, "n": 0}
    total = {"em": 0.0, "f1": 0.0, "n": 0}

    for r in records:
        pred = r.get("predicted_answer") or ""
        golds = r.get("gold_answers") or []
        em = _em_max(pred, golds)
        f1 = _f1_max(pred, golds)

        bucket = ans if r["is_answerable"] else unans
        bucket["em"] += em
        bucket["f1"] += f1
        bucket["n"]  += 1

        src = by_source[r["source"]]
        src["em"] += em
        src["f1"] += f1
        src["n"]  += 1

        total["em"] += em
        total["f1"] += f1
        total["n"]  += 1

    def sd(a, b): return a / b if b > 0 else 0.0

    out_em = {
        "em_overall": sd(total["em"], total["n"]),
        "em_answerable_only":   sd(ans["em"],   ans["n"]),
        "em_unanswerable_only": sd(unans["em"], unans["n"]),
        "n_total":       total["n"],
        "n_answerable":  ans["n"],
        "n_unanswerable": unans["n"],
    }
    out_f1 = {
        "f1_overall": sd(total["f1"], total["n"]),
        "f1_answerable_only":   sd(ans["f1"],   ans["n"]),
        "f1_unanswerable_only": sd(unans["f1"], unans["n"]),
        "n_total":       total["n"],
        "n_answerable":  ans["n"],
        "n_unanswerable": unans["n"],
    }
    for src, v in by_source.items():
        out_em[f"em_{src}"] = sd(v["em"], v["n"])
        out_f1[f"f1_{src}"] = sd(v["f1"], v["n"])
        out_em[f"n_{src}"] = v["n"]
        out_f1[f"n_{src}"] = v["n"]

    return {"exact_match": out_em, "f1": out_f1}


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "evaluation", "qa")
    with open(os.path.join(root, "generated_answers.json"), "r", encoding="utf-8") as f:
        records = json.load(f)
    result = compute(records)
    os.makedirs(os.path.join(root, "report"), exist_ok=True)
    with open(os.path.join(root, "report", "exact_match_scores.json"), "w") as f:
        json.dump(result["exact_match"], f, indent=2)
    with open(os.path.join(root, "report", "f1_scores.json"), "w") as f:
        json.dump(result["f1"], f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
