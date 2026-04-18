"""
Metric 5 — Clause boundary accuracy (full / partial + error breakdown).

For every ANSWERABLE test example where the model produced a
non-empty span, checks whether the predicted span starts and ends
at clause boundaries (sentence terminators, paragraph breaks, or
section markers like "(a)"/"1."/"Article N").

Scoring:
    2 boundaries correct -> 1.0
    1 boundary correct   -> 0.5
    0 correct            -> 0.0

Writes:
    evaluation/qa/report/clause_boundary_accuracy.json
"""

import os
import json
import collections
from typing import List, Dict

from QA.training.legal_metrics import _starts_at_boundary, _ends_at_boundary


def _locate_in_document(document: str, pred_text: str):
    """Return (start_char, end_char) if pred_text appears in document, else (-1,-1)."""
    if not pred_text:
        return -1, -1
    idx = document.find(pred_text)
    if idx < 0:
        return -1, -1
    return idx, idx + len(pred_text)


def compute(records: List[Dict]) -> Dict:
    n = 0
    full_sum = 0.0
    partial_sum = 0.0
    start_err = 0
    end_err = 0

    by_source = collections.defaultdict(lambda: {"n": 0, "full": 0.0, "partial": 0.0})

    for r in records:
        if not r["is_answerable"]:
            continue
        pred = r.get("extracted_span") or ""
        if not pred:
            continue
        doc = r["document"]
        s, e = _locate_in_document(doc, pred)
        if s < 0:
            # can't evaluate boundaries if prediction isn't in the document
            continue

        starts_ok = _starts_at_boundary(doc, s)
        ends_ok   = _ends_at_boundary(doc, e)
        score = (1.0 if starts_ok else 0.0) * 0.5 + (1.0 if ends_ok else 0.0) * 0.5

        n += 1
        full_sum    += 1.0 if (starts_ok and ends_ok) else 0.0
        partial_sum += 1.0 if (starts_ok or ends_ok)  else 0.0
        if not starts_ok: start_err += 1
        if not ends_ok:   end_err   += 1

        src = by_source[r["source"]]
        src["n"] += 1
        src["full"]    += 1.0 if (starts_ok and ends_ok) else 0.0
        src["partial"] += 1.0 if (starts_ok or ends_ok)  else 0.0

    def sd(a, b): return a / b if b > 0 else 0.0

    which_is_worse = (
        "start" if start_err > end_err else "end" if end_err > start_err else "tied"
    )
    return {
        "n_evaluated": n,
        "clause_boundary_accuracy_full":    sd(full_sum,    n),
        "clause_boundary_accuracy_partial": sd(partial_sum, n),
        "boundary_error_analysis": {
            "start_boundary_errors": start_err,
            "end_boundary_errors":   end_err,
            "which_is_worse":        which_is_worse,
        },
        "by_source": {
            s: {
                "n":       v["n"],
                "full":    sd(v["full"],    v["n"]),
                "partial": sd(v["partial"], v["n"]),
            }
            for s, v in by_source.items()
        },
    }


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "evaluation", "qa")
    with open(os.path.join(root, "generated_answers.json")) as f:
        records = json.load(f)
    result = compute(records)
    os.makedirs(os.path.join(root, "report"), exist_ok=True)
    with open(os.path.join(root, "report", "clause_boundary_accuracy.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
