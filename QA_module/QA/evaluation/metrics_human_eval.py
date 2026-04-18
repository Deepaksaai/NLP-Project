"""
Metric 8 (partial) — Human evaluation scaffolding.

Two phases:

  1. `python -m QA.evaluation.metrics_human_eval --sample`
       Randomly draws 100 records from generated_answers.json and
       writes evaluation/qa/human_eval_template.json with blank
       rating fields for a reviewer to fill in. Rerunnable; existing
       ratings are preserved when re-sampling.

  2. `python -m QA.evaluation.metrics_human_eval --ingest`
       Reads the completed template, averages the four rating axes
       across answered examples, and writes
       evaluation/qa/report/end_to_end_human_eval.json.
       Unanswered rows are skipped.

Template schema per row:
    {
      "example_id": str,
      "question":   str,
      "document_excerpt": str,          # first 800 chars
      "gold_answers": [str, ...],
      "predicted_answer": str,
      "rating_correctness":  null,      # 1..5
      "rating_faithfulness": null,
      "rating_clarity":      null,
      "rating_completeness": null,
      "notes":               ""
    }
"""

import os
import json
import random
import argparse
from typing import List, Dict


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(_ROOT, "evaluation", "qa")
TEMPLATE_PATH = os.path.join(EVAL_DIR, "human_eval_template.json")
REPORT_PATH   = os.path.join(EVAL_DIR, "report", "end_to_end_human_eval.json")


def sample_for_review(sample_size: int = 100, seed: int = 0):
    with open(os.path.join(EVAL_DIR, "generated_answers.json")) as f:
        records = json.load(f)
    random.seed(seed)
    picked = random.sample(records, min(sample_size, len(records)))

    # Preserve any existing ratings so re-sampling doesn't wipe them
    existing = {}
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH) as f:
            for row in json.load(f):
                existing[row["example_id"]] = row

    template = []
    for r in picked:
        row = existing.get(r["example_id"], {})
        template.append({
            "example_id":        r["example_id"],
            "source":            r["source"],
            "question":          r["question"],
            "document_excerpt":  r["document"][:800],
            "gold_answers":      r["gold_answers"],
            "predicted_answer":  r.get("predicted_answer", ""),
            "extracted_span":    r.get("extracted_span", ""),
            "rating_correctness":  row.get("rating_correctness",  None),
            "rating_faithfulness": row.get("rating_faithfulness", None),
            "rating_clarity":      row.get("rating_clarity",      None),
            "rating_completeness": row.get("rating_completeness", None),
            "notes":               row.get("notes", ""),
        })

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"[human_eval] wrote {len(template)} rows -> {TEMPLATE_PATH}")
    print("[human_eval] reviewer: fill in rating_* fields (1-5) then run with --ingest")


def ingest() -> Dict:
    if not os.path.exists(TEMPLATE_PATH):
        print(f"[human_eval] template not found: {TEMPLATE_PATH}")
        return {}
    with open(TEMPLATE_PATH) as f:
        rows = json.load(f)

    axes = ["correctness", "faithfulness", "clarity", "completeness"]
    sums = {a: 0.0 for a in axes}
    counts = {a: 0 for a in axes}
    rated_rows = 0

    for row in rows:
        any_rating = False
        for a in axes:
            v = row.get(f"rating_{a}")
            if v is None:
                continue
            any_rating = True
            sums[a] += float(v)
            counts[a] += 1
        if any_rating:
            rated_rows += 1

    def avg(a): return sums[a] / counts[a] if counts[a] else None
    result = {
        "n_rated_rows": rated_rows,
        "n_total_rows": len(rows),
        "mean_correctness":  avg("correctness"),
        "mean_faithfulness": avg("faithfulness"),
        "mean_clarity":      avg("clarity"),
        "mean_completeness": avg("completeness"),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true")
    p.add_argument("--ingest", action="store_true")
    p.add_argument("--sample_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.sample:
        sample_for_review(sample_size=args.sample_size, seed=args.seed)
    elif args.ingest:
        ingest()
    else:
        print("usage: --sample | --ingest")


if __name__ == "__main__":
    main()
