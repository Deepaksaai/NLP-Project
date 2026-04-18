"""
Failure mode analysis — bucketizes every failing prediction into one
of the six categories from the Stage-4 spec:

    1. Retrieval failure        (gold chunk not in retrieved top-k)
    2. Span boundary error      (predicted span high token overlap
                                 with gold but wrong boundaries)
    3. Has-answer false positive (unanswerable question, model answered)
    4. Legal term hallucination (prediction contains entities not
                                 present in gold)
    5. Generation unfaithfulness(plain English contradicts span)
    6. Coreference failure      (only populated by the conversation
                                 evaluator — this module marks those
                                 as a separate 'not applicable' bucket)

Flags a small sample (default 50) per category for manual review.

Writes:
    evaluation/qa/report/flagged_examples.json
    evaluation/qa/report/failure_breakdown.json
"""

import os
import json
import random
import collections
from typing import List, Dict

from QA.training.evaluate import _f1_one, _em_one
from QA.evaluation.metrics_legal_terms import _extract_categorized


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(_ROOT, "evaluation", "qa")


def _classify(record: Dict) -> str:
    pred = record.get("extracted_span") or ""
    plain = record.get("predicted_answer") or ""
    golds = record.get("gold_answers") or []
    gold_chunk = record.get("gold_chunk_idx", -1)
    retrieved = record.get("retrieval_top_k_idxs") or []
    is_answerable = record.get("is_answerable", False)

    # Category 3 — has-answer FP
    if (not is_answerable) and pred.strip():
        return "has_answer_false_positive"

    if not is_answerable:
        return "ok"

    # Category 1 — retrieval failure
    if gold_chunk >= 0 and gold_chunk not in retrieved:
        return "retrieval_failure"

    # No prediction returned for an answerable question -> retrieval or FN
    if not pred.strip():
        return "missed_answer"

    # Compare against each gold answer
    best_f1 = max((_f1_one(pred, g) for g in golds), default=0.0)
    best_em = max((_em_one(pred, g) for g in golds), default=0.0)

    # Category 2 — span boundary error (high F1 but not EM)
    if best_f1 > 0.65 and best_em < 0.5:
        return "span_boundary_error"

    # Category 4 — legal term hallucination
    pred_terms = _extract_categorized(pred)
    gold_terms_union = {k: set() for k in pred_terms}
    for g in golds:
        g_terms = _extract_categorized(g)
        for k in g_terms:
            gold_terms_union[k] |= g_terms[k]
    hallucinated = False
    for k in ("defined", "monetary", "time"):
        extra = pred_terms[k] - gold_terms_union[k]
        if extra:
            hallucinated = True
            break
    if hallucinated and best_f1 < 0.40:
        return "legal_term_hallucination"

    # Category 5 — generation unfaithfulness (only meaningful if plain != span)
    if plain and plain.strip() != pred.strip():
        bow_pred = set(w.lower() for w in pred.split())
        bow_plain = set(w.lower() for w in plain.split())
        if bow_plain and len(bow_pred & bow_plain) / len(bow_plain) < 0.35:
            return "generation_unfaithful"

    if best_f1 < 0.20:
        return "low_quality_span"
    return "ok"


def compute(records: List[Dict], samples_per_category: int = 50) -> Dict:
    counts = collections.Counter()
    buckets = collections.defaultdict(list)

    for r in records:
        label = _classify(r)
        counts[label] += 1
        buckets[label].append(r)

    flagged = {}
    random.seed(0)
    for label, items in buckets.items():
        if label == "ok":
            continue
        sample = random.sample(items, min(samples_per_category, len(items)))
        flagged[label] = [
            {
                "example_id":       x["example_id"],
                "source":           x["source"],
                "question":         x["question"],
                "gold_answers":     x.get("gold_answers", []),
                "extracted_span":   x.get("extracted_span", ""),
                "predicted_answer": x.get("predicted_answer", ""),
                "has_answer_prob":  x.get("has_answer_prob", None),
                "gold_chunk_idx":   x.get("gold_chunk_idx", -1),
                "retrieval_top_k":  x.get("retrieval_top_k_idxs", []),
            }
            for x in sample
        ]

    breakdown = {
        "total": sum(counts.values()),
        "category_counts": dict(counts),
        "category_fractions": {
            k: round(v / max(1, sum(counts.values())), 4)
            for k, v in counts.items()
        },
    }
    return {"breakdown": breakdown, "flagged": flagged}


def main():
    with open(os.path.join(EVAL_DIR, "generated_answers.json")) as f:
        records = json.load(f)
    result = compute(records)
    os.makedirs(os.path.join(EVAL_DIR, "report"), exist_ok=True)
    with open(os.path.join(EVAL_DIR, "report", "failure_breakdown.json"), "w") as f:
        json.dump(result["breakdown"], f, indent=2)
    with open(os.path.join(EVAL_DIR, "report", "flagged_examples.json"), "w") as f:
        json.dump(result["flagged"], f, indent=2)
    print(json.dumps(result["breakdown"], indent=2))


if __name__ == "__main__":
    main()
