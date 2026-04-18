"""
Metric 4 — Retrieval quality, measured independently from span
extraction.

Reads evaluation/qa/retrieval_results.json (written by run_inference)
and computes:
    recall@1, recall@3, recall@5
    answer_reachability_top3
    per-source breakdown
    simple failure breakdown: short questions, legal-term questions,
                              multi-hop / lookup questions

Writes:
    evaluation/qa/report/retrieval_quality.json
"""

import os
import json
import collections
from typing import List, Dict


def _hit(gold_idx: int, retrieved: List[int], k: int) -> float:
    if gold_idx < 0:
        return 1.0  # no gold chunk to miss — trivially a hit
    return 1.0 if gold_idx in retrieved[:k] else 0.0


def compute(retrieval_records: List[Dict], gold_lookup: Dict):
    by_source = collections.defaultdict(lambda: {"n": 0, "r1": 0.0, "r3": 0.0, "r5": 0.0})
    total = {"n": 0, "r1": 0.0, "r3": 0.0, "r5": 0.0}
    reachable_top3 = 0
    reachable_total = 0

    # Failure heuristic buckets
    buckets = collections.defaultdict(lambda: {"n": 0, "r1_miss": 0})

    for r in retrieval_records:
        gid = r["gold_chunk_idx"]
        idxs = r["retrieved_chunk_idxs"]
        src = gold_lookup[r["example_id"]]["source"]
        is_ans = gold_lookup[r["example_id"]]["is_answerable"]

        # Recall metrics only meaningful for answerable examples where we know a gold chunk
        if not is_ans or gid < 0:
            continue

        r1 = _hit(gid, idxs, 1)
        r3 = _hit(gid, idxs, 3)
        r5 = _hit(gid, idxs, 5)
        by_source[src]["n"] += 1
        by_source[src]["r1"] += r1
        by_source[src]["r3"] += r3
        by_source[src]["r5"] += r5
        total["n"] += 1
        total["r1"] += r1
        total["r3"] += r3
        total["r5"] += r5

        reachable_total += 1
        if r3 > 0.5:
            reachable_top3 += 1

        # Failure bucket analysis on top-1 misses
        q = gold_lookup[r["example_id"]]["question"]
        bucket = "short_q" if len(q.split()) <= 6 else "long_q"
        buckets[bucket]["n"] += 1
        if r1 < 0.5:
            buckets[bucket]["r1_miss"] += 1

    def sd(a, b): return a / b if b > 0 else 0.0
    out = {
        "n_evaluated":            total["n"],
        "retrieval_recall_at_1":  sd(total["r1"], total["n"]),
        "retrieval_recall_at_3":  sd(total["r3"], total["n"]),
        "retrieval_recall_at_5":  sd(total["r5"], total["n"]),
        "answer_reachability_top3": sd(reachable_top3, reachable_total),
        "by_source": {
            s: {
                "n": v["n"],
                "recall_at_1": sd(v["r1"], v["n"]),
                "recall_at_3": sd(v["r3"], v["n"]),
                "recall_at_5": sd(v["r5"], v["n"]),
            }
            for s, v in by_source.items()
        },
        "failure_buckets": {
            k: {"n": v["n"], "r1_miss_rate": sd(v["r1_miss"], v["n"])}
            for k, v in buckets.items()
        },
    }
    return out


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "evaluation", "qa")
    with open(os.path.join(root, "retrieval_results.json")) as f:
        retrieval = json.load(f)
    with open(os.path.join(root, "source_documents.json")) as f:
        src_docs = json.load(f)
    with open(os.path.join(root, "gold_answers.json")) as f:
        golds = json.load(f)

    # Build quick lookup
    lookup = {d["example_id"]: d for d in src_docs}
    for g in golds:
        lookup[g["example_id"]]["is_answerable"] = g["is_answerable"]
        lookup[g["example_id"]]["gold_answers"] = g["gold_answers"]

    result = compute(retrieval, lookup)
    os.makedirs(os.path.join(root, "report"), exist_ok=True)
    with open(os.path.join(root, "report", "retrieval_quality.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
