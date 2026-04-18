"""
Chunk retrieval metric for Stage 2 validation.

Given a Stage2QADataset where multiple chunks share the same
question_id, this module runs the model over every chunk, ranks them
by has_answer score, and measures how often the chunk that actually
contains the answer is ranked top-1 / top-3.

Only questions with at least one answerable chunk are counted — if
every chunk for a question is marked unanswerable (e.g. the QASPER
example was yes/no), the question is excluded from this metric.
"""

import torch
from collections import defaultdict
from typing import Dict


@torch.no_grad()
def chunk_retrieval_accuracy(
    model,
    dataset,
    device,
    batch_size: int = 16,
) -> Dict[str, float]:
    model.eval()

    # Group example indices by question_id
    groups = defaultdict(list)
    for i, ex in enumerate(dataset.examples):
        groups[ex["question_id"]].append(i)

    # Score every example once. Accumulate has_answer scores per index
    scores = torch.zeros(len(dataset), dtype=torch.float32)
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        items = [dataset[i] for i in range(start, end)]
        batch = {
            "input_ids":      torch.stack([x["input_ids"]      for x in items]).to(device),
            "segment_ids":    torch.stack([x["segment_ids"]    for x in items]).to(device),
            "attention_mask": torch.stack([x["attention_mask"] for x in items]).to(device),
        }
        _, _, has_answer_logits = model(
            batch["input_ids"], batch["segment_ids"], batch["attention_mask"],
        )
        probs = torch.sigmoid(has_answer_logits).detach().cpu()
        scores[start:end] = probs

    n_valid = 0
    top1 = 0
    top3 = 0
    by_source = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})

    for q_id, idxs in groups.items():
        # Must have at least one positive chunk to be counted
        pos_indices = [i for i in idxs if dataset.examples[i]["is_answerable"]]
        if not pos_indices or len(idxs) < 2:
            continue

        pool_scores = scores[idxs]
        ranked_local = torch.argsort(pool_scores, descending=True).tolist()
        ranked_global = [idxs[k] for k in ranked_local]

        positive_set = set(pos_indices)
        rank = next(
            (r for r, gi in enumerate(ranked_global) if gi in positive_set),
            None,
        )
        if rank is None:
            continue

        src = dataset.examples[idxs[0]]["source"]
        by_source[src]["n"] += 1
        n_valid += 1
        if rank == 0:
            top1 += 1
            by_source[src]["top1"] += 1
        if rank < 3:
            top3 += 1
            by_source[src]["top3"] += 1

    def sd(a, b): return a / b if b > 0 else 0.0

    per_src = {
        s: {
            "n": v["n"],
            "top1": sd(v["top1"], v["n"]),
            "top3": sd(v["top3"], v["n"]),
        }
        for s, v in by_source.items()
    }

    return {
        "n_questions": n_valid,
        "top1": sd(top1, n_valid),
        "top3": sd(top3, n_valid),
        "by_source": per_src,
    }
