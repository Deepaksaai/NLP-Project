"""
Stage 4 Step 3 — Sample 50 random examples for manual qualitative review.

Produces a human-readable file showing source/reference/generated triples
plus a scoring template for the 7 dimensions:
  fluency, coherence, faithfulness, completeness,
  conciseness, legal accuracy, outcome correctness

Score each 1-5, average across 50 examples.

Usage:
    python -m evaluation.qualitative_sample
"""

import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


SCORING_TEMPLATE = """
Scoring (1=poor, 5=excellent):
  fluency        : __  (grammatical correctness)
  coherence      : __  (logical flow between sentences)
  faithfulness   : __  (no contradictions of the source)
  completeness   : __  (all key facts present)
  conciseness    : __  (no unnecessary repetition)
  legal_accuracy : __  (legal terms used correctly)
  outcome        : __  (ruling/verdict stated correctly)
"""


def main():
    eval_dir = "evaluation"
    out_path = "evaluation/qualitative_review.txt"
    json_out = "evaluation/qualitative_review.json"
    n_samples = 50
    seed = 42

    with open(os.path.join(eval_dir, "generated_summaries.json")) as f:
        generated = json.load(f)
    with open(os.path.join(eval_dir, "reference_summaries.json")) as f:
        references = json.load(f)
    with open(os.path.join(eval_dir, "source_documents.json")) as f:
        sources = json.load(f)

    test_ids = [tid for tid in generated if generated[tid].strip()]
    random.seed(seed)
    sampled = random.sample(test_ids, min(n_samples, len(test_ids)))

    json_records = []

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"QUALITATIVE EVALUATION — {len(sampled)} examples\n")
        f.write("=" * 70 + "\n\n")
        f.write(
            "Read each example. Score each dimension 1-5.\n"
            "Average across all examples for the final qualitative report.\n\n"
        )

        for i, tid in enumerate(sampled, 1):
            src = sources.get(tid, "")
            ref = references.get(tid, "")
            gen = generated.get(tid, "")

            # Truncate source to keep file readable
            src_preview = src[:1500] + ("..." if len(src) > 1500 else "")

            f.write("\n" + "─" * 70 + "\n")
            f.write(f"Example {i}/{len(sampled)}  (test_id: {tid})\n")
            f.write("─" * 70 + "\n")
            f.write(f"\n[SOURCE — first 1500 chars of {len(src)} total]\n")
            f.write(src_preview + "\n")
            f.write(f"\n[REFERENCE]\n")
            f.write(ref + "\n")
            f.write(f"\n[GENERATED]\n")
            f.write(gen + "\n")
            f.write(SCORING_TEMPLATE)
            f.write("\n")

            json_records.append({
                "test_id": tid,
                "source_preview": src_preview,
                "reference": ref,
                "generated": gen,
            })

    with open(json_out, "w") as f:
        json.dump(json_records, f, indent=2)

    print(f"Saved {len(sampled)} examples to:")
    print(f"  {out_path}     (human-readable for scoring)")
    print(f"  {json_out}     (JSON for programmatic access)")


if __name__ == "__main__":
    main()
