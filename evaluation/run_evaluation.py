"""
Stage 4 Step 2 — Run all 6 metrics on saved generated outputs.

Loads:
  evaluation/generated_summaries.json
  evaluation/reference_summaries.json
  evaluation/source_documents.json

Runs all 6 metrics and saves:
  evaluation/report/rouge_scores.json
  evaluation/report/bertscore_scores.json
  evaluation/report/faithfulness_scores.json
  evaluation/report/entity_coverage_scores.json
  evaluation/report/outcome_preservation.json
  evaluation/report/length_analysis.json
  evaluation/report/summary_report.json     ← all metrics combined
  evaluation/report/flagged_examples.json   ← examples that failed any metric

Usage:
    python -m evaluation.run_evaluation                       # all metrics
    python -m evaluation.run_evaluation --skip bertscore      # skip slow metric
    python -m evaluation.run_evaluation --only rouge length   # only fast ones
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import (
    compute_rouge,
    compute_bertscore,
    compute_faithfulness,
    compute_entity_coverage,
    compute_outcome_preservation,
    compute_length_analysis,
    detect_outcome,
    split_sentences,
)


METRIC_FNS = {
    "rouge": compute_rouge,
    "bertscore": compute_bertscore,
    "faithfulness": compute_faithfulness,
    "entity_coverage": compute_entity_coverage,
    "outcome": compute_outcome_preservation,
    "length": compute_length_analysis,
}


def load_data(eval_dir):
    print("Loading saved outputs...")
    with open(os.path.join(eval_dir, "generated_summaries.json")) as f:
        generated = json.load(f)
    with open(os.path.join(eval_dir, "reference_summaries.json")) as f:
        references = json.load(f)
    with open(os.path.join(eval_dir, "source_documents.json")) as f:
        sources = json.load(f)
    print(f"  {len(generated)} generated, {len(references)} references, "
          f"{len(sources)} sources")
    return generated, references, sources


def run_metric(name, generated, references, sources):
    """Dispatch to the right metric function with appropriate args."""
    if name == "rouge":
        return compute_rouge(generated, references)
    if name == "bertscore":
        return compute_bertscore(generated, references)
    if name == "faithfulness":
        return compute_faithfulness(generated, sources)
    if name == "entity_coverage":
        return compute_entity_coverage(generated, sources)
    if name == "outcome":
        return compute_outcome_preservation(generated, sources)
    if name == "length":
        return compute_length_analysis(generated, sources)
    return {"error": f"unknown metric {name}"}


def find_flagged_examples(generated, references, sources, max_flag=20):
    """
    Identify examples that fail one or more quality criteria:
      - empty summary
      - very short summary (< 30 words)
      - very long summary (> 500 words)
      - outcome inversion (plaintiff <-> defendant)
    """
    flagged = []
    for tid in generated:
        gen = generated[tid]
        ref = references.get(tid, "")
        src = sources.get(tid, "")
        issues = []

        if not gen.strip():
            issues.append("empty_summary")
        else:
            words = len(gen.split())
            if words < 30:
                issues.append(f"too_short:{words}w")
            elif words > 500:
                issues.append(f"too_long:{words}w")

        if src:
            last = src[int(len(src) * 0.8):]
            src_outcome = detect_outcome(last)
            gen_outcome = detect_outcome(gen)
            if (src_outcome == "plaintiff" and gen_outcome == "defendant") or \
               (src_outcome == "defendant" and gen_outcome == "plaintiff"):
                issues.append(f"outcome_inverted:{src_outcome}->{gen_outcome}")

        if issues:
            flagged.append({
                "test_id": tid,
                "issues": issues,
                "generated": gen[:300],
                "reference": ref[:300],
            })
            if len(flagged) >= max_flag:
                break
    return flagged


def print_target_check(report):
    """Compare against target ranges from the spec."""
    print("\n" + "=" * 60)
    print("TARGET CHECK")
    print("=" * 60)

    targets = {
        "rouge1_f1": (0.30, 0.45, ">"),
        "rouge2_f1": (0.10, 0.20, ">"),
        "rougeL_f1": (0.25, 0.40, ">"),
        "bertscore_f1": (0.80, 1.0, ">"),
        "mean_faithfulness": (0.75, 1.0, ">"),
        "hallucination_rate": (0.0, 0.15, "<"),
        "overall_entity_coverage": (0.60, 1.0, ">"),
        "person_coverage": (0.70, 1.0, ">"),
        "outcome_preservation_rate": (0.70, 1.0, ">"),
        "outcome_inversion_rate": (0.0, 0.05, "<"),
        "mean_summary_words": (100, 300, "range"),
    }

    for key, (lo, hi, kind) in targets.items():
        val = report.get(key)
        if val is None:
            continue
        if kind == ">":
            status = "PASS" if val >= lo else "FAIL"
        elif kind == "<":
            status = "PASS" if val <= hi else "FAIL"
        else:
            status = "PASS" if lo <= val <= hi else "FAIL"
        print(f"  {key:30s} : {val:.4f}  [{status}]  target {kind} {lo}-{hi}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="evaluation")
    parser.add_argument("--report_dir", default="evaluation/report")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Metrics to skip")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these metrics")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)

    generated, references, sources = load_data(args.eval_dir)

    metric_names = list(METRIC_FNS.keys())
    if args.only:
        metric_names = [m for m in metric_names if m in args.only]
    if args.skip:
        metric_names = [m for m in metric_names if m not in args.skip]

    print(f"\nRunning metrics: {metric_names}")

    summary_report = {
        "stage": "stage3_best",
        "test_examples": len(generated),
    }

    all_results = {}
    for name in metric_names:
        print(f"\n{'='*60}")
        print(f"  Metric: {name}")
        print(f"{'='*60}")
        try:
            result = run_metric(name, generated, references, sources)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {"error": str(e)}

        all_results[name] = result

        # Save individual report
        out_path = os.path.join(args.report_dir, f"{name}_scores.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {out_path}")

        # Add key metrics to summary
        for key, val in result.items():
            if isinstance(val, (int, float)) and not key.startswith("n_"):
                summary_report[key] = val

    # Save combined report
    with open(os.path.join(args.report_dir, "summary_report.json"), "w") as f:
        json.dump(summary_report, f, indent=2)

    # Find and save flagged examples
    print("\nFinding flagged examples...")
    flagged = find_flagged_examples(generated, references, sources, max_flag=50)
    with open(os.path.join(args.report_dir, "flagged_examples.json"), "w") as f:
        json.dump(flagged, f, indent=2)
    print(f"  {len(flagged)} flagged")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    for key, val in summary_report.items():
        if isinstance(val, float):
            print(f"  {key:30s} : {val:.4f}")
        else:
            print(f"  {key:30s} : {val}")

    print_target_check(summary_report)

    print("\n" + "=" * 60)
    print(f"Full report: {args.report_dir}/summary_report.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
