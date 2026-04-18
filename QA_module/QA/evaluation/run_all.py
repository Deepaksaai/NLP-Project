"""
Stage 4 orchestrator — runs every metric that can be computed from
the saved inference outputs and writes a consolidated
summary_report.json.

Prerequisites:
    1. checkpoints/qa_stage3_best.pt exists
    2. python -m QA.evaluation.run_inference (writes generated_answers.json etc.)

Then:
    python -m QA.evaluation.run_all
    # followed by (separately, because they need a human / reviewer)
    python -m QA.evaluation.metrics_human_eval --sample
    # <reviewer fills in ratings>
    python -m QA.evaluation.metrics_human_eval --ingest
    # and (optional) re-run to fold human scores into summary_report.json
    python -m QA.evaluation.run_all

The orchestrator is safe to rerun — it overwrites every report file.
"""

import os
import json
import traceback
from typing import Dict

from QA.evaluation import (
    metrics_em_f1,
    metrics_has_answer,
    metrics_retrieval,
    metrics_clause_boundary,
    metrics_legal_terms,
    metrics_generation,
    failure_analysis,
)


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(_ROOT, "evaluation", "qa")
REPORT_DIR = os.path.join(EVAL_DIR, "report")


def _load(name):
    with open(os.path.join(EVAL_DIR, name)) as f:
        return json.load(f)


def _safe(section: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        traceback.print_exc()
        return {"error": f"{type(e).__name__}: {e}", "section": section}


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    generated_path = os.path.join(EVAL_DIR, "generated_answers.json")
    if not os.path.exists(generated_path):
        raise FileNotFoundError(
            "evaluation/qa/generated_answers.json not found — "
            "run `python -m QA.evaluation.run_inference` first"
        )

    records = _load("generated_answers.json")
    print(f"[run_all] {len(records)} records loaded")

    # --- core metrics ---
    em_f1 = _safe("em_f1", metrics_em_f1.compute, records)
    has_ans = _safe("has_answer", metrics_has_answer.compute, records)

    # retrieval needs the side files
    retrieval_records = _load("retrieval_results.json")
    src_docs = _load("source_documents.json")
    golds    = _load("gold_answers.json")
    lookup = {d["example_id"]: d for d in src_docs}
    for g in golds:
        lookup[g["example_id"]]["is_answerable"] = g["is_answerable"]
        lookup[g["example_id"]]["gold_answers"]  = g["gold_answers"]
    retrieval = _safe("retrieval", metrics_retrieval.compute, retrieval_records, lookup)

    clause = _safe("clause_boundary", metrics_clause_boundary.compute, records)
    legal_terms = _safe("legal_terms",  metrics_legal_terms.compute, records)
    generation = _safe("generation",    metrics_generation.compute, records)
    failures = _safe("failures",        failure_analysis.compute, records)

    # --- write per-metric reports ---
    def _dump(name, obj):
        with open(os.path.join(REPORT_DIR, name), "w") as f:
            json.dump(obj, f, indent=2)

    if "exact_match" in em_f1:
        _dump("exact_match_scores.json", em_f1["exact_match"])
        _dump("f1_scores.json", em_f1["f1"])
    _dump("has_answer_analysis.json", {k: v for k, v in has_ans.items()
                                       if k != "calibration_curve"})
    _dump("calibration_curve.json", {
        "sweep": has_ans.get("calibration_curve", []),
        "recommended_threshold": has_ans.get("recommended_threshold"),
    })
    _dump("retrieval_quality.json", retrieval)
    _dump("clause_boundary_accuracy.json", clause)
    _dump("legal_term_preservation.json", legal_terms)
    _dump("generation_quality.json", generation)
    _dump("failure_breakdown.json", failures.get("breakdown", {}))
    _dump("flagged_examples.json", failures.get("flagged", {}))

    # --- optionally fold in previously-written human + conversation reports ---
    human_path = os.path.join(REPORT_DIR, "end_to_end_human_eval.json")
    convo_path = os.path.join(REPORT_DIR, "conversation_memory_eval.json")
    human = json.load(open(human_path)) if os.path.exists(human_path) else None
    convo = json.load(open(convo_path)) if os.path.exists(convo_path) else None

    # --- consolidated summary ---
    em_scores = em_f1.get("exact_match", {}) or {}
    f1_scores = em_f1.get("f1", {}) or {}
    legal_fpr = has_ans.get("legal", {}).get("false_positive_rate")
    general_fpr = has_ans.get("general", {}).get("false_positive_rate")
    summary = {
        "checkpoint": "qa_stage3_best",
        "test_examples_total":       len(records),
        "em_overall":                em_scores.get("em_overall"),
        "f1_overall":                f1_scores.get("f1_overall"),
        "f1_cuad":                   f1_scores.get("f1_cuad"),
        "f1_coliee":                 f1_scores.get("f1_coliee"),
        "f1_ledgar":                 f1_scores.get("f1_ledgar"),
        "f1_squad":                  f1_scores.get("f1_squad"),
        "em_unanswerable_only":      em_scores.get("em_unanswerable_only"),
        "f1_unanswerable_only":      f1_scores.get("f1_unanswerable_only"),
        "has_answer_accuracy":       has_ans.get("overall", {}).get("accuracy"),
        "false_positive_rate_legal":   legal_fpr,
        "false_positive_rate_general": general_fpr,
        "true_negative_rate_legal":  has_ans.get("legal", {}).get("true_negative_rate"),
        "recommended_has_answer_threshold": has_ans.get("recommended_threshold"),
        "retrieval_recall_at_1":     retrieval.get("retrieval_recall_at_1"),
        "retrieval_recall_at_3":     retrieval.get("retrieval_recall_at_3"),
        "retrieval_recall_at_5":     retrieval.get("retrieval_recall_at_5"),
        "answer_reachability_top3":  retrieval.get("answer_reachability_top3"),
        "clause_boundary_accuracy_full":    clause.get("clause_boundary_accuracy_full"),
        "clause_boundary_accuracy_partial": clause.get("clause_boundary_accuracy_partial"),
        "legal_term_preservation":   legal_terms.get("legal_term_preservation_overall"),
        "monetary_term_preservation": legal_terms.get("monetary_term_preservation"),
        "time_period_preservation":  legal_terms.get("time_period_preservation"),
        "defined_term_preservation": legal_terms.get("defined_term_preservation"),
        "generation_faithfulness":   generation.get("mean_faithfulness"),
        "faithfulness_method":       generation.get("faithfulness_method"),
        "simplification_delta":      generation.get("mean_simplification_delta"),
        "information_preservation":  generation.get("information_preservation_rate"),
        "mean_plain_english_reading_level":  generation.get("mean_plain_english_reading_level"),
        "mean_legal_span_reading_level":     generation.get("mean_legal_span_reading_level"),
        "failure_breakdown":         failures.get("breakdown", {}).get("category_fractions"),
        # Slots that require separate invocations:
        "coreference_resolution_rate": (convo or {}).get("coreference_resolution_rate"),
        "context_drift_rate":          (convo or {}).get("context_drift_rate"),
        "memory_depth_accuracy":       (convo or {}).get("memory_depth_accuracy"),
        "human_eval_correctness":      (human or {}).get("mean_correctness"),
        "human_eval_faithfulness":     (human or {}).get("mean_faithfulness"),
        "human_eval_clarity":          (human or {}).get("mean_clarity"),
        "human_eval_completeness":     (human or {}).get("mean_completeness"),
    }

    _dump("summary_report.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"[run_all] wrote all reports under {REPORT_DIR}")


if __name__ == "__main__":
    main()
