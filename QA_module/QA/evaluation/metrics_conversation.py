"""
Metric — Multi-turn conversation evaluation.

Expects a hand-authored dialogues file at:
    evaluation/qa/conversation_dialogues.json

If the file is missing a tiny 2-dialogue example is written as a
starting point and the evaluator exits with a warning.

Dialogue schema:
    [
      {
        "dialogue_id": "d1",
        "document":    "<full text of the source doc>",
        "turns": [
          {"question": "What is the termination clause?",
           "gold_answer": "...",
           "expected_retrieval_hint": "termination"},
          {"question": "How long is the notice period for it?",
           "gold_answer": "30 days",
           "lookback_depth": 1}   # requires turn N-1 context
        ]
      },
      ...
    ]

Metrics:
    coreference_resolution_rate — fraction of follow-up turns
        (lookback_depth >= 1) whose answer F1 is above 0.4
    context_drift_rate — fraction of turns where adding history
        decreased the answer F1 vs running that turn with no history
    memory_depth_accuracy — F1 on turns with lookback_depth >= 2

Writes:
    evaluation/qa/report/conversation_memory_eval.json
"""

import os
import sys
import json
from typing import Dict, List

from QA.training.evaluate import _f1_one


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_DIR = os.path.join(_ROOT, "evaluation", "qa")
DIALOGUES_PATH = os.path.join(EVAL_DIR, "conversation_dialogues.json")


_EXAMPLE_DIALOGUES = [
    {
        "dialogue_id": "example_d1",
        "document": (
            "This Agreement shall commence on January 1 and continue for two years. "
            "Either party may terminate this Agreement upon thirty (30) days prior "
            "written notice to the other party. Payments shall be due net 45 days "
            "after receipt of invoice. Late payments accrue interest at 1.5% per month."
        ),
        "turns": [
            {"question": "What is the termination clause?",
             "gold_answer": "Either party may terminate this Agreement upon thirty (30) days prior written notice to the other party."},
            {"question": "How long is the notice period for it?",
             "gold_answer": "thirty (30) days",
             "lookback_depth": 1},
            {"question": "When are payments due?",
             "gold_answer": "net 45 days after receipt of invoice"},
            {"question": "What happens if they're late?",
             "gold_answer": "accrue interest at 1.5% per month",
             "lookback_depth": 1},
        ],
    },
]


def _ensure_example_file():
    if os.path.exists(DIALOGUES_PATH):
        return
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(DIALOGUES_PATH, "w", encoding="utf-8") as f:
        json.dump(_EXAMPLE_DIALOGUES, f, indent=2)
    print(f"[conversation] wrote example dialogues to {DIALOGUES_PATH}")
    print("[conversation] replace with ~20 real dialogues before reporting final numbers")


def _run_pipeline_turn(pipeline, document: str, question: str, use_memory: bool):
    if not use_memory:
        pipeline.reset_memory()
    result = pipeline.answer(document, question)
    return result.get("raw_span") or result.get("answer") or ""


def evaluate_dialogues(pipeline, dialogues: List[Dict]) -> Dict:
    total_turns = 0
    followup_hits = followup_n = 0
    drift_hits = drift_n = 0
    depth_hits = depth_n = 0

    for dlg in dialogues:
        pipeline.reset_memory()
        history_f1 = []
        nohist_f1  = []

        for i, turn in enumerate(dlg["turns"]):
            total_turns += 1
            lookback = turn.get("lookback_depth", 0)

            # With memory
            pred_with = pipeline.answer(dlg["document"], turn["question"])
            with_text = pred_with.get("raw_span") or pred_with.get("answer") or ""
            f1_with = _f1_one(with_text, turn["gold_answer"])
            history_f1.append(f1_with)

            # Without memory (fresh memory for each turn to isolate)
            pipeline.reset_memory()
            pred_nohist = pipeline.answer(dlg["document"], turn["question"])
            nohist_text = pred_nohist.get("raw_span") or pred_nohist.get("answer") or ""
            f1_no = _f1_one(nohist_text, turn["gold_answer"])
            nohist_f1.append(f1_no)

            # Restore the "with memory" history
            pipeline.reset_memory()
            for j in range(i + 1):
                pipeline.answer(dlg["document"], dlg["turns"][j]["question"])

            if lookback >= 1:
                followup_n += 1
                if f1_with > 0.4:
                    followup_hits += 1
            if lookback >= 2:
                depth_n += 1
                if f1_with > 0.4:
                    depth_hits += 1
            drift_n += 1
            if f1_with + 1e-6 < f1_no:
                drift_hits += 1

    def sd(a, b): return a / b if b > 0 else 0.0
    return {
        "total_dialogues": len(dialogues),
        "total_turns":     total_turns,
        "coreference_resolution_rate": sd(followup_hits, followup_n),
        "context_drift_rate":          sd(drift_hits,    drift_n),
        "memory_depth_accuracy":       sd(depth_hits,    depth_n),
        "n_followup_turns":            followup_n,
        "n_depth_turns":               depth_n,
    }


def main():
    import torch
    from tokenizers import Tokenizer
    from QA.qa_config import QA_TOKENIZER_PATH, load_qa_special_tokens
    from QA.model.qa_model import build_qa_model
    from QA.inference.legal_pipeline import LegalQAPipeline
    from QA.inference.plain_english import NoopRewriter

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(_ROOT, "checkpoints", "qa_stage3_best.pt"))
    args = p.parse_args()

    _ensure_example_file()
    with open(DIALOGUES_PATH) as f:
        dialogues = json.load(f)

    if not os.path.exists(args.checkpoint):
        print(f"[conversation] checkpoint {args.checkpoint} missing — skipping eval")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)
    model = build_qa_model(meta, load_ckpt=None, freeze=False).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state") or state.get("state_dict") or state, strict=False)
    model.eval()

    pipeline = LegalQAPipeline(model, tokenizer, meta, device, rewriter=NoopRewriter())
    result = evaluate_dialogues(pipeline, dialogues)

    os.makedirs(os.path.join(EVAL_DIR, "report"), exist_ok=True)
    with open(os.path.join(EVAL_DIR, "report", "conversation_memory_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
