"""
Qualitative failure analysis for Legal QA.
"""

import json
import os
import logging
from typing import List, Dict, Any

from legal_qa.evaluation.metrics import normalize_answer, exact_match, f1_score, legal_term_preservation

logger = logging.getLogger(__name__)

def categorize_failures(predictions_path: str, out_dir: str):
    """
    Categorizes errors into 5 specific buckets and extracts top 10 examples each.
    Saves to flagged_examples.json.
    """
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    categories = {
        "Retrieval_Failure": [],
        "Span_Boundary_Error": [],
        "Has_Answer_False_Positive": [],
        "Has_Answer_False_Negative": [],
        "Legal_Term_Error": []
    }

    for ex in predictions:
        is_ans_gold = ex["is_answerable_gold"]
        is_ans_pred = 1 if ex["has_answer_prob"] > 0.5 else 0
        
        pred_span = ex["predicted_span"]
        golds = ex["gold_answers"]

        # 1 & 2: Has-Answer False Positives / Negatives
        if is_ans_gold == 0 and is_ans_pred == 1:
            categories["Has_Answer_False_Positive"].append({
                "q": ex["question"], "pred": pred_span,
                "diag": f"Model hallucinated an answer. Prob: {ex['has_answer_prob']:.2f}"
            })
            continue

        if is_ans_gold == 1 and is_ans_pred == 0:
            categories["Has_Answer_False_Negative"].append({
                "q": ex["question"], "gold": golds[0] if golds else "N/A",
                "diag": f"Model abstained. Prob was {ex['has_answer_prob']:.2f}"
            })
            continue

        # If it passed answerability checks correctly, but is actually answerable
        if is_ans_gold == 1 and is_ans_pred == 1:
            # Check Retrieval
            retrieved_texts = " ".join(ex["retrieved_chunks_texts"]).lower()
            gold_clean = " ".join((golds[0] if golds else "").lower().split())
            
            if gold_clean and gold_clean not in retrieved_texts:
                categories["Retrieval_Failure"].append({
                    "q": ex["question"], "gold": golds[0], "pred": pred_span,
                    "diag": "Gold answer text was completely absent from the top 3 retrieved chunks."
                })
                continue
                
            em = exact_match(pred_span, golds)
            f1 = f1_score(pred_span, golds)
            
            if em == 0.0:
                # High F1 but 0 EM = boundary error
                if f1 > 0.6:
                    categories["Span_Boundary_Error"].append({
                        "q": ex["question"], "gold": golds[0], "pred": pred_span,
                        "diag": f"Close, but boundaries mismatched. F1: {f1:.2f}"
                    })
                else:
                    # Generic wrong answer, check if it's a legal term error
                    term_score = legal_term_preservation(pred_span, golds)
                    if term_score < 1.0:
                        categories["Legal_Term_Error"].append({
                            "q": ex["question"], "gold": golds[0], "pred": pred_span,
                            "diag": "Critical numbers, dates, or defined terms were stripped from prediction."
                        })

    # Log out to CLI (Top 10 of each)
    logger.info("\n========== FAILURE ANALYSIS ==========\n")
    for cat, examples in categories.items():
        logger.info(f"--- {cat} ({len(examples)} total) ---")
        for i, curr_ex in enumerate(examples[:10]):
            logger.info(f"  Q: {curr_ex['q']}")
            if 'gold' in curr_ex: logger.info(f"  G: {curr_ex['gold']}")
            if 'pred' in curr_ex: logger.info(f"  P: {curr_ex['pred']}")
            logger.info(f"  D: {curr_ex['diag']}\n")

    # Save to JSON
    report_file = os.path.join(out_dir, "report", "flagged_examples.json")
    with open(report_file, "w") as f:
        json.dump(categories, f, indent=2)
        
    logger.info(f"Failure categorizations saved to {report_file}")
