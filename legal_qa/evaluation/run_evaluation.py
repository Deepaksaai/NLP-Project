"""
Entry point for evaluating the full QA pipeline.
"""

import os
import sys
import argparse
import logging
import json

# Add root project dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from legal_qa.inference.pipeline import LegalQAPipeline
from legal_qa.evaluation.evaluator import QAEvaluator
from legal_qa.evaluation.error_analysis import categorize_failures

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Run Full Pipeline Evaluation")
    parser.add_argument("--quick", action="store_true", help="Run on 200 examples per dataset only")
    parser.add_argument("--threshold", type=float, default=0.5, help="Override default 0.5 has-answer threshold")
    return parser.parse_args()

def dummy_load_test_datasets(quick: bool = False):
    """
    Mock function to simulate loading all test sets.
    In reality, you would connect this to your stage3_dataset loader focusing exclusively on the 'test' splits!
    """
    logger.info("Loading Test datasets...")
    # This structure is what QAEvaluator expects
    return [
        {
            "id": "mock_1", "dataset": "cuad",
            "question": "What is the governing law?", 
            "answers": ["Laws of the State of New York"],
            "is_answerable": 1,
            "document_chunks": [{"body": "This agreement shall be subject to the Laws of the State of New York.", "section": "Governing Law"}]
        },
        {
            "id": "mock_2", "dataset": "squad",
            "question": "What happens if we terminate early?", 
            "answers": [],
            "is_answerable": 0,
            "document_chunks": [{"body": "Termination can occur for breach. Notice must be provided within 10 days.", "section": "Termination"}]
        }
    ] * (100 if quick else 1000)

def print_summary_table(report_path: str):
    """Prints a clean CLI summary table."""
    with open(report_path, "r") as f:
        rep = json.load(f)
        
    print("\n" + "="*80)
    print(" " * 30 + "EVALUATION SUMMARY")
    print("="*80)
    
    # Check if breakdown exists
    breakdowns = rep.get("breakdown", {})
    if breakdowns:
        print(f"{'Dataset':<15} | {'Count':<6} | {'EM':<6} | {'F1':<6} | {'Ans Acc':<7} | {'FPR':<5}")
        print("-" * 80)
        
        for k, v in breakdowns.items():
            if not v: continue
            em = v.get('em', 0)*100
            f1 = v.get('f1', 0)*100
            acc = v.get('accuracy', 0)*100
            fpr = v.get('false_positive_rate', 0)*100
            print(f"{k.upper():<15} | {v.get('count', 0):<6} | {em:<6.2f} | {f1:<6.2f} | {acc:<7.2f} | {fpr:<5.2f}")
    
    print("\n" + "="*80)
    print(f"Optimal Threshold Recommendation: {rep.get('optimal_threshold_recommendation', 'N/A')}")
    print("="*80 + "\n")

def main():
    args = get_args()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(os.path.dirname(out_dir), "checkpoints", "qa_stage3_best.pt")
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Cannot run evaluation. Missing checkpoint: {checkpoint_path}")
        sys.exit(1)

    logger.info("Initializing Legal QA Pipeline for Evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import torch
    pipeline = LegalQAPipeline(checkpoint_path=checkpoint_path, device=device)
    
    # Load data
    test_examples = dummy_load_test_datasets(quick=args.quick)
    
    logger.info(f"Loaded {len(test_examples)} examples. Commencing Full Evaluation Run.")
    evaluator = QAEvaluator(pipeline=pipeline, test_examples=test_examples, out_dir=out_dir)
    
    # 1. Run Complete Evaluation
    evaluator.run_full_evaluation()
    
    # 2. Run Qualitative Error Analysis
    predictions_path = os.path.join(out_dir, "generated_answers.json")
    categorize_failures(predictions_path=predictions_path, out_dir=out_dir)
    
    # 3. Print CLI Summary Table
    report_path = os.path.join(out_dir, "report", "summary_report.json")
    print_summary_table(report_path)

if __name__ == "__main__":
    main()
