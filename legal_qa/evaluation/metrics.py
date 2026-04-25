"""
Automated metrics for Legal QA evaluation.
"""

import string
import re
from typing import List, Dict

def normalize_answer(s: str) -> str:
    """Lowercases, removes punctuation, removes articles, and collapses whitespace."""
    if not s:
        return ""
    
    # Lowercase
    s = s.lower()
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    s = s.translate(translator)
    
    # Remove articles
    tokens = s.split()
    tokens = [t for t in tokens if t not in ["a", "an", "the"]]
    
    # Collapse whitespace
    return " ".join(tokens)

def exact_match(prediction: str, golds: List[str]) -> float:
    """Computes exact match. Returns the max EM across all valid gold answers."""
    if not golds:
        return 1.0 if not prediction else 0.0
    
    norm_pred = normalize_answer(prediction)
    return max([1.0 if norm_pred == normalize_answer(g) else 0.0 for g in golds])

def f1_score(prediction: str, golds: List[str]) -> float:
    """Computes token-overlap F1 score. Returns the max F1 across all gold answers."""
    if not golds:
        return 1.0 if not prediction else 0.0
        
    norm_pred_toks = normalize_answer(prediction).split()
    best_f1 = 0.0
    
    for g in golds:
        norm_gold_toks = normalize_answer(g).split()
        common = set(norm_pred_toks) & set(norm_gold_toks)
        num_same = sum(1 for tok in norm_gold_toks if tok in common)
        
        if len(norm_gold_toks) == 0 or len(norm_pred_toks) == 0:
            f1 = float(norm_gold_toks == norm_pred_toks)
        elif num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(norm_pred_toks)
            recall = 1.0 * num_same / len(norm_gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
            
        if f1 > best_f1:
            best_f1 = f1
            
    return best_f1

def has_answer_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Computes binary classification metrics for answerability."""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must be the same length.")
        
    tp = tn = fp = fn = 0
    for p, l in zip(predictions, labels):
        if p == 1 and l == 1:
            tp += 1
        elif p == 0 and l == 0:
            tn += 1
        elif p == 1 and l == 0:
            fp += 1
        elif p == 0 and l == 1:
            fn += 1
            
    total = len(predictions)
    accuracy = (tp + tn) / total if total > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "true_positive_rate": tpr,
        "true_negative_rate": tnr,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr
    }

def chunk_retrieval_accuracy(retrieved_chunks_list: List[List[str]], gold_answer_texts_list: List[str]) -> Dict[str, float]:
    """
    Computes top-1 and top-3 retrieval accuracy by checking if the gold answer string 
    is contained within the retrieved chunk texts.
    """
    top1 = 0
    top3 = 0
    total = len(retrieved_chunks_list)
    
    if total == 0:
        return {"top1": 0.0, "top3": 0.0}
        
    for chunks, gold_answer in zip(retrieved_chunks_list, gold_answer_texts_list):
        if not gold_answer:
            continue
            
        gold_answer_clean = " ".join(gold_answer.lower().split())
        
        found_in_top1 = False
        found_in_top3 = False
        
        for i, chunk in enumerate(chunks[:3]):
            chunk_clean = " ".join(chunk.lower().split())
            if gold_answer_clean in chunk_clean:
                if i == 0:
                    found_in_top1 = True
                found_in_top3 = True
                break
                
        if found_in_top1:
            top1 += 1
        if found_in_top3:
            top3 += 1
            
    return {"top1": top1 / total, "top3": top3 / total}

def legal_term_preservation(prediction: str, golds: List[str]) -> float:
    """
    Checks if critical monetary, time, and defined terms from the gold answer 
    survived in the prediction.
    """
    if not golds:
        return 1.0

    best_score = 0.0
    
    for gold in golds:
        # Regex patterns
        money_pattern = r'\$[\d,]+|\d+\sdollars?'
        time_pattern = r'\d+\sdays?|\d+\smonths?|\d+\syears?'
        # Capitalized defined terms (two or more consecutive capitalized words to avoid start-of-sentence noise)
        defined_term_pattern = r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b'
        
        gold_terms = []
        gold_terms.extend(re.findall(money_pattern, gold, re.IGNORECASE))
        gold_terms.extend(re.findall(time_pattern, gold, re.IGNORECASE))
        gold_terms.extend(re.findall(defined_term_pattern, gold))
        
        gold_terms = list(set(gold_terms))
        
        if not gold_terms:
            score = 1.0
        else:
            preserved_count = 0
            for term in gold_terms:
                if term.lower() in prediction.lower():
                    preserved_count += 1
            score = preserved_count / len(gold_terms)
            
        if score > best_score:
            best_score = score
            
    return best_score

def clause_boundary_accuracy(prediction: str, source_context: str) -> float:
    """
    Checks if prediction starts/ends at natural clause boundaries inside the source text.
    """
    if not prediction or not source_context:
        return 0.0
        
    start_idx = source_context.find(prediction)
    if start_idx == -1:
        return 0.0
        
    end_idx = start_idx + len(prediction)
    
    # Boundary definitions (simple heuristics)
    boundary_chars = {'.', ';', '!', '?', '\n'}
    section_markers = ["(a)", "(b)", "(c)", "1.", "2.", "3.", "section", "article"]
    
    valid_start = False
    valid_end = False
    
    # Check Start Boundary
    if start_idx == 0:
        valid_start = True
    else:
        prev_chars = source_context[max(0, start_idx-5):start_idx].lower().strip()
        if not prev_chars or prev_chars[-1] in boundary_chars:
            valid_start = True
        else:
            for marker in section_markers:
                if prev_chars.endswith(marker):
                    valid_start = True
                    break
                    
    # Check End Boundary
    if end_idx == len(source_context):
        valid_end = True
    else:
        # If prediction ends in punctuation
        if prediction[-1] in boundary_chars:
            valid_end = True
        else:
            # Check characters immediately after
            next_chars = source_context[end_idx:min(len(source_context), end_idx+5)].strip()
            if not next_chars or next_chars[0] in boundary_chars:
                valid_end = True
                
    if valid_start and valid_end:
        return 1.0
    elif valid_start or valid_end:
        return 0.5
    else:
        return 0.0
