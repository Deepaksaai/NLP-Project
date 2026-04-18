"""
Metric 7 — Plain English generation quality.

Three sub-metrics:

  faithfulness : NLI entailment (extracted_span -> plain_output).
                 Uses cross-encoder/nli-deberta-v3-base via
                 HuggingFace transformers when available;
                 otherwise falls back to bag-of-words overlap with
                 a clearly-labelled 'method = proxy' field so you can
                 tell that the number is not a real NLI score.

  simplification_delta : Flesch reading ease (plain) minus Flesch
                         reading ease (legal span). Positive means
                         easier.

  information_preservation : how many numbers / dates / party names
                             from the legal span survive into the
                             plain English output.

Writes:
    evaluation/qa/report/generation_quality.json

Note: when the Stage-3 pipeline uses NoopRewriter (the default), the
plain output equals the extracted span, so faithfulness is trivially
1.0 and simplification_delta is ~0. That's expected — this metric
becomes meaningful once you plug in a real rewriter.
"""

import os
import re
import json
import math
from typing import List, Dict, Tuple


# -------------------------------------------------------
# Flesch reading ease (no external deps)
# -------------------------------------------------------
_VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def _syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = _VOWEL_RE.findall(word)
    s = len(groups)
    # silent-e heuristic
    if word.endswith("e") and s > 1 and not word.endswith("le"):
        s -= 1
    return max(1, s)


def flesch_reading_ease(text: str) -> float:
    if not text.strip():
        return 0.0
    sentences = [s for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return (
        206.835
        - 1.015 * (len(words) / len(sentences))
        - 84.6  * (syllables / len(words))
    )


def flesch_to_grade(score: float) -> float:
    # rough inverse mapping (grade level ≈ 20 - score/5 as a coarse proxy)
    return max(1.0, 20.0 - score / 5.0)


# -------------------------------------------------------
# Information preservation
# -------------------------------------------------------
_NUM_RE   = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_DATE_RE  = re.compile(r"\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_PARTY_RE = re.compile(r"\b(?:Licensor|Licensee|Buyer|Seller|Lessor|Lessee|Company|Customer|Supplier|Party|Parties)\b")


def _key_facts(text: str):
    return (
        set(_NUM_RE.findall(text))
        | set(_DATE_RE.findall(text))
        | set(m.group() for m in _PARTY_RE.finditer(text))
    )


def information_preservation(span: str, plain: str) -> float:
    facts_in = _key_facts(span)
    if not facts_in:
        return 1.0
    facts_out = _key_facts(plain)
    return len(facts_in & facts_out) / len(facts_in)


# -------------------------------------------------------
# NLI faithfulness with graceful fallback
# -------------------------------------------------------
_NLI_PIPELINE = None
_NLI_STATUS = "uninitialized"


def _lazy_nli_load():
    global _NLI_PIPELINE, _NLI_STATUS
    if _NLI_STATUS != "uninitialized":
        return _NLI_PIPELINE
    try:
        from transformers import pipeline
        _NLI_PIPELINE = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-v3-base",
            return_all_scores=True,
        )
        _NLI_STATUS = "ready"
    except Exception as e:
        _NLI_STATUS = f"unavailable: {type(e).__name__}"
        _NLI_PIPELINE = None
    return _NLI_PIPELINE


def _bow_faithfulness_proxy(premise: str, hypothesis: str) -> float:
    p = set(w.lower() for w in re.findall(r"\w+", premise))
    h = set(w.lower() for w in re.findall(r"\w+", hypothesis))
    if not h:
        return 1.0
    return len(p & h) / len(h)


def faithfulness(premise: str, hypothesis: str) -> Tuple[float, str]:
    """Returns (score, method)."""
    if not premise or not hypothesis:
        return 1.0, "empty"
    pipe = _lazy_nli_load()
    if pipe is not None:
        try:
            out = pipe(f"{premise} </s> {hypothesis}")
            # deberta-nli returns labels entailment/neutral/contradiction
            for entry in out[0] if isinstance(out[0], list) else out:
                if entry["label"].lower() == "entailment":
                    return float(entry["score"]), "nli"
            return 0.0, "nli"
        except Exception:
            pass
    return _bow_faithfulness_proxy(premise, hypothesis), "bow_proxy"


# -------------------------------------------------------
# Aggregator
# -------------------------------------------------------
def compute(records: List[Dict]) -> Dict:
    faith_sum = 0.0
    faith_n = 0
    nli_used = False
    simp_sum = 0.0
    plain_grades = []
    legal_grades = []
    info_sum = 0.0
    info_n = 0

    for r in records:
        span = r.get("extracted_span") or ""
        plain = r.get("predicted_answer") or ""
        if not span or not plain:
            continue
        score, method = faithfulness(span, plain)
        faith_sum += score
        faith_n += 1
        if method == "nli":
            nli_used = True

        plain_score = flesch_reading_ease(plain)
        legal_score = flesch_reading_ease(span)
        simp_sum += plain_score - legal_score
        plain_grades.append(flesch_to_grade(plain_score))
        legal_grades.append(flesch_to_grade(legal_score))

        info_sum += information_preservation(span, plain)
        info_n += 1

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_evaluated":               faith_n,
        "generation_faithfulness_score": avg([faith_sum]) if faith_n else 0.0,
        "mean_faithfulness":         faith_sum / faith_n if faith_n else 0.0,
        "faithfulness_method":       "nli" if nli_used else "bow_proxy",
        "nli_status":                _NLI_STATUS,
        "mean_simplification_delta": simp_sum / faith_n if faith_n else 0.0,
        "mean_plain_english_reading_level": avg(plain_grades),
        "mean_legal_span_reading_level":    avg(legal_grades),
        "information_preservation_rate":    info_sum / info_n if info_n else 0.0,
    }


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "evaluation", "qa")
    with open(os.path.join(root, "generated_answers.json")) as f:
        records = json.load(f)
    result = compute(records)
    os.makedirs(os.path.join(root, "report"), exist_ok=True)
    with open(os.path.join(root, "report", "generation_quality.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
