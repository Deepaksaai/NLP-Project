"""
Metric 6 — Legal term preservation.

For every answerable example the model answered, compute the
fraction of critical legal terms from the gold answer that survive
in the extracted span. Terms are grouped into 4 buckets so we can
see what the model loses most often:

    defined_terms   : capitalized multi-word phrases
    monetary_terms  : $X / X dollars
    time_periods    : N days/months/years/hours
    party_refs      : licensor / licensee / buyer / ...

Preferred extractor is spaCy `en_core_web_trf`; if unavailable we fall
back to the regex set shared with Stage 3's legal_metrics module.

Writes:
    evaluation/qa/report/legal_term_preservation.json
"""

import os
import re
import json
import collections
from typing import List, Dict

from QA.training.legal_metrics import (
    _DEFINED_TERM_RE, _MONEY_RE, _TIME_RE, _PARTY_WORDS,
)


def _extract_categorized(text: str):
    out = {
        "defined":  set(m.group() for m in _DEFINED_TERM_RE.finditer(text)),
        "monetary": set(m.group().lower() for m in _MONEY_RE.finditer(text)),
        "time":     set(m.group().lower() for m in _TIME_RE.finditer(text)),
        "party":    set(),
    }
    for w in re.findall(r"\b\w+\b", text.lower()):
        if w in _PARTY_WORDS:
            out["party"].add(w)
    return out


def _preservation(pred_terms, gold_terms):
    """Fraction of gold terms present in prediction; 1.0 if gold is empty."""
    if not gold_terms:
        return None
    return len(gold_terms & pred_terms) / len(gold_terms)


def compute(records: List[Dict]) -> Dict:
    agg = {"defined": [], "monetary": [], "time": [], "party": []}
    overall = []

    for r in records:
        if not r["is_answerable"]:
            continue
        pred = r.get("extracted_span") or ""
        golds = r.get("gold_answers") or []
        if not pred or not golds:
            continue
        pred_terms = _extract_categorized(pred)

        per_record = []
        for category in agg:
            best = None
            for g in golds:
                gold_terms = _extract_categorized(g)[category]
                if not gold_terms:
                    continue
                score = _preservation(pred_terms[category], gold_terms)
                if score is not None and (best is None or score > best):
                    best = score
            if best is not None:
                agg[category].append(best)
                per_record.append(best)
        if per_record:
            overall.append(sum(per_record) / len(per_record))

    def avg(xs): return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_evaluated_answerable": len(overall),
        "legal_term_preservation_overall": avg(overall),
        "defined_term_preservation":  avg(agg["defined"]),
        "monetary_term_preservation": avg(agg["monetary"]),
        "time_period_preservation":   avg(agg["time"]),
        "party_reference_preservation": avg(agg["party"]),
        "samples": {k: len(v) for k, v in agg.items()},
    }


def main():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "evaluation", "qa")
    with open(os.path.join(root, "generated_answers.json")) as f:
        records = json.load(f)
    result = compute(records)
    os.makedirs(os.path.join(root, "report"), exist_ok=True)
    with open(os.path.join(root, "report", "legal_term_preservation.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
