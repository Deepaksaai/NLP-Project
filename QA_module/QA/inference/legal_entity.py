"""
Legal entity boundary fix — pure-regex fallback (no spaCy required).

If a predicted span cuts through a defined term, monetary amount, or
time period, this module extends the span to cover the full entity.
It only extends — it never shrinks — so it's safe to run on any
prediction.

A spaCy-based version can be dropped in later by swapping
`_find_entities` without touching the extend logic.
"""

import re
from typing import Tuple


_ENTITY_REGEXES = [
    re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)+\b"),                # defined terms
    re.compile(r"\$[0-9][\d,]*(?:\.\d+)?"),                                # money ($X)
    re.compile(r"\b\d[\d,]*\s*(?:dollars?|USD|cents?)\b", re.IGNORECASE),  # money (words)
    re.compile(r"\b\d+\s*(?:days?|weeks?|months?|years?|hours?|minutes?)\b", re.IGNORECASE),
]


def _find_entities(text: str):
    spans = []
    for rx in _ENTITY_REGEXES:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end()))
    spans.sort()
    # Merge overlapping spans
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def extend_span_to_entities(context: str, char_start: int, char_end: int) -> Tuple[int, int]:
    """
    Return (new_start, new_end) such that the span fully contains any
    entity it originally overlapped with.
    """
    if char_start >= char_end or char_end > len(context):
        return char_start, char_end

    entities = _find_entities(context)
    new_s, new_e = char_start, char_end
    for s, e in entities:
        # If entity overlaps span boundaries, extend
        if s < new_e and e > new_s:
            if s < new_s:
                new_s = s
            if e > new_e:
                new_e = e
    return new_s, new_e
