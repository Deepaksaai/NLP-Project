"""
Document chunking utilities — used by QuALITY / QASPER preprocessing
and by the multi-chunk inference pipeline.

Key guarantees:
  - Every chunk records the exact character offsets into the ORIGINAL
    document. That lets callers recompute `answer_start` inside a
    chunk as `original_answer_start - chunk.char_start` and then
    verify the slice really contains the answer text.
  - Word-boundary splits via a regex over non-whitespace runs, so no
    word is ever cut in half.
  - Overlapping windows: each chunk shares `overlap_words` with its
    predecessor, so answers that straddle a boundary still land inside
    at least one chunk entirely.
"""

import re
from dataclasses import dataclass
from typing import List


_WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    text: str
    char_start: int        # inclusive, into original document
    char_end: int          # exclusive, into original document
    word_start: int        # index into the doc's word list (inclusive)
    word_end: int          # index into the doc's word list (exclusive)
    chunk_idx: int
    n_chunks: int


def _words_with_offsets(text: str):
    """Return list of (word, char_start, char_end) tuples."""
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def chunk_document(
    text: str,
    chunk_words: int = 400,
    overlap_words: int = 50,
) -> List[Chunk]:
    """
    Split `text` into overlapping word windows. Returns a list of
    Chunk objects preserving exact char offsets into the original
    document. If the doc fits in one window, returns a single chunk.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    words = _words_with_offsets(text)
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks_raw = []
    i = 0
    while i < len(words):
        j = min(i + chunk_words, len(words))
        chunks_raw.append((i, j))
        if j == len(words):
            break
        i += step

    n = len(chunks_raw)
    out: List[Chunk] = []
    for idx, (i, j) in enumerate(chunks_raw):
        cs = words[i][1]
        ce = words[j - 1][2]
        out.append(Chunk(
            text=text[cs:ce],
            char_start=cs,
            char_end=ce,
            word_start=i,
            word_end=j,
            chunk_idx=idx,
            n_chunks=n,
        ))
    return out


def answer_in_chunk(chunk: Chunk, answer_start_in_doc: int, answer_text: str) -> int:
    """
    If the answer span lies entirely within `chunk`, return the
    answer's char offset INSIDE the chunk text. Otherwise return -1.
    The caller is responsible for verifying the decoded slice.
    """
    if answer_start_in_doc < 0 or not answer_text:
        return -1
    ans_end = answer_start_in_doc + len(answer_text)
    if answer_start_in_doc < chunk.char_start or ans_end > chunk.char_end:
        return -1
    local = answer_start_in_doc - chunk.char_start
    # Defensive check: chunk.text must really contain the answer here.
    if chunk.text[local: local + len(answer_text)] != answer_text:
        return -1
    return local
