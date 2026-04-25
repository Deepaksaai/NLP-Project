"""
Stage 2 preprocessing — QuALITY.

QuALITY is multiple choice over long articles. We convert each
(question, correct_option) pair to span-extraction format by fuzzy-
locating the option text (or a key phrase from it) inside the article.
If no match is found the example is marked unanswerable.

Chunking:
  - Each article is split into overlapping 400-word chunks (50 overlap).
  - The chunk containing the answer becomes the positive example.
  - Up to 4 other chunks become negative examples (unanswerable for
    that question) — capped so negatives don't swamp the batch.
  - If no chunk contains the answer (common for abstractive options),
    ALL chunks become unanswerable for that question, still capped at 4.

Each output example carries:
  source="quality", doc_id, question_id, chunk_idx, n_chunks
so the chunk-retrieval metric can group them at eval time.
"""

import os
import re
import json
import random
from typing import Optional

from tokenizers import Tokenizer

from QA.qa_config import (
    QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens,
)
from QA.data.qa_dataset import build_features
from QA.data.chunking import chunk_document, answer_in_chunk


STAGE_DIR = os.path.join(QA_DATA_ROOT, "stage2")
NEG_CAP_PER_QUESTION = 4


_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "in", "on", "for", "with",
    "is", "was", "were", "be", "are", "that", "this", "it", "as",
    "at", "by", "from", "or", "but", "not", "no", "did", "do", "does",
    "has", "have", "had", "will", "would", "could", "should", "can",
    "may", "might", "he", "she", "they", "we", "i", "you",
}


def _strip_option_prefix(opt: str) -> str:
    """Drop leading letter prefixes like 'A: ' or '(B) '."""
    return re.sub(r"^\s*[\(\[]?[A-Da-d][\)\].:\-]\s*", "", opt).strip()


def _key_phrases(text: str, min_len: int = 2, max_phrases: int = 6):
    """Content-word bigrams / trigrams — used for fuzzy fallback search."""
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS]
    phrases = []
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            phrases.append(" ".join(words[i: i + n]))
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for p in phrases:
        if p not in seen and len(p) >= min_len:
            seen.add(p)
            uniq.append(p)
        if len(uniq) >= max_phrases:
            break
    return uniq


def _locate_answer_in_article(article: str, option_text: str) -> Optional[tuple]:
    """
    Return (char_start, answer_text) if we can find the option (or a
    close variant) somewhere in the article; None otherwise.
    """
    option_text = option_text.strip()
    if not option_text:
        return None

    low_article = article.lower()
    low_opt = option_text.lower()

    # 1) direct case-insensitive substring match
    idx = low_article.find(low_opt)
    if idx >= 0:
        return idx, article[idx: idx + len(option_text)]

    # 2) fall back to scanning for key phrases
    for phrase in _key_phrases(option_text):
        idx = low_article.find(phrase)
        if idx >= 0:
            # Anchor to that phrase — the matched text is whatever
            # substring of the article has that length.
            return idx, article[idx: idx + len(phrase)]

    return None


def _pick_correct_option(ex: dict) -> Optional[str]:
    """QuALITY shape varies by mirror — try the common layouts."""
    options = ex.get("options") or ex.get("choices") or []
    # gold index: several possible field names
    for key in ("gold_label", "label", "answer_index", "correct_answer", "answer"):
        if key in ex and ex[key] is not None:
            val = ex[key]
            if isinstance(val, int) and 0 <= val < len(options):
                return _strip_option_prefix(options[val])
            if isinstance(val, str) and len(val) == 1 and val.upper() in "ABCD":
                return _strip_option_prefix(options[ord(val.upper()) - ord("A")])
    return None


def _expand_chunks(
    tokenizer, meta, question: str, article: str, doc_id: str, question_id: str,
    answer_char_in_article: int, answer_text: str,
):
    """
    Build one feature dict per (question, chunk) pair. At most 1 positive
    + NEG_CAP_PER_QUESTION negatives are returned.
    """
    chunks = chunk_document(article, chunk_words=400, overlap_words=50)
    if not chunks:
        return []

    positive_idx = -1
    if answer_char_in_article >= 0 and answer_text:
        for c in chunks:
            local = answer_in_chunk(c, answer_char_in_article, answer_text)
            if local >= 0:
                positive_idx = c.chunk_idx
                break

    feats = []

    # Positive example (if the answer landed cleanly in some chunk)
    if positive_idx >= 0:
        c = chunks[positive_idx]
        local = answer_char_in_article - c.char_start
        raw = {
            "question":      question,
            "context":       c.text,
            "answer_text":   answer_text,
            "answer_start":  local,
            "is_answerable": True,
            "domain":        "scientific",
            "source":        "quality",
            "doc_id":        doc_id,
            "question_id":   question_id,
            "chunk_idx":     c.chunk_idx,
            "n_chunks":      c.n_chunks,
        }
        f = build_features(raw, tokenizer, meta["cls_id"], meta["sep_id"], meta["pad_id"])
        if f is not None:
            feats.append(f)

    # Negative chunks (capped)
    candidates = [c for c in chunks if c.chunk_idx != positive_idx]
    random.shuffle(candidates)
    for c in candidates[:NEG_CAP_PER_QUESTION]:
        raw = {
            "question":      question,
            "context":       c.text,
            "answer_text":   "",
            "answer_start":  -1,
            "is_answerable": False,
            "domain":        "scientific",
            "source":        "quality",
            "doc_id":        doc_id,
            "question_id":   question_id,
            "chunk_idx":     c.chunk_idx,
            "n_chunks":      c.n_chunks,
        }
        f = build_features(raw, tokenizer, meta["cls_id"], meta["sep_id"], meta["pad_id"])
        if f is not None:
            feats.append(f)

    return feats


def preprocess_split(split: str, limit: int = None):
    from datasets import load_dataset

    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)

    # QuALITY is mirrored under several names on HF; try the most common.
    last_err = None
    for name in ("emozilla/quality", "quality", "tau/quality"):
        try:
            ds = load_dataset(name, split=split)
            print(f"[quality] using dataset '{name}'")
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"QuALITY not reachable under any known name: {last_err}")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    kept = []
    n_positive = n_negative = n_dropped_noloc = 0
    random.seed(0)

    for ex in ds:
        article = ex.get("article") or ex.get("context") or ex.get("passage") or ""
        question = ex.get("question") or ex.get("query") or ""
        if not article or not question:
            continue

        opt_text = _pick_correct_option(ex)
        doc_id = str(ex.get("article_id") or ex.get("id") or ex.get("_id") or hash(article))
        q_id = f"{doc_id}::q{ex.get('question_id', len(kept))}"

        located = _locate_answer_in_article(article, opt_text) if opt_text else None
        if located is None:
            n_dropped_noloc += 1
            answer_char = -1
            answer_text = ""
        else:
            answer_char, answer_text = located

        feats = _expand_chunks(
            tokenizer, meta, question, article, doc_id, q_id,
            answer_char, answer_text,
        )
        for f in feats:
            kept.append(f)
            if f["is_answerable"]:
                n_positive += 1
            else:
                n_negative += 1

    os.makedirs(STAGE_DIR, exist_ok=True)
    base = "train" if split.startswith("train") else "val"
    out_path = os.path.join(STAGE_DIR, f"{base}_quality.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"[quality] {split}: kept {len(kept)} "
          f"(+{n_positive}/-{n_negative})  no-locate {n_dropped_noloc}  -> {out_path}")


if __name__ == "__main__":
    preprocess_split("train")
    preprocess_split("validation")
