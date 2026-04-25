"""
Stage 3 preprocessing — CuAD.

CuAD examples are (question, clause_context, answer_span). We:

  1. Emit one answerable feature per example (the positive).
  2. Build a pool of contexts grouped by (doc_id, clause_title) so
     we can generate legal unanswerable examples:

        - Cross-document negatives: question from contract A, context
          from contract B (different doc_id).
        - Wrong-clause negatives: question from category X, context
          from category Y in the same doc.

  3. Legal contexts get `prepend_legal=True` during tokenization so
     the <legal> signal token appears in the input sequence.

Target negative ratio (spec): ~35% of legal examples. With ~13k CuAD
positives, we generate up to ~7k cross-doc negatives + ~3k wrong-clause
negatives, which lands near the target after sampling.
"""

import os
import re
import json
import random
from collections import defaultdict
from typing import Optional

from tokenizers import Tokenizer

from QA.qa_config import (
    QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens, LEGAL_TOKEN,
)
from QA.data.qa_dataset import build_features


STAGE_DIR = os.path.join(QA_DATA_ROOT, "stage3")
NEG_CROSSDOC_PER_QUESTION = 1
NEG_WRONGCLAUSE_PER_QUESTION = 1


_CLAUSE_TITLE_RE = re.compile(r"['\"]([^'\"]+)['\"]")


def _extract_clause_title(question: str) -> Optional[str]:
    """CuAD questions embed the clause category in quotes."""
    m = _CLAUSE_TITLE_RE.search(question)
    return m.group(1).strip().lower() if m else None


def _build_legal_feature(raw, tokenizer, meta, legal_id):
    return build_features(
        raw, tokenizer,
        cls_id=meta["cls_id"], sep_id=meta["sep_id"], pad_id=meta["pad_id"],
        prepend_legal=True, legal_id=legal_id,
    )


def preprocess_split(split: str, limit: int = None, seed: int = 0):
    from datasets import load_dataset

    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)
    legal_id = tokenizer.token_to_id(LEGAL_TOKEN)
    if legal_id is None:
        raise RuntimeError(f"{LEGAL_TOKEN} not found in tokenizer")

    ds = load_dataset("theatticusproject/cuad-qa", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    random.seed(seed)

    positives = []               # answerable features
    pool_by_doc  = defaultdict(list)   # doc_id -> [(clause_title, context)]
    ex_pool      = []            # (doc_id, clause_title, question, context)

    # -------- pass 1: emit positives --------
    for ex in ds:
        question = ex["question"]
        context  = ex["context"]
        doc_id   = ex.get("title") or ex.get("id") or str(hash(context))
        clause_title = _extract_clause_title(question) or "unknown"
        answers  = ex["answers"]
        texts    = answers.get("text") or []
        starts   = answers.get("answer_start") or []
        is_ans   = len(texts) > 0 and len(starts) > 0 and texts[0].strip() != ""

        if is_ans:
            raw = {
                "question":      question,
                "context":       context,
                "answer_text":   texts[0],
                "answer_start":  int(starts[0]),
                "is_answerable": True,
                "domain":        "legal",
                "source":        "cuad",
                "doc_id":        doc_id,
                "question_id":   f"{doc_id}::{clause_title}",
                "chunk_idx":     0,
                "n_chunks":      1,
            }
            feat = _build_legal_feature(raw, tokenizer, meta, legal_id)
            if feat is not None:
                positives.append(feat)

        pool_by_doc[doc_id].append((clause_title, context))
        ex_pool.append((doc_id, clause_title, question, context))

    # -------- pass 2: generate negatives --------
    negatives = []

    all_doc_ids = list(pool_by_doc.keys())

    def _sample_other_doc_context(current_doc):
        if len(all_doc_ids) < 2:
            return None
        for _ in range(10):
            other = random.choice(all_doc_ids)
            if other != current_doc and pool_by_doc[other]:
                return random.choice(pool_by_doc[other])
        return None

    def _sample_other_clause_same_doc(current_doc, current_clause):
        choices = [
            (ct, cx) for (ct, cx) in pool_by_doc[current_doc]
            if ct != current_clause
        ]
        return random.choice(choices) if choices else None

    for doc_id, clause_title, question, context in ex_pool:
        # Cross-document negative
        for _ in range(NEG_CROSSDOC_PER_QUESTION):
            picked = _sample_other_doc_context(doc_id)
            if picked is None:
                continue
            _, neg_ctx = picked
            raw = {
                "question":      question,
                "context":       neg_ctx,
                "answer_text":   "",
                "answer_start":  -1,
                "is_answerable": False,
                "domain":        "legal",
                "source":        "cuad",  # counted against cuad pool for the mix
                "doc_id":        f"{doc_id}_xdoc",
                "question_id":   f"{doc_id}::{clause_title}::xdoc",
                "chunk_idx":     0,
                "n_chunks":      1,
                "neg_kind":      "cross_document",
            }
            feat = _build_legal_feature(raw, tokenizer, meta, legal_id)
            if feat is not None:
                negatives.append(feat)

        # Wrong-clause negative
        for _ in range(NEG_WRONGCLAUSE_PER_QUESTION):
            picked = _sample_other_clause_same_doc(doc_id, clause_title)
            if picked is None:
                continue
            _, neg_ctx = picked
            raw = {
                "question":      question,
                "context":       neg_ctx,
                "answer_text":   "",
                "answer_start":  -1,
                "is_answerable": False,
                "domain":        "legal",
                "source":        "cuad",
                "doc_id":        f"{doc_id}_wclause",
                "question_id":   f"{doc_id}::{clause_title}::wclause",
                "chunk_idx":     0,
                "n_chunks":      1,
                "neg_kind":      "wrong_clause",
            }
            feat = _build_legal_feature(raw, tokenizer, meta, legal_id)
            if feat is not None:
                negatives.append(feat)

    # Cap negatives so the overall answerable / unanswerable ratio
    # lands near the Stage-3 target of 45% answerable / 35% unanswerable
    # (the rest comes from the anchor pools). With CuAD only, that's
    # ~55% answerable / 45% unanswerable in the legal subset.
    target_neg = int(round(len(positives) * 45 / 55))
    if len(negatives) > target_neg:
        random.shuffle(negatives)
        negatives = negatives[:target_neg]

    all_feats = positives + negatives
    random.shuffle(all_feats)

    os.makedirs(STAGE_DIR, exist_ok=True)
    base = "train" if split == "train" else "val"
    out_path = os.path.join(STAGE_DIR, f"{base}_cuad.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_feats, f)
    print(f"[cuad] {split}: positives={len(positives)}  "
          f"negatives={len(negatives)}  -> {out_path}")


if __name__ == "__main__":
    preprocess_split("train")
    preprocess_split("test")
