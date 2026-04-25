"""
Stage 2 preprocessing — QASPER (allenai/qasper).

Filter rule:
  - answer type "extractive"  -> keep as answerable example
  - answer type "yes" / "no"  -> unanswerable (model is a span extractor)
  - answer type "abstractive" -> unanswerable

Chunking: same 400/50 overlap as QuALITY. For extractive answers we
pick the first chunk that contains the answer string; for non-
extractive we emit up to NEG_CAP negative chunks for the question.
"""

import os
import json
import random
from tokenizers import Tokenizer

from QA.qa_config import (
    QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens,
)
from QA.data.qa_dataset import build_features
from QA.data.chunking import chunk_document


STAGE_DIR = os.path.join(QA_DATA_ROOT, "stage2")
NEG_CAP_PER_QUESTION = 4


def _flatten_paper(paper: dict) -> str:
    """Join the QASPER paper's sections into a single body of text."""
    parts = []
    title = paper.get("title")
    if title:
        parts.append(title)
    abstract = paper.get("abstract")
    if abstract:
        parts.append(abstract)
    for section in paper.get("full_text", {}).get("section_name", []) or []:
        if section:
            parts.append(section)
    # full_text sometimes comes as dict of lists; handle both shapes
    ft = paper.get("full_text")
    if isinstance(ft, dict):
        paragraphs = ft.get("paragraphs") or []
        if paragraphs and isinstance(paragraphs[0], list):
            for para_list in paragraphs:
                parts.extend(para_list)
        else:
            parts.extend(paragraphs)
    elif isinstance(ft, list):
        for sec in ft:
            if isinstance(sec, dict):
                parts.extend(sec.get("paragraphs", []))
    return "\n\n".join(p for p in parts if isinstance(p, str) and p.strip())


def _first_extractive_answer(answer_struct):
    """
    Given a QASPER answer entry, return (answer_text, type).
    type is 'extractive', 'boolean', or 'abstractive'.
    """
    if not answer_struct:
        return "", "abstractive"
    ans = answer_struct.get("answer", answer_struct)
    if ans.get("unanswerable"):
        return "", "abstractive"
    spans = ans.get("extractive_spans") or []
    if spans:
        return spans[0], "extractive"
    if ans.get("yes_no") is not None:
        return "", "boolean"
    if ans.get("free_form_answer"):
        return "", "abstractive"
    return "", "abstractive"


def _build_examples_for_question(
    tokenizer, meta, question: str, paper_text: str, doc_id: str, q_id: str,
    answer_text: str, is_extractive: bool,
):
    chunks = chunk_document(paper_text, chunk_words=400, overlap_words=50)
    if not chunks:
        return []

    positive_idx = -1
    answer_char_in_chunk = -1
    if is_extractive and answer_text:
        low_ans = answer_text.lower()
        for c in chunks:
            idx = c.text.lower().find(low_ans)
            if idx >= 0:
                positive_idx = c.chunk_idx
                answer_char_in_chunk = idx
                break

    feats = []

    if positive_idx >= 0:
        c = chunks[positive_idx]
        raw = {
            "question":      question,
            "context":       c.text,
            "answer_text":   c.text[answer_char_in_chunk: answer_char_in_chunk + len(answer_text)],
            "answer_start":  answer_char_in_chunk,
            "is_answerable": True,
            "domain":        "scientific",
            "source":        "qasper",
            "doc_id":        doc_id,
            "question_id":   q_id,
            "chunk_idx":     c.chunk_idx,
            "n_chunks":      c.n_chunks,
        }
        f = build_features(raw, tokenizer, meta["cls_id"], meta["sep_id"], meta["pad_id"])
        if f is not None:
            feats.append(f)

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
            "source":        "qasper",
            "doc_id":        doc_id,
            "question_id":   q_id,
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

    ds = load_dataset("allenai/qasper", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    kept = []
    n_ext = n_nonext = 0
    random.seed(0)

    for paper in ds:
        doc_id = str(paper.get("id") or paper.get("paper_id") or hash(paper.get("title", "")))
        paper_text = _flatten_paper(paper)
        if not paper_text:
            continue

        qas = paper.get("qas") or {}
        questions = qas.get("question") or []
        answer_lists = qas.get("answers") or []

        for qi, q in enumerate(questions):
            q_id = f"{doc_id}::q{qi}"
            answers = answer_lists[qi] if qi < len(answer_lists) else {}
            # answers is usually a dict with a list under "answer"
            if isinstance(answers, dict) and "answer" in answers:
                first = answers["answer"][0] if answers["answer"] else {}
                ans_text, ans_type = _first_extractive_answer({"answer": first})
            else:
                ans_text, ans_type = _first_extractive_answer(answers)

            is_ext = ans_type == "extractive" and bool(ans_text)
            if is_ext: n_ext += 1
            else:      n_nonext += 1

            feats = _build_examples_for_question(
                tokenizer, meta, q, paper_text, doc_id, q_id,
                ans_text, is_ext,
            )
            kept.extend(feats)

    os.makedirs(STAGE_DIR, exist_ok=True)
    base = "train" if split.startswith("train") else "val"
    out_path = os.path.join(STAGE_DIR, f"{base}_qasper.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"[qasper] {split}: kept {len(kept)}  "
          f"extractive_qs={n_ext}  other_qs={n_nonext}  -> {out_path}")


if __name__ == "__main__":
    preprocess_split("train")
    preprocess_split("validation")
