"""
Stage 3 preprocessing — legal QA (CUAD / PrivacyQA / contract clauses).

Stub: wire this to whatever legal QA corpus you settle on. The
contract is already defined in qa_dataset.py — all you need to do is
produce dicts with the REQUIRED_RAW_FIELDS and pass them through
build_features().
"""

import os
import json
from tokenizers import Tokenizer

from QA.qa_config import QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens
from QA.data.qa_dataset import build_features


STAGE_DIR = os.path.join(QA_DATA_ROOT, "stage3")


def preprocess_cuad(limit: int = None):
    """CUAD: Contract Understanding Atticus Dataset."""
    from datasets import load_dataset
    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)

    try:
        ds = load_dataset("cuad", split="train")
    except Exception as e:
        raise RuntimeError(f"CUAD load failed: {e}")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    kept, dropped = [], 0
    for ex in ds:
        answers = ex["answers"]
        texts = answers["text"]
        starts = answers["answer_start"]
        is_ans = len(texts) > 0
        raw = {
            "question":      ex["question"],
            "context":       ex["context"],
            "answer_text":   texts[0] if is_ans else "",
            "answer_start":  int(starts[0]) if is_ans else -1,
            "is_answerable": bool(is_ans),
            "domain":        "legal",
        }
        feat = build_features(
            raw, tokenizer,
            cls_id=meta["cls_id"], sep_id=meta["sep_id"], pad_id=meta["pad_id"],
        )
        if feat is None:
            dropped += 1
            continue
        kept.append(feat)

    os.makedirs(STAGE_DIR, exist_ok=True)
    out_path = os.path.join(STAGE_DIR, "train.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"CUAD: kept {len(kept)}  dropped {dropped}  -> {out_path}")


if __name__ == "__main__":
    preprocess_cuad(limit=500)
