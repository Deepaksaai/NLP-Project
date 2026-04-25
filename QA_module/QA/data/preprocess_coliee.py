"""
Stage 3 preprocessing — COLIEE (stub).

COLIEE is not reliably mirrored on HuggingFace. This preprocessor
reads a locally-placed JSON file at data/coliee/{split}.json with
records of the shape:

    {
      "question":    str,
      "context":     str,
      "answer_text": str,   # optional
      "answer_start":int,   # optional char offset
      "is_answerable": bool
    }

and converts them to Stage-3 features. If the file is missing this
script exits cleanly with a warning so training can still run on
CuAD + LEDGAR alone.
"""

import os
import json
from tokenizers import Tokenizer

from QA.qa_config import (
    QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens, LEGAL_TOKEN,
)
from QA.data.qa_dataset import build_features


STAGE_DIR = os.path.join(QA_DATA_ROOT, "stage3")
LOCAL_ROOT = os.path.join(os.path.dirname(QA_DATA_ROOT), "coliee")


def preprocess_split(split: str):
    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)
    legal_id = tokenizer.token_to_id(LEGAL_TOKEN)

    src_path = os.path.join(LOCAL_ROOT, f"{split}.json")
    if not os.path.exists(src_path):
        print(f"[coliee] {src_path} missing — skipping (stub)")
        return

    with open(src_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    kept = []
    for i, ex in enumerate(items):
        raw = {
            "question":      ex["question"],
            "context":       ex["context"],
            "answer_text":   ex.get("answer_text", ""),
            "answer_start":  ex.get("answer_start", -1),
            "is_answerable": bool(ex.get("is_answerable", False)),
            "domain":        "legal",
            "source":        "coliee",
            "doc_id":        f"coliee_{i}",
            "question_id":   f"coliee_{i}",
            "chunk_idx":     0,
            "n_chunks":      1,
        }
        feat = build_features(
            raw, tokenizer,
            cls_id=meta["cls_id"], sep_id=meta["sep_id"], pad_id=meta["pad_id"],
            prepend_legal=True, legal_id=legal_id,
        )
        if feat is not None:
            kept.append(feat)

    os.makedirs(STAGE_DIR, exist_ok=True)
    base = "train" if split.startswith("train") else "val"
    out_path = os.path.join(STAGE_DIR, f"{base}_coliee.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"[coliee] {split}: kept {len(kept)}  -> {out_path}")


if __name__ == "__main__":
    preprocess_split("train")
    preprocess_split("validation")
