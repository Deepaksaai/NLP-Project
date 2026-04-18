"""
Stage 1 preprocessing — SQuAD 2.0 (general domain, answerable + unanswerable).

Loads the HuggingFace dataset, converts each example to the unified
raw format, tokenizes with the QA tokenizer, and writes train/val
JSON files under data/qa/stage1/.
"""

import os
import json
from tokenizers import Tokenizer

from QA.qa_config import (
    QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens,
)
from QA.data.qa_dataset import build_features


STAGE_DIR = os.path.join(QA_DATA_ROOT, "stage1")


def _to_raw(ex, domain="general"):
    answers = ex["answers"]
    texts   = answers["text"]
    starts  = answers["answer_start"]
    is_ans  = len(texts) > 0 and len(starts) > 0
    return {
        "question":      ex["question"],
        "context":       ex["context"],
        "answer_text":   texts[0] if is_ans else "",
        "answer_start":  int(starts[0]) if is_ans else -1,
        "is_answerable": bool(is_ans),
        "domain":        domain,
    }


def preprocess_split(split: str, limit: int = None):
    from datasets import load_dataset

    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)

    ds = load_dataset("squad_v2", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    kept, dropped = [], 0
    for ex in ds:
        raw = _to_raw(ex)
        feat = build_features(
            raw, tokenizer,
            cls_id=meta["cls_id"], sep_id=meta["sep_id"], pad_id=meta["pad_id"],
        )
        if feat is None:
            dropped += 1
            continue
        kept.append(feat)

    os.makedirs(STAGE_DIR, exist_ok=True)
    # Normalize split name to the training loader's convention:
    #   train      -> train_squad.json
    #   validation -> val_squad.json
    base = "train" if split.startswith("train") else "val"
    out_path = os.path.join(STAGE_DIR, f"{base}_squad.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"{split}: kept {len(kept)}  dropped {dropped}  -> {out_path}")


if __name__ == "__main__":
    preprocess_split("train", limit=50000)
    preprocess_split("validation", limit=2000)
