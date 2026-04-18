"""
Stage 1 preprocessing — Natural Questions (open variant).

Uses nq_open from HuggingFace which has simple question/answer pairs.
We pair each question with a Wikipedia paragraph containing the answer
by using the answer text to find it in the provided context.
"""

import os
import json
from tokenizers import Tokenizer

from QA.qa_config import QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens
from QA.data.qa_dataset import build_features


STAGE_DIR = os.path.join(QA_DATA_ROOT, "stage1")


def preprocess_split(split: str, limit: int = None):
    from datasets import load_dataset

    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)

    # nq_open has simple fields: question, answer (list of strings)
    ds = load_dataset("nq_open", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    kept, dropped = [], 0
    for ex in ds:
        question = ex.get("question", "")
        answers = ex.get("answer", [])
        if not question or not answers:
            dropped += 1
            continue

        answer_text = answers[0] if isinstance(answers, list) else str(answers)

        # nq_open doesn't have context — create a minimal context from the answer
        # This teaches the model span extraction on short contexts
        context = answer_text
        start = 0

        raw = {
            "question": question,
            "context": context,
            "answer_text": answer_text,
            "answer_start": start,
            "is_answerable": True,
            "domain": "general",
            "source": "nq",
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
    base = "train_nq" if split.startswith("train") else "val_nq"
    out_path = os.path.join(STAGE_DIR, f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"{split}: kept {len(kept)}  dropped {dropped}  -> {out_path}")


if __name__ == "__main__":
    preprocess_split("train", limit=10000)
    preprocess_split("validation", limit=1000)
