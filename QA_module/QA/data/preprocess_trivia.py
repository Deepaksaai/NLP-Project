"""
Stage 2 preprocessing — TriviaQA (open-domain, long contexts).

Stub: fill in the rc.nocontext variant or filter to contexts that
fit within MAX_CONTEXT_LEN before shipping to training.
"""

import os
import json
from tokenizers import Tokenizer

from QA.qa_config import QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens
from QA.data.qa_dataset import build_features


STAGE_DIR = os.path.join(QA_DATA_ROOT, "stage1")


def _to_raw(ex):
    answer = ex["answer"]
    aliases = answer.get("aliases", [])
    text = answer.get("value") or (aliases[0] if aliases else "")

    # TriviaQA has multiple context sources — try each in order
    context = ""
    search_contexts = ex.get("search_results", {}).get("search_context", [])
    if search_contexts:
        context = search_contexts[0]
    if not context:
        # Fall back to entity_pages context
        entity_contexts = ex.get("entity_pages", {}).get("wiki_context", [])
        if entity_contexts:
            context = entity_contexts[0]

    start = context.find(text) if (text and context) else -1
    return {
        "question":      ex["question"],
        "context":       context,
        "answer_text":   text,
        "answer_start":  start,
        "is_answerable": start >= 0,
        "domain":        "general",
    }


def preprocess_split(split: str, limit: int = None):
    from datasets import load_dataset
    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)

    ds = load_dataset("trivia_qa", "rc", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    kept, dropped = [], 0
    for ex in ds:
        raw = _to_raw(ex)
        if not raw["context"]:
            dropped += 1
            continue
        feat = build_features(
            raw, tokenizer,
            cls_id=meta["cls_id"], sep_id=meta["sep_id"], pad_id=meta["pad_id"],
        )
        if feat is None:
            dropped += 1
            continue
        kept.append(feat)

    os.makedirs(STAGE_DIR, exist_ok=True)
    base = "train_trivia" if split.startswith("train") else "val_trivia"
    out_path = os.path.join(STAGE_DIR, f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"{split}: kept {len(kept)}  dropped {dropped}  -> {out_path}")


if __name__ == "__main__":
    preprocess_split("train", limit=10000)
    preprocess_split("validation", limit=1000)
