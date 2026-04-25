"""
Stage 3 preprocessing — LEDGAR (lex_glue subset).

LEDGAR is a classification dataset: (provision_text, label). We
convert it to span-extraction QA by:

  1. Looking up a fixed question template per label (see
     legal_templates.LEDGAR_TEMPLATES). Labels without a template
     are discarded.
  2. Using the provision text as both context and answer, capped at
     200 words to avoid absurdly long targets.
  3. Marking every converted example is_answerable=True.
  4. Prepending the <legal> token during tokenization.
"""

import os
import json
from tokenizers import Tokenizer

from QA.qa_config import (
    QA_TOKENIZER_PATH, QA_DATA_ROOT, load_qa_special_tokens, LEGAL_TOKEN,
)
from QA.data.qa_dataset import build_features
from QA.data.legal_templates import template_for


STAGE_DIR = os.path.join(QA_DATA_ROOT, "stage3")
MAX_ANSWER_WORDS = 200


def _truncate_to_words(text: str, n: int) -> str:
    words = text.split()
    if len(words) <= n:
        return text
    return " ".join(words[:n])


def _label_name(ds, label_idx):
    """lex_glue returns label as an int; look up the class name."""
    try:
        feat = ds.features["label"]
        return feat.int2str(label_idx)
    except Exception:
        return str(label_idx)


def preprocess_split(split: str, limit: int = None):
    from datasets import load_dataset

    meta = load_qa_special_tokens()
    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)
    legal_id = tokenizer.token_to_id(LEGAL_TOKEN)
    if legal_id is None:
        raise RuntimeError(f"{LEGAL_TOKEN} not found in tokenizer")

    ds = load_dataset("lex_glue", "ledgar", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    kept = []
    dropped_label = 0
    dropped_build = 0

    for i, ex in enumerate(ds):
        text   = ex["text"]
        label  = _label_name(ds, ex["label"])
        question = template_for(label)
        if question is None:
            dropped_label += 1
            continue

        answer_text = _truncate_to_words(text, MAX_ANSWER_WORDS)
        raw = {
            "question":      question,
            "context":       text,
            "answer_text":   answer_text,
            "answer_start":  0,
            "is_answerable": True,
            "domain":        "legal",
            "source":        "ledgar",
            "doc_id":        f"ledgar_{i}",
            "question_id":   f"ledgar_{i}",
            "chunk_idx":     0,
            "n_chunks":      1,
            "ledgar_label":  label,
        }
        feat = build_features(
            raw, tokenizer,
            cls_id=meta["cls_id"], sep_id=meta["sep_id"], pad_id=meta["pad_id"],
            prepend_legal=True, legal_id=legal_id,
        )
        if feat is None:
            dropped_build += 1
            continue
        kept.append(feat)

    os.makedirs(STAGE_DIR, exist_ok=True)
    base = "train" if split.startswith("train") else "val"
    out_path = os.path.join(STAGE_DIR, f"{base}_ledgar.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f)
    print(f"[ledgar] {split}: kept {len(kept)}  "
          f"dropped(label={dropped_label}, build={dropped_build})  -> {out_path}")


if __name__ == "__main__":
    preprocess_split("train")
    preprocess_split("validation")
