"""
Extend the summarizer's BPE tokenizer with QA-specific special tokens.

Adds [CLS] and [SEP] via `add_special_tokens` which appends them to
the end of the vocabulary. The HuggingFace `tokenizers` library does
not let us force them into ids 5/6 without retraining, so their real
ids land at 32000/32001. Everything downstream reads ids from
qa_special_tokens.json instead of hardcoding.

Input format the tokenizer is prepared for:
    [CLS] <question tokens> [SEP] <context tokens> [SEP]
"""

import os
import json
import shutil

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from QA.qa_config import (
    SUMMARIZER_TOKENIZER_PATH,
    QA_TOKENIZER_PATH,
    QA_SPECIAL_TOKENS_JSON,
    ORIG_VOCAB_SIZE,
)


def main():
    assert os.path.exists(SUMMARIZER_TOKENIZER_PATH), \
        f"Summarizer tokenizer not found: {SUMMARIZER_TOKENIZER_PATH}"

    os.makedirs(os.path.dirname(QA_TOKENIZER_PATH), exist_ok=True)

    tokenizer = Tokenizer.from_file(SUMMARIZER_TOKENIZER_PATH)
    old_vocab = tokenizer.get_vocab_size()
    print(f"Loaded summarizer tokenizer — vocab size: {old_vocab}")

    added = tokenizer.add_special_tokens(["[CLS]", "[SEP]"])
    print(f"Added {added} special tokens")

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("<pad>")
    new_vocab = tokenizer.get_vocab_size()

    # Post-processor: auto-wraps encode(question, context) with [CLS]/[SEP]
    tokenizer.post_processor = TemplateProcessing(
        single=f"[CLS] $A [SEP]",
        pair=f"[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    tokenizer.save(QA_TOKENIZER_PATH)
    print(f"Saved QA tokenizer -> {QA_TOKENIZER_PATH}")

    meta = {
        "pad_id":       pad_id,
        "cls_id":       cls_id,
        "sep_id":       sep_id,
        "old_vocab":    old_vocab,
        "new_vocab":    new_vocab,
        "source":       SUMMARIZER_TOKENIZER_PATH,
        "note": (
            "The spec requested [CLS]=5/[SEP]=6 but those ids already belong "
            "to BPE tokens in the summarizer tokenizer. Since retraining is "
            "forbidden, the new special tokens are appended to the end of the "
            "vocabulary. All downstream code reads their ids from this file."
        ),
    }
    with open(QA_SPECIAL_TOKENS_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata -> {QA_SPECIAL_TOKENS_JSON}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
