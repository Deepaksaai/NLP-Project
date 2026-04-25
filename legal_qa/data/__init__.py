"""
Data preprocessing and dataset pipeline for legal QA.

Exports the three stage datasets and shared utilities.
"""

from legal_qa.data.dataset_utils import (
    get_tokenizer,
    load_and_validate_squad,
    convert_to_features,
    verify_span,
    normalize_text,
    MODEL_NAME,
    MAX_SEQ_LEN,
)

__all__ = [
    "get_tokenizer",
    "load_and_validate_squad",
    "convert_to_features",
    "verify_span",
    "normalize_text",
    "MODEL_NAME",
    "MAX_SEQ_LEN",
]
