"""
Shared utilities for the Legal QA data pipeline.

Provides:
    - get_tokenizer:          Lazily load & cache the DeBERTa-v3-large tokenizer
    - load_and_validate_squad: Load SQuAD 2.0, validate answer offsets, return
                               normalised train / val example lists
    - convert_to_features:    Convert one raw QA example → model-ready tensors
    - verify_span:            Decode a predicted span and compare to gold answer
    - normalize_text:         Lowercase, strip punctuation, collapse whitespace
"""

import logging
import re
import string
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_SEQ_LEN = 512

# Singleton cache so the tokenizer is loaded only once per process.
_tokenizer_cache: dict = {}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def get_tokenizer() -> AutoTokenizer:
    """Lazily load and cache the DeBERTa-v3-large tokenizer."""
    if MODEL_NAME not in _tokenizer_cache:
        logger.info("Loading tokenizer: %s", MODEL_NAME)
        _tokenizer_cache[MODEL_NAME] = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer_cache[MODEL_NAME]


# ---------------------------------------------------------------------------
# Text normalisation (used for span verification)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# SQuAD 2.0 loader with validation
# ---------------------------------------------------------------------------
def load_and_validate_squad(max_train: Optional[int] = None, max_val: Optional[int] = None) -> Tuple[List[dict], List[dict]]:
    """
    Load SQuAD 2.0 from HuggingFace and validate every answerable example.

    Validation rule: for answerable examples
        context[answer_start : answer_start + len(answer_text)]
    must exactly equal answer_text.  Invalid examples are discarded with a
    warning.

    Returns:
        (train_examples, val_examples) — each item is a dict with keys:
            question, context, answer_text, answer_start, is_answerable
    """
    logger.info("Loading SQuAD 2.0 …")
    ds = load_dataset("squad_v2")

    def _convert_split(split_name: str) -> List[dict]:
        raw = ds[split_name]
        examples: List[dict] = []
        discarded = 0

        for row in raw:
            answers = row["answers"]
            if answers["text"]:
                # Answerable
                answer_text = answers["text"][0]
                answer_start = answers["answer_start"][0]

                # --- Validation ---
                ctx = row["context"]
                end = answer_start + len(answer_text)
                if answer_start < 0 or end > len(ctx):
                    logger.warning(
                        "SQuAD %s: answer offset out of bounds "
                        "(start=%d, len_ans=%d, len_ctx=%d) — discarding",
                        split_name, answer_start, len(answer_text), len(ctx),
                    )
                    discarded += 1
                    continue
                extracted = ctx[answer_start:end]
                if extracted != answer_text:
                    logger.warning(
                        "SQuAD %s: span mismatch — expected %r, got %r — discarding",
                        split_name, answer_text, extracted,
                    )
                    discarded += 1
                    continue

                examples.append({
                    "question": row["question"],
                    "context": ctx,
                    "answer_text": answer_text,
                    "answer_start": answer_start,
                    "is_answerable": True,
                })
            else:
                # Unanswerable
                examples.append({
                    "question": row["question"],
                    "context": row["context"],
                    "answer_text": "",
                    "answer_start": -1,
                    "is_answerable": False,
                })

        logger.info(
            "SQuAD %s: %d valid examples, %d discarded",
            split_name, len(examples), discarded,
        )
        return examples

    train_examples = _convert_split("train")
    val_examples = _convert_split("validation")
    if max_train:
        train_examples = train_examples[:max_train]
    if max_val:
        val_examples = val_examples[:max_val]
    return train_examples, val_examples


# ---------------------------------------------------------------------------
# Feature conversion  (raw example → DeBERTa-tokenised tensors)
# ---------------------------------------------------------------------------
def convert_to_features(
    example: dict,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = MAX_SEQ_LEN,
    domain_prefix: str = "",
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Tokenise a single QA example with DeBERTa and map the character-level
    answer span to token positions using offset_mapping.

    Args:
        example: dict with keys question, context, answer_text, answer_start,
                 is_answerable.
        tokenizer: DeBERTa tokenizer (defaults to get_tokenizer()).
        max_length: Maximum sequence length (default 512).
        domain_prefix: Optional string prepended to the context before
                       tokenisation (e.g. "legal: ").

    Returns:
        Feature dict with: input_ids, attention_mask, token_type_ids,
        start_positions, end_positions, is_answerable  (all tensors).
        Returns None if the answer span falls outside the tokenised window
        (i.e. it was truncated away).
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    question = example["question"]
    context = example["context"]
    answer_text = example.get("answer_text", "")
    answer_start = example.get("answer_start", -1)
    is_answerable = example.get("is_answerable", bool(answer_text))

    # Optional domain prefix
    if domain_prefix:
        context = domain_prefix + context
        # Shift answer_start by the prefix length so char offsets stay valid
        if is_answerable and answer_start >= 0:
            answer_start += len(domain_prefix)

    # ---- Tokenise ----
    encoding = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].squeeze(0)          # (L,)
    attention_mask = encoding["attention_mask"].squeeze(0)  # (L,)
    offset_mapping = encoding["offset_mapping"].squeeze(0)  # (L, 2)

    # ---- Build token_type_ids from sequence_ids ----
    # DeBERTa-v3 may not return meaningful token_type_ids, so we build
    # them from the tokenizer's sequence_ids():
    #   None → 0  (special tokens)
    #   0    → 0  (question)
    #   1    → 1  (context)
    seq_ids = encoding.sequence_ids(batch_index=0)
    token_type_ids = torch.tensor(
        [0 if s is None or s == 0 else 1 for s in seq_ids],
        dtype=torch.long,
    )

    # ---- Map character offsets → token positions ----
    if not is_answerable or answer_start < 0 or not answer_text:
        # Unanswerable → CLS convention: start = end = 0
        start_position = 0
        end_position = 0
    else:
        answer_end_char = answer_start + len(answer_text)

        start_position = None
        end_position = None

        for idx in range(len(seq_ids)):
            if seq_ids[idx] != 1:
                continue
            tok_start = offset_mapping[idx][0].item()
            tok_end = offset_mapping[idx][1].item()

            # Start token: first context token whose range covers answer_start
            if start_position is None:
                if tok_start <= answer_start and tok_end > answer_start:
                    start_position = idx

            # End token: last context token whose range covers the answer end
            if tok_start < answer_end_char and tok_end >= answer_end_char:
                end_position = idx

        # If we didn't find the answer (truncated away), return None
        if start_position is None or end_position is None:
            logger.debug(
                "Answer span truncated away — question: %s", question[:80]
            )
            return None

        # Sanity: start must come before or at end
        if start_position > end_position:
            logger.warning(
                "start_position (%d) > end_position (%d) — falling back to CLS",
                start_position, end_position,
            )
            start_position = 0
            end_position = 0

    return {
        "input_ids": input_ids,                         # (L,)
        "attention_mask": attention_mask,                # (L,)
        "token_type_ids": token_type_ids,                # (L,)
        "start_positions": torch.tensor(start_position, dtype=torch.long),
        "end_positions": torch.tensor(end_position, dtype=torch.long),
        "is_answerable": torch.tensor(
            1.0 if is_answerable else 0.0, dtype=torch.float
        ),
    }


# ---------------------------------------------------------------------------
# Span verification
# ---------------------------------------------------------------------------
def verify_span(
    input_ids: torch.Tensor,
    start_position: int,
    end_position: int,
    answer_text: str,
    tokenizer: Optional[AutoTokenizer] = None,
) -> bool:
    """
    Decode the span ``input_ids[start_position : end_position + 1]`` and
    compare to the expected *answer_text* after normalisation.

    Returns True if they match, False otherwise.  Mismatches are logged
    but never cause a crash.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    # CLS convention for unanswerable
    if start_position == 0 and end_position == 0:
        return answer_text == "" or not answer_text

    span_ids = input_ids[start_position : end_position + 1].tolist()
    decoded = tokenizer.decode(span_ids, skip_special_tokens=True)

    norm_decoded = normalize_text(decoded)
    norm_answer = normalize_text(answer_text)

    if norm_decoded == norm_answer:
        return True

    # Check containment (sub-word tokenisation may add/drop characters)
    if norm_answer in norm_decoded or norm_decoded in norm_answer:
        logger.debug(
            "Partial span match (acceptable): decoded=%r, answer=%r",
            norm_decoded, norm_answer,
        )
        return True

    logger.warning(
        "Span verification FAILED — decoded=%r, expected=%r",
        norm_decoded, norm_answer,
    )
    return False


# ---------------------------------------------------------------------------
# __main__ quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tok = get_tokenizer()
    print(f"Tokenizer loaded — vocab size: {tok.vocab_size}")

    train, val = load_and_validate_squad()
    print(f"SQuAD 2.0 — train: {len(train)}, val: {len(val)}")

    # Quick feature-conversion test on first 10 examples
    passed, failed = 0, 0
    for ex in train[:10]:
        feats = convert_to_features(ex, tok)
        if feats is None:
            failed += 1
            continue
        ok = verify_span(
            feats["input_ids"],
            feats["start_positions"].item(),
            feats["end_positions"].item(),
            ex["answer_text"],
            tok,
        )
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"Quick verify: {passed} passed, {failed} failed out of 10")
