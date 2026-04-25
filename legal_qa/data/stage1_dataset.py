"""
Stage 1 — General QA Dataset.

Combines SQuAD 2.0, TriviaQA (RC), and Natural Questions into a single
PyTorch Dataset with a WeightedBalancedSampler that enforces a 65 / 35
answerable / unanswerable ratio per batch.

Usage:
    from legal_qa.data.stage1_dataset import GeneralQADataset, WeightedBalancedSampler, collate_fn
    ds  = GeneralQADataset()
    sam = WeightedBalancedSampler(ds)
    dl  = DataLoader(ds, batch_size=16, sampler=sam, collate_fn=collate_fn)
"""

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datasets import load_dataset

from legal_qa.data.dataset_utils import (
    get_tokenizer,
    load_and_validate_squad,
    convert_to_features,
    verify_span,
    normalize_text,
    MAX_SEQ_LEN,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Internal loaders — each returns List[dict] in the uniform schema:
#   { question, context, answer_text, answer_start, is_answerable }
# =========================================================================

def _load_squad_examples(split: str = "train", max_examples: Optional[int] = None) -> List[dict]:
    """Load SQuAD 2.0 via the shared validator."""
    train, val = load_and_validate_squad(
        max_train=max_examples if split == "train" else None,
        max_val=max_examples if split == "validation" else None,
    )
    return train if split == "train" else val


# -------------------------------------------------------------------------
# TriviaQA  (Reading Comprehension split)
# -------------------------------------------------------------------------
def _load_triviaqa_examples(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """
    Load TriviaQA RC.  For every example take the first valid answer
    occurrence in the first available context document.
    """
    logger.info("Loading TriviaQA (rc) — %s …", split)
    try:
        ds = load_dataset("trivia_qa", "rc", split=split)
    except Exception as e:
        logger.warning("Could not load TriviaQA: %s — skipping", e)
        return []

    examples: List[dict] = []
    skipped = 0

    for idx, row in enumerate(ds):
        if max_examples and len(examples) >= max_examples:
            break

        answer_aliases = row["answer"].get("aliases", [])
        answer_value = row["answer"].get("value", "")
        all_answers = [answer_value] + answer_aliases if answer_value else answer_aliases

        if not all_answers:
            skipped += 1
            continue

        # Try to find one answer in the search-result contexts
        contexts = []
        sr = row.get("search_results", {})
        if sr and sr.get("search_context"):
            contexts.extend(sr["search_context"])
        ep = row.get("entity_pages", {})
        if ep and ep.get("wiki_context"):
            contexts.extend(ep["wiki_context"])

        found = False
        for ctx_text in contexts:
            if not ctx_text:
                continue
            # Truncate extremely long contexts to 10 000 chars for search
            ctx_text = ctx_text[:10_000]
            for ans in all_answers:
                pos = ctx_text.find(ans)
                if pos >= 0:
                    examples.append({
                        "question": row["question"],
                        "context": ctx_text,
                        "answer_text": ans,
                        "answer_start": pos,
                        "is_answerable": True,
                    })
                    found = True
                    break
            if found:
                break

        if not found:
            skipped += 1

    logger.info(
        "TriviaQA %s: %d examples extracted, %d skipped (no valid context match)",
        split, len(examples), skipped,
    )
    return examples


# -------------------------------------------------------------------------
# Natural Questions
# -------------------------------------------------------------------------
def _load_nq_examples(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """
    Load Google Natural Questions.  Extract short answers when available;
    examples without short answers are marked unanswerable.
    """
    logger.info("Loading Natural Questions — %s …", split)
    try:
        ds = load_dataset("natural_questions", "default", split=split)
    except Exception as e:
        logger.warning("Could not load Natural Questions: %s — skipping", e)
        return []

    examples: List[dict] = []
    skipped = 0

    for idx, row in enumerate(ds):
        if max_examples and len(examples) >= max_examples:
            break

        # Reconstruct plain text from document tokens
        doc_tokens = row.get("document", {}).get("tokens", {})
        token_texts = doc_tokens.get("token", [])
        is_html = doc_tokens.get("is_html", [])

        if not token_texts:
            skipped += 1
            continue

        # Build plain text from non-HTML tokens
        plain_tokens = []
        token_char_starts = []  # char offset of each plain token in the text
        char_pos = 0
        for t, html_flag in zip(token_texts, is_html):
            if not html_flag:
                token_char_starts.append(char_pos)
                plain_tokens.append(t)
                char_pos += len(t) + 1  # +1 for space
            else:
                token_char_starts.append(-1)

        plain_text = " ".join(plain_tokens)
        if not plain_text.strip():
            skipped += 1
            continue

        # Truncate for manageability
        plain_text = plain_text[:10_000]

        question_text = row.get("question", {}).get("text", "")
        if not question_text:
            skipped += 1
            continue

        annotations = row.get("annotations", {})
        short_answers_list = annotations.get("short_answers", [])
        yes_no = annotations.get("yes_no_answer", [])

        # Try to get the first short answer
        found_answer = False
        if short_answers_list:
            for sa_block in short_answers_list:
                start_tokens = sa_block.get("start_token", [])
                end_tokens = sa_block.get("end_token", [])
                if start_tokens and end_tokens:
                    st = start_tokens[0]
                    et = end_tokens[0]
                    # Extract answer text from the original token list
                    ans_tokens = []
                    for ti in range(st, min(et, len(token_texts))):
                        if ti < len(is_html) and not is_html[ti]:
                            ans_tokens.append(token_texts[ti])
                    if ans_tokens:
                        ans_text = " ".join(ans_tokens)
                        # Find this text in our plain_text
                        pos = plain_text.find(ans_text)
                        if pos >= 0:
                            examples.append({
                                "question": question_text,
                                "context": plain_text,
                                "answer_text": ans_text,
                                "answer_start": pos,
                                "is_answerable": True,
                            })
                            found_answer = True
                            break

        if not found_answer:
            # Mark as unanswerable
            examples.append({
                "question": question_text,
                "context": plain_text[:5_000],
                "answer_text": "",
                "answer_start": -1,
                "is_answerable": False,
            })

    logger.info(
        "NQ %s: %d examples extracted, %d skipped",
        split, len(examples), skipped,
    )
    return examples


# =========================================================================
# PyTorch Dataset
# =========================================================================
class GeneralQADataset(Dataset):
    """
    Combined Stage-1 dataset: SQuAD 2.0 + TriviaQA + Natural Questions.

    Every example is pre-tokenised with DeBERTa at construction time and
    stored as a feature dict.
    """

    def __init__(
        self,
        split: str = "train",
        max_squad: Optional[int] = None,
        max_triviaqa: Optional[int] = None,
        max_nq: Optional[int] = None,
        tokenizer=None,
    ):
        super().__init__()
        if tokenizer is None:
            tokenizer = get_tokenizer()

        logger.info("Building GeneralQADataset (stage 1) — split=%s", split)

        # ---- Collect raw examples ----
        raw_examples: List[dict] = []

        squad = _load_squad_examples(split, max_examples=max_squad)
        logger.info("  SQuAD 2.0: %d examples", len(squad))
        raw_examples.extend(squad)

        triviaqa = _load_triviaqa_examples(split, max_examples=max_triviaqa)
        logger.info("  TriviaQA:  %d examples", len(triviaqa))
        raw_examples.extend(triviaqa)

        # NQ skipped — SQuAD + TriviaQA (248k) is sufficient for stage 1
        # nq = _load_nq_examples(split, max_examples=max_nq)
        # raw_examples.extend(nq)

        # ---- Tokenise ----
        self.features: List[dict] = []
        self.is_answerable_flags: List[bool] = []
        discarded = 0

        for ex in raw_examples:
            feats = convert_to_features(ex, tokenizer)
            if feats is None:
                discarded += 1
                continue
            self.features.append(feats)
            self.is_answerable_flags.append(ex["is_answerable"])

        logger.info(
            "GeneralQADataset ready — %d features (%d discarded), "
            "answerable=%d, unanswerable=%d",
            len(self.features), discarded,
            sum(self.is_answerable_flags),
            len(self.is_answerable_flags) - sum(self.is_answerable_flags),
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# =========================================================================
# Weighted Balanced Sampler  (65 % answerable, 35 % unanswerable)
# =========================================================================
class WeightedBalancedSampler:
    """
    Wraps ``torch.utils.data.WeightedRandomSampler`` to enforce a
    65 / 35 answerable / unanswerable balance in every epoch.

    Usage:
        sampler = WeightedBalancedSampler(dataset)
        loader  = DataLoader(dataset, batch_size=16, sampler=sampler)
    """

    def __new__(cls, dataset: GeneralQADataset, num_samples: Optional[int] = None):
        flags = dataset.is_answerable_flags
        n_pos = sum(flags)
        n_neg = len(flags) - n_pos

        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Cannot balance — answerable=%d, unanswerable=%d. "
                "Falling back to uniform sampling.",
                n_pos, n_neg,
            )
            return WeightedRandomSampler(
                weights=[1.0] * len(flags),
                num_samples=num_samples or len(flags),
                replacement=True,
            )

        w_pos = 0.65 / n_pos
        w_neg = 0.35 / n_neg
        weights = [w_pos if ans else w_neg for ans in flags]

        return WeightedRandomSampler(
            weights=weights,
            num_samples=num_samples or len(flags),
            replacement=True,
        )


# =========================================================================
# Collation
# =========================================================================
def collate_fn(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """
    Stack a list of feature dicts into a batched dict of tensors.

    All sequences are already padded to MAX_SEQ_LEN by the tokenizer,
    so a simple ``torch.stack`` suffices — but we verify shapes first.
    """
    keys = ["input_ids", "attention_mask", "token_type_ids",
            "start_positions", "end_positions", "is_answerable"]

    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [item[k] for item in batch]
        # Scalars (start/end positions, is_answerable) are 0-dim → stack directly
        if tensors[0].dim() == 0:
            out[k] = torch.stack(tensors)
        else:
            # Verify all have the same shape
            shape0 = tensors[0].shape
            for i, t in enumerate(tensors):
                if t.shape != shape0:
                    raise ValueError(
                        f"Shape mismatch in batch for key '{k}': "
                        f"item 0 has {shape0}, item {i} has {t.shape}"
                    )
            out[k] = torch.stack(tensors)

    return out


# =========================================================================
# Main — quick smoke test
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tok = get_tokenizer()
    print(f"Tokenizer: {tok.__class__.__name__}, vocab size: {tok.vocab_size}\n")

    # Load a small slice for quick verification
    print("=" * 60)
    print("STAGE 1 — General QA Dataset  (100-example smoke test)")
    print("=" * 60)

    # --- SQuAD only for fast test ---
    squad_train, _ = load_and_validate_squad()
    test_examples = squad_train[:100]

    passed = 0
    failed = 0
    discarded = 0
    n_answerable = 0
    n_unanswerable = 0

    for ex in test_examples:
        feats = convert_to_features(ex, tok)
        if feats is None:
            discarded += 1
            continue

        if ex["is_answerable"]:
            n_answerable += 1
        else:
            n_unanswerable += 1

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

    total = passed + failed
    print(f"\nVerification summary (100 examples):")
    print(f"  Passed:         {passed}/{total}")
    print(f"  Failed:         {failed}/{total}")
    print(f"  Discarded:      {discarded}")
    print(f"  Answerable:     {n_answerable}")
    print(f"  Unanswerable:   {n_unanswerable}")
    print(f"  Ratio ans/unans: {n_answerable}/{n_unanswerable}")
