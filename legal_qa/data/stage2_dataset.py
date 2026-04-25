"""
Stage 2 — Long-Document QA Dataset.

Loads QuALITY and QASPER, converts them to span-extraction format, wraps
long documents in a ChunkingDataset that creates overlapping 400-word chunks,
and mixes in 20 % SQuAD 2.0 anchor examples to prevent catastrophic
forgetting.

Usage:
    from legal_qa.data.stage2_dataset import LongDocQADataset, collate_fn
    ds = LongDocQADataset(split="train")
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
"""

import difflib
import logging
import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from legal_qa.data.dataset_utils import (
    get_tokenizer,
    load_and_validate_squad,
    convert_to_features,
    verify_span,
    MAX_SEQ_LEN,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Internal loaders — produce List[dict] in the uniform schema
# =========================================================================

# -------------------------------------------------------------------------
# QuALITY  (multiple-choice → span extraction via fuzzy match)
# -------------------------------------------------------------------------
def _load_quality_examples(
    split: str = "train",
    match_threshold: float = 0.6,
    max_examples: Optional[int] = None,
) -> List[dict]:
    """
    Load QuALITY and convert MC questions to span extraction.

    For each question we fuzzy-search the correct option text inside the
    article.  If the best SequenceMatcher ratio ≥ ``match_threshold`` we
    treat the best-matching substring as the answer span; otherwise the
    example is marked unanswerable.
    """
    logger.info("Loading QuALITY — %s …", split)
    try:
        ds = load_dataset("emrecan/quality", split=split)
    except Exception:
        try:
            ds = load_dataset("QuALITY", split=split)
        except Exception as e:
            logger.warning("Could not load QuALITY: %s — skipping", e)
            return []

    examples: List[dict] = []

    for idx, row in enumerate(ds):
        if max_examples and len(examples) >= max_examples:
            break

        article = row.get("article", "") or row.get("context", "")
        question = row.get("question", "")
        options = row.get("options", [])
        gold_label = row.get("gold_label", row.get("answer", -1))

        if not article or not question or not options:
            continue

        # Identify the correct option text
        try:
            correct_idx = int(gold_label)
            # QuALITY labels might be 1-indexed
            if correct_idx >= len(options):
                correct_idx = correct_idx - 1
            if correct_idx < 0 or correct_idx >= len(options):
                continue
            correct_text = options[correct_idx]
        except (ValueError, IndexError):
            continue

        # Fuzzy-match the correct option inside the article
        article_lower = article.lower()
        option_lower = correct_text.lower()

        best_ratio = 0.0
        best_start = -1
        best_len = len(option_lower)

        # Sliding window fuzzy search — use a window width proportional
        # to the option length (±50 %)
        win_min = max(5, int(best_len * 0.5))
        win_max = min(len(article_lower), int(best_len * 1.5) + 1)

        for win_size in range(win_min, win_max):
            step = max(1, win_size // 4)
            for start in range(0, len(article_lower) - win_size + 1, step):
                candidate = article_lower[start : start + win_size]
                ratio = difflib.SequenceMatcher(
                    None, option_lower, candidate
                ).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = start
                    best_len = win_size

        if best_ratio >= match_threshold and best_start >= 0:
            answer_text = article[best_start : best_start + best_len]
            examples.append({
                "question": question,
                "context": article,
                "answer_text": answer_text,
                "answer_start": best_start,
                "is_answerable": True,
            })
        else:
            examples.append({
                "question": question,
                "context": article,
                "answer_text": "",
                "answer_start": -1,
                "is_answerable": False,
            })

    logger.info("QuALITY %s: %d examples", split, len(examples))
    return examples


# -------------------------------------------------------------------------
# QASPER
# -------------------------------------------------------------------------
def _load_qasper_examples(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[dict]:
    """
    Load QASPER and keep only extractive QA examples.
    yes/no and abstractive answers are marked unanswerable.
    """
    logger.info("Loading QASPER — %s …", split)
    try:
        ds = load_dataset("allenai/qasper", split=split)
    except Exception as e:
        logger.warning("Could not load QASPER: %s — skipping", e)
        return []

    examples: List[dict] = []

    for row in ds:
        if max_examples and len(examples) >= max_examples:
            break

        # Build full text from paragraphs in sections
        full_text_parts = []
        sections = row.get("full_text", {})
        section_names = sections.get("section_name", [])
        paragraphs_list = sections.get("paragraphs", [])

        for sec_name, paras in zip(section_names, paragraphs_list):
            if sec_name:
                full_text_parts.append(sec_name)
            if isinstance(paras, list):
                full_text_parts.extend(paras)
            elif isinstance(paras, str):
                full_text_parts.append(paras)

        full_text = "\n".join(full_text_parts)
        if not full_text.strip():
            continue

        # Process QAs
        qas = row.get("qas", {})
        questions = qas.get("question", [])
        answers_list = qas.get("answers", [])

        for q, ans_block in zip(questions, answers_list):
            if not q:
                continue

            answer_entries = ans_block.get("answer", [])
            if not answer_entries:
                continue

            found_extractive = False
            for ae in answer_entries:
                ans_type = ae.get("type", "")
                extractive_spans = ae.get("extractive_spans", [])

                if ans_type == "extractive" and extractive_spans:
                    ans_text = extractive_spans[0]
                    if not ans_text:
                        continue
                    pos = full_text.find(ans_text)
                    if pos >= 0:
                        examples.append({
                            "question": q,
                            "context": full_text,
                            "answer_text": ans_text,
                            "answer_start": pos,
                            "is_answerable": True,
                        })
                        found_extractive = True
                        break

            if not found_extractive:
                # yes/no or abstractive → unanswerable for span extraction
                examples.append({
                    "question": q,
                    "context": full_text[:10_000],
                    "answer_text": "",
                    "answer_start": -1,
                    "is_answerable": False,
                })

    logger.info("QASPER %s: %d examples", split, len(examples))
    return examples


# =========================================================================
# Chunking wrapper — splits long docs into overlapping chunks
# =========================================================================
class ChunkingDataset(Dataset):
    """
    Wraps a list of long-document QA examples and splits each document
    into overlapping word-level chunks.

    For each document:
      • The chunk containing the gold answer is the *positive* example.
      • Up to ``neg_per_doc`` other chunks become *unanswerable* negatives.

    Args:
        raw_examples: List[dict] with question, context, answer_text,
                      answer_start, is_answerable.
        chunk_words:  Chunk size in white-space tokens (default 400).
        overlap_words: Overlap between successive chunks (default 50).
        neg_per_doc:  Max negative chunks to keep per document (default 4).
        tokenizer:    DeBERTa tokenizer.
    """

    def __init__(
        self,
        raw_examples: List[dict],
        chunk_words: int = 400,
        overlap_words: int = 50,
        neg_per_doc: int = 4,
        tokenizer=None,
    ):
        super().__init__()
        if tokenizer is None:
            tokenizer = get_tokenizer()

        self.features: List[dict] = []
        self.is_answerable_flags: List[bool] = []

        discarded = 0

        for ex in raw_examples:
            context = ex["context"]
            question = ex["question"]
            is_ans = ex["is_answerable"]
            answer_text = ex.get("answer_text", "")
            answer_start = ex.get("answer_start", -1)

            # --- Chunk the context ---
            words = context.split()
            if not words:
                discarded += 1
                continue

            chunk_specs = []  # (char_start, char_end) of each chunk
            step = max(1, chunk_words - overlap_words)

            # Build char offset map for word boundaries
            word_char_starts = []
            pos = 0
            for w in words:
                idx = context.find(w, pos)
                word_char_starts.append(idx)
                pos = idx + len(w)

            for wi in range(0, len(words), step):
                wj = min(wi + chunk_words, len(words))
                cs = word_char_starts[wi]
                # End of last word in chunk
                ce = word_char_starts[wj - 1] + len(words[wj - 1])
                chunk_specs.append((cs, ce))
                if wj >= len(words):
                    break

            # --- Find the chunk containing the answer ---
            pos_chunk_idx = -1
            if is_ans and answer_start >= 0 and answer_text:
                answer_end = answer_start + len(answer_text)
                for ci, (cs, ce) in enumerate(chunk_specs):
                    if cs <= answer_start and answer_end <= ce:
                        pos_chunk_idx = ci
                        break

            # --- Build features ---
            neg_indices = [
                i for i in range(len(chunk_specs)) if i != pos_chunk_idx
            ]
            if len(neg_indices) > neg_per_doc:
                neg_indices = random.sample(neg_indices, neg_per_doc)

            for ci, (cs, ce) in enumerate(chunk_specs):
                chunk_text = context[cs:ce]

                if ci == pos_chunk_idx:
                    # Positive — adjust answer_start relative to chunk
                    local_start = answer_start - cs
                    chunk_ex = {
                        "question": question,
                        "context": chunk_text,
                        "answer_text": answer_text,
                        "answer_start": local_start,
                        "is_answerable": True,
                    }
                elif ci in neg_indices:
                    chunk_ex = {
                        "question": question,
                        "context": chunk_text,
                        "answer_text": "",
                        "answer_start": -1,
                        "is_answerable": False,
                    }
                else:
                    continue  # Skip chunks beyond the negative budget

                feats = convert_to_features(chunk_ex, tokenizer)
                if feats is None:
                    discarded += 1
                    continue
                self.features.append(feats)
                self.is_answerable_flags.append(chunk_ex["is_answerable"])

        logger.info(
            "ChunkingDataset: %d features (%d discarded), ans=%d, unans=%d",
            len(self.features), discarded,
            sum(self.is_answerable_flags),
            len(self.is_answerable_flags) - sum(self.is_answerable_flags),
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# =========================================================================
# Full Stage-2 dataset (long-doc chunks + 20 % SQuAD anchor)
# =========================================================================
class LongDocQADataset(Dataset):
    """
    Stage-2 dataset combining chunked QuALITY + QASPER examples with
    20 % SQuAD 2.0 anchor data.
    """

    def __init__(
        self,
        split: str = "train",
        squad_fraction: float = 0.20,
        max_quality: Optional[int] = None,
        max_qasper: Optional[int] = None,
        tokenizer=None,
    ):
        super().__init__()
        if tokenizer is None:
            tokenizer = get_tokenizer()

        # ---- Long-document sources ----
        quality_raw = _load_quality_examples(split, max_examples=max_quality)
        qasper_raw = _load_qasper_examples(split, max_examples=max_qasper)

        long_raw = quality_raw + qasper_raw
        logger.info("Stage 2 long-doc raw: %d (QuALITY=%d, QASPER=%d)",
                     len(long_raw), len(quality_raw), len(qasper_raw))

        # ---- Chunk long documents ----
        chunked = ChunkingDataset(long_raw, tokenizer=tokenizer)

        # ---- SQuAD anchor ----
        squad_train, _ = load_and_validate_squad()
        n_anchor = int(len(chunked) * squad_fraction / (1 - squad_fraction))
        n_anchor = min(n_anchor, len(squad_train))
        squad_sample = random.sample(squad_train, n_anchor)

        anchor_features: List[dict] = []
        for ex in squad_sample:
            feats = convert_to_features(ex, tokenizer)
            if feats is not None:
                anchor_features.append(feats)

        logger.info("SQuAD anchor: %d examples (target %d)",
                     len(anchor_features), n_anchor)

        # ---- Combine ----
        self.features = chunked.features + anchor_features
        self.is_answerable_flags = (
            chunked.is_answerable_flags
            + [True] * len(anchor_features)  # SQuAD anchors may include unans
        )

        # Shuffle once
        combined = list(zip(self.features, self.is_answerable_flags))
        random.shuffle(combined)
        self.features = [c[0] for c in combined]
        self.is_answerable_flags = [c[1] for c in combined]

        logger.info(
            "LongDocQADataset ready — %d features, ans=%d, unans=%d",
            len(self.features),
            sum(self.is_answerable_flags),
            len(self.is_answerable_flags) - sum(self.is_answerable_flags),
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# =========================================================================
# Collation
# =========================================================================
def collate_fn(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """Stack feature dicts into a batched tensor dict."""
    keys = ["input_ids", "attention_mask", "token_type_ids",
            "start_positions", "end_positions", "is_answerable"]

    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [item[k] for item in batch]
        if tensors[0].dim() == 0:
            out[k] = torch.stack(tensors)
        else:
            shape0 = tensors[0].shape
            for i, t in enumerate(tensors):
                if t.shape != shape0:
                    raise ValueError(
                        f"Shape mismatch for key '{k}': "
                        f"item 0 has {shape0}, item {i} has {t.shape}"
                    )
            out[k] = torch.stack(tensors)
    return out


# =========================================================================
# Main — smoke test
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tok = get_tokenizer()
    print(f"Tokenizer: {tok.__class__.__name__}, vocab size: {tok.vocab_size}\n")

    print("=" * 60)
    print("STAGE 2 — Long-Document QA Dataset  (100-example smoke test)")
    print("=" * 60)

    # Test QASPER only for speed
    qasper_raw = _load_qasper_examples("train", max_examples=50)
    # Also grab some SQuAD
    squad_train, _ = load_and_validate_squad()
    test_raw = qasper_raw[:50] + squad_train[:50]

    passed = 0
    failed = 0
    discarded = 0
    n_answerable = 0
    n_unanswerable = 0

    for ex in test_raw:
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
