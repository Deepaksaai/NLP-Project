"""
Stage 0 Part 1 — Train a BPE tokenizer from scratch on ALL domains.

Datasets used:
  - CNN/DailyMail  (news articles + highlights)
  - XSum           (news documents + summaries)
  - arXiv          (scientific articles + abstracts)
  - PubMed         (biomedical articles + abstracts)
  - MultiLexSum    (legal case sources + summaries)  [local JSONL files]
  - BillSum        (legislative bills + summaries)

The tokenizer sees every domain upfront so its vocabulary covers
general, scientific, and legal language equally.

No pretrained weights — only the BPE algorithm implementation from
the HuggingFace `tokenizers` library.
"""

import os
import sys
import json

# -------------------------------------------------------
# Config
# -------------------------------------------------------
VOCAB_SIZE = 32000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<legal>"]

# MultiLexSum local files (two releases)
MULTILEXSUM_FILES = [
    {"train": "../datasets/multilexsum/train_1.json", "test": "../datasets/multilexsum/test_1.json", "sources": "../datasets/multilexsum/sources_1.json"},
    {"train": "../datasets/multilexsum/train_2.json", "test": "../datasets/multilexsum/test_2.json", "sources": "../datasets/multilexsum/sources_2.json"},
]

OUTPUT_DIR = "."  # saves into tokenizer/


# -------------------------------------------------------
# Dataset iterators — each yields strings without loading all into RAM
# -------------------------------------------------------
def iter_cnn_dailymail():
    """CNN/DailyMail 3.0.0 — articles + highlights."""
    from datasets import load_dataset
    print("  Loading CNN/DailyMail...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
    count = 0
    for ex in ds:
        yield ex["article"]
        yield ex["highlights"]
        count += 1
    print(f"    CNN/DailyMail: {count} examples streamed")


def iter_xsum():
    """XSum — document + summary."""
    from datasets import load_dataset
    print("  Loading XSum...")
    ds = load_dataset("EdinburghNLP/xsum", split="train")
    count = 0
    for ex in ds:
        yield ex["document"]
        yield ex["summary"]
        count += 1
    print(f"    XSum: {count} examples streamed")


def iter_arxiv():
    """arXiv — article + abstract."""
    from datasets import load_dataset
    print("  Loading arXiv...")
    try:
        ds = load_dataset("ccdv/arxiv-summarization", split="train")
    except Exception:
        try:
            ds = load_dataset("tomasg25/scientific_lay_summarisation", "arxiv", split="train")
        except Exception as e:
            print(f"    arXiv unavailable ({e}), skipping")
            return
    count = 0
    # Field names vary by dataset version
    text_key = "article" if "article" in ds.column_names else "text"
    summ_key = "abstract" if "abstract" in ds.column_names else "summary"
    for ex in ds:
        yield ex[text_key]
        yield ex[summ_key]
        count += 1
    print(f"    arXiv: {count} examples streamed")


def iter_pubmed():
    """PubMed — article + abstract."""
    from datasets import load_dataset
    print("  Loading PubMed...")
    try:
        ds = load_dataset("ccdv/pubmed-summarization", split="train")
    except Exception:
        try:
            ds = load_dataset("tomasg25/scientific_lay_summarisation", "plos", split="train")
        except Exception as e:
            print(f"    PubMed unavailable ({e}), skipping")
            return
    count = 0
    text_key = "article" if "article" in ds.column_names else "text"
    summ_key = "abstract" if "abstract" in ds.column_names else "summary"
    for ex in ds:
        yield ex[text_key]
        yield ex[summ_key]
        count += 1
    print(f"    PubMed: {count} examples streamed")


def iter_billsum():
    """BillSum — text + summary."""
    from datasets import load_dataset
    print("  Loading BillSum...")
    ds = load_dataset("billsum", split="train")
    count = 0
    for ex in ds:
        yield ex["text"]
        yield ex["summary"]
        count += 1
    print(f"    BillSum: {count} examples streamed")


def iter_multilexsum():
    """MultiLexSum — source documents + case summaries from local JSONL files."""
    print("  Loading MultiLexSum (local files)...")
    seen_cases = set()
    total_docs = 0
    total_summaries = 0

    for fg in MULTILEXSUM_FILES:
        sources_path = fg["sources"]
        if not os.path.exists(sources_path):
            print(f"    Skipping {sources_path} (not found)")
            continue

        # Load sources dict
        with open(sources_path, "r", encoding="utf-8") as f:
            sources = json.load(f)

        # Yield all source document texts
        for doc_id, entry in sources.items():
            text = entry.get("doc_text") or entry.get("text", "")
            if text and len(text) > 50:
                yield text
                total_docs += 1

        # Yield summaries from train + test splits
        for split_path in [fg["train"], fg["test"]]:
            if not os.path.exists(split_path):
                continue
            with open(split_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    case = json.loads(line)
                    case_id = case.get("case_id")
                    if case_id and case_id in seen_cases:
                        continue
                    if case_id:
                        seen_cases.add(case_id)

                    for key in ["summary/long", "summary/short", "summary/tiny"]:
                        s = case.get(key)
                        if s:
                            yield s
                            total_summaries += 1

        # Free sources dict memory before loading next release
        del sources

    print(f"    MultiLexSum: {total_docs} docs + {total_summaries} summaries streamed")


def all_data_iterator():
    """Chain all dataset iterators into one stream."""
    yield from iter_cnn_dailymail()
    yield from iter_xsum()
    yield from iter_arxiv()
    yield from iter_pubmed()
    yield from iter_billsum()
    yield from iter_multilexsum()


# -------------------------------------------------------
# Train the tokenizer
# -------------------------------------------------------
def train_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.normalizers import NFC
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    print("=" * 60)
    print("Stage 0 Part 1: Training BPE Tokenizer")
    print("=" * 60)

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # ByteLevel pre-tokenizer: handles any unicode gracefully,
    # never produces <unk> for unseen characters
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.normalizer = NFC()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    print("\nStreaming data from all 6 datasets...")
    tokenizer.train_from_iterator(all_data_iterator(), trainer=trainer)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"\nTokenizer trained — vocab size: {actual_vocab}")

    # Verify special token IDs
    for i, tok in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(tok)
        assert actual_id == i, f"Special token {tok} has id {actual_id}, expected {i}"
        print(f"  {tok:10s} = {actual_id}")

    # Save tokenizer
    tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))
    print(f"\nTokenizer saved to {OUTPUT_DIR}/tokenizer.json")

    # ---- Validation: tokenize sample sentences ----
    print("\n--- Validation ---")

    legal_text = "The plaintiff filed a motion for summary judgment pursuant to Rule 56."
    news_text = "The prime minister announced new economic reforms on Monday."
    science_text = "We propose a novel architecture for abstractive summarization."

    for label, text in [("Legal", legal_text), ("News", news_text), ("Science", science_text)]:
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        ids = encoding.ids
        decoded = tokenizer.decode(ids)
        print(f"\n  [{label}] {text}")
        print(f"  Tokens ({len(tokens)}): {tokens}")
        print(f"  IDs:    {ids}")
        print(f"  Decoded: {decoded}")

    print("\n" + "=" * 60)
    print("Stage 0 Part 1 COMPLETE")
    print("=" * 60)

    return tokenizer


if __name__ == "__main__":
    train_tokenizer()
