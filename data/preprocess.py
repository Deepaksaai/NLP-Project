"""
Tokenization and preprocessing utilities shared across all stages.
"""

from tokenizers import Tokenizer


def load_tokenizer(path="tokenizer/tokenizer.json"):
    """Load the Stage 0 BPE tokenizer from disk."""
    tokenizer = Tokenizer.from_file(path)
    print(f"Tokenizer loaded from {path} (vocab={tokenizer.get_vocab_size()})")
    return tokenizer


# Fixed special token IDs (must match Stage 0 training)
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
LEGAL_ID = 4
LEGAL_ID = 4


def tokenize_pair(source_text, target_text, tokenizer, max_src=400, max_tgt=128):
    """
    Tokenize a (source, target) pair for seq2seq training.

    Source: raw BPE token IDs, truncated to max_src.
    Target: wrapped with <s> ... </s>, truncated to max_tgt total.

    Returns dict with input_ids, target_ids, and their lengths.
    """
    src_ids = tokenizer.encode(source_text.strip()).ids[:max_src]

    tgt_tokens = tokenizer.encode(target_text.strip()).ids[:max_tgt - 2]
    tgt_ids = [SOS_ID] + tgt_tokens + [EOS_ID]

    return {
        "input_ids": src_ids,
        "target_ids": tgt_ids,
        "src_len": len(src_ids),
        "tgt_len": len(tgt_ids),
    }
