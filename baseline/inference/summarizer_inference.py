"""
Summarizer inference wrapper.

Thin adapter around inference/hierarchical.py that accepts pre-chunked
document dicts (as produced by baseline/preprocessing.py) and returns a
single clean summary.

Exports:
    load_summarizer(checkpoint_path, device)
    run_summarization(chunks, model, tokenizer, device)
"""

import os
import sys
import re
import torch

# Ensure project root is on sys.path so model / inference / data
# modules resolve correctly when called from any cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.transformer import build_model
from inference.hierarchical import beam_search_generate, post_process
from data.preprocess import LEGAL_ID


# -------------------------------------------------------
# Model loader
# -------------------------------------------------------
def load_summarizer(checkpoint_path: str, device: str):
    """
    Load the Stage 3 summarizer Transformer onto the given device.

    Args:
        checkpoint_path: Path to stage3_best.pt
        device: "cuda" or "cpu"

    Returns:
        Transformer model in eval mode.
    """
    model = build_model(
        device=device,
        vocab_size=32000,
        d_model=384,
        n_heads=6,
        n_encoder_layers=6,
        n_decoder_layers=4,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.0,
        pad_idx=0,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# -------------------------------------------------------
# Internal helpers
# -------------------------------------------------------
def _beam_pass(text: str, model, tokenizer, device: str,
               max_len: int = 256, min_len: int = 80,
               max_src: int = 400):
    """Tokenize with <legal> prefix, run beam search, post-process."""
    ids = tokenizer.encode(text.strip()).ids[: max_src - 1]
    ids = [LEGAL_ID] + ids
    src = torch.tensor([ids], dtype=torch.long, device=device)

    out_ids = beam_search_generate(
        model, src,
        beam_width=4,
        max_len=max_len,
        min_len=min_len,
        length_penalty=0.6,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    return post_process(tokenizer.decode(out_ids))


def _truncate_to_sentence_boundary(text: str, max_words: int) -> str:
    """Truncate text to the last complete sentence under max_words."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = []
    count = 0
    for s in sentences:
        sw = len(s.split())
        if count + sw > max_words:
            break
        kept.append(s)
        count += sw
    if not kept:
        return " ".join(text.split()[:max_words])
    return " ".join(kept)


# -------------------------------------------------------
# Main entry point
# -------------------------------------------------------
def run_summarization(chunks, model, tokenizer, device: str):
    """
    Run full hierarchical summarization over pre-chunked input.

    Args:
        chunks: list of chunk dicts from preprocessing.py. Each chunk must
                contain 'text' (section header + body) and 'word_count'.
        model: loaded summarizer Transformer
        tokenizer: shared BPE tokenizer (32 000 vocab)
        device: "cuda" or "cpu"

    Returns:
        {
          "summary":     final summary string,
          "chunk_count": number of chunk-level summaries produced,
          "path":        "direct" | "grouped" | "single" | "none",
          "word_count":  word count of the final summary,
        }
    """

    # -------------------------------------------------------
    # Step 1 — chunk-level summaries
    # -------------------------------------------------------
    chunk_summaries = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        wc = chunk.get("word_count") or len(text.split())

        # Skip stubs that carry no real content
        if wc < 30 or not text:
            continue

        chunk_sum = _beam_pass(
            text, model, tokenizer, device,
            max_len=150, min_len=20, max_src=400,
        )
        if chunk_sum:
            chunk_summaries.append(chunk_sum)

    total = len(chunk_summaries)
    if total == 0:
        return {"summary": "", "chunk_count": 0, "path": "none", "word_count": 0}

    # -------------------------------------------------------
    # Step 2 — progressive combination
    # -------------------------------------------------------
    if total == 1:
        final = chunk_summaries[0]
        path = "single"
    elif total <= 6:
        combined = " ".join(chunk_summaries)
        final = _beam_pass(
            combined, model, tokenizer, device,
            max_len=256, min_len=80, max_src=400,
        )
        path = "direct"
    else:
        group_summaries = []
        for i in range(0, total, 3):
            group_text = " ".join(chunk_summaries[i:i + 3])
            group_sum = _beam_pass(
                group_text, model, tokenizer, device,
                max_len=180, min_len=40, max_src=400,
            )
            group_summaries.append(group_sum)
        combined_groups = " ".join(group_summaries)
        final = _beam_pass(
            combined_groups, model, tokenizer, device,
            max_len=256, min_len=80, max_src=400,
        )
        path = "grouped"

    # -------------------------------------------------------
    # Step 3 — post-processing
    # -------------------------------------------------------
    final = post_process(final)
    wc = len(final.split())

    if wc < 50:
        print(f"  [warn] summary is short ({wc} words)")
    elif wc > 400:
        final = _truncate_to_sentence_boundary(final, 400)
        wc = len(final.split())

    return {
        "summary": final,
        "chunk_count": total,
        "path": path,
        "word_count": wc,
    }
