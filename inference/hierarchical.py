"""
Hierarchical inference pipeline for Stage 3 model.

Implements:
  - Beam search with repetition penalty, no-repeat n-gram, min length, length penalty
  - Hierarchical chunked summarization (chunk → summarize → concat → re-summarize)
  - <legal> token prepending
  - Post-processing (special token strip, dedup, whitespace)

Usage:
    from inference.hierarchical import HierarchicalSummarizer
    summarizer = HierarchicalSummarizer(model, tokenizer, device)
    summary = summarizer.summarize(legal_document_text)
"""

import re
import torch
import torch.nn.functional as F

from data.preprocess import PAD_ID, SOS_ID, EOS_ID, LEGAL_ID


# -------------------------------------------------------
# Beam search
# -------------------------------------------------------
@torch.no_grad()
def beam_search_generate(
    model,
    src,
    beam_width=4,
    max_len=256,
    min_len=30,
    length_penalty=0.6,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    sos_idx=SOS_ID,
    eos_idx=EOS_ID,
):
    """
    Beam search decoding with repetition penalty, no-repeat n-gram constraint,
    and length normalization.

    Args:
        src: (1, src_len) source token tensor
    Returns:
        list[int] of generated token IDs (excluding SOS)
    """
    model.eval()
    device = src.device
    assert src.size(0) == 1, "beam_search_generate expects batch_size=1"

    src_mask = model.make_src_mask(src)
    memory = model.encode(src, src_mask)

    # Each beam: dict with tokens, score, done flag
    beams = [{"tokens": [sos_idx], "score": 0.0, "done": False}]
    completed = []

    def length_normalized_score(beam):
        n = len(beam["tokens"])
        return beam["score"] / (((5 + n) / 6) ** length_penalty)

    for step in range(max_len):
        if all(b["done"] for b in beams):
            break

        candidates = []

        for beam in beams:
            if beam["done"]:
                completed.append(beam)
                continue

            tokens = beam["tokens"]
            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_mask = model.make_tgt_mask(tgt)
            out = model.decode(tgt, memory, tgt_mask, src_mask)
            logits = model.project(out[:, -1]).squeeze(0)  # (vocab,)

            # Repetition penalty: reduce probability of already-generated tokens
            if repetition_penalty != 1.0:
                for tok in set(tokens):
                    if logits[tok] > 0:
                        logits[tok] /= repetition_penalty
                    else:
                        logits[tok] *= repetition_penalty

            # No-repeat n-gram: ban tokens that would form repeated n-grams
            if no_repeat_ngram_size > 0 and len(tokens) >= no_repeat_ngram_size:
                prefix = tuple(tokens[-(no_repeat_ngram_size - 1):])
                banned = set()
                for i in range(len(tokens) - no_repeat_ngram_size + 1):
                    if tuple(tokens[i:i + no_repeat_ngram_size - 1]) == prefix:
                        banned.add(tokens[i + no_repeat_ngram_size - 1])
                for t in banned:
                    logits[t] = -1e9

            # Min length: prevent EOS until min_len reached
            if (len(tokens) - 1) < min_len:
                logits[eos_idx] = -1e9

            # Top-k expansion
            log_probs = F.log_softmax(logits, dim=-1)
            top_lp, top_idx = log_probs.topk(beam_width)

            for k in range(beam_width):
                tok = top_idx[k].item()
                new_score = beam["score"] + top_lp[k].item()
                new_tokens = tokens + [tok]
                done = (tok == eos_idx)
                candidates.append({
                    "tokens": new_tokens,
                    "score": new_score,
                    "done": done,
                })

        # Keep top-k by length-normalized score
        candidates.sort(key=length_normalized_score, reverse=True)
        beams = candidates[:beam_width]

    completed.extend(beams)
    if not completed:
        return [sos_idx]

    completed.sort(key=length_normalized_score, reverse=True)
    best_tokens = completed[0]["tokens"]

    # Strip leading SOS
    if best_tokens and best_tokens[0] == sos_idx:
        best_tokens = best_tokens[1:]
    # Strip trailing EOS
    if best_tokens and best_tokens[-1] == eos_idx:
        best_tokens = best_tokens[:-1]

    return best_tokens


# -------------------------------------------------------
# Text utilities
# -------------------------------------------------------
def chunk_words(text, chunk_words=350, overlap_words=50, max_chunks=6):
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    stride = chunk_words - overlap_words
    for start in range(0, len(words), stride):
        chunk = " ".join(words[start:start + chunk_words])
        chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
        if len(chunks) >= max_chunks:
            break
    return chunks


def remove_repeated_sentences(text):
    """Remove duplicate sentences while preserving order."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for s in sentences:
        s_norm = s.strip().lower()
        if s_norm and s_norm not in seen:
            seen.add(s_norm)
            unique.append(s.strip())
    return " ".join(unique)


def post_process(text):
    """Clean up generated summary."""
    # Remove special token strings if they leaked through decode
    text = text.replace("<s>", "").replace("</s>", "")
    text = text.replace("<legal>", "").replace("<pad>", "")
    text = text.replace("<unk>", "")
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove repeated sentences
    text = remove_repeated_sentences(text)
    return text


# -------------------------------------------------------
# Hierarchical summarizer
# -------------------------------------------------------
class HierarchicalSummarizer:

    def __init__(self, model, tokenizer, device,
                 max_src_len=400, max_gen_len=256, min_gen_len=80,
                 beam_width=4, length_penalty=1.2,
                 repetition_penalty=1.2, no_repeat_ngram_size=3,
                 chunk_size=350, overlap=50, max_chunks=6):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_src_len = max_src_len
        self.max_gen_len = max_gen_len
        self.min_gen_len = min_gen_len
        self.beam_width = beam_width
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chunks = max_chunks

        self.model.eval()

    def encode_with_legal(self, text):
        """Tokenize text with <legal> prefix, truncate to max_src_len."""
        ids = self.tokenizer.encode(text.strip()).ids[:self.max_src_len - 1]
        ids = [LEGAL_ID] + ids
        ids = ids[:self.max_src_len]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def generate_one(self, text, gen_min=None, gen_max=None):
        """Run beam search on a single text input."""
        src = self.encode_with_legal(text)
        out_ids = beam_search_generate(
            self.model, src,
            beam_width=self.beam_width,
            max_len=gen_max or self.max_gen_len,
            min_len=gen_min or self.min_gen_len,
            length_penalty=self.length_penalty,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
        )
        text_out = self.tokenizer.decode(out_ids)
        return post_process(text_out)

    def summarize(self, document):
        """
        Full hierarchical pipeline:
          1. Chunk the document
          2. Beam search each chunk
          3. Concat chunk summaries
          4. Beam search the concatenation (second pass)
          5. Post-process and return
        """
        # Step 1-2: chunk and summarize each
        chunks = chunk_words(
            document,
            chunk_words=self.chunk_size,
            overlap_words=self.overlap,
            max_chunks=self.max_chunks,
        )

        if len(chunks) == 1:
            # Single chunk — no need for second pass
            return self.generate_one(chunks[0])

        chunk_summaries = []
        for chunk in chunks:
            summary = self.generate_one(chunk, gen_min=20, gen_max=120)
            chunk_summaries.append(summary)

        # Step 3-4: concatenate and re-summarize
        concatenated = " ".join(chunk_summaries)
        final = self.generate_one(
            concatenated,
            gen_min=self.min_gen_len,
            gen_max=self.max_gen_len,
        )

        return final
