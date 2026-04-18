"""
Multi-chunk inference for long documents.

Given (question, document), this pipeline:

    1. Splits the document into overlapping 400/50 word chunks.
    2. Builds the [CLS] q [SEP] chunk [SEP] input for each chunk.
    3. Runs the model on every chunk to get start/end/has_answer logits.
    4. Filters chunks whose sigmoid(has_answer) < 0.4 (permissive).
    5. On each surviving chunk, picks the best (i, j) via joint span
       selection with a -1e9 floor on invalid positions.
    6. Scores each chunk by
            final = sigmoid(has_answer) * best_span_softmax_score
       where best_span_softmax_score is exp(start+end) after subtracting
       the chunk's maximum span score to stay numerically stable.
    7. Chooses the chunk with the highest final score. If the winning
       chunk's has_answer score is still < 0.5, returns the explicit
       'not found in document' response.

The same tokenizer and QAModel used during training must be passed in.
"""

import math
import torch
from dataclasses import dataclass
from typing import List, Optional

from QA.qa_config import (
    MAX_QUESTION_LEN, MAX_CONTEXT_LEN, MAX_TOTAL_LEN, PAD_ID, MAX_ANSWER_LEN,
)
from QA.data.chunking import chunk_document
from QA.training.span_select import joint_span_select


NOT_FOUND_MESSAGE = "This information was not found in the document."

_HAS_ANS_KEEP_THRESHOLD = 0.3
_HAS_ANS_FINAL_THRESHOLD = 0.35


@dataclass
class ChunkPrediction:
    chunk_idx: int
    has_answer_score: float
    best_span_score: float
    final_score: float
    start_tok: int
    end_tok: int
    span_text: str


def _build_input(tokenizer, question: str, chunk_text: str, meta):
    q_ids = tokenizer.encode(question).ids[:MAX_QUESTION_LEN]
    c_ids = tokenizer.encode(chunk_text).ids[:MAX_CONTEXT_LEN]
    input_ids = [meta["cls_id"]] + q_ids + [meta["sep_id"]] + c_ids + [meta["sep_id"]]
    seg = [0] * (1 + len(q_ids) + 1) + [1] * (len(c_ids) + 1)
    attn = [1] * len(input_ids)

    pad = MAX_TOTAL_LEN - len(input_ids)
    if pad > 0:
        input_ids += [PAD_ID] * pad
        seg += [1] * pad
        attn += [0] * pad
    else:
        input_ids = input_ids[:MAX_TOTAL_LEN]
        seg = seg[:MAX_TOTAL_LEN]
        attn = attn[:MAX_TOTAL_LEN]
    return input_ids, seg, attn


@torch.no_grad()
def answer_question(
    model,
    tokenizer,
    qa_meta: dict,
    question: str,
    document: str,
    device,
    batch_size: int = 8,
    max_answer_len: int = MAX_ANSWER_LEN,
    return_all_chunks: bool = False,
) -> dict:
    model.eval()

    chunks = chunk_document(document, chunk_words=400, overlap_words=50)
    if not chunks:
        return {"answer": NOT_FOUND_MESSAGE, "best_chunk": None, "chunks": []}

    # Stack all chunks into one tensor and run in mini-batches
    inputs, segs, attns = [], [], []
    for c in chunks:
        ii, ss, aa = _build_input(tokenizer, question, c.text, qa_meta)
        inputs.append(ii); segs.append(ss); attns.append(aa)

    input_ids   = torch.tensor(inputs, dtype=torch.long, device=device)
    segment_ids = torch.tensor(segs,   dtype=torch.long, device=device)
    attn_mask   = torch.tensor(attns,  dtype=torch.long, device=device)

    all_start, all_end, all_has = [], [], []
    for s in range(0, len(chunks), batch_size):
        e = min(s + batch_size, len(chunks))
        sl, el, hl = model(input_ids[s:e], segment_ids[s:e], attn_mask[s:e])
        all_start.append(sl); all_end.append(el); all_has.append(hl)
    start_logits = torch.cat(all_start, dim=0)   # (n_chunks, L)
    end_logits   = torch.cat(all_end,   dim=0)
    has_logits   = torch.cat(all_has,   dim=0)   # (n_chunks,)

    has_probs = torch.sigmoid(has_logits)

    # Joint span selection per chunk
    s_idx, e_idx = joint_span_select(start_logits, end_logits, max_answer_len)

    preds: List[ChunkPrediction] = []
    for k in range(len(chunks)):
        h = has_probs[k].item()
        i = int(s_idx[k].item())
        j = int(e_idx[k].item())
        span_score = (start_logits[k, i] + end_logits[k, j]).item()
        # Skip trivially [CLS]-only spans for scoring purposes
        if i == 0 and j == 0:
            final = h * 0.0
            span_text = ""
        else:
            # Squash the raw logit sum into (0, 1) via a stable sigmoid
            squashed = 1.0 / (1.0 + math.exp(-span_score / 10.0))
            final = h * squashed
            span_ids = input_ids[k, i: j + 1].tolist()
            span_text = tokenizer.decode(span_ids).strip()

        preds.append(ChunkPrediction(
            chunk_idx=chunks[k].chunk_idx,
            has_answer_score=h,
            best_span_score=span_score,
            final_score=final,
            start_tok=i,
            end_tok=j,
            span_text=span_text,
        ))

    # Filter by permissive keep-threshold, fall back to all if nothing survives
    kept = [p for p in preds if p.has_answer_score >= _HAS_ANS_KEEP_THRESHOLD and p.span_text]
    pool = kept if kept else [p for p in preds if p.span_text]

    if not pool:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "best_chunk": None,
            "chunks": preds if return_all_chunks else [],
        }

    best = max(pool, key=lambda p: p.final_score)

    if best.has_answer_score < _HAS_ANS_FINAL_THRESHOLD:
        answer = NOT_FOUND_MESSAGE
    else:
        answer = best.span_text

    return {
        "answer": answer,
        "best_chunk": best,
        "chunks": preds if return_all_chunks else [],
    }
