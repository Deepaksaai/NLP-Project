"""
Stage 4 — generate all model outputs to disk BEFORE any metric runs.

For every test example this script runs the full Stage-3 inference
pipeline (TF-IDF retrieval -> QA model -> legal entity fix -> plain
English rewrite) and serializes:

    evaluation/qa/generated_answers.json

containing the canonical per-example record consumed by every metric
script. Lean auxiliary files are also emitted so downstream code can
skip loading the huge main file:

    gold_answers.json
    source_documents.json
    has_answer_predictions.json
    retrieval_results.json
    plain_english_outputs.json

Run ONCE per checkpoint — metrics read from disk, never regenerate.

Usage:
    python -m QA.evaluation.run_inference \\
        --checkpoint checkpoints/qa_stage3_best.pt \\
        --limit 200
"""

import os
import sys
import json
import math
import argparse
import time
from typing import List, Dict

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from QA.qa_config import (
    QA_TOKENIZER_PATH, load_qa_special_tokens,
    MAX_QUESTION_LEN, MAX_CONTEXT_LEN, MAX_TOTAL_LEN, PAD_ID,
    LEGAL_MAX_ANSWER_LEN, LEGAL_LENGTH_PENALTY, LEGAL_TOKEN,
)
from QA.model.qa_model import build_qa_model
from QA.data.chunking import chunk_document, answer_in_chunk
from QA.inference.retriever import HybridChunkRetriever
from QA.inference.legal_entity import extend_span_to_entities
from QA.inference.plain_english import NoopRewriter
from QA.training.span_select import joint_span_select
from QA.evaluation.test_loaders import load_all


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(_ROOT, "evaluation", "qa")


def _load_model(ckpt_path: str, device):
    meta = load_qa_special_tokens()
    model = build_qa_model(meta, load_ckpt=None, freeze=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[inference] loaded {ckpt_path}  missing={len(missing)}  unexpected={len(unexpected)}")
    model.eval()
    return model, meta


def _build_input(tokenizer, meta, legal_id, question: str, chunk_text: str, use_legal: bool):
    q_ids = tokenizer.encode(question).ids[:MAX_QUESTION_LEN]
    c_ids = tokenizer.encode(chunk_text).ids[:MAX_CONTEXT_LEN]
    if use_legal:
        input_ids = ([meta["cls_id"]] + q_ids + [meta["sep_id"]]
                     + [legal_id] + c_ids + [meta["sep_id"]])
        seg = [0] * (1 + len(q_ids) + 1) + [1] * (1 + len(c_ids) + 1)
    else:
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


def _gold_chunk_hit(chunks, gold_answers: List[str]) -> int:
    """Return the index of the first chunk containing any gold answer, else -1."""
    if not gold_answers:
        return -1
    for c in chunks:
        low = c.text.lower()
        for g in gold_answers:
            if g and g.lower() in low:
                return c.chunk_idx
    return -1


@torch.no_grad()
def process_example(model, tokenizer, meta, legal_id, rewriter, example, device,
                    top_k: int = 3, max_answer_len: int = LEGAL_MAX_ANSWER_LEN,
                    length_penalty: float = LEGAL_LENGTH_PENALTY) -> Dict:
    use_legal = example["source"] in ("cuad", "ledgar", "coliee")
    retriever = HybridChunkRetriever(example["document"])
    if not retriever.chunks:
        return {
            **example,
            "predicted_answer":    "",
            "extracted_span":      "",
            "has_answer_prob":     0.0,
            "retrieved_chunks":    [],
            "winning_chunk":       -1,
            "gold_chunk_idx":      -1,
            "retrieval_top_k_idxs":[],
        }

    all_top = retriever.top_k(
        example["question"], k=min(top_k, len(retriever.chunks)),
    )
    gold_chunk_idx = _gold_chunk_hit(retriever.chunks, example["gold_answers"])

    # Batch the top-k chunks through the model in one forward pass
    inputs, segs, attns = [], [], []
    for r in all_top:
        ii, ss, aa = _build_input(tokenizer, meta, legal_id,
                                  example["question"], r.chunk.text, use_legal)
        inputs.append(ii); segs.append(ss); attns.append(aa)

    input_ids   = torch.tensor(inputs, dtype=torch.long, device=device)
    segment_ids = torch.tensor(segs,   dtype=torch.long, device=device)
    attn_mask   = torch.tensor(attns,  dtype=torch.long, device=device)

    start_logits, end_logits, has_logits = model(input_ids, segment_ids, attn_mask)
    has_probs = torch.sigmoid(has_logits).cpu().tolist()
    s_idx, e_idx = joint_span_select(
        start_logits, end_logits,
        max_answer_len=max_answer_len,
        length_penalty=length_penalty,
    )
    s_idx = s_idx.cpu().tolist()
    e_idx = e_idx.cpu().tolist()

    chunk_records = []
    best = None
    for k, r in enumerate(all_top):
        i, j = int(s_idx[k]), int(e_idx[k])
        if i == 0 and j == 0:
            span_text = ""
            span_score = float(start_logits[k, 0].item() + end_logits[k, 0].item())
            final = 0.0
        else:
            span_score = float(start_logits[k, i].item() + end_logits[k, j].item())
            squashed = 1.0 / (1.0 + math.exp(-span_score / 10.0))
            final = has_probs[k] * squashed
            span_ids = input_ids[k, i: j + 1].tolist()
            span_text = tokenizer.decode(span_ids).strip()

        rec = {
            "chunk_idx":         r.chunk.chunk_idx,
            "tfidf_score":       r.score,
            "has_answer_prob":   has_probs[k],
            "best_span_score":   span_score,
            "final_score":       final,
            "start_tok":         i,
            "end_tok":           j,
            "span_text":         span_text,
        }
        chunk_records.append(rec)
        if best is None or final > best["final_score"]:
            best = rec

    # Decide the extracted span
    if best and best["span_text"]:
        chunk = next(r.chunk for r in all_top if r.chunk.chunk_idx == best["chunk_idx"])
        idx_in_chunk = chunk.text.find(best["span_text"])
        if idx_in_chunk >= 0:
            new_s, new_e = extend_span_to_entities(
                chunk.text, idx_in_chunk, idx_in_chunk + len(best["span_text"]),
            )
            extracted = chunk.text[new_s: new_e]
        else:
            extracted = best["span_text"]
        top_has_prob = best["has_answer_prob"]
    else:
        extracted = ""
        top_has_prob = max(has_probs) if has_probs else 0.0

    predicted = rewriter.rewrite(example["question"], extracted) if extracted else ""

    return {
        **example,
        "predicted_answer":     predicted,
        "extracted_span":       extracted,
        "has_answer_prob":      float(top_has_prob),
        "retrieved_chunks":     chunk_records,
        "winning_chunk":        best["chunk_idx"] if best else -1,
        "gold_chunk_idx":       gold_chunk_idx,
        "retrieval_top_k_idxs": [r.chunk.chunk_idx for r in all_top],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default=os.path.join(_ROOT, "checkpoints", "qa_stage3_best.pt"))
    p.add_argument("--limit", type=int, default=None,
                   help="Per-source cap for smoke tests.")
    p.add_argument("--top_k", type=int, default=3)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] device={device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    tokenizer = Tokenizer.from_file(QA_TOKENIZER_PATH)
    legal_id = tokenizer.token_to_id(LEGAL_TOKEN)
    model, meta = _load_model(args.checkpoint, device)
    rewriter = NoopRewriter()

    data_by_source = load_all(limit_per_source=args.limit)

    all_records = []
    t0 = time.time()
    for src, items in data_by_source.items():
        for i, ex in enumerate(items):
            rec = process_example(model, tokenizer, meta, legal_id, rewriter,
                                  ex, device, top_k=args.top_k)
            all_records.append(rec)
            if (i + 1) % 25 == 0:
                dt = time.time() - t0
                print(f"  {src} {i+1}/{len(items)}  ({dt:.1f}s elapsed)")

    # Canonical record
    with open(os.path.join(OUT_DIR, "generated_answers.json"), "w", encoding="utf-8") as f:
        json.dump(all_records, f)

    # Auxiliary files (spec requirement)
    with open(os.path.join(OUT_DIR, "gold_answers.json"), "w") as f:
        json.dump([{"example_id": r["example_id"], "gold_answers": r["gold_answers"],
                    "is_answerable": r["is_answerable"]} for r in all_records], f)
    with open(os.path.join(OUT_DIR, "source_documents.json"), "w") as f:
        json.dump([{"example_id": r["example_id"], "source": r["source"],
                    "question": r["question"], "document": r["document"]}
                   for r in all_records], f)
    with open(os.path.join(OUT_DIR, "has_answer_predictions.json"), "w") as f:
        json.dump([{"example_id": r["example_id"], "has_answer_prob": r["has_answer_prob"],
                    "is_answerable": r["is_answerable"], "source": r["source"]}
                   for r in all_records], f)
    with open(os.path.join(OUT_DIR, "retrieval_results.json"), "w") as f:
        json.dump([{"example_id": r["example_id"],
                    "retrieved_chunk_idxs": r["retrieval_top_k_idxs"],
                    "gold_chunk_idx":       r["gold_chunk_idx"],
                    "retrieved_chunks":     r["retrieved_chunks"]}
                   for r in all_records], f)
    with open(os.path.join(OUT_DIR, "plain_english_outputs.json"), "w") as f:
        json.dump([{"example_id": r["example_id"],
                    "extracted_span": r["extracted_span"],
                    "predicted_answer": r["predicted_answer"]}
                   for r in all_records], f)

    print(f"[inference] wrote {len(all_records)} records to {OUT_DIR}")


if __name__ == "__main__":
    main()
