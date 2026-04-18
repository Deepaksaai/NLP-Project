# Legal QA Model Card

> Fill in the blanks from `evaluation/qa/report/summary_report.json`
> after Stage 4 finishes. Numbers in angle brackets are placeholders.

## Overview
- **Model**: custom 6-layer Transformer encoder + three QA heads
  (start, end, has_answer) + 2-row segment embedding.
- **Base weights**: summarizer `stage3_best.pt` encoder (no external
  pretrained weights).
- **Tokenizer**: 32k BPE from the summarizer pipeline, extended with
  `[CLS]` and `[SEP]` (see `tokenizer/qa_special_tokens.json`).
- **Intended use**: extractive question answering over legal
  contracts, statutes, and case law with a faithfulness-first
  disposition (correct abstention is valued over speculative answers).

## Training stages

| Stage | Datasets | Key idea |
|-------|----------|----------|
| 1 — general QA | SQuAD 2.0 + TriviaQA (NQ optional) | Span extraction mechanics + unanswerable handling; encoder frozen → gradually unfrozen over 12 epochs. |
| 2 — long-document QA | QuALITY + QASPER, SQuAD anchor | Learns to read long formal documents; chunked training + chunk retrieval metric; discriminative LR. |
| 3 — legal fine-tuning | CuAD + LEDGAR + COLIEE, QASPER and SQuAD anchors | Adds `<legal>` signal token, cross-document and wrong-clause negatives, length-penalized long-span decoding, stricter faithfulness bar. |

## Evaluation summary (fill from `summary_report.json`)

| Metric | Value |
|---|---|
| EM (overall)                       | `<em_overall>` |
| F1 (overall / CuAD / COLIEE / LEDGAR) | `<f1_overall>` / `<f1_cuad>` / `<f1_coliee>` / `<f1_ledgar>` |
| Has-answer accuracy                 | `<has_answer_accuracy>` |
| Legal false-positive rate           | `<false_positive_rate_legal>` |
| Retrieval recall @1 / @3            | `<retrieval_recall_at_1>` / `<retrieval_recall_at_3>` |
| Clause boundary accuracy (full)     | `<clause_boundary_accuracy_full>` |
| Legal term preservation             | `<legal_term_preservation>` |
| Monetary term preservation          | `<monetary_term_preservation>` |
| Time period preservation            | `<time_period_preservation>` |
| Generation faithfulness             | `<generation_faithfulness>` |
| Simplification delta                | `<simplification_delta>` |
| Information preservation            | `<information_preservation>` |
| Coreference resolution rate         | `<coreference_resolution_rate>` |
| Context drift rate                  | `<context_drift_rate>` |
| Human — correctness / faithfulness / clarity / completeness | `<human_eval_*>` |

## Deployment configuration

- **Has-answer decision threshold**: `<recommended_has_answer_threshold>`
  (chosen by sweep to keep legal FPR < 0.08 at TPR > 0.75).
- **Max answer length (legal)**: 150 tokens.
- **Length penalty**: 0.05.
- **Retrieval**: TF-IDF over 400/50 word chunks, top-3 passed to QA.

## Known limitations

- English only. The tokenizer covers bytes, but the training corpora
  are all English.
- Span extraction only. The `PlainEnglishRewriter` interface exists
  but no dedicated rewriter has been trained; the default `NoopRewriter`
  returns the extracted span unchanged.
- Retrieval uses term overlap — semantic rewording of questions will
  sometimes miss the relevant chunk. If `answer_reachability_top3 <
  0.85`, rerun with BM25 or add a semantic index.
- `<legal>` token is not masked in the span logits; it is passively
  learned to be low-scoring rather than hard-constrained.
- Evaluation metrics for legal-term preservation and clause boundary
  accuracy are heuristic (regex + simple rules). For production use
  they should be benchmarked against a human-labeled audit set.

## Use cases where this model should NOT be trusted

- Any situation where a wrong answer has direct legal or financial
  consequences without human review.
- Questions requiring multi-hop reasoning across multiple contracts.
- Numerical reasoning beyond direct quotation ("is $50k more than
  half of $120k?" — out of scope for extractive QA).
- Non-English documents.
- Clauses where the correct answer is an explicit negation ("the
  contract does not impose indemnification") — the model may still
  surface the clause as if it were an affirmative answer.
