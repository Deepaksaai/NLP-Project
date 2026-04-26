# NLP-Project

Legal Document Assistant — abstractive summarization and extractive question answering over legal documents.

## Overview

Given a legal PDF (contracts, court orders, legislative bills), the system:
1. Extracts and chunks the document
2. Produces an abstractive summary using a custom-trained Transformer
3. Answers natural-language questions using a fine-tuned DeBERTa-v3 model
4. Provides a TextRank extractive baseline for comparison

A Streamlit UI ties everything together.

## Models

### Summarizer
- Custom encoder-decoder Transformer (~12M parameters, built from scratch)
- 32k BPE tokenizer trained from scratch
- 3-stage curriculum: Stage 1 (CNN/DM + XSum) → Stage 2 (arXiv + PubMed) → Stage 3 (legal: MultiLexSum + BillSum)
- Hierarchical beam search (beam=4, length penalty=0.6, no-repeat 3-gram)
- Coverage loss to reduce repetition (λ=0.5 from epoch 3)

### QA Model
- Backbone: `deepset/deberta-v3-base-squad2` (DeBERTa-v3-base fine-tuned on SQuAD 2.0)
- Custom span extraction heads + has-answer head
- Fine-tuned on CUAD (510 contracts, 13k QA pairs, 41 clause categories)
- Sliding window inference: 384 context tokens, stride 128
- Hybrid retrieval: TF-IDF (1,2)-gram + all-MiniLM-L6-v2 dense embeddings fused via RRF

## Evaluation Results

### Summarizer (BillSum test set, 500 examples)

| Metric | Score |
|--------|-------|
| ROUGE-1 F1 | 0.388 |
| ROUGE-2 F1 | 0.206 |
| ROUGE-L F1 | 0.281 |
| Hallucination Rate | 7.6% |
| Outcome Preservation | 100% |
| Mean Summary Length | ~79 words |

### QA Model (CUAD, 5-epoch fine-tune)

| Epoch | Train Loss | Val F1 | Val EM |
|-------|-----------|--------|--------|
| 1 | 1.3197 | 0.8661 | 0.8454 |
| 3 | 0.4381 | 0.8962 | 0.8752 |
| 5 | 0.2860 | 0.8936 | 0.8752 |

Best checkpoint: epoch 4 — Val F1 = 0.9024, Val EM = 0.8865

## Model Downloads

Checkpoints are too large for GitHub and must be downloaded separately:

**[Download Checkpoints/Model](https://docs.google.com/document/d/1pOZkPvzvGQ-XEg3hxDDEC94hc6x47Yx1UzGilg3scDg/edit?usp=sharing)**

After downloading, place files as:
```
CHECKPOINTS/
├── summarizer_checkpoints/
│   ├── stage3_best.pt       ← required (deployed summarizer)
│   └── tokenizer.json       ← required (32k BPE tokenizer)
└── qa_checkpoints/
    └── deberta_best.pt      ← required (deployed QA model)
```

## Project Structure

```
├── pipeline.py                  Main pipeline (summarizer + QA + retrieval)
├── streamlit_app.py             Streamlit web UI
├── text_rank_summarizer.py      TextRank extractive baseline
│
├── baseline/
│   ├── preprocessing.py         PDF extraction, cleaning, chunking, TF-IDF
│   └── inference/
│       └── summarizer_inference.py  Hierarchical beam-search inference
│
├── model/                       Transformer architecture (from scratch)
│   ├── transformer.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention.py
│   └── positional.py
│
├── QA_deberta/                  DeBERTa QA model
│   ├── model.py                 DebertaQAModel + predict_span
│   ├── heads.py                 SpanHead + HasAnswerHead
│   ├── loss.py                  QA loss
│   └── train.py                 Fine-tuning script
│
├── training/                    Summarizer training scripts (Stages 1-3)
├── data/                        Dataset preprocessing
├── evaluation/                  Evaluation scripts + results
│   └── report/                  ROUGE, faithfulness, entity coverage JSONs
└── CHECKPOINTS/                 Model weights + logs (download from Drive)
```

## Quick Start

```bash
pip install torch transformers tokenizers sentence-transformers \
            streamlit scikit-learn spacy pdfplumber rouge-score \
            sentencepiece protobuf
python -m spacy download en_core_web_sm

# Download checkpoints and place them as shown above

streamlit run streamlit_app.py
```

Then upload a legal PDF in the sidebar and click **Process Document**.

## Datasets Used

| Dataset | Stage | Purpose |
|---------|-------|---------|
| CNN/DailyMail | 1 | General summarization |
| XSum | 1 | Aggressive compression |
| arXiv | 2 | Long formal documents |
| PubMed | 2 | Scientific language |
| MultiLexSum | 3 | Legal case documents |
| BillSum | 3 + eval | Legislative bills |
| CUAD | QA fine-tune + eval | Contract clause QA |
