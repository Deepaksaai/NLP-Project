# NLP-Project

Legal document summarization and question answering system built entirely from scratch — no pretrained weights.

## Overview

Given a legal document (court orders, contracts, legislative bills), the system produces a concise abstractive summary preserving key legal facts, parties, and outcomes. It also supports question answering over legal documents.

## Model Downloads

Model checkpoints and large dataset files are hosted on Google Drive (too large for GitHub):

**[Download Model & Data](PASTE_YOUR_GOOGLE_DRIVE_LINK_HERE)**

After downloading, place files as follows:
```
checkpoints/
├── stage3_best.pt        ← final summarization model (required)
├── stage2_best.pt        ← Stage 2 checkpoint (optional)
└── stage1_best.pt        ← Stage 1 checkpoint (optional)

datasets/multilexsum/
├── sources_1.json        ← MultiLexSum v1 source documents
└── sources_2.json        ← MultiLexSum v2 source documents
```

## Summarizer Architecture

- **Model**: Encoder-decoder Transformer (32M parameters, built from scratch)
- **Training**: 4-stage curriculum learning
  - Stage 1: General summarization (CNN/DailyMail + XSum, ~491k pairs)
  - Stage 2: Long formal documents (arXiv + PubMed, ~2.3M aligned chunks)
  - Stage 3: Legal fine-tuning (MultiLexSum + BillSum, ~221k chunks)
- **Tokenizer**: 32k BPE trained from scratch on all 6 datasets
- **No pretrained weights used anywhere**

## Evaluation Results (BillSum Test Set, 500 examples)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| ROUGE-1 F1 | 0.3877 | > 0.30 | PASS |
| ROUGE-2 F1 | 0.2060 | > 0.10 | PASS |
| ROUGE-L F1 | 0.2813 | > 0.25 | PASS |
| Outcome Preservation | 1.0000 | > 0.70 | PASS |
| Outcome Inversion | 0.0000 | < 0.05 | PASS |
| Hallucination Rate | 0.0760 | < 0.15 | PASS |
| Compression Ratio | 0.068 | 0.05-0.15 | PASS |

## Project Structure

```
├── model/              Transformer architecture (from scratch)
├── data/               Data pipeline + preprocessing
├── training/           Training scripts (Stages 1-3)
├── inference/          Hierarchical beam search pipeline
├── evaluation/         6-metric evaluation framework
├── tokenizer/          BPE tokenizer training
├── QA/                 Question answering module
├── checkpoints/        Model checkpoints (download from Drive)
├── datasets/           Raw datasets
├── logs/               Training logs + metrics
└── validate_stage0.py  Foundation validation
```

## Quick Start

```bash
# Install dependencies
pip install torch tokenizers datasets rouge-score sentence-transformers spacy
python -m spacy download en_core_web_sm

# Download checkpoints from Google Drive link above
# Place stage3_best.pt in checkpoints/

# Run evaluation
python -m evaluation.generate_outputs --n 100 --skip_mlx
python -m evaluation.run_evaluation
```

## Datasets Used

| Dataset | Stage | Purpose |
|---------|-------|---------|
| CNN/DailyMail | 1 | General summarization |
| XSum | 1 | Aggressive compression |
| arXiv | 2 | Long formal documents |
| PubMed | 2 | Scientific language |
| MultiLexSum | 3 | Legal case documents |
| BillSum | 3 + eval | Legislative bills |

## Known Limitations

- Entity coverage is lower than pretrained models (generic CRS-style paraphrasing from training data)
- Summaries average ~79 words (shorter than ideal 100-300 range)
- NLI-based faithfulness metric underperforms due to long-document premise limitations
- Evaluated on BillSum test only; MultiLexSum evaluation excluded due to infrastructure constraints