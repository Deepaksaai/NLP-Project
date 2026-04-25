# Legal QA System

This module contains a complete end-to-end framework designed to fine-tune, deploy, and evaluate a Question Answering system specifically targeting massive Legal Document analysis. It is powered by `microsoft/deberta-v3-large` for highly accurate bounding box extraction and `facebook/bart-base` for Plain English inference translations.

---

## 📂 Architecture Overview

The system is separated into four primary packages:

### 1. `legal_qa/data/` (Data Preprocessing & Dataset Loading)
Handles the highly complex data pipelining required. 
* **`dataset_utils.py`**: Shared logic for the Tokenizer, alignment mapping (character offsets to tokens), and span verification.
* **`stage1_dataset.py`**: Data Sampler for SQuAD 2.0, TriviaQA, and NQ. Enforces a 65% Answerable / 35% Unanswerable constraint per batch.
* **`stage2_dataset.py`**: The `ChunkingDataset`. Converts huge documents into 400-word blocks with 50-word overlaps. Retrieves one positive chunk and four negative chunks per batch. 
* **`stage3_dataset.py`**: Legal Domain specifically (CUAD, COLIEE, LEDGAR). Automatically applies cross-document negative chunking to train against legal hallucination, and pre-pends `legal:` context flags.

### 2. `legal_qa/model/` (Network Architecture)
* **`qa_model.py`**: The primary `LegalQAModel`. Attaches three parallel heads (Start Span, End Span, and Is-Answerable) onto DeBERTa. Employs mathematically optimal joint span selection at inference time (`O(L^2)`) to ensure End indices naturally follow Start indices while applying length penalties.
* **`heads.py`**: Swappable Linear architecture isolating the classifiers.

### 3. `legal_qa/training/` (Optimizers, Loops, and Scripts)
* **`loss.py`**: Blends CrossEntropy gradients with BCE gradients.
* **`discriminative_optimizer.py`**: Layer-wise Learning Rate assignments. DeBERTa encoder layers 0-7 decay slowly to protect grammatical syntax, while the untrained Head wrappers adapt aggressively.
* **`trainer.py`**: The heavy-lifting training cycle. Handles Mixed Precision (`torch.cuda.amp`), Gradient Accumulation, and Early-stopping checkpoint evaluation checks.

### 4. `legal_qa/inference/` (Live Deployment Environment)
This is what actually executes when a client uploads a document today.
* **`retriever.py`**: Converts raw document strings into TF-IDF vector matrices. Expands user queries dynamically using legal synonyms (`cancel` -> `terminate`) and scores matching chunks.
* **`predictor.py`**: Runs DeBERTa over the Top K chunks retrieved. Evaluates combined scores, fixes token entity boundaries dynamically, and outputs calibrated confidence bandings.
* **`generator.py`**: Translates the raw legal prediction span outputted by `predictor` into plain English using `BART-Base`. Applies automatic strict faithfulness overrides if the generator drops dates, monetary values, or defined terms.
* **`pipeline.py`**: Interlinks `Retriever -> Predictor -> Generator`. Manages conversation history context automatically.

### 5. `legal_qa/evaluation/` (Testing Pipeline)
* **`metrics.py`**: Exact Match, F1, Chunk Retrieval Accuracy, False Positive classification curves, Clause Boundary verification, and Term Preservation logic.
* **`evaluator.py`**: Runs full metric evaluations across the CuAD, SQuad, LEDGAR, and COLIEE subset boundaries. Calibrates optimal answerability thresholds.
* **`error_analysis.py`**: Qualitatively buckets failures into distinct logical piles: Retrieval Loss, Border Misalignment, Hallucination, and Legal Deletions. Logs out exactly why the model is failing.

---

## 🚀 How to Run Training (The 3 Stages)

Due to catastrophic forgetting, the model cannot be zero-shot trained directly on complex legal text. You **must** trace this lineage.

1. **Stage 1 (General Knowledge & SQuAD Formatting)**
   ```bash
   python legal_qa/training/train_stage1.py
   ```
   **Expected Metrics**: Val F1 should very rapidly ascend to `~88.0%`. 

2. **Stage 2 (Long Document Chunking Strategy)**
   ```bash
   python legal_qa/training/train_stage2.py
   ```
   **Expected Metrics**: Val F1 will likely decay to `~84.0%`. The key target is checking the terminal logs for **Val Chunk Rank Accuracy** rising above `80%`.

3. **Stage 3 (Legal Fine Tuning)**
   ```bash
   python legal_qa/training/train_stage3.py
   ```
   **Expected Metrics**: The metric subsets for `CUAD` and `LEDGAR` must stabilize over `0.85 F1`. Carefully monitor the **Cross-Doc False Positive Rate**; if it exceeds `5%`, the model is hallucinating too frequently, and you must pause training.

**Flags**: 
* Use `--dry_run` to test code stability on only 10 batches. 
* Use `--resume` to bypass starting fresh if the GPU crashes.

---

## 📊 How to Run Operations & Evaluation

Once Stage 3 produces the `qa_stage3_best.pt` file, execute evaluation sweeps:

```bash
python legal_qa/evaluation/run_evaluation.py --quick
```

**Interpreting the Report**:
The terminal will generate a clean Summary Table breaking down EM and False Positive limits across the four test bounds. It will subsequently output an `Optimal Threshold Recommendation`.

Inside `legal_qa/evaluation/report/flagged_examples.json`, you will find your Top 10 worst errors grouped chronologically by category. Use these qualitative outputs to manually observe where the Retriever failed versus where the Extractor failed. `Retrieval Failure` means `retriever.py` needs tuning; `Has_Answer_False_Positive` means `loss.py` needs its answerability weighting increased.
