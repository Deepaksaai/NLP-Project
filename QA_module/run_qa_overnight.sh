#!/bin/bash
# ================================================================
# QA Module — Complete overnight training pipeline
#
# This script runs ALL preprocessing + training + evaluation
# for the QA module from scratch, with all bug fixes applied.
#
# Usage:
#   cd ~/nlp_project/temp
#   chmod +x run_qa_overnight.sh
#   nohup ./run_qa_overnight.sh > qa_overnight.log 2>&1 &
#   tail -f qa_overnight.log
#
# Expected runtime: ~8-12 hours total
# ================================================================

set -e  # Exit on any error

echo "================================================================"
echo "QA Overnight Pipeline — $(date)"
echo "================================================================"

# Navigate to project root (parent of temp/)
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# Verify critical files exist
echo ""
echo "--- Verifying files ---"
for f in checkpoints/stage3_best.pt checkpoints/tokenizer.json tokenizer/qa_tokenizer.json tokenizer/qa_special_tokens.json; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
        if [[ "$f" == *"qa_tokenizer"* ]] || [[ "$f" == *"qa_special"* ]]; then
            echo "  -> Running add_qa_tokens.py to generate..."
            python -m QA.add_qa_tokens
        else
            echo "  FATAL: Cannot continue without $f"
            exit 1
        fi
    fi
done

# Check GPU
echo ""
echo "--- GPU Check ---"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# ================================================================
# PHASE 1: Data Preprocessing (~30 min)
# ================================================================
echo ""
echo "================================================================"
echo "PHASE 1: Data Preprocessing — $(date)"
echo "================================================================"

# Clean old preprocessed data (limits changed) and old checkpoints
rm -f data/qa/stage1/*.json data/qa/stage2/*.json data/qa/stage3/*.json
rm -f checkpoints/qa_stage1_*.pt checkpoints/qa_stage2_*.pt checkpoints/qa_stage3_*.pt
mkdir -p data/qa/stage1 data/qa/stage2 data/qa/stage3

echo ""
echo "--- Stage 1 data: SQuAD 2.0 ---"
python -m QA.data.preprocess_squad

echo ""
echo "--- Stage 1 data: TriviaQA ---"
python -m QA.data.preprocess_trivia

echo ""
echo "--- Stage 1 data: Natural Questions ---"
python -m QA.data.preprocess_nq

echo ""
echo "--- Stage 2 data: QASPER ---"
python -m QA.data.preprocess_qasper

echo ""
echo "--- Stage 2 data: QuALITY ---"
python -m QA.data.preprocess_quality || echo "  (QuALITY skipped — dataset may not be available)"

echo ""
echo "--- Stage 3 data: CuAD (train + validation) ---"
python -m QA.data.preprocess_cuad

echo ""
echo "--- Stage 3 data: LEDGAR ---"
python -m QA.data.preprocess_ledgar

echo ""
echo "--- Stage 3 data: COLIEE ---"
python -m QA.data.preprocess_coliee || echo "  (COLIEE skipped — optional dataset)"

echo ""
echo "--- Verifying preprocessed data ---"
echo "Stage 1 files:"
ls -la data/qa/stage1/ 2>/dev/null || echo "  (none)"
echo "Stage 2 files:"
ls -la data/qa/stage2/ 2>/dev/null || echo "  (none)"
echo "Stage 3 files:"
ls -la data/qa/stage3/ 2>/dev/null || echo "  (none)"

# ================================================================
# PHASE 2: Stage 1 Training — General QA (~3-4 hours)
# ================================================================
echo ""
echo "================================================================"
echo "PHASE 2: Stage 1 Training — General QA — $(date)"
echo "================================================================"

# Sanity check first
echo "--- Running sanity check ---"
python -m QA.training.train_qa_stage1 --sanity

echo ""
echo "--- Full Stage 1 training ---"
python -m QA.training.train_qa_stage1 \
    --epochs 12 \
    --batch_size 32 \
    --warmup_steps 5000 \
    --patience 5 \
    --grad_clip 1.0 \
    --log_every 100

echo ""
echo "--- Stage 1 complete ---"
ls -la checkpoints/qa_stage1_best.pt

# ================================================================
# PHASE 3: Stage 2 Training — Scientific QA (~2-3 hours)
# ================================================================
echo ""
echo "================================================================"
echo "PHASE 3: Stage 2 Training — Scientific QA — $(date)"
echo "================================================================"

python -m QA.training.train_qa_stage2 \
    --epochs 8 \
    --batch_size 16 \
    --warmup_steps 2000 \
    --patience 4 \
    --grad_clip 1.0 \
    --log_every 100

echo ""
echo "--- Stage 2 complete ---"
ls -la checkpoints/qa_stage2_best.pt

# ================================================================
# PHASE 4: Stage 3 Training — Legal QA (~2-3 hours)
# ================================================================
echo ""
echo "================================================================"
echo "PHASE 4: Stage 3 Training — Legal QA — $(date)"
echo "================================================================"

python -m QA.training.train_qa_stage3 \
    --epochs 10 \
    --batch_size 16 \
    --warmup_steps 500 \
    --patience 4 \
    --dropout 0.3 \
    --grad_clip 1.0 \
    --log_every 100

echo ""
echo "--- Stage 3 complete ---"
ls -la checkpoints/qa_stage3_best.pt

# ================================================================
# PHASE 5: Evaluation (~30 min)
# ================================================================
echo ""
echo "================================================================"
echo "PHASE 5: Evaluation — $(date)"
echo "================================================================"

python -m QA.evaluation.run_inference --limit 200

echo ""
python -m QA.evaluation.run_all

# ================================================================
# Done
# ================================================================
echo ""
echo "================================================================"
echo "QA OVERNIGHT PIPELINE COMPLETE — $(date)"
echo "================================================================"
echo ""
echo "Results:"
echo "  Stage 1 metrics: QA/logs/stage1_metrics.json"
echo "  Stage 2 metrics: QA/logs/stage2_metrics.json"
echo "  Stage 3 metrics: QA/logs/stage3_metrics.json"
echo "  Evaluation:      QA/evaluation/report/"
echo "  Checkpoints:     checkpoints/qa_stage3_best.pt"
echo ""
echo "Download metrics:"
echo "  scp user10@172.16.121.34:~/nlp_project/QA/logs/*.json ."
