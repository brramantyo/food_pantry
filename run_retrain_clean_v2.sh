#!/bin/bash
#SBATCH --job-name=v11c_v9
#SBATCH --output=logs/retrain_clean_v9_%j.log
#SBATCH --error=logs/retrain_clean_v9_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00

echo "=========================================="
echo "Re-train v11 Clean FROM v9 Checkpoint (18 categories)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

cd ~/food_pantry
mkdir -p logs

# ── Ensure v5 symlinks exist ─────────────────────────────────────────────
cd florence2_data_clean
ln -sf train.jsonl train_v5.jsonl 2>/dev/null
ln -sf valid.jsonl valid_v5.jsonl 2>/dev/null
ln -sf test.jsonl test_v5.jsonl 2>/dev/null
cd ~/food_pantry

# ── Verify v9 checkpoint exists ──────────────────────────────────────────
if [ ! -d "checkpoints_v9/best_model" ]; then
    echo "ERROR: checkpoints_v9/best_model not found!"
    exit 1
fi
echo "✓ v9 checkpoint found"
echo ""

# ── Train v11 from v9 checkpoint on clean data ───────────────────────────
echo "============================================"
echo "Training: v9 checkpoint → clean 18-cat data"
echo "============================================"

python3 train_florence2_v11.py \
    --train-jsonl florence2_data_clean/train.jsonl \
    --valid-jsonl florence2_data_clean/valid.jsonl \
    --data-dir cleaned_data \
    --output-dir checkpoints_v11_clean_v2 \
    --checkpoint checkpoints_v9/best_model \
    --epochs 15 \
    --batch-size 2 \
    --lr 5e-6 \
    --bf16 \
    --gradient-checkpointing \
    --patience 5

echo ""
echo "Training complete."
echo ""

# ── Evaluate ─────────────────────────────────────────────────────────────
echo "============================================"
echo "Evaluating on clean test set..."
echo "============================================"

python3 evaluate_florence2.py \
    --checkpoint checkpoints_v11_clean_v2/best_model \
    --base-model microsoft/Florence-2-large-ft \
    --jsonl florence2_data_clean/test.jsonl \
    --data-dir cleaned_data \
    --output eval_results_v11_clean_v2.json \
    --bf16

echo ""

# ── Fuzzy re-eval ────────────────────────────────────────────────────────
echo "============================================"
echo "Fuzzy category matching re-evaluation..."
echo "============================================"

python3 eval_fuzzy.py \
    --input eval_results_v11_clean_v2.json \
    --output eval_results_v11_clean_v2_fuzzy.json

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  Strict: eval_results_v11_clean_v2.json"
echo "  Fuzzy:  eval_results_v11_clean_v2_fuzzy.json"
echo ""
echo "Compare with:"
echo "  Original v11 (dirty, 21 cats): 76.5% F1"
echo "  v11 clean fresh LoRA: 53.9% F1"
echo "  This run (v9 pretrained → clean): ???"
