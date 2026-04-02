#!/bin/bash
#SBATCH --job-name=retrain_clean
#SBATCH --output=logs/retrain_clean_%j.log
#SBATCH --error=logs/retrain_clean_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00

echo "=========================================="
echo "Re-train on Cleaned Dataset (18 categories)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

cd ~/food_pantry
mkdir -p logs

# ── Step 1: Convert cleaned COCO → Florence-2 JSONL ──────────────────────
echo "============================================"
echo "[Step 1/3] Converting cleaned data to Florence-2 format..."
echo "============================================"

python3 convert_coco_to_florence2.py \
    --data-dir cleaned_data \
    --mapping usda_mapping_clean.json \
    --output-dir florence2_data_clean

echo ""
echo "Checking output:"
wc -l florence2_data_clean/*.jsonl 2>/dev/null
echo ""

# ── Step 2: Train v11 on cleaned data ────────────────────────────────────
echo "============================================"
echo "[Step 2/3] Training Florence-2 v11 on cleaned data..."
echo "============================================"

# Train from scratch (no v9 checkpoint — categories changed)
python3 train_florence2_v11.py \
    --train-jsonl florence2_data_clean/train.jsonl \
    --valid-jsonl florence2_data_clean/valid.jsonl \
    --data-dir cleaned_data \
    --output-dir checkpoints_v11_clean \
    --checkpoint none \
    --epochs 15 \
    --batch-size 2 \
    --lr 1e-5 \
    --bf16 \
    --gradient-checkpointing \
    --patience 5

echo ""
echo "Training complete."
echo ""

# ── Step 3: Evaluate on clean test set ───────────────────────────────────
echo "============================================"
echo "[Step 3/3] Evaluating on clean test set..."
echo "============================================"

python3 evaluate_florence2.py \
    --checkpoint checkpoints_v11_clean/best_model \
    --base-model microsoft/Florence-2-large-ft \
    --jsonl florence2_data_clean/test.jsonl \
    --data-dir cleaned_data \
    --output eval_results_v11_clean.json \
    --bf16

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
echo ""
echo "Results: eval_results_v11_clean.json"
echo "Compare with original: eval_results_v11.json"
