#!/bin/bash
#SBATCH --job-name=contr_clean
#SBATCH --output=logs/contrastive_clean_%j.log
#SBATCH --error=logs/contrastive_clean_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

echo "=========================================="
echo "Contrastive Learning on Cleaned Dataset (18 categories)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

cd ~/food_pantry
mkdir -p logs

# ── Step 1: Check if florence2_data_clean exists ──────────────────────────
if [ ! -d "florence2_data_clean" ]; then
    echo "[Step 0] Converting cleaned data to Florence-2 format first..."
    python3 convert_coco_to_florence2.py \
        --data-dir cleaned_data \
        --mapping usda_mapping_clean.json \
        --output-dir florence2_data_clean
    echo ""
fi

echo "Data check:"
wc -l florence2_data_clean/*.jsonl 2>/dev/null
echo ""

# ── Step 2: Train contrastive on clean data ───────────────────────────────
echo "============================================"
echo "Training Contrastive (frozen encoder + projection + classifier)"
echo "============================================"

python3 train_contrastive.py \
    --base-model microsoft/Florence-2-large-ft \
    --data-dir cleaned_data \
    --jsonl-dir florence2_data_clean \
    --output-dir checkpoints_contrastive_clean \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-3 \
    --alpha 0.5 \
    --temperature 0.07 \
    --proj-dim 128 \
    --threshold 0.5 \
    --min-samples 15 \
    --bf16

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
echo ""
echo "Model: checkpoints_contrastive_clean/best_model.pt"
echo "Next: run ensemble eval with v11_clean + contrastive_clean"
