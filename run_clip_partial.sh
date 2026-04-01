#!/bin/bash
#SBATCH --job-name=clip_partial
#SBATCH --output=logs/clip_partial_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

echo "=========================================="
echo "CLIP Partial Fine-Tune (Last 4 Layers + Focal Loss)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

mkdir -p logs

# Method 2: Partial fine-tune (unfreeze last 4 encoder layers)
# Good balance between overfitting risk and adaptation
python train_clip_classifier.py \
    --crop-dir crop_data/ \
    --train-jsonl crop_data/train_crops.jsonl \
    --val-jsonl crop_data/val_crops.jsonl \
    --output-dir clip_output_partial/ \
    --model openai/clip-vit-base-patch32 \
    --epochs 25 \
    --batch-size 24 \
    --lr 1e-4 \
    --unfreeze-layers 4 \
    --focal-loss \
    --focal-gamma 2.0 \
    --balanced-sampling

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
