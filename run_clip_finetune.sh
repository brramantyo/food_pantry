#!/bin/bash
#SBATCH --job-name=clip_ft
#SBATCH --output=logs/clip_finetune_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "=========================================="
echo "CLIP Fine-Tune (Full Encoder + Focal Loss)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

mkdir -p logs

# Method 1: Full fine-tune (unfreeze all layers)
# Lower batch size to fit in memory with gradients
python train_clip_classifier.py \
    --crop-dir crop_data/ \
    --train-jsonl crop_data/train_crops.jsonl \
    --val-jsonl crop_data/val_crops.jsonl \
    --output-dir clip_output_finetune/ \
    --model openai/clip-vit-base-patch32 \
    --epochs 30 \
    --batch-size 16 \
    --lr 5e-5 \
    --unfreeze-layers -1 \
    --focal-loss \
    --focal-gamma 2.0 \
    --balanced-sampling

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
