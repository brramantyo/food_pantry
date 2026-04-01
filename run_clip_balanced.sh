#!/bin/bash
#SBATCH --job-name=clip_balanced
#SBATCH --output=logs/clip_balanced_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

echo "=========================================="
echo "CLIP Classifier with Focal Loss + Balanced Sampling"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

mkdir -p logs

python train_clip_classifier.py \
    --crop-dir crop_data/ \
    --train-jsonl crop_data/train_crops.jsonl \
    --val-jsonl crop_data/val_crops.jsonl \
    --output-dir clip_output_balanced/ \
    --model openai/clip-vit-base-patch32 \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --freeze-encoder \
    --focal-loss \
    --focal-gamma 2.0 \
    --balanced-sampling

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
