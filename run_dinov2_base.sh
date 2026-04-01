#!/bin/bash
#SBATCH --job-name=dino_b14
#SBATCH --output=logs/dinov2_base_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

echo "=========================================="
echo "DINOv2 ViT-B/14 Classifier (Baseline)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

mkdir -p logs

python train_dinov2_classifier.py \
    --crop-dir crop_data/ \
    --train-jsonl crop_data/train_crops.jsonl \
    --val-jsonl crop_data/val_crops.jsonl \
    --output-dir dinov2_output_base/ \
    --model dinov2_vitb14 \
    --epochs 30 \
    --batch-size 24 \
    --lr 5e-5 \
    --img-size 224 \
    --unfreeze-layers 4 \
    --focal-gamma 2.0 \
    --label-smoothing 0.1 \
    --grad-accum 1 \
    --balanced-sampling

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
