#!/bin/bash
#SBATCH --job-name=dino_l14
#SBATCH --output=logs/dinov2_large_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00

echo "=========================================="
echo "DINOv2 ViT-L/14 Classifier"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

mkdir -p logs

python train_dinov2_classifier.py \
    --crop-dir crop_data/ \
    --train-jsonl crop_data/train_crops.jsonl \
    --val-jsonl crop_data/val_crops.jsonl \
    --output-dir dinov2_output_large/ \
    --model dinov2_vitl14 \
    --epochs 30 \
    --batch-size 16 \
    --lr 5e-5 \
    --img-size 224 \
    --unfreeze-layers 6 \
    --focal-gamma 2.0 \
    --label-smoothing 0.1 \
    --grad-accum 2 \
    --balanced-sampling

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
