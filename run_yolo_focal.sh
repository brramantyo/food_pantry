#!/bin/bash
#SBATCH --job-name=yolo_focal
#SBATCH --output=logs/yolo_focal_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00

echo "=========================================="
echo "Training YOLO with Focal Loss"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

mkdir -p logs

python train_yolo_focal.py \
    --data-dir . \
    --output-dir runs/detect \
    --epochs 50 \
    --batch 16 \
    --imgsz 640

echo ""
echo "=========================================="
echo "Training complete!"
echo "End: $(date)"
echo "=========================================="
