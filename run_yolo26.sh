#!/bin/bash
#SBATCH --job-name=yolo26
#SBATCH --output=logs/yolo26_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00

echo "=========================================="
echo "YOLOv26 Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

mkdir -p logs

python train_yolo_detector.py \
    --data-dir . \
    --output-dir runs/detect_yolo26/ \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640 \
    --model yolo26m.pt \
    --patience 15 \
    --mixup 0.15 \
    --cutmix 0.15

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
