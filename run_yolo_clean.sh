#!/bin/bash
#SBATCH --job-name=yolo_clean
#SBATCH --output=logs/yolo_clean_%j.log
#SBATCH --error=logs/yolo_clean_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00

echo "=========================================="
echo "YOLO Training on Cleaned Dataset (18 categories)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

cd ~/food_pantry
mkdir -p logs

pip install --user ultralytics 2>/dev/null

# ── Train YOLO on cleaned data ───────────────────────────────────────────
echo "============================================"
echo "Training YOLOv11m on cleaned data (18 categories)"
echo "============================================"

python3 train_yolo_detector.py \
    --data-dir cleaned_data \
    --output-dir yolo_output_clean \
    --epochs 50 \
    --batch-size 16 \
    --imgsz 640 \
    --model yolo11m.pt \
    --patience 10

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
echo ""
echo "Results in: runs/detect/yolo_output_clean/"
echo "Compare with original: runs/detect/yolo_output/ (75.2% mAP50, 21 cats)"
