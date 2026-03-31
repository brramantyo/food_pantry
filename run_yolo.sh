#!/bin/bash
#SBATCH --job-name=yolo_det
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --output=yolo_train_%j.log

cd ~/food_pantry

echo "=== Fine-tuning YOLOv11 for pantry item detection ==="
echo "=== 21 categories, COCO format annotations ==="

pip install --user ultralytics 2>/dev/null

python train_yolo_detector.py \
    --data-dir . \
    --output-dir ./yolo_output \
    --epochs 50 \
    --batch-size 16 \
    --imgsz 640 \
    --model yolo11m.pt
