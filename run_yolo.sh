#!/bin/bash
#SBATCH --job-name=yolo_detector_train
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --gres=gpu:1

cd ~/food_pantry

python3 train_yolo_detector.py \
    --data-dir train_val_data/ \
    --output-dir yolo_output/ \
    --epochs 50 \
    --batch-size 16 \
    --imgsz 640 \
    --model yolo11m.pt
