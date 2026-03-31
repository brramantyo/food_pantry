#!/bin/bash
#SBATCH --job-name=gen_crops
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --output=gen_crops_%j.log

cd ~/food_pantry

echo "=== Step 1: Generate crop dataset from GT bboxes + OD detections ==="
echo "=== GT crops: use COCO annotations directly ==="
echo "=== OD crops: run fine-tuned OD, match to GT by IoU, fallback to v11 classifier ==="
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python generate_crop_dataset.py \
  --data-dir . \
  --od-checkpoint ./checkpoints_od_v1/best_model \
  --cls-checkpoint ./checkpoints_v11/best_model \
  --base-model microsoft/Florence-2-large-ft \
  --output-dir ./crop_data \
  --padding 0.15 \
  --iou-threshold 0.3 \
  --bf16
