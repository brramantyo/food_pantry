#!/bin/bash
#SBATCH --job-name=train_v13
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --output=train_v13_%j.log

cd ~/food_pantry

echo "=== Training v13: Mixed full-image + crop classifier ==="
echo "=== Continue from v11, learn to handle both shelf photos and individual crops ==="
echo "=== This fixes precision drop in OD→crop→classify pipeline ==="
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_florence2_v13.py \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --crop-jsonl ./crop_data/train_crops.jsonl \
  --crop-val-jsonl ./crop_data/val_crops.jsonl \
  --crop-data-dir ./crop_data \
  --checkpoint ./checkpoints_v11/best_model \
  --output-dir ./checkpoints_v13 \
  --model microsoft/Florence-2-large-ft \
  --epochs 12 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lr 1e-5 \
  --patience 5 \
  --min-samples-per-class 40 \
  --label-smoothing 0.03 \
  --max-length 512 \
  --bf16
