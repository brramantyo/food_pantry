#!/bin/bash
#SBATCH --job-name=contrastive
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=contrastive_%j.log

cd ~/food_pantry

echo "=== Supervised Contrastive Learning on Florence-2 Features ==="
echo "=== Frozen encoder + projection head + classifier head ==="
echo ""

python train_contrastive.py \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --output-dir ./checkpoints_contrastive \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-3 \
  --alpha 0.5 \
  --temperature 0.07 \
  --proj-dim 128 \
  --threshold 0.5 \
  --min-samples 20 \
  --bf16
