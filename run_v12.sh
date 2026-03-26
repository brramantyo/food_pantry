#!/bin/bash
#SBATCH --job-name=train_v12
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=train_v12_%j.log

cd ~/food_pantry

echo "=== Training v12: Multi-label recall focus ==="
echo "=== Continue from v11 checkpoint ==="
echo "=== 15 epochs, LR=2e-5, weighted CE, 3x multi-item boost ==="

python train_florence2_v12.py \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --checkpoint ./checkpoints_v11/best_model \
  --output-dir ./checkpoints_v12 \
  --model microsoft/Florence-2-large-ft \
  --epochs 15 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lr 2e-5 \
  --patience 6 \
  --min-samples-per-class 60 \
  --multi-item-boost 3 \
  --boost-intensity 0.6 \
  --label-smoothing 0.03 \
  --max-length 512 \
  --bf16
