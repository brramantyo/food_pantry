#!/bin/bash
#SBATCH --job-name=train_v12b
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=train_v12b_%j.log

cd ~/food_pantry

echo "=== Training v12b: Conservative multi-label from v11 checkpoint ==="
echo "=== 12 epochs, LR=1.5e-5, standard CE, 1.5x multi-item boost ==="
echo "=== 5 worst classes boosted at 30% intensity ==="

python train_florence2_v12b.py \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --checkpoint ./checkpoints_v11/best_model \
  --output-dir ./checkpoints_v12b \
  --model microsoft/Florence-2-large-ft \
  --epochs 12 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lr 1.5e-5 \
  --patience 5 \
  --min-samples-per-class 50 \
  --focal-gamma 0.0 \
  --label-smoothing 0.03 \
  --max-length 512 \
  --bf16
