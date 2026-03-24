#!/bin/bash
#SBATCH --job-name=train_v11
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=train_v11_%j.log

cd ~/food_pantry

echo "=== Training v11: Continue from v9 checkpoint with surgical fixes ==="
echo "=== 10 epochs, LR=1e-5, standard CE, min 50 samples/class ==="

python train_florence2_v11.py \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --checkpoint ./checkpoints_v9/best_model \
  --output-dir ./checkpoints_v11 \
  --model microsoft/Florence-2-large-ft \
  --epochs 10 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lr 1e-5 \
  --patience 5 \
  --min-samples-per-class 50 \
  --focal-gamma 0.0 \
  --label-smoothing 0.03 \
  --max-length 512 \
  --bf16
