#!/bin/bash
#SBATCH --job-name=train_v71
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=train_v7_1_%j.log

cd ~/food_pantry

python train_florence2_v7_1.py \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --output-dir ./checkpoints_v7_1 \
  --model microsoft/Florence-2-large-ft \
  --epochs 30 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lr 2e-5 \
  --lora-r 48 \
  --lora-alpha 96 \
  --patience 7 \
  --min-samples-per-class 25 \
  --focal-gamma 1.0 \
  --label-smoothing 0.05 \
  --max-length 512 \
  --bf16
