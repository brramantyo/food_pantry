#!/bin/bash
#SBATCH --job-name=train_v6
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=train_v6_%j.log

cd ~/food_pantry

python train_florence2_v6.py \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --output-dir ./checkpoints_v6 \
  --model microsoft/Florence-2-large-ft \
  --epochs 25 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --lr 3e-5 \
  --lora-r 32 \
  --lora-alpha 64 \
  --patience 5 \
  --max-length 512 \
  --bf16
