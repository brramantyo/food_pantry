#!/bin/bash
#SBATCH --job-name=train_v9_1
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=train_v9_1_%j.log

cd ~/food_pantry

echo "=== Step 1: Fix case mismatch in training data ==="
python fix_case_v9.py ./grocery_data/merged_train.jsonl

echo ""
echo "=== Step 2: Training v9.1: Case-fixed Grocery + Food Pantry ==="
python train_florence2_v9.py \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --train-jsonl ./grocery_data/merged_train.jsonl \
  --output-dir ./checkpoints_v9_1 \
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
