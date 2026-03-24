#!/bin/bash
#SBATCH --job-name=train_v10
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=train_v10_%j.log

cd ~/food_pantry

# Check which grocery data file exists
GROCERY_JSONL="./grocery_data/grocery_train.jsonl"
if [ ! -f "$GROCERY_JSONL" ]; then
  echo "grocery_train.jsonl not found, using merged_train.jsonl for stage 1"
  GROCERY_JSONL="./grocery_data/merged_train.jsonl"
fi

echo "=== Training v10: Two-Stage (Grocery pre-train → Food Pantry fine-tune) ==="
echo "=== Stage 1: 8 epochs on Grocery data (LR=5e-5) ==="
echo "=== Stage 2: 25 epochs on Food Pantry data (LR=2e-5) ==="
echo "=== Grocery file: $GROCERY_JSONL ==="

python train_florence2_v10.py \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --grocery-jsonl "$GROCERY_JSONL" \
  --output-dir ./checkpoints_v10 \
  --model microsoft/Florence-2-large-ft \
  --epochs 25 \
  --stage1-epochs 8 \
  --stage1-lr 5e-5 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lr 2e-5 \
  --lora-r 48 \
  --lora-alpha 96 \
  --patience 7 \
  --min-samples-per-class 40 \
  --focal-gamma 1.0 \
  --label-smoothing 0.05 \
  --max-length 512 \
  --bf16
