#!/bin/bash
#SBATCH --job-name=con_v2b
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=contrastive_v2b_%j.log

cd ~/food_pantry

echo "=== Contrastive v2B: Partial unfreeze (last 2 encoder layers) ==="
echo "=== bs=16, alpha=0.3, lr=1e-4, temp=0.1, unfreeze=2, auto-threshold ==="
echo ""

python train_contrastive_v2.py \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl-dir ./florence2_data \
  --output-dir ./checkpoints_contrastive_v2b \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4 \
  --alpha 0.3 \
  --temperature 0.1 \
  --auto-threshold \
  --unfreeze-layers 2 \
  --min-samples 20 \
  --bf16
