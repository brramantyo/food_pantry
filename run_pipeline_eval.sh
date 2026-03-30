#!/bin/bash
#SBATCH --job-name=pipeline_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --output=pipeline_eval_%j.log

cd ~/food_pantry

echo "=== Full End-to-End Pipeline Evaluation ==="
echo "=== Task 1 (v11 classify) → Task 2 (USDA match) → Nutrition ==="
echo "=== Full test set (166 samples) ==="
echo ""

python evaluate_pipeline_full.py \
  --jsonl ./florence2_data/test_v5.jsonl \
  --data-dir . \
  --base-model microsoft/Florence-2-large-ft \
  --checkpoint ./checkpoints_v11/best_model \
  --usda-dir ./usda_data \
  --top-k 5 \
  --output ./eval_pipeline_full.json \
  --bf16
