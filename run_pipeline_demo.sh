#!/bin/bash
#SBATCH --job-name=pipeline_demo
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=pipeline_demo_%j.log

cd ~/food_pantry

echo "=== End-to-End Pipeline Demo ==="
echo "=== Image → Task 1 (classify) → Task 2 (USDA match) → Nutrition ==="

python pipeline_end_to_end.py \
  --jsonl ./florence2_data/test_v5.jsonl \
  --data-dir . \
  --base-model microsoft/Florence-2-large-ft \
  --checkpoint ./checkpoints_v11/best_model \
  --usda-dir ./usda_data \
  --top-k 3 \
  --max-samples 10 \
  --output ./pipeline_results_demo.json \
  --bf16
