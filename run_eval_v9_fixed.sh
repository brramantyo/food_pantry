#!/bin/bash
#SBATCH --job-name=fix_eval_v9
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=eval_v9_fixed_%j.log

cd ~/food_pantry

# Step 1: Fix case in merged training data (for future re-training)
echo "=== Fixing case mismatch in training data ==="
python fix_case_v9.py ./grocery_data/merged_train.jsonl

# Step 2: Re-eval v9 with case-normalized eval script
echo ""
echo "=== Re-evaluating v9 with case normalization ==="
python evaluate_florence2.py \
  --checkpoint ./checkpoints_v9/best_model \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_v9_fixed.json \
  --bf16 --show-errors --show-predictions 10
