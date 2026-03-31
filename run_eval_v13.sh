#!/bin/bash
#SBATCH --job-name=eval_v13
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=40G
#SBATCH --output=eval_v13_%j.log

cd ~/food_pantry

echo "=== Evaluating v13 (direct classification on full images) ==="
echo "=== This checks that v13 didn't regress on full-image classification ==="
echo ""

python evaluate_florence2.py \
  --checkpoint ./checkpoints_v13/best_model \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_v13.json \
  --bf16 --show-errors --show-predictions 10
