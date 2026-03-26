#!/bin/bash
#SBATCH --job-name=eval_v12
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eval_v12_%j.log

cd ~/food_pantry

echo "=== Evaluating v12 ==="

python evaluate_florence2.py \
  --data-dir . \
  --test-jsonl ./florence2_data/test_v5.jsonl \
  --checkpoint ./checkpoints_v12/best_model \
  --model microsoft/Florence-2-large-ft \
  --output ./eval_results_v12.json \
  --bf16
