#!/bin/bash
#SBATCH --job-name=eval_v12b
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eval_v12b_%j.log

cd ~/food_pantry

echo "=== Evaluating v12b ==="

python evaluate_florence2.py \
  --base-model microsoft/Florence-2-large-ft \
  --checkpoint ./checkpoints_v12b/best_model \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_v12b.json \
  --bf16
