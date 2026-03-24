#!/bin/bash
#SBATCH --job-name=eval_v9_tta
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eval_v9_tta_%j.log

cd ~/food_pantry

echo "=== Evaluating v9 with Test-Time Augmentation ==="

python evaluate_tta.py \
  --checkpoint ./checkpoints_v9/best_model \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_v9_tta.json \
  --bf16 \
  --threshold 2 \
  --show-predictions 10
