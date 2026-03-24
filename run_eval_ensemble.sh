#!/bin/bash
#SBATCH --job-name=eval_ensemble
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=eval_ensemble_%j.log

cd ~/food_pantry

echo "=== Ensemble: v9 (best recall) + v11 (best precision) — UNION mode ==="

python evaluate_ensemble.py \
  --checkpoint1 ./checkpoints_v9/best_model \
  --checkpoint2 ./checkpoints_v11/best_model \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_ensemble_union.json \
  --mode union \
  --bf16 \
  --show-predictions 10
