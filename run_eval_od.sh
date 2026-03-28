#!/bin/bash
#SBATCH --job-name=eval_od
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eval_od_v1_%j.log

cd ~/food_pantry

echo "=== Evaluating Florence-2 OD (fine-tuned) ==="

python evaluate_od.py \
  --base-model microsoft/Florence-2-large-ft \
  --checkpoint ./checkpoints_od_v1/best_model \
  --data-dir . \
  --output ./eval_results_od_v1.json \
  --bf16
