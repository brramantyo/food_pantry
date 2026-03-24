#!/bin/bash
#SBATCH --job-name=eval_v9_1
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=eval_v9_1_%j.log

cd ~/food_pantry

python evaluate_florence2.py \
  --checkpoint ./checkpoints_v9_1/best_model \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_v9_1.json \
  --bf16 --show-errors --show-predictions 10
