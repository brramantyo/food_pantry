#!/bin/bash
#SBATCH --job-name=eval_v7
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=eval_v7_%j.log

cd ~/food_pantry

python evaluate_florence2.py \
  --checkpoint ./checkpoints_v7/best_model \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_v7.json \
  --bf16 --show-errors --show-predictions 10
