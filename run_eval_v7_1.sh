#!/bin/bash
#SBATCH --job-name=eval_v71
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=eval_v7_1_%j.log

cd ~/food_pantry

python evaluate_florence2.py \
  --checkpoint ./checkpoints_v7_1/best_model \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_v7_1.json \
  --bf16 --show-errors --show-predictions 10
