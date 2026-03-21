#!/bin/bash
#SBATCH --job-name=eval_zeroshot
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eval_zeroshot_%j.log

cd ~/food_pantry

python evaluate_zeroshot.py \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_zeroshot.json \
  --bf16 --show-predictions 10
