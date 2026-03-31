#!/bin/bash
#SBATCH --job-name=od_contrastive_eval
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --gres=gpu:1

cd ~/food_pantry

python3 evaluate_od_contrastive.py \
    --base-model microsoft/florence-2-base \
    --od-checkpoint checkpoints_od_v1/best_model \
    --contrastive-checkpoint checkpoints_contrastive/best_model.pt \
    --data-dir test_data/ \
    --jsonl test_data/test.jsonl \
    --output eval_od_contrastive_results.json \
    --threshold 0.5 \
    --crop-threshold 0.6 \
    --bf16
