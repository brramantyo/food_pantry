#!/bin/bash
#SBATCH --job-name=od_contr
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --output=od_contrastive_%j.log

cd ~/food_pantry

echo "=== OD → Crop → Contrastive Classifier Pipeline ==="
echo "=== Using existing OD + existing contrastive model ==="

python evaluate_od_contrastive.py \
    --base-model microsoft/Florence-2-large-ft \
    --od-checkpoint ./checkpoints_od_v1/best_model \
    --contrastive-checkpoint ./checkpoints_contrastive/best_model.pt \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_od_contrastive.json \
    --threshold 0.5 \
    --crop-threshold 0.6 \
    --bf16
