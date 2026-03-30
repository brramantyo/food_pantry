#!/bin/bash
#SBATCH --job-name=pipe_ens
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --output=pipeline_ensemble_%j.log

cd ~/food_pantry

echo "=== End-to-End Pipeline with Ensemble Union (v11 + Contrastive) ==="
echo "=== Task 1: Ensemble classify → Task 2: USDA match → Nutrition ==="
echo ""

python evaluate_pipeline_ensemble.py \
  --jsonl ./florence2_data/test_v5.jsonl \
  --data-dir . \
  --base-model microsoft/Florence-2-large-ft \
  --v11-checkpoint ./checkpoints_v11/best_model \
  --contrastive-checkpoint ./checkpoints_contrastive/best_model.pt \
  --usda-dir ./usda_data \
  --top-k 5 \
  --output ./eval_pipeline_ensemble.json \
  --threshold 0.5 \
  --bf16
