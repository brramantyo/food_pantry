#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --output=ensemble_v11_con_%j.log

cd ~/food_pantry

echo "=== Ensemble: v11 (generative) + Contrastive (discriminative) ==="
echo "=== Strategies: union, intersection ==="
echo ""

python evaluate_ensemble.py \
  --base-model microsoft/Florence-2-large-ft \
  --v11-checkpoint ./checkpoints_v11/best_model \
  --contrastive-checkpoint ./checkpoints_contrastive/best_model.pt \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_ensemble_v11_contrastive.json \
  --threshold 0.5 \
  --bf16
