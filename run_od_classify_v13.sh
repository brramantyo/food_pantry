#!/bin/bash
#SBATCH --job-name=od_v13
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --output=od_classify_v13_%j.log

cd ~/food_pantry

echo "=== OD → Crop → v13 Classify Pipeline ==="
echo "=== Key test: does v13 (trained on crops) fix the precision problem? ==="
echo "=== Comparing: v11 pipeline (56.4% F1) vs v13 pipeline ==="
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python evaluate_od_classify.py \
  --base-model microsoft/Florence-2-large-ft \
  --od-checkpoint ./checkpoints_od_v1/best_model \
  --cls-checkpoint ./checkpoints_v13/best_model \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_od_classify_v13.json \
  --bf16
