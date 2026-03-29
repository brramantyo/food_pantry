#!/bin/bash
#SBATCH --job-name=od_classify
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --output=od_classify_%j.log

cd ~/food_pantry

echo "=== Fine-tuned OD → Crop → v11 Classify Pipeline ==="
echo "=== OD: checkpoints_od_v1/best_model ==="
echo "=== CLS: checkpoints_v11/best_model ==="
echo ""

python evaluate_od_classify.py \
  --base-model microsoft/Florence-2-large-ft \
  --od-checkpoint ./checkpoints_od_v1/best_model \
  --cls-checkpoint ./checkpoints_v11/best_model \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_od_classify.json \
  --bf16
