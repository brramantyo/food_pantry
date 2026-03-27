#!/bin/bash
#SBATCH --job-name=detect_classify
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=detect_classify_%j.log

cd ~/food_pantry

echo "=== Detection-first pipeline: detect objects → crop → classify ==="
echo "=== Using vanilla Florence-2 for detection + v11 for classification ==="

python evaluate_detect_classify.py \
  --base-model microsoft/Florence-2-large-ft \
  --checkpoint ./checkpoints_v11/best_model \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_detect_classify.json \
  --bf16
