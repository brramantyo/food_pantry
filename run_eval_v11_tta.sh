#!/bin/bash
#SBATCH --job-name=eval_v11_tta
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=eval_v11_tta_%j.log

cd ~/food_pantry

echo "=== Evaluating v11 with Test-Time Augmentation (5 augmentations) ==="
echo "=== Each image: original + hflip + bright+ + bright- + crop90 ==="
echo "=== Threshold: 2/5 (class must appear in >=2 augmentations) ==="

python evaluate_tta.py \
  --checkpoint ./checkpoints_v11/best_model \
  --base-model microsoft/Florence-2-large-ft \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_v11_tta.json \
  --bf16 \
  --threshold 2 \
  --show-predictions 10
