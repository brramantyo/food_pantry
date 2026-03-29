#!/bin/bash
#SBATCH --job-name=ocr_v2
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=ocr_input_v2_%j.log

cd ~/food_pantry

echo "=== OCR Input v2: Two-stage (OCR → classify → merge) ==="
echo "=== v3 fixes: blacklist, specific keywords, min 2 matches ==="

python evaluate_ocr_input_v2.py \
  --base-model microsoft/Florence-2-large-ft \
  --checkpoint ./checkpoints_v11/best_model \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_ocr_input_v3.json \
  --min-ocr-score 0.5 \
  --min-matches 2 \
  --bf16
