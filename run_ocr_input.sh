#!/bin/bash
#SBATCH --job-name=ocr_input
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=ocr_input_%j.log

cd ~/food_pantry

echo "=== OCR-input experiment: OCR text as additional input signal ==="
echo "=== Comparing image-only vs OCR+image classification ==="

python evaluate_ocr_input.py \
  --base-model microsoft/Florence-2-large-ft \
  --checkpoint ./checkpoints_v11/best_model \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_ocr_input.json \
  --bf16
