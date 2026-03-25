#!/bin/bash
#SBATCH --job-name=ocr_boost
#SBATCH --output=ocr_boost_%j.log
#SBATCH --error=ocr_boost_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G

cd ~/food_pantry

echo "=== OCR Boost: Post-processing v11 with text detection ==="
echo "=== Using vanilla Florence-2 for OCR (not fine-tuned) ==="

python3 evaluate_ocr_boost.py \
    --eval-results ./eval_results_v11.json \
    --data-dir . \
    --model microsoft/Florence-2-large-ft \
    --bf16 \
    --show-ocr \
    --min-keyword-matches 1
