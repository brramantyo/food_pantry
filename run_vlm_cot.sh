#!/bin/bash
#SBATCH --job-name=vlm_cot
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=vlm_cot_%j.log

cd ~/food_pantry

echo "=== VLM Chain-of-Thought: Qwen2.5-VL-7B-Instruct ==="
echo "=== Zero-shot classification with step-by-step reasoning ==="
echo ""

# Install qwen-vl-utils if needed
pip install --user qwen-vl-utils 2>/dev/null

python evaluate_vlm_cot.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_vlm_cot.json \
  --max-new-tokens 1024 \
  --bf16
