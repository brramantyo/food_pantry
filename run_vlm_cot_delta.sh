#!/bin/bash
#SBATCH --job-name=vlm_cot
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgnn-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --output=vlm_cot_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1

cd ~/food_pantry
source ~/usd_env/bin/activate

echo "=== VLM Chain-of-Thought: Qwen2.5-VL-7B-Instruct ==="
echo "=== Zero-shot classification with step-by-step reasoning ==="
echo "=== Running on Delta (A100 GPU) ==="
echo ""

python evaluate_vlm_cot.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_vlm_cot.json \
  --max-new-tokens 1024 \
  --bf16
