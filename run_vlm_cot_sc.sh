#!/bin/bash
#SBATCH --job-name=vlm_sc
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgnn-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=40G
#SBATCH --output=vlm_sc_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1

cd ~/food_pantry
source ~/usd_env/bin/activate

echo "=== VLM Self-Consistency: 3x sampling + majority vote ==="
echo "=== Qwen2.5-VL-7B-Instruct, temperature=0.7, threshold=2/3 ==="
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python evaluate_vlm_cot.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_vlm_cot_sc.json \
  --max-new-tokens 512 \
  --bf16 \
  --self-consistency 3 \
  --sc-temperature 0.7 \
  --sc-threshold 2
