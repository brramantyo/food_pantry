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

# Install flash-attn if not present (saves massive VRAM)
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn not installed, will use sdpa"

echo "=== VLM Chain-of-Thought v2: Improved Prompt ==="
echo "=== Few-shot + label-reading + category disambiguation ==="
echo "=== Running on Delta (A100 GPU) ==="
echo ""

# Set memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python evaluate_vlm_cot.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_vlm_cot_v2.json \
  --max-new-tokens 1024 \
  --bf16
