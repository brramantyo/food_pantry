#!/bin/bash
#SBATCH --job-name=full_cot
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --output=pipeline_full_cot_%j.log

cd ~/food_pantry

echo "=== BEST-OF-EVERYTHING Full Pipeline ==="
echo "=== Task 1: Ensemble (v11 ∪ Contrastive) = 80.3% F1 ==="
echo "=== Task 2: CoT-enhanced USDA matching (VLM reads labels → specific query → rerank) ==="
echo ""
echo "Models loaded: Florence-2 v11 + Contrastive + Qwen2.5-VL-7B + USDA embeddings"
echo "This needs ~40GB VRAM, may be tight on single GPU"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python evaluate_pipeline_full_cot.py \
  --jsonl ./florence2_data/test_v5.jsonl \
  --data-dir . \
  --base-model microsoft/Florence-2-large-ft \
  --v11-checkpoint ./checkpoints_v11/best_model \
  --contrastive-checkpoint ./checkpoints_contrastive/best_model.pt \
  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
  --usda-dir ./usda_data \
  --top-k 5 \
  --output ./eval_pipeline_full_cot.json \
  --bf16
