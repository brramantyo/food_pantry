#!/bin/bash
#SBATCH --job-name=pipe_cot
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=40G
#SBATCH --output=pipeline_cot_%j.log

cd ~/food_pantry

echo "=== End-to-End Pipeline with CoT-Enhanced USDA Matching ==="
echo "=== Task 1: Florence-2 v11 classifier (unchanged) ==="
echo "=== Task 2: Qwen2.5-VL-7B reads labels → specific USDA query → rerank ==="
echo ""
echo "Comparing:"
echo "  Baseline: category name → USDA semantic search"
echo "  CoT: VLM reads product label → specific query → USDA search → VLM rerank"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python evaluate_pipeline_cot.py \
  --jsonl ./florence2_data/test_v5.jsonl \
  --data-dir . \
  --cls-model microsoft/Florence-2-large-ft \
  --cls-checkpoint ./checkpoints_v11/best_model \
  --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
  --usda-dir ./usda_data \
  --top-k 5 \
  --output ./eval_pipeline_cot.json \
  --bf16
