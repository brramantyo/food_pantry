#!/bin/bash
#SBATCH --job-name=od_v13
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgnn-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --output=od_classify_v13_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1

cd ~/food_pantry
source ~/usd_env/bin/activate

echo "=== OD → Crop → v13 Classify Pipeline ==="
echo "=== Key test: does v13 (trained on crops) fix the precision problem? ==="
echo "=== Comparing: v11 pipeline (56.4% F1) vs v13 pipeline ==="
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python evaluate_od_classify.py \
  --base-model microsoft/Florence-2-large-ft \
  --od-checkpoint ./checkpoints_od_v1/best_model \
  --cls-checkpoint ./checkpoints_v13/best_model \
  --data-dir . \
  --jsonl ./florence2_data/test_v5.jsonl \
  --output ./eval_results_od_classify_v13.json \
  --bf16
