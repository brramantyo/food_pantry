#!/bin/bash
#SBATCH --job-name=ens_clean
#SBATCH --output=logs/ensemble_clean_%j.log
#SBATCH --error=logs/ensemble_clean_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

echo "=========================================="
echo "Ensemble Evaluation on Cleaned Dataset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd ~/food_pantry
mkdir -p logs

# ── Check both models exist ──────────────────────────────────────────────
echo "Checking models..."
if [ ! -d "checkpoints_v11_clean/best_model" ]; then
    echo "ERROR: checkpoints_v11_clean/best_model not found!"
    echo "Run run_retrain_clean.sh first"
    exit 1
fi

if [ ! -f "checkpoints_contrastive_clean/best_model.pt" ]; then
    echo "ERROR: checkpoints_contrastive_clean/best_model.pt not found!"
    echo "Run run_contrastive_clean.sh first"
    exit 1
fi

echo "  ✓ v11 clean model found"
echo "  ✓ contrastive clean model found"
echo ""

# ── Run ensemble evaluation ──────────────────────────────────────────────
echo "============================================"
echo "Evaluating v11_clean ∪ contrastive_clean ensemble"
echo "============================================"

python3 evaluate_ensemble.py \
    --base-model microsoft/Florence-2-large-ft \
    --v11-checkpoint checkpoints_v11_clean/best_model \
    --contrastive-checkpoint checkpoints_contrastive_clean/best_model.pt \
    --data-dir cleaned_data \
    --jsonl florence2_data_clean/test.jsonl \
    --output eval_ensemble_clean.json \
    --bf16

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
echo ""
echo "Results: eval_ensemble_clean.json"
echo "Compare with original: eval_ensemble_v11_contrastive.json (80.3% F1, 21 cats)"
