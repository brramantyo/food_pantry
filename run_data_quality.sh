#!/bin/bash
#SBATCH --job-name=data_qual
#SBATCH --output=logs/data_quality_%j.log
#SBATCH --error=logs/data_quality_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

echo "=========================================="
echo "Data Quality Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd ~/food_pantry
mkdir -p logs data_quality_report

# Run analysis with predictions for mislabel detection
python3 analyze_data_quality.py \
    --data-dir . \
    --predictions eval_results_v11.json \
    --output-dir data_quality_report

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
echo ""
echo "Output:"
echo "  data_quality_report/quality_summary.json"
echo "  data_quality_report/bad_images.json"
echo "  data_quality_report/duplicates.json"
echo "  data_quality_report/mislabel_candidates.json"
