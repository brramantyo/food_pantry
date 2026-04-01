#!/bin/bash
#SBATCH --job-name=gen_failures
#SBATCH --output=logs/generate_failures_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

echo "=========================================="
echo "Generating 5 Failure Examples for Report"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

mkdir -p figures/failure_examples

python3 generate_failure_examples_cluster.py \
    --checkpoint checkpoints/best_model \
    --base-model microsoft/Florence-2-large-ft \
    --test-jsonl florence2_data/test.jsonl \
    --data-dir . \
    --output-dir figures/failure_examples \
    --n-examples 5 \
    --run-inference \
    --device cuda

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
echo ""
echo "Output:"
echo "  figures/failure_examples/failure_example_*.png"
echo "  figures/failure_examples/latex_failure_examples.tex"
