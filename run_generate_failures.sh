#!/bin/bash
#SBATCH --job-name=gen_failures
#SBATCH --output=logs/generate_failures_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Generate 5 failure examples for the report
# This script loads evaluation results and creates annotated visualizations

echo "=========================================="
echo "Generating Failure Examples"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Activate environment if needed
# source ~/venv/bin/activate

# Option 1: Use existing evaluation results (RECOMMENDED - faster)
# Replace with your actual evaluation results file
EVAL_RESULTS="eval_results_v11.json"

if [ -f "$EVAL_RESULTS" ]; then
    echo "Using existing predictions from: $EVAL_RESULTS"
    python3 generate_failure_examples_cluster.py \
        --predictions "$EVAL_RESULTS" \
        --test-jsonl florence2_data/test.jsonl \
        --data-dir . \
        --output-dir figures/failure_examples \
        --n-examples 5
else
    echo "⚠ Evaluation results not found: $EVAL_RESULTS"
    echo "Running inference instead..."
    
    # Option 2: Run inference (slower, but works if you don't have eval results)
    python3 generate_failure_examples_cluster.py \
        --checkpoint checkpoints/v11_best \
        --base-model microsoft/Florence-2-large-ft \
        --test-jsonl florence2_data/test.jsonl \
        --data-dir . \
        --output-dir figures/failure_examples \
        --n-examples 5 \
        --run-inference \
        --device cuda
fi

echo ""
echo "=========================================="
echo "Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Output files:"
echo "  • figures/failure_examples/failure_example_*.png"
echo "  • figures/failure_examples/failure_descriptions.json"
echo "  • figures/failure_examples/latex_failure_examples.tex"
echo ""
echo "Next: Copy latex_failure_examples.tex into your report!"
