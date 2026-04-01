#!/bin/bash
# Quick script to check if you have the required files

echo "Checking requirements for failure example generation..."
echo ""

# Check test.jsonl
if [ -f "florence2_data/test.jsonl" ]; then
    echo "✓ test.jsonl found"
    NUM_SAMPLES=$(wc -l < florence2_data/test.jsonl)
    echo "  → $NUM_SAMPLES test samples"
else
    echo "✗ test.jsonl NOT FOUND"
    echo "  Expected: florence2_data/test.jsonl"
fi

echo ""

# Check for evaluation results
echo "Looking for evaluation results..."
EVAL_FILES=$(find . -maxdepth 1 -name "eval_*.json" -o -name "*results*.json" 2>/dev/null)

if [ -z "$EVAL_FILES" ]; then
    echo "✗ No evaluation results found"
    echo "  You'll need to run inference (slower)"
else
    echo "✓ Found evaluation results:"
    for f in $EVAL_FILES; do
        echo "  → $f"
        # Check if it has per_sample_results
        HAS_PREDICTIONS=$(python3 -c "import json; d=json.load(open('$f')); print('per_sample_results' in d)" 2>/dev/null)
        if [ "$HAS_PREDICTIONS" = "True" ]; then
            echo "    ✓ Has per-sample predictions"
        else
            echo "    ✗ Missing per-sample predictions"
        fi
    done
fi

echo ""

# Check for model checkpoint
echo "Looking for model checkpoints..."
if [ -d "checkpoints" ]; then
    echo "✓ checkpoints/ directory found"
    ls -d checkpoints/*/ 2>/dev/null | head -5
else
    echo "✗ checkpoints/ directory not found"
fi

echo ""

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import matplotlib" 2>/dev/null && echo "✓ matplotlib" || echo "✗ matplotlib (pip install matplotlib)"
python3 -c "import PIL" 2>/dev/null && echo "✓ PIL/Pillow" || echo "✗ PIL/Pillow (pip install pillow)"
python3 -c "import torch" 2>/dev/null && echo "✓ torch" || echo "✗ torch"
python3 -c "import transformers" 2>/dev/null && echo "✓ transformers" || echo "✗ transformers"

echo ""
echo "=========================================="
echo "RECOMMENDATION:"
echo "=========================================="

if [ -n "$EVAL_FILES" ]; then
    echo "You have evaluation results! Use Option 1 (fast):"
    echo ""
    echo "  1. Edit run_generate_failures.sh"
    echo "  2. Set EVAL_RESULTS to your eval file"
    echo "  3. Run: sbatch run_generate_failures.sh"
else
    echo "No evaluation results found. Use Option 2 (slower):"
    echo ""
    echo "  1. Make sure you have a model checkpoint"
    echo "  2. Edit run_generate_failures.sh"
    echo "  3. Update checkpoint path"
    echo "  4. Run: sbatch run_generate_failures.sh"
fi

echo ""
