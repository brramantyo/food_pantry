#!/bin/bash
#SBATCH --job-name=clean_data
#SBATCH --output=logs/clean_data_%j.log
#SBATCH --error=logs/clean_data_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

echo "=========================================="
echo "Dataset Cleaning & Re-split"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

cd ~/food_pantry
mkdir -p logs

# ── Run cleaning ──────────────────────────────────────────────────────────
# Config:
#   - Remove blurry images (blur score < 50)
#   - Drop tiny categories: Frozen Mix Veg (10), Oil (16), Spices (16), Carton of Eggs (35), Baby Food (11)
#   - Merge: Meat Canned + Seafood Canned → Canned Protein
#   - Merge: Vegetables Fresh + Fresh Fruit → Fresh Produce
#   - Re-split 70/15/15 with seed 42
#   - Deduplicate (fixes 92 cross-split leaks)

python3 clean_and_resplit.py \
    --data-dir . \
    --output-dir cleaned_data \
    --blur-threshold 50 \
    --drop-categories "Frozen Mix Vegetable,Oil,Spices Seasonings and Mixes,Carton of Eggs,Baby Food" \
    --merge "Meat and Poultry - Canned+Seafood - Canned=Canned Protein" \
    --merge "Vegetables - Fresh+Fresh Fruit=Fresh Produce" \
    --split-ratio 0.70 0.15 0.15 \
    --seed 42

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review: cat cleaned_data/cleaning_report.json"
echo "  2. Convert to Florence-2 format:"
echo "     python3 convert_coco_to_florence2.py --data-dir cleaned_data --mapping usda_mapping.json --output-dir florence2_data_clean"
echo "  3. Re-train best model (v11) on cleaned data"
echo "  4. Compare results: clean vs original"
