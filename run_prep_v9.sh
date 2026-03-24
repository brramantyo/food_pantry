#!/bin/bash
#SBATCH --job-name=prep_v9
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --output=prep_v9_%j.log

# Step 1: Download and convert Grocery Store Dataset
cd ~/food_pantry

echo "=== Downloading Grocery Store Dataset ==="
python download_grocery_dataset.py \
  --output-dir ./grocery_data \
  --repo-dir ./GroceryStoreDataset

echo ""
echo "=== Files created ==="
ls -la ./grocery_data/
wc -l ./grocery_data/*.jsonl

echo ""
echo "=== Done! Now run: sbatch run_v9.sh ==="
