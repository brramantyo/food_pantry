#!/bin/bash
#SBATCH --job-name=build_usda_idx
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=build_usda_index_%j.log

cd ~/food_pantry

echo "=== Step 1: Preprocess USDA data ==="
python preprocess_usda.py --usda-dir ./usda_data

echo ""
echo "=== Step 2: Build embedding index ==="
pip install -q sentence-transformers 2>/dev/null

python build_usda_index.py \
  --usda-dir ./usda_data \
  --model all-MiniLM-L6-v2 \
  --batch-size 512
