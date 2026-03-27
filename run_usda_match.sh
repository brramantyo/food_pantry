#!/bin/bash
#SBATCH --job-name=usda_match
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=usda_match_%j.log

cd ~/food_pantry

echo "=== Test 1: Semantic search examples ==="
python usda_matcher.py --usda-dir ./usda_data --query "granola bar cinnamon raisin" --top-k 5
echo ""
python usda_matcher.py --usda-dir ./usda_data --query "canned soup chicken noodle" --top-k 5
echo ""
python usda_matcher.py --usda-dir ./usda_data --query "peanut butter creamy" --top-k 5

echo ""
echo "=== Test 2: Nutrition summary for all 21 pantry categories ==="
python usda_matcher.py --usda-dir ./usda_data --nutrition-summary
