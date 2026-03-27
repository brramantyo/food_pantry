#!/bin/bash
#SBATCH --job-name=preprocess_usda
#SBATCH --partition=general
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=preprocess_usda_%j.log

cd ~/food_pantry

echo "=== Preprocessing USDA FoodData Central ==="

python preprocess_usda.py --usda-dir ./usda_data
