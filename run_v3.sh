#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=florence2_v3
#SBATCH --output=train_v3_%j.log

cd ~/food_pantry
python train_florence2_v3.py --data-dir . --jsonl-dir ./florence2_data --output-dir ./checkpoints_v3 --model microsoft/Florence-2-base-ft --epochs 25 --batch-size 4 --bf16 --gradient-checkpointing
