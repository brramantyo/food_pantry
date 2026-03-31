#!/bin/bash
#SBATCH --job-name=clip_crop_classifier_train
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --partition=general
#SBATCH --gres=gpu:1

cd ~/food_pantry

python3 train_clip_classifier.py \
    --crop-dir crop_data/ \
    --train-jsonl crop_data/train_crops.jsonl \
    --val-jsonl crop_data/val_crops.jsonl \
    --output-dir clip_output/ \
    --model openai/clip-vit-base-patch32 \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --freeze-encoder \
    --bf16
