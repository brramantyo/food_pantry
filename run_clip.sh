#!/bin/bash
#SBATCH --job-name=clip_crop
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=40G
#SBATCH --output=clip_train_%j.log

cd ~/food_pantry

echo "=== Training CLIP for single-label crop classification ==="
echo "=== 21 pantry categories, trained on GT crops ==="

python train_clip_classifier.py \
    --crop-dir ./crop_data \
    --train-jsonl ./crop_data/train_crops.jsonl \
    --val-jsonl ./crop_data/val_crops.jsonl \
    --output-dir ./clip_output \
    --model openai/clip-vit-base-patch32 \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --freeze-encoder \
    --bf16
