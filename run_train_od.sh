#!/bin/bash
#SBATCH --job-name=train_od
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=train_od_v1_%j.log

cd ~/food_pantry

echo "=== Training Florence-2 OD (Object Detection) on pantry COCO data ==="
echo "=== 15 epochs, LR=2e-5, LoRA r=32 ==="

python train_florence2_od.py \
  --data-dir . \
  --output-dir ./checkpoints_od_v1 \
  --model microsoft/Florence-2-large-ft \
  --epochs 15 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lr 2e-5 \
  --patience 5 \
  --max-length 1024 \
  --lora-r 32 \
  --lora-alpha 64 \
  --bf16
