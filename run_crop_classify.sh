#!/bin/bash
#SBATCH --job-name=crop_cls
#SBATCH --output=logs/crop_classify_%j.log
#SBATCH --error=logs/crop_classify_%j.log
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00

echo "=========================================="
echo "YOLO → Crop Classification Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo ""

cd ~/food_pantry
mkdir -p logs

# ── Find YOLO weights ────────────────────────────────────────────────────
# Try clean model first, fall back to original
YOLO_WEIGHTS=""
if [ -f "runs/detect/yolo_output_clean/yolo_detector/weights/best.pt" ]; then
    YOLO_WEIGHTS="runs/detect/yolo_output_clean/yolo_detector/weights/best.pt"
elif [ -f "runs/detect/yolo_output/yolo_detector/weights/best.pt" ]; then
    YOLO_WEIGHTS="runs/detect/yolo_output/yolo_detector/weights/best.pt"
else
    echo "ERROR: No YOLO weights found!"
    echo "Tried:"
    echo "  runs/detect/yolo_output_clean/yolo_detector/weights/best.pt"
    echo "  runs/detect/yolo_output/yolo_detector/weights/best.pt"
    ls -la runs/detect/yolo_output_clean/yolo_detector/weights/ 2>/dev/null
    ls -la runs/detect/yolo_output/yolo_detector/weights/ 2>/dev/null
    exit 1
fi
echo "YOLO weights: $YOLO_WEIGHTS"

# ── Experiment 1: YOLO-only (baseline) ───────────────────────────────────
echo ""
echo "============================================"
echo "Experiment 1: YOLO-only classification"
echo "============================================"

python3 evaluate_crop_classify.py \
    --yolo-weights "$YOLO_WEIGHTS" \
    --test-jsonl florence2_data_clean/test.jsonl \
    --data-dir cleaned_data \
    --output eval_crop_yolo_only.json \
    --conf-threshold 0.25 \
    --yolo-only

echo ""

# ── Experiment 2: YOLO → Florence-2 crop classification ─────────────────
echo "============================================"
echo "Experiment 2: YOLO → Florence-2 crop classify"
echo "============================================"

# Use v11 clean v2 checkpoint (from v9)
FLORENCE_CKPT="checkpoints_v11_clean_v2/best_model"
if [ ! -d "$FLORENCE_CKPT" ]; then
    # Fall back to v11 clean (fresh LoRA)
    FLORENCE_CKPT="checkpoints_v11_clean/best_model"
fi
if [ ! -d "$FLORENCE_CKPT" ]; then
    echo "WARNING: No clean Florence-2 checkpoint found, using original v11"
    FLORENCE_CKPT="checkpoints_v11/best_model"
fi
echo "Florence-2 checkpoint: $FLORENCE_CKPT"

python3 evaluate_crop_classify.py \
    --yolo-weights "$YOLO_WEIGHTS" \
    --florence-checkpoint "$FLORENCE_CKPT" \
    --base-model microsoft/Florence-2-large-ft \
    --test-jsonl florence2_data_clean/test.jsonl \
    --data-dir cleaned_data \
    --output eval_crop_florence2.json \
    --conf-threshold 0.25 \
    --bf16

echo ""
echo "=========================================="
echo "DONE! End time: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  YOLO-only:           eval_crop_yolo_only.json"
echo "  YOLO → Florence-2:   eval_crop_florence2.json"
