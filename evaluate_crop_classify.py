#!/usr/bin/env python3
"""
evaluate_crop_classify.py
=========================
Two-stage pipeline: YOLO detection → Florence-2 per-crop classification.

Stage 1: YOLO detects bounding boxes in each test image
Stage 2: Each crop is fed to Florence-2 for single-item classification
Results: per-image multi-label predictions (union of per-crop labels)

Usage:
    python evaluate_crop_classify.py \
        --yolo-weights runs/detect/yolo_output_clean/yolo_detector/weights/best.pt \
        --florence-checkpoint checkpoints_v11_clean_v2/best_model \
        --base-model microsoft/Florence-2-large-ft \
        --test-jsonl florence2_data_clean/test.jsonl \
        --data-dir cleaned_data \
        --output eval_crop_classify.json \
        --conf-threshold 0.25 \
        --bf16
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# ── 18 Clean Categories ────────────────────────────────────────────────────────

CLEAN_CATEGORIES = [
    "Beans and Legumes - Canned or Dried",
    "Bread and Bakery Products",
    "Canned Protein",
    "Canned Tomato Products",
    "Carbohydrate Meal",
    "Condiments and Sauces",
    "Dairy and Dairy Alternatives",
    "Desserts and Sweets",
    "Drinks",
    "Fresh Produce",
    "Fruits - Canned or Processed",
    "Granola Products",
    "Meat and Poultry - Fresh",
    "Nut Butters and Nuts",
    "Ready Meals",
    "Savory Snacks and Crackers",
    "Soup",
    "Vegetables - Canned",
]

CANONICAL_MAP = {c.lower(): c for c in CLEAN_CATEGORIES}

# Fuzzy aliases for model output normalization
ALIASES = {
    'vegetables - canned or dried': 'Vegetables - Canned',
    'vegetables - canned and dried': 'Vegetables - Canned',
    'fruits - canned or dried': 'Fruits - Canned or Processed',
    'fruits - canned and dried': 'Fruits - Canned or Processed',
    'beans and legumes': 'Beans and Legumes - Canned or Dried',
    'beans - canned or dried': 'Beans and Legumes - Canned or Dried',
    'canned protein products': 'Canned Protein',
    'canned proteins': 'Canned Protein',
    'canned meat': 'Canned Protein',
    'canned seafood': 'Canned Protein',
    'meat and poultry - canned': 'Canned Protein',
    'seafood - canned': 'Canned Protein',
    'fresh produce items': 'Fresh Produce',
    'fresh fruit': 'Fresh Produce',
    'fresh fruits': 'Fresh Produce',
    'vegetables - fresh': 'Fresh Produce',
    'fresh vegetables': 'Fresh Produce',
    'snacks and crackers': 'Savory Snacks and Crackers',
    'savory snacks': 'Savory Snacks and Crackers',
    'nut butters': 'Nut Butters and Nuts',
    'granola': 'Granola Products',
    'ready meals and entrees': 'Ready Meals',
    'dairy': 'Dairy and Dairy Alternatives',
    'desserts': 'Desserts and Sweets',
    'sweets and desserts': 'Desserts and Sweets',
    'carbohydrate meals': 'Carbohydrate Meal',
    'bread and bakery': 'Bread and Bakery Products',
    'canned tomato': 'Canned Tomato Products',
    'condiments': 'Condiments and Sauces',
}

for alias, canon in ALIASES.items():
    CANONICAL_MAP[alias.lower()] = canon


def normalize_category(name):
    """Normalize a category name to canonical form."""
    n = name.strip().lower()
    if n in CANONICAL_MAP:
        return CANONICAL_MAP[n]
    # Substring match
    for key, val in sorted(CANONICAL_MAP.items(), key=lambda x: -len(x[0])):
        if key in n or n in key:
            return val
    return name


def load_yolo_model(weights_path):
    """Load YOLO model for inference."""
    from ultralytics import YOLO
    print(f"Loading YOLO model: {weights_path}")
    model = YOLO(weights_path)
    return model


def load_florence2(base_model_name, checkpoint_path, device, use_bf16=False):
    """Load Florence-2 with LoRA checkpoint."""
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import PeftModel

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Loading Florence-2: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True, torch_dtype=dtype
    )

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading LoRA checkpoint: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
        model = model.merge_and_unload()
        print("  ✓ LoRA merged")

    model.to(device)
    model.eval()
    return model, processor


def yolo_detect(yolo_model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """Run YOLO detection on an image, return list of (box, class_name, conf)."""
    results = yolo_model.predict(
        image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    detections = []
    if results and len(results) > 0:
        r = results[0]
        names = r.names  # {idx: name}
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().tolist()  # [x1, y1, x2, y2]
            cls_name = names.get(cls_id, f"class_{cls_id}")
            detections.append({
                'box': xyxy,
                'class_name': cls_name,
                'confidence': conf,
            })
    return detections


def crop_and_classify(florence_model, processor, image, box, device, dtype):
    """Crop image to bounding box and classify with Florence-2."""
    x1, y1, x2, y2 = [int(c) for c in box]
    
    # Add padding (10% of box size)
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
    img_w, img_h = image.size
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)
    
    crop = image.crop((x1, y1, x2, y2))
    
    # Skip tiny crops
    if crop.size[0] < 20 or crop.size[1] < 20:
        return None
    
    # Florence-2 classification prompt for single item
    prompt = '<CLASSIFY>'
    task_prompt = (
        'Identify the food pantry item in this image. '
        'Return a JSON object: {"name": "<category>", "category": "<USDA group>", '
        '"package_type": "<type>", "count": 1}'
    )
    full_prompt = f"{prompt} {task_prompt}"
    
    inputs = processor(
        text=full_prompt,
        images=crop,
        return_tensors="pt",
    ).to(device)
    
    # Convert to correct dtype
    if dtype == torch.bfloat16:
        for k in inputs:
            if inputs[k].dtype == torch.float32:
                inputs[k] = inputs[k].to(torch.bfloat16)
    
    with torch.no_grad():
        generated = florence_model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=3,
            do_sample=False,
        )
    
    output = processor.batch_decode(generated, skip_special_tokens=True)[0]
    
    # Parse the category from output
    try:
        # Try JSON parse
        import re
        json_match = re.search(r'\{[^}]+\}', output)
        if json_match:
            parsed = json.loads(json_match.group())
            return normalize_category(parsed.get('name', ''))
    except:
        pass
    
    # Fallback: try to extract category name directly
    output_clean = output.strip()
    normalized = normalize_category(output_clean)
    if normalized in set(CLEAN_CATEGORIES):
        return normalized
    
    return output_clean


def parse_gt_categories(target_text):
    """Parse ground truth categories from JSONL target."""
    try:
        parsed = json.loads(target_text)
        items = parsed.get('items', [])
        return set(normalize_category(item['name']) for item in items if 'name' in item)
    except:
        return set()


def main():
    parser = argparse.ArgumentParser(description="YOLO → Florence-2 crop classification pipeline")
    parser.add_argument('--yolo-weights', required=True, help='Path to YOLO best.pt')
    parser.add_argument('--florence-checkpoint', default=None, help='Path to Florence-2 LoRA checkpoint')
    parser.add_argument('--base-model', default='microsoft/Florence-2-large-ft')
    parser.add_argument('--test-jsonl', required=True, help='Test JSONL file')
    parser.add_argument('--data-dir', default='.', help='Base directory for image paths')
    parser.add_argument('--output', default='eval_crop_classify.json')
    parser.add_argument('--conf-threshold', type=float, default=0.25)
    parser.add_argument('--crop-padding', type=float, default=0.1)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--yolo-only', action='store_true', help='Skip Florence-2, use YOLO class names directly')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    # Load models
    yolo_model = load_yolo_model(args.yolo_weights)

    florence_model, processor = None, None
    if not args.yolo_only:
        florence_model, processor = load_florence2(
            args.base_model, args.florence_checkpoint, device, args.bf16
        )

    # Load test data
    print(f"\nLoading test data: {args.test_jsonl}")
    test_samples = []
    with open(args.test_jsonl) as f:
        for line in f:
            sample = json.loads(line)
            test_samples.append(sample)
    print(f"  {len(test_samples)} test samples")

    # Evaluate
    print(f"\n{'='*70}")
    print("YOLO → Crop Classification Pipeline")
    print(f"{'='*70}\n")

    tp, fp, fn = 0, 0, 0
    per_class_tp = Counter()
    per_class_fp = Counter()
    per_class_fn = Counter()
    per_class_support = Counter()
    exact_match = 0
    total_detections = 0
    predictions_log = []

    for i, sample in enumerate(test_samples):
        image_path = os.path.join(args.data_dir, sample['image'])
        if not os.path.exists(image_path):
            print(f"  [WARN] Image not found: {image_path}")
            continue

        # Ground truth
        gt_cats = parse_gt_categories(sample.get('target', sample.get('suffix', '{}')))

        # Stage 1: YOLO detection
        detections = yolo_detect(yolo_model, image_path, conf_threshold=args.conf_threshold)
        total_detections += len(detections)

        # Stage 2: Classify each crop
        pred_cats = set()
        crop_details = []

        if args.yolo_only:
            # Use YOLO class names directly
            for det in detections:
                cat = normalize_category(det['class_name'])
                pred_cats.add(cat)
                crop_details.append({
                    'yolo_class': det['class_name'],
                    'normalized': cat,
                    'confidence': det['confidence'],
                })
        else:
            # Florence-2 per-crop classification
            image = Image.open(image_path).convert('RGB')
            for det in detections:
                cat = crop_and_classify(
                    florence_model, processor, image, det['box'], device, dtype
                )
                if cat:
                    norm_cat = normalize_category(cat)
                    pred_cats.add(norm_cat)
                    crop_details.append({
                        'yolo_class': det['class_name'],
                        'florence_class': cat,
                        'normalized': norm_cat,
                        'confidence': det['confidence'],
                        'box': det['box'],
                    })

        # Score
        matched = gt_cats & pred_cats
        missed = gt_cats - pred_cats
        extra = pred_cats - gt_cats

        if not missed and not extra and len(gt_cats) > 0:
            exact_match += 1

        tp += len(matched)
        fn += len(missed)
        fp += len(extra)

        for c in gt_cats:
            per_class_support[c] += 1
        for c in matched:
            per_class_tp[c] += 1
        for c in missed:
            per_class_fn[c] += 1
        for c in extra:
            per_class_fp[c] += 1

        predictions_log.append({
            'image': sample['image'],
            'gt': sorted(gt_cats),
            'pred': sorted(pred_cats),
            'matched': sorted(matched),
            'missed': sorted(missed),
            'extra': sorted(extra),
            'num_detections': len(detections),
            'crops': crop_details,
        })

        if (i + 1) % 20 == 0:
            curr_p = tp / (tp + fp) if (tp + fp) > 0 else 0
            curr_r = tp / (tp + fn) if (tp + fn) > 0 else 0
            curr_f1 = 2 * curr_p * curr_r / (curr_p + curr_r) if (curr_p + curr_r) > 0 else 0
            print(f"  [{i+1}/{len(test_samples)}] Running F1: {curr_f1*100:.1f}% "
                  f"(P={curr_p*100:.1f}%, R={curr_r*100:.1f}%)")

    # Final metrics
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    n_mismatch = sum(1 for p in predictions_log if p['missed'] or p['extra'])

    print(f"\n{'='*70}")
    mode = "YOLO-only" if args.yolo_only else "YOLO → Florence-2 Crop"
    print(f"RESULTS: {mode} Classification")
    print(f"{'='*70}")
    print(f"")
    print(f"  Micro Precision: {micro_p*100:.1f}%")
    print(f"  Micro Recall:    {micro_r*100:.1f}%")
    print(f"  Micro F1:        {micro_f1*100:.1f}%")
    print(f"  Exact Match:     {exact_match}/{len(test_samples)} ({100*exact_match/len(test_samples):.1f}%)")
    print(f"")
    print(f"  Total detections: {total_detections} across {len(test_samples)} images")
    print(f"  Avg detections/image: {total_detections/len(test_samples):.1f}")
    print(f"  Mismatched images: {n_mismatch}/{len(test_samples)}")
    print(f"")

    # Per-class
    all_cats = sorted(set(list(per_class_support.keys()) + list(per_class_fp.keys())))
    print(f"  {'Category':<40} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>5}")
    print("  " + "-" * 64)
    macro_f1_sum = 0
    macro_count = 0
    for cat in all_cats:
        t = per_class_tp[cat]
        f_p = per_class_fp[cat]
        f_n = per_class_fn[cat]
        sup = per_class_support[cat]
        p = t / (t + f_p) if (t + f_p) > 0 else 0
        r = t / (t + f_n) if (t + f_n) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if sup > 0:
            macro_f1_sum += f1
            macro_count += 1
        print(f"  {cat:<40} {p*100:>5.1f}% {r*100:>5.1f}% {f1*100:>5.1f}% {sup:>5}")

    macro_f1 = macro_f1_sum / macro_count if macro_count > 0 else 0
    print(f"\n  Macro F1: {macro_f1*100:.1f}%")

    # Sample mismatches
    mismatches = [p for p in predictions_log if p['missed'] or p['extra']]
    if mismatches:
        print(f"\n  Sample mismatches:")
        for m in mismatches[:8]:
            print(f"    {m['image'][-60:]}")
            print(f"      GT:   {m['gt']}")
            print(f"      Pred: {m['pred']}")
            if m['missed']:
                print(f"      Miss: {m['missed']}")
            if m['extra']:
                print(f"      Extra:{m['extra']}")

    # Save results
    result = {
        'mode': mode,
        'micro_precision': round(micro_p, 4),
        'micro_recall': round(micro_r, 4),
        'micro_f1': round(micro_f1, 4),
        'macro_f1': round(macro_f1, 4),
        'exact_match_rate': round(exact_match / len(test_samples), 4),
        'tp': tp, 'fp': fp, 'fn': fn,
        'total_samples': len(test_samples),
        'total_detections': total_detections,
        'conf_threshold': args.conf_threshold,
        'predictions': predictions_log,
        'per_class': {cat: {
            'precision': round(per_class_tp[cat] / (per_class_tp[cat] + per_class_fp[cat]), 4) if (per_class_tp[cat] + per_class_fp[cat]) > 0 else 0,
            'recall': round(per_class_tp[cat] / (per_class_tp[cat] + per_class_fn[cat]), 4) if (per_class_tp[cat] + per_class_fn[cat]) > 0 else 0,
            'support': per_class_support[cat],
        } for cat in all_cats},
    }
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {args.output}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
