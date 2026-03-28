#!/usr/bin/env python3
"""
evaluate_od.py
==============
Evaluate fine-tuned Florence-2 Object Detection on pantry test set.

Computes:
  1. Detection metrics: mAP, per-class AP
  2. Classification from detections: aggregate detected categories per image → F1
  3. Comparison with image-level classification (v11)

Usage:
  python evaluate_od.py \
    --base-model microsoft/Florence-2-large-ft \
    --checkpoint ./checkpoints_od_v1/best_model \
    --data-dir . \
    --output ./eval_results_od_v1.json \
    --bf16
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel


TASK_PROMPT = "<OD>"

PANTRY_CATEGORIES = {
    "Baby Food", "Beans and Legumes - Canned or Dried", "Bread and Bakery Products",
    "Canned Tomato Products", "Carbohydrate Meal", "Condiments and Sauces",
    "Dairy and Dairy Alternatives", "Desserts and Sweets", "Drinks", "Fresh Fruit",
    "Fruits - Canned or Processed", "Granola Products", "Meat and Poultry - Canned",
    "Meat and Poultry - Fresh", "Nut Butters and Nuts", "Ready Meals",
    "Savory Snacks and Crackers", "Seafood - Canned", "Soup",
    "Vegetables - Canned", "Vegetables - Fresh",
}

CANONICAL_CATEGORIES = {c.lower(): c for c in PANTRY_CATEGORIES}


def normalize_category(name):
    return CANONICAL_CATEGORIES.get(name.lower().strip(), name)


def parse_od_output(text):
    """
    Parse Florence-2 OD output format.
    
    Expected format: "category_name<loc_x1><loc_y1><loc_x2><loc_y2>..."
    Returns list of (category, [x1, y1, x2, y2]) tuples.
    """
    detections = []
    
    # Pattern: text followed by 4 loc tokens
    pattern = r'([^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    matches = re.finditer(pattern, text)
    
    for m in matches:
        name = m.group(1).strip()
        x1, y1, x2, y2 = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        name = normalize_category(name)
        if name in PANTRY_CATEGORIES:
            detections.append({
                "name": name,
                "bbox": [x1, y1, x2, y2],
            })
    
    return detections


def load_coco_ground_truth(coco_json_path):
    """Load COCO annotations and return per-image ground truth."""
    with open(coco_json_path) as f:
        coco = json.load(f)
    
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    
    # Excluded categories
    excluded = {6, 12, 18, 23}
    
    img_lookup = {img["id"]: img for img in coco["images"]}
    
    gt = {}  # filename → list of {name, bbox_norm}
    for ann in coco["annotations"]:
        if ann["category_id"] in excluded:
            continue
        
        img_id = ann["image_id"]
        if img_id not in img_lookup:
            continue
        
        img_info = img_lookup[img_id]
        fname = img_info["file_name"]
        
        if fname not in gt:
            gt[fname] = {"categories": set(), "annotations": []}
        
        cat_name = cat_names[ann["category_id"]]
        gt[fname]["categories"].add(cat_name)
        gt[fname]["annotations"].append({
            "name": cat_name,
            "bbox": [float(v) for v in ann["bbox"]],  # COCO: x, y, w, h
        })
    
    return gt


def compute_classification_metrics(all_targets, all_preds):
    """Compute set-based classification metrics (same as image-level eval)."""
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()
    class_support = Counter()
    exact_match = 0
    
    for target_set, pred_set in zip(all_targets, all_preds):
        target_set = {c for c in target_set if c in PANTRY_CATEGORIES}
        pred_set = {c for c in pred_set if c in PANTRY_CATEGORIES}
        
        if target_set == pred_set:
            exact_match += 1
        
        for cls in target_set | pred_set:
            if cls in target_set and cls in pred_set:
                class_tp[cls] += 1
            elif cls in pred_set:
                class_fp[cls] += 1
            elif cls in target_set:
                class_fn[cls] += 1
        
        for cls in target_set:
            class_support[cls] += 1
    
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    micro_p = total_tp / max(total_tp + total_fp, 1)
    micro_r = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    
    per_class = {}
    precisions, recalls, f1s = [], [], []
    all_classes = sorted(set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys())))
    
    for cls in all_classes:
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        per_class[cls] = {"precision": p, "recall": r, "f1": f1, "support": class_support[cls]}
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    
    n = len(all_targets)
    
    return {
        "exact_match": exact_match / max(n, 1),
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {
            "precision": sum(precisions) / max(len(precisions), 1),
            "recall": sum(recalls) / max(len(recalls), 1),
            "f1": sum(f1s) / max(len(f1s), 1),
        },
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Florence-2 OD on pantry test set")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_od_v1/best_model")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--output", type=str, default="./eval_results_od_v1.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32
    )
    
    print(f"Loading checkpoint: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()
    print("  Model loaded ✓")
    
    # Load ground truth
    test_coco = os.path.join(args.data_dir, "test", "_annotations.coco.json")
    print(f"\nLoading ground truth from {test_coco}")
    gt = load_coco_ground_truth(test_coco)
    print(f"  {len(gt)} images with annotations")
    
    # Get test images
    test_dir = os.path.join(args.data_dir, "test")
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if args.max_samples:
        test_images = test_images[:args.max_samples]
    
    print(f"  Evaluating {len(test_images)} images...")
    
    # Run inference
    all_targets = []
    all_preds = []
    all_detections = []
    
    t0 = time.time()
    for i, fname in enumerate(test_images):
        img_path = os.path.join(test_dir, fname)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed: {fname}: {e}")
            continue
        
        # Ground truth categories for this image
        if fname in gt:
            target_cats = gt[fname]["categories"]
        else:
            target_cats = set()
        
        all_targets.append(target_cats)
        
        # Run OD inference
        inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
        
        gen_kwargs = dict(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            early_stopping=True,
        )
        
        if amp_dtype:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                generated_ids = model.generate(**gen_kwargs)
        else:
            generated_ids = model.generate(**gen_kwargs)
        
        text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Parse detections
        detections = parse_od_output(text)
        all_detections.append({"image": fname, "detections": detections, "raw": text[:500]})
        
        # Extract unique categories from detections
        pred_cats = {d["name"] for d in detections}
        all_preds.append(pred_cats)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(test_images) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(test_images)} ({rate:.1f} img/s, ETA: {eta:.0f}s)")
    
    elapsed = time.time() - t0
    print(f"\n  Inference complete: {len(all_targets)} images in {elapsed:.1f}s")
    
    # Detection stats
    total_dets = sum(len(d["detections"]) for d in all_detections)
    avg_dets = total_dets / max(len(all_detections), 1)
    print(f"  Total detections: {total_dets}")
    print(f"  Average detections per image: {avg_dets:.1f}")
    
    # Compute classification metrics from detections
    metrics = compute_classification_metrics(all_targets, all_preds)
    
    # Print report
    print(f"\n{'='*70}")
    print(f"  OD-BASED CLASSIFICATION RESULTS (fine-tuned detector)")
    print(f"{'='*70}")
    print(f"  Samples: {len(all_targets)}")
    print(f"  Exact Match: {metrics['exact_match']:.1%}")
    print(f"\n  Micro:  P={metrics['micro']['precision']:.1%}  R={metrics['micro']['recall']:.1%}  F1={metrics['micro']['f1']:.1%}")
    print(f"  Macro:  P={metrics['macro']['precision']:.1%}  R={metrics['macro']['recall']:.1%}  F1={metrics['macro']['f1']:.1%}")
    
    print(f"\n  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    
    for cls in sorted(metrics["per_class"].keys()):
        m = metrics["per_class"][cls]
        print(f"  {cls:<45} {m['precision']:>5.1%} {m['recall']:>5.1%} {m['f1']:>5.1%} {m['support']:>8}")
    
    # Show sample detections
    print(f"\n{'='*70}")
    print(f"  SAMPLE DETECTIONS (first 5)")
    print(f"{'='*70}")
    for det in all_detections[:5]:
        print(f"\n  Image: {det['image']}")
        print(f"  Raw output: {det['raw'][:200]}...")
        for d in det["detections"][:5]:
            print(f"    → {d['name']} at [{d['bbox'][0]}, {d['bbox'][1]}, {d['bbox'][2]}, {d['bbox'][3]}]")
    
    # Save results
    results = {
        "metrics": metrics,
        "total_detections": total_dets,
        "avg_detections_per_image": avg_dets,
        "sample_detections": all_detections[:20],
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
