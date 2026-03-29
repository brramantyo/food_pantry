#!/usr/bin/env python3
"""
evaluate_od_classify.py
=======================
Fine-tuned OD → Crop → v11 Classify pipeline.

Key difference from evaluate_detect_classify.py:
  - OLD: vanilla Florence-2 OD (generic objects) → v11 classify
  - NEW: fine-tuned Florence-2 OD (pantry-specific bboxes) → v11 classify
  
The fine-tuned OD model knows pantry items, so bbox quality should be
much better. We IGNORE the OD labels (which are noisy/hallucinated)
and only use the bounding boxes for cropping.

Pipeline:
  1. Run fine-tuned OD model → get bounding boxes (ignore labels)
  2. Crop each detected region (with padding)
  3. Run v11 classifier on each crop → get categories
  4. Also run v11 on full image (fallback for missed detections)
  5. Merge all predictions (union of categories)

Usage:
  python evaluate_od_classify.py \
    --base-model microsoft/Florence-2-large-ft \
    --od-checkpoint ./checkpoints_od_v1/best_model \
    --cls-checkpoint ./checkpoints_v11/best_model \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_results_od_classify.json \
    --bf16
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel


# ── Constants ──────────────────────────────────────────────────────────────────

OD_PROMPT = "<OD>"
CLS_PROMPT = "<OD>"  # v11 uses <OD> prompt for classification

VALID_CATEGORIES = {
    "Baby Food", "Beans and Legumes - Canned or Dried", "Bread and Bakery Products",
    "Canned Tomato Products", "Carbohydrate Meal", "Condiments and Sauces",
    "Dairy and Dairy Alternatives", "Desserts and Sweets", "Drinks", "Fresh Fruit",
    "Fruits - Canned or Processed", "Granola Products", "Meat and Poultry - Canned",
    "Meat and Poultry - Fresh", "Nut Butters and Nuts", "Ready Meals",
    "Savory Snacks and Crackers", "Seafood - Canned", "Soup",
    "Vegetables - Canned", "Vegetables - Fresh",
}

CANONICAL_MAP = {c.lower(): c for c in VALID_CATEGORIES}


def normalize_category(name):
    return CANONICAL_MAP.get(name.lower().strip(), name)


# ── OD Parsing ─────────────────────────────────────────────────────────────────

def parse_od_bboxes(text, img_width=1000, img_height=1000):
    """
    Parse Florence-2 OD output and extract bounding boxes ONLY.
    Labels are ignored (they're noisy from OD training).
    
    Florence-2 loc tokens are normalized 0-999, need to scale to image size.
    
    Returns list of [x1, y1, x2, y2] in pixel coordinates.
    """
    bboxes = []
    pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    
    for m in re.finditer(pattern, text):
        # Florence-2 locations are normalized to 0-999
        x1 = int(m.group(1)) / 999 * img_width
        y1 = int(m.group(2)) / 999 * img_height
        x2 = int(m.group(3)) / 999 * img_width
        y2 = int(m.group(4)) / 999 * img_height
        
        # Ensure valid bbox
        if x2 > x1 and y2 > y1:
            bboxes.append([x1, y1, x2, y2])
    
    return bboxes


def deduplicate_bboxes(bboxes, iou_threshold=0.5):
    """Remove duplicate/overlapping bboxes using NMS-like approach."""
    if not bboxes:
        return []
    
    # Sort by area (larger first)
    bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    
    keep = []
    for bbox in bboxes:
        is_dup = False
        for kept in keep:
            iou = compute_iou(bbox, kept)
            if iou > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(bbox)
    
    return keep


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


# ── Classification Parsing ─────────────────────────────────────────────────────

def parse_classification(text):
    """
    Parse v11 classifier output (JSON format).
    Returns set of category names.
    """
    text = text.strip()
    
    # Strategy 1: Direct JSON parse
    for candidate in [text, text.replace("'", '"')]:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict) and "items" in result:
                cats = set()
                for item in result["items"]:
                    name = normalize_category(item.get("name", ""))
                    if name in VALID_CATEGORIES:
                        cats.add(name)
                return cats
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Strategy 2: Find JSON in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict) and "items" in result:
                cats = set()
                for item in result["items"]:
                    name = normalize_category(item.get("name", ""))
                    if name in VALID_CATEGORIES:
                        cats.add(name)
                return cats
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Strategy 3: Fix truncated JSON
    if start >= 0:
        fragment = text[start:]
        for suffix in ['"}]}', '"}]}}', ']}}', ']}', '}]}', '"}}', '"}', '}',
                       ', "confidence": "high"}]}']:
            try:
                result = json.loads(fragment + suffix)
                if isinstance(result, dict) and "items" in result:
                    cats = set()
                    for item in result["items"]:
                        if isinstance(item, dict) and "name" in item:
                            name = normalize_category(item["name"])
                            if name in VALID_CATEGORIES:
                                cats.add(name)
                    return cats
            except (json.JSONDecodeError, TypeError):
                pass
    
    # Strategy 4: Regex for known categories
    cats = set()
    for cat in VALID_CATEGORIES:
        if cat.lower() in text.lower():
            cats.add(cat)
    return cats


# ── Crop Helper ────────────────────────────────────────────────────────────────

def crop_with_padding(image, bbox, padding_ratio=0.15):
    """Crop image with padding around bbox. Returns None if too small."""
    x1, y1, x2, y2 = bbox
    w, h = image.size
    
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    
    # Minimum size check
    if crop.width < 32 or crop.height < 32:
        return None
    return crop


# ── Model Inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_od(model, processor, image, device, amp_dtype=None):
    """Run OD model, return raw text output."""
    inputs = processor(text=OD_PROMPT, images=image, return_tensors="pt").to(device)
    
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
    
    return processor.batch_decode(generated_ids, skip_special_tokens=False)[0]


@torch.no_grad()
def run_classify(model, processor, image, device, amp_dtype=None):
    """Run classification model, return raw text output."""
    inputs = processor(text=CLS_PROMPT, images=image, return_tensors="pt").to(device)
    
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
    
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


# ── Pipeline ───────────────────────────────────────────────────────────────────

def evaluate_image(image, od_model, cls_model, processor, device, amp_dtype,
                   min_box_area=1000, max_detections=20, iou_threshold=0.5):
    """
    Full pipeline for one image:
    1. Fine-tuned OD → bboxes (ignore labels)
    2. Deduplicate overlapping bboxes
    3. Crop each → v11 classify
    4. Also classify full image
    5. Union all categories
    
    Returns: (predicted_categories: set, n_bboxes: int, n_valid_crops: int, details: dict)
    """
    w, h = image.size
    
    # Step 1: Run fine-tuned OD
    od_text = run_od(od_model, processor, image, device, amp_dtype)
    bboxes = parse_od_bboxes(od_text, img_width=w, img_height=h)
    
    # Step 2: Deduplicate
    bboxes = deduplicate_bboxes(bboxes, iou_threshold=iou_threshold)
    
    # Filter by area
    bboxes = [b for b in bboxes if (b[2]-b[0])*(b[3]-b[1]) >= min_box_area]
    
    # Limit detections
    bboxes = bboxes[:max_detections]
    
    n_bboxes = len(bboxes)
    
    # Step 3: Crop and classify each bbox
    crop_categories = set()
    n_valid = 0
    crop_details = []
    
    for bbox in bboxes:
        crop = crop_with_padding(image, bbox)
        if crop is None:
            continue
        
        n_valid += 1
        cls_text = run_classify(cls_model, processor, crop, device, amp_dtype)
        cats = parse_classification(cls_text)
        crop_categories.update(cats)
        
        crop_details.append({
            "bbox": [round(v, 1) for v in bbox],
            "crop_size": [crop.width, crop.height],
            "categories": sorted(cats),
        })
    
    # Step 4: Also classify full image (important fallback!)
    full_cls_text = run_classify(cls_model, processor, image, device, amp_dtype)
    full_categories = parse_classification(full_cls_text)
    
    # Step 5: Union all
    all_categories = crop_categories | full_categories
    
    details = {
        "n_bboxes_raw": n_bboxes,
        "n_valid_crops": n_valid,
        "od_raw": od_text[:300],
        "crop_categories": sorted(crop_categories),
        "full_image_categories": sorted(full_categories),
        "final_categories": sorted(all_categories),
        "crop_details": crop_details[:10],  # limit for output size
    }
    
    return all_categories, n_bboxes, n_valid, details


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(all_targets, all_preds):
    """Compute micro/macro precision, recall, F1."""
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()
    class_support = Counter()
    exact_match = 0
    
    for target_set, pred_set in zip(all_targets, all_preds):
        target_set = {c for c in target_set if c in VALID_CATEGORIES}
        pred_set = {c for c in pred_set if c in VALID_CATEGORIES}
        
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
    all_classes = sorted(set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys())))
    precisions, recalls, f1s = [], [], []
    
    for cls in all_classes:
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        per_class[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": class_support[cls],
        }
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    
    n = len(all_targets)
    n_cls = max(len(all_classes), 1)
    
    return {
        "total_samples": n,
        "exact_match": round(exact_match / max(n, 1), 4),
        "micro": {
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4),
        },
        "macro": {
            "precision": round(sum(precisions) / n_cls, 4),
            "recall": round(sum(recalls) / n_cls, 4),
            "f1": round(sum(f1s) / n_cls, 4),
        },
        "per_class": per_class,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tuned OD → Crop → v11 Classify pipeline")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--od-checkpoint", type=str, default="./checkpoints_od_v1/best_model",
                        help="Fine-tuned OD model checkpoint")
    parser.add_argument("--cls-checkpoint", type=str, default="./checkpoints_v11/best_model",
                        help="Fine-tuned classifier (v11) checkpoint")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--output", type=str, default="./eval_results_od_classify.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-box-area", type=int, default=1000)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--no-full-image", action="store_true",
                        help="Skip full-image classification fallback (crops only)")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Device: {device}")
    if amp_dtype:
        print("Using bf16")
    
    # ── Load OD model ──────────────────────────────────────────────────────
    print(f"\n[1/2] Loading fine-tuned OD model: {args.od_checkpoint}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    
    od_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    od_model = PeftModel.from_pretrained(od_model, args.od_checkpoint)
    od_model = od_model.merge_and_unload()
    od_model = od_model.to(device).eval()
    print("  OD model loaded ✓")
    
    # ── Load classifier model ──────────────────────────────────────────────
    print(f"[2/2] Loading classifier (v11): {args.cls_checkpoint}")
    cls_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    cls_model = PeftModel.from_pretrained(cls_model, args.cls_checkpoint)
    cls_model = cls_model.merge_and_unload()
    cls_model = cls_model.to(device).eval()
    print("  Classifier loaded ✓")
    
    # ── Load test data ─────────────────────────────────────────────────────
    print(f"\nLoading test data from {args.jsonl}...")
    samples = []
    with open(args.jsonl, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Evaluating {len(samples)} samples...")
    
    # ── Evaluate ───────────────────────────────────────────────────────────
    all_targets = []
    all_preds = []
    all_details = []
    total_bboxes = 0
    total_valid = 0
    
    t0 = time.time()
    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed: {img_path}: {e}")
            continue
        
        # Ground truth
        target = json.loads(sample["target"])
        target_cats = {normalize_category(item["name"]) for item in target.get("items", [])
                       if normalize_category(item["name"]) in VALID_CATEGORIES}
        
        # Pipeline
        pred_cats, n_bbox, n_valid, details = evaluate_image(
            image, od_model, cls_model, processor, device, amp_dtype,
            min_box_area=args.min_box_area,
            max_detections=args.max_detections,
            iou_threshold=args.iou_threshold,
        )
        
        # If --no-full-image, remove full image categories from prediction
        if args.no_full_image:
            pred_cats = pred_cats - set(details["full_image_categories"]) | \
                        set(details["crop_categories"])
        
        all_targets.append(target_cats)
        all_preds.append(pred_cats)
        total_bboxes += n_bbox
        total_valid += n_valid
        
        details["image"] = img_rel
        details["target"] = sorted(target_cats)
        details["match"] = target_cats == pred_cats
        all_details.append(details)
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(samples)} ({rate:.2f} img/s, ETA: {eta:.0f}s)")
    
    elapsed = time.time() - t0
    n = len(all_targets)
    
    # ── Compute metrics ────────────────────────────────────────────────────
    metrics = compute_metrics(all_targets, all_preds)
    
    # ── Print report ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FINE-TUNED OD → CROP → V11 CLASSIFY PIPELINE")
    print(f"{'='*70}")
    print(f"  OD checkpoint: {args.od_checkpoint}")
    print(f"  CLS checkpoint: {args.cls_checkpoint}")
    print(f"  Full-image fallback: {'OFF' if args.no_full_image else 'ON'}")
    print(f"  Samples: {n}")
    print(f"  Time: {elapsed:.1f}s ({n/elapsed:.2f} img/s)")
    print(f"  Avg bboxes/image: {total_bboxes/max(n,1):.1f}")
    print(f"  Avg valid crops/image: {total_valid/max(n,1):.1f}")
    print(f"  Exact Match: {metrics['exact_match']:.1%}")
    
    print(f"\n  Micro:  P={metrics['micro']['precision']:.1%}  R={metrics['micro']['recall']:.1%}  F1={metrics['micro']['f1']:.1%}")
    print(f"  Macro:  P={metrics['macro']['precision']:.1%}  R={metrics['macro']['recall']:.1%}  F1={metrics['macro']['f1']:.1%}")
    
    print(f"\n  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
    for cls in sorted(metrics["per_class"].keys()):
        m = metrics["per_class"][cls]
        print(f"  {cls:<45} {m['precision']:>5.1%} {m['recall']:>5.1%} {m['f1']:>5.1%} {m['support']:>5}")
    
    # ── Comparison: crops-only vs full+crops ───────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ABLATION: Crops-only vs Full+Crops")
    print(f"{'='*70}")
    
    # Compute crops-only metrics
    crops_only_preds = [set(d["crop_categories"]) for d in all_details]
    crops_metrics = compute_metrics(all_targets, crops_only_preds)
    
    # Full-only (v11 baseline equivalent)
    full_only_preds = [set(d["full_image_categories"]) for d in all_details]
    full_metrics = compute_metrics(all_targets, full_only_preds)
    
    print(f"  {'Method':<30} {'Micro P':>8} {'Micro R':>8} {'Micro F1':>9} {'Macro F1':>9}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
    print(f"  {'Full image only':<30} {full_metrics['micro']['precision']:>7.1%} {full_metrics['micro']['recall']:>7.1%} {full_metrics['micro']['f1']:>8.1%} {full_metrics['macro']['f1']:>8.1%}")
    print(f"  {'Crops only':<30} {crops_metrics['micro']['precision']:>7.1%} {crops_metrics['micro']['recall']:>7.1%} {crops_metrics['micro']['f1']:>8.1%} {crops_metrics['macro']['f1']:>8.1%}")
    print(f"  {'Full + Crops (union)':<30} {metrics['micro']['precision']:>7.1%} {metrics['micro']['recall']:>7.1%} {metrics['micro']['f1']:>8.1%} {metrics['macro']['f1']:>8.1%}")
    
    # ── Sample details ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SAMPLE PREDICTIONS (first 5)")
    print(f"{'='*70}")
    for d in all_details[:5]:
        print(f"\n  Image: {d['image']}")
        print(f"  Target:     {d['target']}")
        print(f"  Full image: {d['full_image_categories']}")
        print(f"  Crops:      {d['crop_categories']}")
        print(f"  Final:      {d['final_categories']}")
        print(f"  Bboxes: {d['n_bboxes_raw']}, Valid crops: {d['n_valid_crops']}")
        match_str = "✓ MATCH" if d['match'] else "✗ MISMATCH"
        print(f"  {match_str}")
    
    # ── Save ───────────────────────────────────────────────────────────────
    results = {
        "pipeline": "finetuned_od_crop_v11_classify",
        "od_checkpoint": args.od_checkpoint,
        "cls_checkpoint": args.cls_checkpoint,
        "full_image_fallback": not args.no_full_image,
        "n_samples": n,
        "metrics": metrics,
        "crops_only_metrics": crops_metrics,
        "full_only_metrics": full_metrics,
        "avg_bboxes": round(total_bboxes / max(n, 1), 2),
        "avg_valid_crops": round(total_valid / max(n, 1), 2),
        "details": all_details[:50],  # save first 50 for analysis
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
