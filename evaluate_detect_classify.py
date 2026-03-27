#!/usr/bin/env python3
"""
evaluate_detect_classify.py
===========================
Detection-first pipeline: detect objects → crop → classify each crop.

Pipeline:
  1. Run Florence-2 <OD> on full image → get bounding boxes
  2. Crop each detected region
  3. Run fine-tuned Florence-2 classifier on each crop
  4. Merge predictions (deduplicate categories)

This directly addresses the multi-label under-detection problem:
instead of asking the model to predict ALL categories at once,
we detect individual items first, then classify each one.

Usage:
  python evaluate_detect_classify.py \
    --base-model microsoft/Florence-2-large-ft \
    --checkpoint ./checkpoints_v11/best_model \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_results_detect_classify.json \
    --bf16
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

TASK_PROMPT = "<OD>"
CLASSIFY_PROMPT = "<OD>"  # same prompt for classification

CANONICAL_CATEGORIES = {
    "baby food": "Baby Food",
    "beans and legumes - canned or dried": "Beans and Legumes - Canned or Dried",
    "bread and bakery products": "Bread and Bakery Products",
    "canned tomato products": "Canned Tomato Products",
    "carbohydrate meal": "Carbohydrate Meal",
    "condiments and sauces": "Condiments and Sauces",
    "dairy and dairy alternatives": "Dairy and Dairy Alternatives",
    "desserts and sweets": "Desserts and Sweets",
    "drinks": "Drinks",
    "fresh fruit": "Fresh Fruit",
    "fruits - canned or processed": "Fruits - Canned or Processed",
    "granola products": "Granola Products",
    "meat and poultry - canned": "Meat and Poultry - Canned",
    "meat and poultry - fresh": "Meat and Poultry - Fresh",
    "nut butters and nuts": "Nut Butters and Nuts",
    "ready meals": "Ready Meals",
    "savory snacks and crackers": "Savory Snacks and Crackers",
    "seafood - canned": "Seafood - Canned",
    "soup": "Soup",
    "vegetables - canned": "Vegetables - Canned",
    "vegetables - fresh": "Vegetables - Fresh",
}

VALID_CATEGORIES = set(CANONICAL_CATEGORIES.values())


def normalize_category(name):
    return CANONICAL_CATEGORIES.get(name.lower().strip(), name)


def detect_objects_florence2(model_vanilla, processor, image, device, amp_dtype=None):
    """
    Use vanilla Florence-2 <OD> to detect objects in the image.
    Returns list of bounding boxes: [{"bbox": [x1,y1,x2,y2], "label": str}, ...]
    """
    inputs = processor(text="<OD>", images=image, return_tensors="pt").to(device)

    if amp_dtype:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            generated_ids = model_vanilla.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=True,
            )
    else:
        generated_ids = model_vanilla.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            early_stopping=True,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Parse Florence-2 OD output format
    # Florence-2 OD output: <OD> label<loc_x1><loc_y1><loc_x2><loc_y2> ...
    detections = []
    try:
        result = processor.post_process_generation(
            text, task="<OD>", image_size=(image.width, image.height)
        )
        if "<OD>" in result:
            od_result = result["<OD>"]
            bboxes = od_result.get("bboxes", [])
            labels = od_result.get("labels", [])
            for bbox, label in zip(bboxes, labels):
                detections.append({"bbox": bbox, "label": label})
    except Exception as e:
        # Fallback: try to parse raw text
        pass

    return detections


def classify_crop(model_finetuned, processor, crop_image, device, amp_dtype=None):
    """
    Classify a single cropped image using the fine-tuned model.
    Returns parsed prediction dict or None.
    """
    inputs = processor(text=CLASSIFY_PROMPT, images=crop_image, return_tensors="pt").to(device)

    if amp_dtype:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            generated_ids = model_finetuned.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                early_stopping=True,
            )
    else:
        generated_ids = model_finetuned.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3,
            early_stopping=True,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    try:
        pred = json.loads(text)
        return pred
    except json.JSONDecodeError:
        return None


def crop_with_padding(image, bbox, padding_ratio=0.1):
    """Crop image with some padding around the bbox."""
    x1, y1, x2, y2 = bbox
    w, h = image.size

    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))

    # Ensure minimum size
    if crop.width < 32 or crop.height < 32:
        return None
    return crop


def merge_predictions(crop_predictions):
    """
    Merge predictions from multiple crops into a single prediction.
    Deduplicate categories, keep highest confidence.
    """
    seen_categories = {}
    for pred in crop_predictions:
        if not pred:
            continue
        items = pred.get("items", [])
        for item in items:
            name = normalize_category(item.get("name", ""))
            if name not in VALID_CATEGORIES:
                continue
            conf = item.get("confidence", "medium")
            pkg = item.get("package_type", "unknown")

            if name not in seen_categories:
                seen_categories[name] = {"name": name, "package_type": pkg, "confidence": conf}
            else:
                # Keep higher confidence
                conf_order = {"high": 3, "medium": 2, "low": 1}
                if conf_order.get(conf, 0) > conf_order.get(seen_categories[name]["confidence"], 0):
                    seen_categories[name] = {"name": name, "package_type": pkg, "confidence": conf}

    return {"items": list(seen_categories.values())}


def evaluate_sample(image, model_vanilla, model_finetuned, processor, device, amp_dtype,
                    min_box_area=1000, max_detections=15):
    """
    Full pipeline for one image:
    1. Detect objects
    2. Crop each
    3. Classify each crop
    4. Also classify full image (fallback)
    5. Merge all predictions
    """
    # Step 1: Detect objects using vanilla Florence-2
    detections = detect_objects_florence2(model_vanilla, processor, image, device, amp_dtype)

    # Step 2 & 3: Crop and classify each detection
    crop_predictions = []
    valid_detections = 0

    for det in detections[:max_detections]:
        bbox = det["bbox"]
        # Filter tiny boxes
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < min_box_area:
            continue

        crop = crop_with_padding(image, bbox)
        if crop is None:
            continue

        valid_detections += 1
        pred = classify_crop(model_finetuned, processor, crop, device, amp_dtype)
        if pred:
            crop_predictions.append(pred)

    # Step 4: Always also classify the full image as fallback
    full_pred = classify_crop(model_finetuned, processor, image, device, amp_dtype)
    if full_pred:
        crop_predictions.append(full_pred)

    # Step 5: Merge all predictions
    merged = merge_predictions(crop_predictions)

    return merged, len(detections), valid_detections


def compute_metrics(all_targets, all_predictions):
    """Compute precision, recall, F1 (micro and macro)."""
    # Per-class TP, FP, FN
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()

    for target_classes, pred_classes in zip(all_targets, all_predictions):
        for cls in pred_classes:
            if cls in target_classes:
                class_tp[cls] += 1
            else:
                class_fp[cls] += 1
        for cls in target_classes:
            if cls not in pred_classes:
                class_fn[cls] += 1

    all_classes = sorted(set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys())))

    # Micro
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    # Macro
    per_class = {}
    macro_p_sum, macro_r_sum, macro_f1_sum = 0, 0, 0
    for cls in all_classes:
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_class[cls] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}
        macro_p_sum += p
        macro_r_sum += r
        macro_f1_sum += f1

    n_classes = len(all_classes) if all_classes else 1
    macro_p = macro_p_sum / n_classes
    macro_r = macro_r_sum / n_classes
    macro_f1 = macro_f1_sum / n_classes

    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Detection-first classification pipeline")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--output", type=str, default="./eval_results_detect_classify.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-box-area", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    amp_dtype = None
    if args.bf16 and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("Using bf16")
    elif args.fp16:
        amp_dtype = torch.float16
        print("Using fp16")

    # Load vanilla Florence-2 for detection
    print(f"\nLoading vanilla Florence-2 for detection: {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model_vanilla = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    print("  Vanilla model loaded ✓")

    # Load fine-tuned Florence-2 for classification
    print(f"Loading fine-tuned model for classification: {args.checkpoint}...")
    model_finetuned = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32,
        attn_implementation="eager",
    )
    from peft import PeftModel
    model_finetuned = PeftModel.from_pretrained(model_finetuned, args.checkpoint)
    model_finetuned = model_finetuned.merge_and_unload()
    model_finetuned.to(device).eval()
    print("  Fine-tuned model loaded ✓")

    # Load test data
    print(f"\nLoading test data from {args.jsonl}...")
    samples = []
    with open(args.jsonl, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Evaluating {len(samples)} samples...")

    # Evaluate
    all_targets = []
    all_predictions = []
    results = []
    total_detections = 0
    total_valid = 0

    start = time.time()
    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to load {img_path}: {e}")
            continue

        # Get target classes
        target = json.loads(sample["target"])
        target_classes = {normalize_category(item["name"]) for item in target.get("items", [])}

        # Run pipeline
        with torch.no_grad():
            merged, n_det, n_valid = evaluate_sample(
                image, model_vanilla, model_finetuned, processor, device, amp_dtype,
                min_box_area=args.min_box_area,
            )

        pred_classes = {normalize_category(item["name"]) for item in merged.get("items", [])
                        if normalize_category(item["name"]) in VALID_CATEGORIES}

        all_targets.append(target_classes)
        all_predictions.append(pred_classes)
        total_detections += n_det
        total_valid += n_valid

        results.append({
            "image": img_rel,
            "target": sample["target"],
            "predicted": json.dumps(merged),
            "n_detections": n_det,
            "n_valid_crops": n_valid,
            "target_classes": sorted(target_classes),
            "pred_classes": sorted(pred_classes),
            "match": target_classes == pred_classes,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(samples)} ({rate:.1f} img/s, ETA: {eta:.0f}s)")

    elapsed = time.time() - start

    # Compute metrics
    metrics = compute_metrics(all_targets, all_predictions)
    exact_match = sum(1 for r in results if r["match"]) / len(results) if results else 0

    # Print report
    print(f"\n{'='*70}")
    print("DETECTION-FIRST PIPELINE EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"\n  Total Samples: {len(results)}")
    print(f"  Time: {elapsed:.1f}s ({len(results)/elapsed:.1f} img/s)")
    print(f"  Avg detections per image: {total_detections/len(results):.1f}")
    print(f"  Avg valid crops per image: {total_valid/len(results):.1f}")
    print(f"  Exact Match Accuracy: {100*exact_match:.1f}%")
    print(f"\n  Micro-Averaged:")
    print(f"    Precision: {100*metrics['micro']['precision']:.1f}%")
    print(f"    Recall:    {100*metrics['micro']['recall']:.1f}%")
    print(f"    F1:        {100*metrics['micro']['f1']:.1f}%")
    print(f"\n  Macro-Averaged:")
    print(f"    Precision: {100*metrics['macro']['precision']:.1f}%")
    print(f"    Recall:    {100*metrics['macro']['recall']:.1f}%")
    print(f"    F1:        {100*metrics['macro']['f1']:.1f}%")

    print(f"\n{'='*70}")
    print("PER-CLASS METRICS")
    print(f"{'='*70}")
    print(f"  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for cls in sorted(metrics["per_class"].keys()):
        m = metrics["per_class"][cls]
        print(f"  {cls:<45} {100*m['precision']:5.1f}% {100*m['recall']:5.1f}% {100*m['f1']:5.1f}% {m['support']:>7}")

    # Save
    output = {
        "pipeline": "detect_then_classify",
        "checkpoint": args.checkpoint,
        "n_samples": len(results),
        "metrics": metrics,
        "exact_match": exact_match,
        "avg_detections": total_detections / len(results) if results else 0,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
