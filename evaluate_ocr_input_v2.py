#!/usr/bin/env python3
"""
evaluate_ocr_input_v2.py
========================
Fixed OCR-input experiment: Two-stage approach.

Problem with v1: Florence-2 processor asserts task token must be the ONLY text.
Cannot concat OCR text to <OD> prompt.

V2 Strategy:
  1. Run vanilla Florence-2 <OCR> → extract text from image
  2. Run fine-tuned Florence-2 <OD> → get image-only classification
  3. Use OCR text to RE-RANK or CORRECT predictions via text similarity matching
  
  This is "OCR as a post-classification correction signal" —
  different from the failed OCR-boost (which tried to override low-confidence preds).
  Here we use OCR to ADD missing classes by checking if OCR text matches known products.

Comparison:
  - Image-only (v11 baseline): Just run classifier
  - OCR-corrected: Run classifier + use OCR to catch missed items

Usage:
  python evaluate_ocr_input_v2.py \
    --base-model microsoft/Florence-2-large-ft \
    --checkpoint ./checkpoints_v11/best_model \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_results_ocr_input_v2.json \
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


# ── OCR Text → Category Mapping ────────────────────────────────────────────────
# Keywords that suggest a category when found in OCR text
# This is the "text-aware cue" that Prof Zheng suggested

OCR_CATEGORY_KEYWORDS = {
    "Baby Food": ["baby", "gerber", "infant", "toddler", "beech-nut"],
    "Beans and Legumes - Canned or Dried": [
        "beans", "lentils", "chickpea", "kidney", "black bean", "pinto",
        "garbanzo", "navy bean", "refried", "goya"
    ],
    "Bread and Bakery Products": [
        "bread", "bagel", "muffin", "roll", "bun", "tortilla", "pita",
        "croissant", "biscuit", "loaf", "bakery", "wonder", "sara lee"
    ],
    "Canned Tomato Products": [
        "tomato", "marinara", "salsa", "pasta sauce", "hunt's",
        "ro-tel", "rotel", "diced tomato", "crushed tomato", "tomato paste",
        "tomato sauce", "contadina", "muir glen"
    ],
    "Carbohydrate Meal": [
        "pasta", "rice", "noodle", "macaroni", "spaghetti", "penne",
        "ramen", "rice-a-roni", "hamburger helper", "velveeta",
        "kraft dinner", "uncle ben", "minute rice", "barilla",
        "mac & cheese", "mac and cheese", "fettuccine", "linguine"
    ],
    "Condiments and Sauces": [
        "ketchup", "mustard", "mayo", "mayonnaise", "sauce", "dressing",
        "vinegar", "soy sauce", "hot sauce", "bbq", "barbecue",
        "worcestershire", "tabasco", "heinz", "french's", "ranch"
    ],
    "Dairy and Dairy Alternatives": [
        "milk", "cheese", "yogurt", "butter", "cream", "almond milk",
        "oat milk", "soy milk", "lactose", "dairy", "kraft singles",
        "velveeta", "horizon", "silk", "chobani"
    ],
    "Desserts and Sweets": [
        "cake", "cookie", "brownie", "candy", "chocolate", "sugar",
        "frosting", "pudding", "jello", "gummy", "snack cake",
        "little debbie", "hostess", "oreo", "chips ahoy"
    ],
    "Drinks": [
        "juice", "water", "soda", "tea", "coffee", "drink", "beverage",
        "lemonade", "gatorade", "kool-aid", "capri sun", "tropicana"
    ],
    "Fresh Fruit": [
        "apple", "banana", "orange", "grape", "strawberry", "blueberry",
        "fresh fruit", "organic fruit", "pear", "peach", "melon"
    ],
    "Fruits - Canned or Processed": [
        "fruit cocktail", "mandarin", "pineapple", "peach", "pear",
        "applesauce", "dole", "del monte", "fruit cup"
    ],
    "Granola Products": [
        "granola", "oat", "nature valley", "quaker", "clif bar",
        "kind bar", "trail mix", "muesli", "granola bar"
    ],
    "Meat and Poultry - Canned": [
        "spam", "vienna sausage", "canned chicken", "canned meat",
        "corned beef", "potted meat", "libby", "armour", "hormel"
    ],
    "Meat and Poultry - Fresh": [
        "chicken", "beef", "pork", "turkey", "ground meat", "steak",
        "breast", "thigh", "drumstick", "sausage", "bacon",
        "tyson", "perdue", "oscar mayer"
    ],
    "Nut Butters and Nuts": [
        "peanut butter", "almond butter", "cashew", "nut", "jif",
        "skippy", "peter pan", "planters", "mixed nuts"
    ],
    "Ready Meals": [
        "ready to eat", "microwaveable", "frozen dinner", "lean cuisine",
        "stouffer", "hungry man", "banquet", "marie callender",
        "hot pocket", "chef boyardee", "ravioli"
    ],
    "Savory Snacks and Crackers": [
        "chips", "crackers", "pretzels", "popcorn", "cheez-it",
        "goldfish", "ritz", "triscuit", "doritos", "lays", "cheetos",
        "pringles", "wheat thins", "saltine"
    ],
    "Seafood - Canned": [
        "tuna", "salmon", "sardine", "anchovy", "crab", "clam",
        "starkist", "bumble bee", "chicken of the sea", "seafood"
    ],
    "Soup": [
        "soup", "broth", "chowder", "stew", "campbell", "progresso",
        "chunky", "ramen", "cup of noodles", "chicken noodle"
    ],
    "Vegetables - Canned": [
        "corn", "peas", "green bean", "carrots", "mixed vegetables",
        "del monte", "green giant", "libby", "canned vegetable"
    ],
    "Vegetables - Fresh": [
        "lettuce", "spinach", "kale", "broccoli", "celery", "onion",
        "potato", "tomato", "pepper", "cucumber", "fresh vegetable"
    ],
}


def extract_ocr_text(model, processor, image, device, amp_dtype=None):
    """Extract text from image using vanilla Florence-2 <OCR> task."""
    inputs = processor(text="<OCR>", images=image, return_tensors="pt").to(device)
    
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,
        num_beams=3,
        early_stopping=True,
    )
    
    if amp_dtype:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            generated_ids = model.generate(**gen_kwargs)
    else:
        generated_ids = model.generate(**gen_kwargs)
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text


def classify_image_only(model, processor, image, device, amp_dtype=None):
    """Standard image-only classification using fine-tuned model."""
    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
    
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,
        num_beams=3,
        early_stopping=True,
    )
    
    if amp_dtype:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            generated_ids = model.generate(**gen_kwargs)
    else:
        generated_ids = model.generate(**gen_kwargs)
    
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    try:
        pred = json.loads(text)
        return pred, True
    except json.JSONDecodeError:
        return {"items": []}, False


def ocr_suggest_categories(ocr_text, existing_predictions, threshold=2):
    """
    Analyze OCR text and suggest additional categories not already predicted.
    
    Args:
        ocr_text: Raw OCR output from the image
        existing_predictions: Set of categories already predicted by classifier
        threshold: Minimum number of keyword matches to suggest a category
    
    Returns:
        List of (category, confidence_score, matched_keywords) tuples
    """
    if not ocr_text:
        return []
    
    ocr_lower = ocr_text.lower()
    suggestions = []
    
    for category, keywords in OCR_CATEGORY_KEYWORDS.items():
        if category in existing_predictions:
            continue  # Already predicted, skip
        
        matched = []
        for kw in keywords:
            # Check for keyword in OCR text (word boundary aware)
            if kw.lower() in ocr_lower:
                matched.append(kw)
        
        if len(matched) >= 1:
            # Score: more matches = higher confidence
            score = min(len(matched) / 3.0, 1.0)  # Normalize to 0-1
            suggestions.append((category, score, matched))
    
    # Sort by score descending
    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions


def merge_predictions(image_pred, ocr_suggestions, min_ocr_score=0.3):
    """
    Merge image-only predictions with OCR-suggested categories.
    
    Only add OCR suggestions that meet the minimum score threshold.
    """
    items = list(image_pred.get("items", []))
    existing = {normalize_category(item.get("name", "")) for item in items}
    
    for category, score, keywords in ocr_suggestions:
        if score >= min_ocr_score and category not in existing:
            items.append({
                "name": category,
                "package_type": "unknown",
                "confidence": "ocr-suggested",
                "ocr_keywords": keywords,
                "ocr_score": round(score, 2),
            })
            existing.add(category)
    
    return {"items": items}


def compute_metrics(all_targets, all_preds):
    """Compute micro/macro precision, recall, F1 and per-class metrics."""
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()
    class_support = Counter()
    exact_match = 0
    
    for target_items, pred_items in zip(all_targets, all_preds):
        target_set = {normalize_category(it.get("name", "")) for it in target_items}
        pred_set = {normalize_category(it.get("name", "")) for it in pred_items}
        
        # Filter to valid categories
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
    
    # Micro
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    micro_p = total_tp / max(total_tp + total_fp, 1)
    micro_r = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    
    # Macro
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
    macro_p = sum(precisions) / max(len(precisions), 1)
    macro_r = sum(recalls) / max(len(recalls), 1)
    macro_f1 = sum(f1s) / max(len(f1s), 1)
    
    return {
        "exact_match": exact_match / max(n, 1),
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "per_class": per_class,
    }


def print_report(title, metrics, n_samples):
    """Pretty-print evaluation report."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Samples: {n_samples}")
    print(f"  Exact Match: {metrics['exact_match']:.1%}")
    print(f"\n  Micro:  P={metrics['micro']['precision']:.1%}  R={metrics['micro']['recall']:.1%}  F1={metrics['micro']['f1']:.1%}")
    print(f"  Macro:  P={metrics['macro']['precision']:.1%}  R={metrics['macro']['recall']:.1%}  F1={metrics['macro']['f1']:.1%}")
    
    print(f"\n  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    
    per_class = metrics["per_class"]
    for cls in sorted(per_class.keys()):
        m = per_class[cls]
        print(f"  {cls:<45} {m['precision']:>5.1%} {m['recall']:>5.1%} {m['f1']:>5.1%} {m['support']:>8}")


def main():
    parser = argparse.ArgumentParser(description="OCR-input experiment v2 (two-stage)")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--output", type=str, default="./eval_results_ocr_input_v2.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--min-ocr-score", type=float, default=0.3,
                        help="Minimum OCR keyword score to add a suggestion")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Device: {device}")
    if amp_dtype:
        print(f"Using bf16")
    
    # Load vanilla Florence-2 for OCR
    print(f"\nLoading vanilla Florence-2 for OCR: {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model_vanilla = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device)
    model_vanilla.eval()
    print("  Vanilla model loaded ✓")
    
    # Load fine-tuned model for classification
    print(f"Loading fine-tuned model: {args.checkpoint}...")
    model_finetuned = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32
    )
    model_finetuned = PeftModel.from_pretrained(model_finetuned, args.checkpoint)
    model_finetuned = model_finetuned.merge_and_unload()
    model_finetuned = model_finetuned.to(device)
    model_finetuned.eval()
    print("  Fine-tuned model loaded ✓")
    
    # Load test data
    print(f"\nLoading test data from {args.jsonl}...")
    samples = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Evaluating {len(samples)} samples...")
    
    # Track results for both modes
    targets_all = []
    preds_image_only = []
    preds_ocr_corrected = []
    ocr_additions = 0
    ocr_texts_log = []
    
    t0 = time.time()
    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to load {img_path}: {e}")
            continue
        
        # Parse target
        target = json.loads(sample["target"])
        target_items = target.get("items", [])
        targets_all.append(target_items)
        
        # Step 1: Image-only classification
        pred_img, _ = classify_image_only(model_finetuned, processor, image, device, amp_dtype)
        preds_image_only.append(pred_img.get("items", []))
        
        # Step 2: OCR text extraction
        with torch.no_grad():
            ocr_text = extract_ocr_text(model_vanilla, processor, image, device, amp_dtype)
        
        # Step 3: OCR-based category suggestions
        existing_preds = {normalize_category(it.get("name", "")) for it in pred_img.get("items", [])}
        ocr_suggestions = ocr_suggest_categories(ocr_text, existing_preds)
        
        # Step 4: Merge
        merged = merge_predictions(pred_img, ocr_suggestions, min_ocr_score=args.min_ocr_score)
        preds_ocr_corrected.append(merged.get("items", []))
        
        if ocr_suggestions:
            ocr_additions += len([s for s in ocr_suggestions if s[1] >= args.min_ocr_score])
        
        ocr_texts_log.append({
            "image": img_rel,
            "ocr_text": ocr_text[:300],
            "image_pred": [it.get("name") for it in pred_img.get("items", [])],
            "ocr_suggestions": [(s[0], s[1], s[2]) for s in ocr_suggestions],
            "merged_pred": [it.get("name") for it in merged.get("items", [])],
        })
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(samples)} ({rate:.1f} img/s, ETA: {eta:.0f}s)")
    
    elapsed = time.time() - t0
    print(f"\n  Inference complete: {len(targets_all)} images in {elapsed:.1f}s")
    
    # Compute metrics for both modes
    metrics_img = compute_metrics(targets_all, preds_image_only)
    metrics_ocr = compute_metrics(targets_all, preds_ocr_corrected)
    
    # Print comparison
    print_report("IMAGE-ONLY CLASSIFICATION (v11 baseline)", metrics_img, len(targets_all))
    print_report("OCR-CORRECTED CLASSIFICATION (image + OCR keywords)", metrics_ocr, len(targets_all))
    
    # Print delta
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: OCR-Corrected vs Image-Only")
    print(f"{'=' * 70}")
    d_em = metrics_ocr["exact_match"] - metrics_img["exact_match"]
    d_f1 = metrics_ocr["micro"]["f1"] - metrics_img["micro"]["f1"]
    d_p = metrics_ocr["micro"]["precision"] - metrics_img["micro"]["precision"]
    d_r = metrics_ocr["micro"]["recall"] - metrics_img["micro"]["recall"]
    print(f"  Exact Match: {d_em:+.1%}")
    print(f"  Micro P:     {d_p:+.1%}")
    print(f"  Micro R:     {d_r:+.1%}")
    print(f"  Micro F1:    {d_f1:+.1%}")
    print(f"  OCR additions: {ocr_additions} categories added across {len(targets_all)} images")
    
    # Show some OCR examples
    print(f"\n{'=' * 70}")
    print(f"  OCR TEXT EXAMPLES (first 10 with suggestions)")
    print(f"{'=' * 70}")
    shown = 0
    for log in ocr_texts_log:
        if log["ocr_suggestions"] and shown < 10:
            print(f"\n  Image: {os.path.basename(log['image'])}")
            print(f"  OCR: {log['ocr_text'][:150]}...")
            print(f"  Image pred: {log['image_pred']}")
            print(f"  OCR suggests: {[(s[0], f'{s[1]:.1f}', s[2]) for s in log['ocr_suggestions']]}")
            print(f"  Merged: {log['merged_pred']}")
            shown += 1
    
    # Per-class comparison
    print(f"\n{'=' * 70}")
    print(f"  PER-CLASS DELTA (OCR-corrected minus Image-only)")
    print(f"{'=' * 70}")
    print(f"  {'Class':<45} {'ΔPrec':>7} {'ΔRec':>7} {'ΔF1':>7}")
    print(f"  {'-'*45} {'-'*7} {'-'*7} {'-'*7}")
    
    all_classes = sorted(set(list(metrics_img["per_class"].keys()) + list(metrics_ocr["per_class"].keys())))
    for cls in all_classes:
        img_m = metrics_img["per_class"].get(cls, {"precision": 0, "recall": 0, "f1": 0})
        ocr_m = metrics_ocr["per_class"].get(cls, {"precision": 0, "recall": 0, "f1": 0})
        dp = ocr_m["precision"] - img_m["precision"]
        dr = ocr_m["recall"] - img_m["recall"]
        df = ocr_m["f1"] - img_m["f1"]
        marker = " ⬆" if df > 0.01 else (" ⬇" if df < -0.01 else "")
        print(f"  {cls:<45} {dp:>+6.1%} {dr:>+6.1%} {df:>+6.1%}{marker}")
    
    # Save results
    results = {
        "image_only": {
            "exact_match": metrics_img["exact_match"],
            "micro": metrics_img["micro"],
            "macro": metrics_img["macro"],
            "per_class": metrics_img["per_class"],
        },
        "ocr_corrected": {
            "exact_match": metrics_ocr["exact_match"],
            "micro": metrics_ocr["micro"],
            "macro": metrics_ocr["macro"],
            "per_class": metrics_ocr["per_class"],
        },
        "ocr_additions": ocr_additions,
        "ocr_log": ocr_texts_log[:20],
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
