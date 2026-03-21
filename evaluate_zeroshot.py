#!/usr/bin/env python3
"""
evaluate_zeroshot.py
====================
Zero-shot baseline evaluation of vanilla Florence-2 on the food pantry test set.

Since a vanilla model doesn't know our custom output format, we use two approaches:
  1. <OD> (object detection) — Florence-2 native task, outputs bounding box labels
  2. <DENSE_REGION_CAPTION> — detailed region captions
  3. <MORE_DETAILED_CAPTION> — free-form image description

For each, we fuzzy-match detected labels to our 25 food categories and compute metrics.

Usage:
  python evaluate_zeroshot.py \
    --base-model microsoft/Florence-2-large-ft \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_results_zeroshot.json \
    --bf16
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


# ── Our food categories ────────────────────────────────────────────────────────

CATEGORIES = [
    "Baby Food",
    "Beans and Legumes - Canned or Dried",
    "Bread and Bakery Products",
    "Canned Tomato Products",
    "Carbohydrate Meal",
    "Condiments and Sauces",
    "Dairy and Dairy Alternatives",
    "Desserts and Sweets",
    "Drinks",
    "Fresh Fruit",
    "Fruits - Canned or Processed",
    "Granola Products",
    "Meat and Poultry - Canned",
    "Meat and Poultry - Fresh",
    "Nut Butters and Nuts",
    "Ready Meals",
    "Savory Snacks and Crackers",
    "Seafood - Canned",
    "Soup",
    "Vegetables - Canned",
    "Vegetables - Fresh",
]

# ── Keyword mapping: fuzzy terms → our 21 categories ──────────────────────────
# Each category gets a list of keywords/phrases that, if found in Florence-2 output,
# would indicate that category is present.

KEYWORD_MAP = {
    "Baby Food": ["baby food", "infant food", "baby formula", "gerber", "baby cereal",
                   "baby jar"],
    "Beans and Legumes - Canned or Dried": [
        "beans", "legumes", "lentils", "chickpeas", "black beans",
        "kidney beans", "pinto beans", "canned beans", "dried beans",
        "navy beans", "great northern", "lima beans", "garbanzo"],
    "Bread and Bakery Products": [
        "bread", "bakery", "loaf", "baguette", "roll", "bun",
        "muffin", "croissant", "tortilla", "pita", "flatbread", "bagel"],
    "Canned Tomato Products": [
        "canned tomato", "tomato sauce", "tomato paste", "diced tomatoes",
        "crushed tomatoes", "tomato puree", "stewed tomatoes", "marinara",
        "tomato can"],
    "Carbohydrate Meal": [
        "pasta", "noodle", "spaghetti", "macaroni", "penne", "ramen",
        "linguine", "fettuccine", "lasagna", "rice", "grain", "quinoa",
        "couscous", "barley", "bag of rice", "box of rice", "mac and cheese",
        "instant noodle"],
    "Condiments and Sauces": [
        "condiment", "sauce", "ketchup", "mustard", "mayonnaise",
        "soy sauce", "hot sauce", "salsa", "bbq sauce", "dressing",
        "vinegar", "relish", "worcestershire"],
    "Dairy and Dairy Alternatives": [
        "dairy", "milk", "cheese", "yogurt", "butter", "cream",
        "sour cream", "cottage cheese", "cream cheese", "oat milk",
        "almond milk", "soy milk", "eggs", "egg carton", "carton of eggs"],
    "Desserts and Sweets": [
        "dessert", "sweet", "candy", "chocolate", "cookie", "cookies",
        "cake", "brownie", "pudding", "jello", "ice cream", "pie",
        "frosting", "marshmallow", "sugar", "syrup", "honey"],
    "Drinks": [
        "drink", "beverage", "juice", "soda", "water bottle",
        "tea", "coffee", "lemonade", "gatorade", "bottle of water",
        "can of soda", "energy drink", "bottle of juice", "milk jug"],
    "Fresh Fruit": [
        "apple", "banana", "orange", "grape", "strawberry", "blueberry",
        "pear", "peach", "plum", "watermelon", "melon", "mango", "kiwi",
        "cherry", "lemon", "lime", "fresh fruit", "fruit basket",
        "pineapple", "avocado"],
    "Fruits - Canned or Processed": [
        "canned fruit", "fruit cup", "applesauce", "fruit cocktail",
        "canned peach", "canned pear", "fruit preserves", "jam", "jelly",
        "dried fruit", "raisins", "cranberries dried"],
    "Granola Products": [
        "granola", "granola bar", "cereal", "oatmeal", "breakfast bar",
        "energy bar", "protein bar", "cheerios", "cornflakes", "muesli",
        "cereal box"],
    "Meat and Poultry - Canned": [
        "canned meat", "spam", "canned chicken", "canned tuna",
        "vienna sausage", "canned ham", "potted meat", "corned beef",
        "canned turkey"],
    "Meat and Poultry - Fresh": [
        "meat", "chicken", "beef", "pork", "turkey", "steak",
        "ground beef", "chicken breast", "sausage", "bacon",
        "ham", "hot dog", "poultry", "lamb", "ribs"],
    "Nut Butters and Nuts": [
        "peanut butter", "almond butter", "nut butter", "nutella",
        "cashew butter", "sunflower butter", "peanuts", "almonds",
        "cashews", "walnuts", "mixed nuts", "pecans", "jar of peanut"],
    "Ready Meals": [
        "ready meal", "frozen meal", "tv dinner", "microwave meal",
        "frozen dinner", "lean cuisine", "hungry man", "stouffer",
        "frozen pizza", "pizza", "burrito", "frozen entree",
        "meal kit", "canned meal"],
    "Savory Snacks and Crackers": [
        "snack", "cracker", "chips", "pretzel", "popcorn",
        "cheese crackers", "goldfish", "tortilla chips",
        "potato chips", "doritos", "cheetos", "lays"],
    "Seafood - Canned": [
        "seafood", "fish", "shrimp", "tuna", "salmon", "crab",
        "sardine", "canned tuna", "canned salmon", "canned fish",
        "anchovies", "clam"],
    "Soup": [
        "soup", "broth", "stew", "chili", "canned soup", "chicken soup",
        "tomato soup", "noodle soup", "campbell", "progresso"],
    "Vegetables - Canned": [
        "canned vegetable", "canned corn", "canned peas",
        "canned green beans", "canned carrots", "canned beets",
        "canned spinach", "canned mixed vegetables"],
    "Vegetables - Fresh": [
        "lettuce", "onion", "potato", "carrot", "broccoli", "celery",
        "cucumber", "spinach", "cabbage", "zucchini", "squash",
        "mushroom", "garlic", "ginger", "corn on the cob",
        "fresh vegetable", "bell pepper", "tomato", "green beans"],
}


def match_categories(text: str) -> Set[str]:
    """
    Given free-form text from Florence-2, find which of our categories are mentioned.
    Uses keyword matching (case-insensitive).
    """
    text_lower = text.lower()
    matched = set()
    for category, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                matched.add(category)
                break
    return matched


def parse_florence2_od(text: str) -> List[str]:
    """
    Parse Florence-2 <OD> output format.
    Typical format: "label1<loc_x1><loc_y1><loc_x2><loc_y2>label2<loc_...>..."
    Returns list of detected labels.
    """
    # Remove location tokens
    cleaned = re.sub(r'<loc_\d+>', '', text)
    # Split on common separators
    labels = [l.strip() for l in cleaned.split('\n') if l.strip()]
    if len(labels) <= 1:
        # Try splitting differently — sometimes all on one line
        labels = [l.strip() for l in re.split(r'[,;]', cleaned) if l.strip()]
    return labels


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, processor, image_path, device, prompt="<OD>", amp_dtype=None):
    """Run inference on a single image and return generated text."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    if amp_dtype is not None:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=True,
            )
    else:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            early_stopping=True,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


# ── Metrics (reused logic) ─────────────────────────────────────────────────────

def extract_class_set_from_target(target_str: str) -> Set[str]:
    """Extract set of class names from ground-truth JSON string."""
    try:
        parsed = json.loads(target_str)
        return {item["name"] for item in parsed.get("items", []) if "name" in item}
    except (json.JSONDecodeError, KeyError):
        return set()


def compute_metrics(results: List[dict]) -> dict:
    """Compute P/R/F1 metrics from prediction results."""
    n = len(results)
    if n == 0:
        return {"error": "No predictions"}

    all_classes = set()
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()
    category_exact = 0

    for rec in results:
        target_classes = rec["target_classes"]
        pred_classes = rec["pred_classes"]
        all_classes.update(target_classes)
        all_classes.update(pred_classes)

        if target_classes == pred_classes:
            category_exact += 1

        for cls in target_classes & pred_classes:
            class_tp[cls] += 1
        for cls in pred_classes - target_classes:
            class_fp[cls] += 1
        for cls in target_classes - pred_classes:
            class_fn[cls] += 1

    # Per-class
    per_class = {}
    for cls in sorted(all_classes):
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[cls] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "support": tp + fn,
        }

    # Micro
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # Macro
    macro_p = sum(m["precision"] for m in per_class.values()) / len(per_class) if per_class else 0.0
    macro_r = sum(m["recall"] for m in per_class.values()) / len(per_class) if per_class else 0.0
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(per_class) if per_class else 0.0

    return {
        "total_samples": n,
        "category_detection_accuracy": round(category_exact / n, 4),
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro": {"precision": round(macro_p, 4), "recall": round(macro_r, 4), "f1": round(macro_f1, 4)},
        "per_class": per_class,
    }


def print_report(metrics: dict, prompt_name: str):
    print(f"\n{'='*70}")
    print(f"ZERO-SHOT EVALUATION — Prompt: {prompt_name}")
    print(f"{'='*70}")
    print(f"\n  Total Samples:                {metrics['total_samples']}")
    print(f"  Category Detection Accuracy:  {metrics['category_detection_accuracy']*100:.1f}%")
    print(f"\n  Micro-Averaged:")
    print(f"    Precision: {metrics['micro']['precision']*100:.1f}%")
    print(f"    Recall:    {metrics['micro']['recall']*100:.1f}%")
    print(f"    F1:        {metrics['micro']['f1']*100:.1f}%")
    print(f"\n  Macro-Averaged:")
    print(f"    Precision: {metrics['macro']['precision']*100:.1f}%")
    print(f"    Recall:    {metrics['macro']['recall']*100:.1f}%")
    print(f"    F1:        {metrics['macro']['f1']*100:.1f}%")

    print(f"\n  {'Class':<40s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>7s}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for cls, m in sorted(metrics["per_class"].items()):
        print(f"  {cls:<40s} {m['precision']*100:5.1f}% {m['recall']*100:5.1f}% "
              f"{m['f1']*100:5.1f}% {m['support']:>7d}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    print(f"  {'MICRO AVG':<40s} {metrics['micro']['precision']*100:5.1f}% "
          f"{metrics['micro']['recall']*100:5.1f}% {metrics['micro']['f1']*100:5.1f}%")
    print(f"  {'MACRO AVG':<40s} {metrics['macro']['precision']*100:5.1f}% "
          f"{metrics['macro']['recall']*100:5.1f}% {metrics['macro']['f1']*100:5.1f}%")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Zero-shot Florence-2 baseline evaluation.")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft",
                        help="Base model name (no LoRA checkpoint loaded).")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Root data directory with image folders.")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl",
                        help="Path to JSONL file (uses same test set for fair comparison).")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON results.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for quick testing.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16.")
    parser.add_argument("--show-predictions", type=int, default=10,
                        help="Number of sample predictions to display.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    amp_dtype = None
    if args.bf16:
        amp_dtype = torch.bfloat16
        print("Using bf16")
    elif args.fp16:
        amp_dtype = torch.float16
        print("Using fp16")

    # ── Load vanilla model (NO LoRA) ───────────────────────────────────────
    print(f"\nLoading vanilla model: {args.base_model} (NO fine-tuning)...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=amp_dtype if amp_dtype else torch.float32,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    print("  Model loaded (vanilla, no LoRA).")

    # ── Load test data ─────────────────────────────────────────────────────
    print(f"\nLoading test data from {args.jsonl}...")
    samples = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Evaluating {len(samples)} samples...")

    # ── Prompts to test ────────────────────────────────────────────────────
    prompts = {
        "<OD>": "Object Detection",
        "<MORE_DETAILED_CAPTION>": "Detailed Caption",
    }

    all_metrics = {}

    for prompt_token, prompt_name in prompts.items():
        print(f"\n{'='*70}")
        print(f"Running inference with prompt: {prompt_token} ({prompt_name})")
        print(f"{'='*70}")

        results = []
        start_time = time.time()

        for i, sample in enumerate(samples):
            img_rel = sample["image"].replace("\\", "/")
            img_path = os.path.join(args.data_dir, img_rel)

            if not os.path.exists(img_path):
                print(f"  [SKIP] Image not found: {img_path}", file=sys.stderr)
                continue

            pred_text = run_inference(model, processor, img_path, device,
                                     prompt=prompt_token, amp_dtype=amp_dtype)

            target_classes = extract_class_set_from_target(sample["target"])
            pred_classes = match_categories(pred_text)

            results.append({
                "image": sample["image"],
                "target_classes": target_classes,
                "pred_classes": pred_classes,
                "raw_output": pred_text,
                "target_text": sample["target"],
            })

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(samples) - i - 1) / rate
                print(f"  Progress: {i+1}/{len(samples)} ({rate:.1f} img/s, ETA: {eta:.0f}s)")

        elapsed = time.time() - start_time
        print(f"\n  Inference complete: {len(results)} images in {elapsed:.1f}s "
              f"({len(results)/elapsed:.1f} img/s)")

        # Show sample predictions
        if args.show_predictions > 0:
            print(f"\n  Sample predictions ({prompt_name}):")
            for rec in results[:args.show_predictions]:
                print(f"\n    Image: {rec['image']}")
                print(f"    Raw output: {rec['raw_output'][:200]}...")
                print(f"    Target categories: {rec['target_classes']}")
                print(f"    Matched categories: {rec['pred_classes']}")

        metrics = compute_metrics(results)
        print_report(metrics, prompt_name)
        all_metrics[prompt_token] = {
            "prompt_name": prompt_name,
            "metrics": metrics,
            "predictions": [
                {
                    "image": r["image"],
                    "target": list(r["target_classes"]),
                    "predicted": list(r["pred_classes"]),
                    "raw_output": r["raw_output"],
                }
                for r in results
            ],
        }

    # ── Summary comparison ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ZERO-SHOT BASELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Prompt':<30s} {'Cat Acc':>8s} {'Micro F1':>9s} {'Macro F1':>9s}")
    print(f"  {'-'*30} {'-'*8} {'-'*9} {'-'*9}")
    for prompt_token, data in all_metrics.items():
        m = data["metrics"]
        print(f"  {data['prompt_name']:<30s} "
              f"{m['category_detection_accuracy']*100:7.1f}% "
              f"{m['micro']['f1']*100:8.1f}% "
              f"{m['macro']['f1']*100:8.1f}%")
    print()
    print("NOTE: These are zero-shot baselines (no fine-tuning).")
    print("Categories are detected via keyword matching on model free-text output.")
    print("This gives an upper bound on what keyword matching can extract from vanilla Florence-2.")

    # ── Save ───────────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
