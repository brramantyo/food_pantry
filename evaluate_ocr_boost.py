#!/usr/bin/env python3
"""
evaluate_ocr_boost.py
=====================
Post-processing boost: Run Florence-2 OCR on test images and use detected text
to correct/boost classification predictions from v11.

Strategy:
  1. Load v11 predictions (already computed)
  2. Run Florence-2 OCR on each test image
  3. Match OCR text against category keywords
  4. If OCR strongly suggests a category that v11 missed → add it
  5. If OCR contradicts v11's prediction → flag but don't remove (precision > recall)

Zero training cost — just inference + post-processing.
"""

import json
import os
import re
import sys
import time
import argparse
from collections import Counter

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ── OCR keyword → category mapping ────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "Baby Food": [
        "baby", "infant", "gerber", "beech-nut", "toddler", "puree",
    ],
    "Beans and Legumes - Canned or Dried": [
        "beans", "bean", "lentil", "chickpea", "garbanzo", "kidney", "pinto",
        "black bean", "navy bean", "great northern",
    ],
    "Bread and Bakery Products": [
        "bread", "bagel", "tortilla", "muffin", "bun", "roll", "bakery",
        "loaf", "wheat bread", "white bread", "sourdough",
    ],
    "Canned Tomato Products": [
        "tomato sauce", "tomato paste", "diced tomato", "crushed tomato",
        "tomatoes", "marinara", "tomato puree", "stewed tomato",
    ],
    "Carbohydrate Meal": [
        "pasta", "rice", "noodle", "macaroni", "spaghetti", "penne",
        "linguine", "fettuccine", "ramen", "couscous", "quinoa",
    ],
    "Condiments and Sauces": [
        "ketchup", "mustard", "mayonnaise", "mayo", "soy sauce", "hot sauce",
        "bbq sauce", "barbecue", "salad dressing", "vinegar", "sriracha",
        "worcestershire", "tabasco", "ranch", "seasoning", "spice",
        "pepper", "salt", "garlic powder", "onion powder", "cumin",
        "paprika", "oregano", "basil", "chili powder", "cinnamon",
    ],
    "Dairy and Dairy Alternatives": [
        "milk", "cheese", "yogurt", "yoghurt", "butter", "cream",
        "almond milk", "oat milk", "soy milk", "egg", "eggs",
    ],
    "Desserts and Sweets": [
        "cookie", "cake", "chocolate", "candy", "brownie", "pudding",
        "ice cream", "frosting", "wafer", "oreo", "snack cake",
        "sweet", "sugar", "fudge", "caramel",
    ],
    "Drinks": [
        "juice", "soda", "water", "coffee", "tea", "lemonade",
        "gatorade", "sport drink", "energy drink", "coca-cola", "pepsi",
    ],
    "Fresh Fruit": [
        "apple", "banana", "orange", "grape", "strawberry", "blueberry",
        "mango", "pineapple", "watermelon", "peach", "pear", "kiwi",
        "fresh fruit",
    ],
    "Fruits - Canned or Processed": [
        "fruit cocktail", "peaches in", "pears in", "mandarin",
        "applesauce", "fruit cup", "pineapple chunk", "canned fruit",
        "mixed fruit", "diced peach",
    ],
    "Granola Products": [
        "granola", "granola bar", "oat", "oats", "muesli", "cereal",
        "cheerios", "nature valley", "quaker",
    ],
    "Meat and Poultry - Canned": [
        "canned chicken", "spam", "corned beef", "vienna sausage",
        "potted meat", "canned meat", "canned turkey",
    ],
    "Meat and Poultry - Fresh": [
        "chicken breast", "ground beef", "pork", "turkey", "steak",
        "ground turkey", "chicken thigh", "beef", "fresh chicken",
    ],
    "Nut Butters and Nuts": [
        "peanut butter", "almond butter", "cashew", "almond", "walnut",
        "pecan", "mixed nuts", "sunflower seed", "peanut", "jif", "skippy",
    ],
    "Ready Meals": [
        "ready meal", "frozen dinner", "mac and cheese", "microwave",
        "heat and eat", "meal kit", "frozen entree", "hungry man",
        "lean cuisine", "stouffer", "banquet",
    ],
    "Savory Snacks and Crackers": [
        "chip", "chips", "cracker", "pretzel", "popcorn", "goldfish",
        "cheez-it", "ritz", "doritos", "lays", "pringles", "tortilla chip",
        "rice cake",
    ],
    "Seafood - Canned": [
        "tuna", "salmon", "sardine", "anchovy", "crab", "canned fish",
        "starkist", "bumble bee", "chicken of the sea",
    ],
    "Soup": [
        "soup", "broth", "chowder", "stew", "campbell", "progresso",
        "chicken noodle soup", "tomato soup", "cream of",
    ],
    "Vegetables - Canned": [
        "canned corn", "green beans", "canned peas", "canned carrot",
        "canned beet", "del monte", "green giant", "libby",
        "mixed vegetable", "sweet corn",
    ],
    "Vegetables - Fresh": [
        "broccoli", "carrot", "lettuce", "onion", "bell pepper",
        "celery", "spinach", "cucumber", "zucchini", "potato",
        "fresh vegetable",
    ],
}

# Compile patterns for faster matching
COMPILED_PATTERNS = {}
for category, keywords in CATEGORY_KEYWORDS.items():
    # Sort by length descending (match longer phrases first)
    sorted_kw = sorted(keywords, key=len, reverse=True)
    pattern = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in sorted_kw) + r')', re.IGNORECASE)
    COMPILED_PATTERNS[category] = pattern


def match_ocr_to_categories(ocr_text):
    """Match OCR text against category keywords. Returns dict of category -> match count."""
    matches = Counter()
    for category, pattern in COMPILED_PATTERNS.items():
        found = pattern.findall(ocr_text)
        if found:
            matches[category] = len(found)
    return matches


def run_ocr(model, processor, image_path, device, amp_dtype=None):
    """Run Florence-2 OCR on an image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text="<OCR>", images=image, return_tensors="pt").to(device)

    if amp_dtype:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            gen = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                early_stopping=True,
            )
    else:
        gen = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3,
            early_stopping=True,
        )

    text = processor.batch_decode(gen, skip_special_tokens=True)[0]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", default="./eval_results_v11.json")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--model", default="microsoft/Florence-2-large-ft")
    parser.add_argument("--checkpoint", default=None,
                        help="LoRA checkpoint (optional, vanilla model may have better OCR)")
    parser.add_argument("--min-keyword-matches", type=int, default=1,
                        help="Min keyword matches to add a category")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--show-ocr", action="store_true",
                        help="Print OCR text for each image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    amp_dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else None
    if amp_dtype:
        print("Using bf16")

    # Load model (vanilla Florence-2 for OCR — fine-tuned model may have lost OCR ability)
    print(f"\nLoading model: {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    print("  Model loaded.")

    # Load v11 eval results
    with open(args.eval_results) as f:
        data = json.load(f)
    predictions = data["predictions"]
    print(f"\nLoaded {len(predictions)} predictions from {args.eval_results}")

    # Run OCR + boost
    boosted = 0
    total_ocr_matches = 0
    results_boosted = []

    start = time.time()
    for i, rec in enumerate(predictions):
        img_rel = rec["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)

        if not os.path.exists(img_path):
            results_boosted.append(rec)
            continue

        # Parse original predictions
        try:
            target = json.loads(rec["target"])
        except:
            target = {"items": []}
        try:
            pred = json.loads(rec["predicted"])
        except:
            pred = {"items": []}

        target_classes = {item["name"] for item in target.get("items", []) if isinstance(item, dict)}
        pred_classes = {item["name"] for item in pred.get("items", []) if isinstance(item, dict)}

        # Run OCR
        ocr_text = run_ocr(model, processor, img_path, device, amp_dtype)
        ocr_matches = match_ocr_to_categories(ocr_text)

        if args.show_ocr and ocr_text.strip():
            print(f"\n  [{i+1}] OCR: {ocr_text[:100]}")
            if ocr_matches:
                print(f"       Matches: {dict(ocr_matches)}")

        # Boost: add categories found by OCR but missed by v11
        added = set()
        for cat, count in ocr_matches.items():
            if count >= args.min_keyword_matches and cat not in pred_classes:
                added.add(cat)

        if added:
            boosted += 1
            # Build new prediction with added classes
            new_items = list(pred.get("items", []))
            for cat in added:
                new_items.append({
                    "name": cat,
                    "package_type": "package",
                    "confidence": "medium",
                    "source": "ocr_boost",
                })
            pred = {"items": new_items}

        total_ocr_matches += len(ocr_matches)

        results_boosted.append({
            "image": rec["image"],
            "target": json.dumps(target),
            "predicted": json.dumps(pred),
            "ocr_text": ocr_text,
            "ocr_matches": dict(ocr_matches),
            "ocr_added": list(added),
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(predictions) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(predictions)} ({rate:.1f} img/s, ETA: {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"\n  OCR complete: {len(predictions)} images in {elapsed:.1f}s")
    print(f"  Images boosted: {boosted}/{len(predictions)} ({100*boosted/len(predictions):.1f}%)")
    print(f"  Total OCR category matches: {total_ocr_matches}")

    # ── Recompute metrics with boosted predictions ─────────────────────────

    from evaluate_florence2 import compute_metrics, print_report

    # Reformat for compute_metrics
    reformatted = []
    for rec in results_boosted:
        try:
            tp = json.loads(rec["target"]) if isinstance(rec["target"], str) else rec["target"]
        except:
            tp = {"items": []}
        try:
            pp = json.loads(rec["predicted"]) if isinstance(rec["predicted"], str) else rec["predicted"]
        except:
            pp = {"items": []}

        reformatted.append({
            "image": rec["image"],
            "target_text": rec["target"] if isinstance(rec["target"], str) else json.dumps(rec["target"]),
            "pred_text": rec["predicted"] if isinstance(rec["predicted"], str) else json.dumps(rec["predicted"]),
            "target_parsed": tp,
            "pred_parsed": pp,
        })

    metrics = compute_metrics(reformatted)
    print_report(metrics)

    # Save results
    out_path = "./eval_results_v11_ocr_boost.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "predictions": results_boosted}, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
