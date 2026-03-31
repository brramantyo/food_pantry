#!/usr/bin/env python3
"""
evaluate_vlm_cot.py
===================
Zero-shot VLM evaluation with chain-of-thought reasoning.

Uses Qwen2.5-VL-7B (open-source) to classify pantry images with
step-by-step reasoning. No training required — pure inference.

This serves as a comparison baseline:
  - How does a general-purpose VLM with reasoning compare to
    our fine-tuned Florence-2 specialist?
  - Does chain-of-thought help with multi-label detection?

The prompt asks the VLM to:
  1. Describe what it sees in the image
  2. Identify individual food items
  3. Map each item to one of the 21 categories
  4. Output structured JSON

Usage:
  python evaluate_vlm_cot.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_vlm_cot.json \
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


# ── Categories ─────────────────────────────────────────────────────────────────

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

# Also build fuzzy matching for common VLM outputs
FUZZY_MAP = {
    "beans": "Beans and Legumes - Canned or Dried",
    "legumes": "Beans and Legumes - Canned or Dried",
    "bread": "Bread and Bakery Products",
    "bakery": "Bread and Bakery Products",
    "tomato": "Canned Tomato Products",
    "tomatoes": "Canned Tomato Products",
    "pasta": "Carbohydrate Meal",
    "rice": "Carbohydrate Meal",
    "noodles": "Carbohydrate Meal",
    "carbohydrate": "Carbohydrate Meal",
    "condiment": "Condiments and Sauces",
    "sauce": "Condiments and Sauces",
    "ketchup": "Condiments and Sauces",
    "mustard": "Condiments and Sauces",
    "dairy": "Dairy and Dairy Alternatives",
    "milk": "Dairy and Dairy Alternatives",
    "cheese": "Dairy and Dairy Alternatives",
    "yogurt": "Dairy and Dairy Alternatives",
    "dessert": "Desserts and Sweets",
    "sweets": "Desserts and Sweets",
    "candy": "Desserts and Sweets",
    "chocolate": "Desserts and Sweets",
    "cookies": "Desserts and Sweets",
    "drinks": "Drinks",
    "juice": "Drinks",
    "soda": "Drinks",
    "beverage": "Drinks",
    "water": "Drinks",
    "fruit": "Fresh Fruit",
    "apple": "Fresh Fruit",
    "banana": "Fresh Fruit",
    "canned fruit": "Fruits - Canned or Processed",
    "applesauce": "Fruits - Canned or Processed",
    "granola": "Granola Products",
    "oats": "Granola Products",
    "cereal": "Granola Products",
    "canned meat": "Meat and Poultry - Canned",
    "spam": "Meat and Poultry - Canned",
    "canned chicken": "Meat and Poultry - Canned",
    "fresh meat": "Meat and Poultry - Fresh",
    "chicken": "Meat and Poultry - Fresh",
    "beef": "Meat and Poultry - Fresh",
    "pork": "Meat and Poultry - Fresh",
    "peanut butter": "Nut Butters and Nuts",
    "nuts": "Nut Butters and Nuts",
    "almonds": "Nut Butters and Nuts",
    "ready meal": "Ready Meals",
    "frozen dinner": "Ready Meals",
    "microwave": "Ready Meals",
    "chips": "Savory Snacks and Crackers",
    "crackers": "Savory Snacks and Crackers",
    "snack": "Savory Snacks and Crackers",
    "pretzels": "Savory Snacks and Crackers",
    "popcorn": "Savory Snacks and Crackers",
    "tuna": "Seafood - Canned",
    "salmon": "Seafood - Canned",
    "sardines": "Seafood - Canned",
    "canned fish": "Seafood - Canned",
    "seafood": "Seafood - Canned",
    "soup": "Soup",
    "broth": "Soup",
    "canned vegetables": "Vegetables - Canned",
    "canned corn": "Vegetables - Canned",
    "canned beans": "Beans and Legumes - Canned or Dried",
    "green beans": "Vegetables - Canned",
    "fresh vegetables": "Vegetables - Fresh",
    "lettuce": "Vegetables - Fresh",
    "broccoli": "Vegetables - Fresh",
    "carrots": "Vegetables - Fresh",
}


def normalize_category(name):
    name_lower = name.lower().strip()
    # Exact match first
    if name_lower in CANONICAL_MAP:
        return CANONICAL_MAP[name_lower]
    # Fuzzy match
    for key, cat in FUZZY_MAP.items():
        if key in name_lower:
            return cat
    return name


# ── Chain-of-Thought Prompt ────────────────────────────────────────────────────

CATEGORIES_LIST = "\n".join(f"  - {c}" for c in sorted(VALID_CATEGORIES))

COT_PROMPT = f"""You are an expert food pantry item classifier. Your task is to look at an image of food pantry items and classify every visible item into the correct categories.

**IMPORTANT: Read all text, labels, and brand names on the packaging carefully.** The text on packages is the most reliable signal for classification.

**The 21 valid categories are:**
{CATEGORIES_LIST}

**Classification rules:**
- Each IMAGE may contain MULTIPLE items from DIFFERENT categories
- Only output categories from the 21 listed above (exact spelling)
- Read package labels to determine contents — don't guess from shape/color alone
- "Carbohydrate Meal" = pasta, rice, noodles, mac & cheese, ramen
- "Ready Meals" = complete frozen/shelf-stable meals (e.g., canned ravioli, frozen dinners, meal kits)
- "Granola Products" = granola bars, oatmeal, cereal, breakfast bars
- "Desserts and Sweets" = cookies, cake, candy, chocolate, pudding, icing
- "Condiments and Sauces" = ketchup, mustard, salad dressing, cooking sauce, seasoning mix
- "Dairy and Dairy Alternatives" = milk, cheese, yogurt, butter, non-dairy milk
- Canned vegetables with NO meat = "Vegetables - Canned"
- Canned tomato paste/sauce/diced = "Canned Tomato Products"

**Example 1:**
Image shows: A box of Barilla spaghetti, two cans of Hunt's tomato sauce, and a jar of Skippy peanut butter.
Answer: {{"categories": ["Carbohydrate Meal", "Canned Tomato Products", "Nut Butters and Nuts"]}}

**Example 2:**
Image shows: A bag of Doritos chips, a can of Campbell's chicken noodle soup, and a package of Oreo cookies.
Answer: {{"categories": ["Savory Snacks and Crackers", "Soup", "Desserts and Sweets"]}}

**Now analyze the image below. Follow these steps:**
1. Read ALL text/labels visible on packages
2. Identify each distinct food item
3. Map each item to exactly one of the 21 categories
4. Output ONLY a JSON object: {{"categories": ["Category1", "Category2", ...]}}"""


# ── Parse VLM Output ───────────────────────────────────────────────────────────

def parse_vlm_output(text):
    """Extract categories from VLM chain-of-thought output."""
    categories = set()
    
    # Strategy 1: Find JSON block in output
    json_match = re.search(r'\{[^}]*"categories"\s*:\s*\[([^\]]*)\][^}]*\}', text, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            for cat in result.get("categories", []):
                normalized = normalize_category(cat)
                if normalized in VALID_CATEGORIES:
                    categories.add(normalized)
            if categories:
                return categories
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Strategy 2: Find any JSON array
    array_match = re.search(r'"categories"\s*:\s*\[([^\]]*)\]', text, re.DOTALL)
    if array_match:
        items_str = array_match.group(1)
        for item in re.findall(r'"([^"]+)"', items_str):
            normalized = normalize_category(item)
            if normalized in VALID_CATEGORIES:
                categories.add(normalized)
        if categories:
            return categories
    
    # Strategy 3: Look for category names mentioned anywhere in text
    for cat in VALID_CATEGORIES:
        if cat.lower() in text.lower():
            categories.add(cat)
    
    # Strategy 4: Fuzzy match keywords in text
    if not categories:
        text_lower = text.lower()
        for key, cat in FUZZY_MAP.items():
            if key in text_lower:
                categories.add(cat)
    
    return categories


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(all_targets, all_preds):
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()
    exact = 0
    
    for t, p in zip(all_targets, all_preds):
        if t == p:
            exact += 1
        for c in t | p:
            if c in t and c in p:
                class_tp[c] += 1
            elif c in p:
                class_fp[c] += 1
            else:
                class_fn[c] += 1
    
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    micro_p = total_tp / max(total_tp + total_fp, 1)
    micro_r = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    
    per_class = {}
    all_classes = sorted(set(list(class_tp) + list(class_fp) + list(class_fn)))
    for cls in all_classes:
        tp, fp, fn = class_tp[cls], class_fp[cls], class_fn[cls]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        per_class[cls] = {"precision": round(p, 4), "recall": round(r, 4), 
                          "f1": round(f1, 4), "support": tp + fn}
    
    macro_f1 = sum(m["f1"] for m in per_class.values()) / max(len(per_class), 1)
    
    return {
        "micro_p": round(micro_p, 4), "micro_r": round(micro_r, 4),
        "micro_f1": round(micro_f1, 4), "macro_f1": round(macro_f1, 4),
        "exact_match": round(exact / max(len(all_targets), 1), 4),
        "per_class": per_class,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLM Chain-of-Thought evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--output", type=str, default="./eval_vlm_cot.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--self-consistency", type=int, default=1,
                        help="Number of samples for self-consistency voting (1=greedy, 3+=majority vote)")
    parser.add_argument("--sc-temperature", type=float, default=0.7,
                        help="Temperature for self-consistency sampling")
    parser.add_argument("--sc-threshold", type=int, default=2,
                        help="Min votes needed (default: majority = ceil(n/2))")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    
    # ── Load Model ─────────────────────────────────────────────────────
    print(f"\nLoading {args.model}...")
    
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    # Use sdpa attention (saves massive VRAM vs eager)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )
    print("  Using sdpa attention")
    
    # Limit image resolution to avoid OOM on vision encoder
    # Default Qwen2.5-VL max_pixels=1003520 → reduce to ~200K pixels
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=128*28*28,    # ~100K
        max_pixels=256*28*28,    # ~200K (aggressive reduction for 40GB GPU)
    )
    model.eval()
    print("  Model loaded ✓")
    
    # ── Load test data ─────────────────────────────────────────────────
    samples = []
    with open(args.jsonl, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Evaluating {len(samples)} samples...\n")
    
    # ── Evaluate ───────────────────────────────────────────────────────
    all_targets = []
    all_preds = []
    results = []
    
    t0 = time.time()
    
    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)
        
        if not os.path.exists(img_path):
            if i < 3:  # Print first few missing paths for debugging
                print(f"  [WARN] Image not found: {img_path}")
            continue
        
        image = Image.open(img_path).convert("RGB")
        
        # Ground truth
        target = json.loads(sample["target"])
        target_cats = {normalize_category(item["name"]) for item in target.get("items", [])
                       if normalize_category(item.get("name", "")) in VALID_CATEGORIES}
        
        # Build Qwen2-VL message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": COT_PROMPT},
                ],
            }
        ]
        
        # Process
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_input], images=[image], 
            return_tensors="pt", padding=True,
        ).to(device)
        
        n_samples = args.self_consistency
        if n_samples <= 1:
            # Greedy decoding
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            pred_cats = parse_vlm_output(output_text)
        else:
            # Self-consistency: sample n times, majority vote
            from collections import Counter as Ctr
            all_votes = Counter()  # category -> vote count
            all_outputs = []
            for s in range(n_samples):
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.sc_temperature,
                        top_p=0.9,
                    )
                generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
                out_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                all_outputs.append(out_text[:200])
                sample_cats = parse_vlm_output(out_text)
                for cat in sample_cats:
                    all_votes[cat] += 1
            
            # Majority vote: category needs >= threshold votes
            threshold = args.sc_threshold if args.sc_threshold else (n_samples // 2 + 1)
            pred_cats = {cat for cat, count in all_votes.items() if count >= threshold}
            output_text = f"[SC {n_samples}x] votes: {dict(all_votes)} | outputs: {all_outputs}"
        
        all_targets.append(target_cats)
        all_preds.append(pred_cats)
        
        results.append({
            "image": img_rel,
            "target": sorted(target_cats),
            "predicted": sorted(pred_cats),
            "match": target_cats == pred_cats,
            "vlm_output": output_text[:500],  # Truncate for storage
        })
        
        # Print progress
        match_str = "✓" if target_cats == pred_cats else "✗"
        print(f"  [{i+1}/{len(samples)}] {match_str} Target: {sorted(target_cats)}")
        print(f"           Pred:   {sorted(pred_cats)}")
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"\n  --- Progress: {i+1}/{len(samples)} ({rate:.2f} img/s, ETA: {eta:.0f}s) ---\n")
    
    elapsed = time.time() - t0
    n = len(all_targets)
    
    # ── Compute metrics ────────────────────────────────────────────────
    metrics = compute_metrics(all_targets, all_preds)
    
    # ── Print report ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  VLM CHAIN-OF-THOUGHT EVALUATION")
    print(f"{'='*70}")
    print(f"  Model: {args.model}")
    print(f"  Samples: {n}")
    print(f"  Time: {elapsed:.1f}s ({n/elapsed:.2f} img/s)")
    
    print(f"\n  Micro P:     {metrics['micro_p']:.1%}")
    print(f"  Micro R:     {metrics['micro_r']:.1%}")
    print(f"  Micro F1:    {metrics['micro_f1']:.1%}")
    print(f"  Macro F1:    {metrics['macro_f1']:.1%}")
    print(f"  Exact Match: {metrics['exact_match']:.1%}")
    
    print(f"\n  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
    for cls in sorted(metrics["per_class"].keys()):
        m = metrics["per_class"][cls]
        print(f"  {cls:<45} {m['precision']:>5.1%} {m['recall']:>5.1%} {m['f1']:>5.1%} {m['support']:>5}")
    
    # ── Comparison with v11 ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Method':<40} {'Micro F1':>9}")
    print(f"  {'-'*40} {'-'*9}")
    print(f"  {'Vanilla Florence-2 (zero-shot)':<40} {'27.4%':>9}")
    print(f"  {'Fine-tuned Florence-2 (v11)':<40} {'76.5%':>9}")
    model_short = args.model.split("/")[-1]
    print(f"  {f'{model_short} (CoT, zero-shot)':<40} {metrics['micro_f1']:>8.1%}")
    
    # ── Sample CoT outputs ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SAMPLE CHAIN-OF-THOUGHT OUTPUTS (first 3)")
    print(f"{'='*70}")
    for r in results[:3]:
        print(f"\n  📷 {os.path.basename(r['image'])}")
        print(f"  Target: {r['target']}")
        print(f"  Predicted: {r['predicted']}")
        print(f"  VLM reasoning:")
        # Print first 300 chars of reasoning
        reasoning = r['vlm_output'][:300].replace('\n', '\n    ')
        print(f"    {reasoning}...")
    
    # ── Save ───────────────────────────────────────────────────────────
    output = {
        "model": args.model,
        "approach": "vlm_chain_of_thought_zero_shot",
        "n_samples": n,
        "time_seconds": round(elapsed, 1),
        "images_per_second": round(n / elapsed, 2),
        "metrics": metrics,
        "results": results,
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
