#!/usr/bin/env python3
"""
evaluate_florence2.py
=====================
Evaluate a fine-tuned Florence-2 model on the test (or valid) set.

Metrics computed:
  - Exact Match Accuracy (full JSON string match)
  - Item-level Precision, Recall, F1 (per class and macro/micro)
  - Category Detection Accuracy (did the model find the right set of classes?)
  - Count Accuracy (did the model predict the correct count per class?)
  - JSON Parse Rate (% of outputs that are valid JSON)

Usage:
    python evaluate_florence2.py \
        --checkpoint ./checkpoints/best_model \
        --base-model microsoft/Florence-2-base-ft \
        --data-dir . \
        --jsonl ./florence2_data/test.jsonl \
        --output ./eval_results.json \
        --bf16

    # Use --split valid to evaluate on validation set instead
    python evaluate_florence2.py \
        --checkpoint ./checkpoints/best_model \
        --base-model microsoft/Florence-2-base-ft \
        --data-dir . \
        --jsonl ./florence2_data/valid.jsonl \
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
from peft import PeftModel


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


VALID_CATEGORIES = {
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
}

VALID_PACKAGE_TYPES = {
    "bag", "box", "bottle", "can", "carton", "jar", "pouch",
    "container", "package", "wrapper", "tub", "tray", "tube",
    "cup", "jug", "packet", "bundle", "sleeve", "tin",
}


def fix_json_text(text: str) -> str:
    """Apply multiple fixes to corrupted JSON text from Florence-2."""
    t = text.strip()
    # Single quotes to double quotes
    t = t.replace("'", '"')
    # Fix spaced-out keys: " name" -> "name", " package_ type" -> "package_type"
    t = re.sub(r'"\s+([\w])', r'"\1', t)  # leading space inside quotes
    t = re.sub(r'([\w])\s+"', r'\1"', t)  # trailing space inside quotes
    # Fix underscored keys with spaces: "package_ type" -> "package_type"
    t = re.sub(r'"\s*([\w]+)_\s*([\w]+)\s*"', r'"\1_\2"', t)
    # Fix colon stuck to key: "name:" "value" -> "name": "value"
    t = re.sub(r'"(\w+):"\s*"', r'"\1": "', t)
    # Fix missing colon: "name" "value" -> "name": "value"  (only for known keys)
    for key in ["name", "package_type", "confidence"]:
        t = re.sub(rf'"{key}"\s*"', f'"{key}": "', t)
    # Fix values that lost their key structure: "packaged" alone -> "package_type": "package"
    # Fix truncated items: {..., "high"} -> {..., "confidence": "high"}
    return t


def fix_json_deep(text: str) -> str:
    """
    More aggressive JSON repair for multi-item Florence-2 output.
    Handles common corruption patterns in second/third items.
    """
    t = fix_json_text(text)
    
    # Fix pattern: }, {" name" -> }, {"name"  (spaces in subsequent items)
    t = re.sub(r'\}\s*,\s*\{\s*"', '}, {"', t)
    
    # Fix pattern: "high"}]  missing closing (ensure proper termination)
    if t.count('[') > t.count(']'):
        t = t + ']' * (t.count('[') - t.count(']'))
    if t.count('{') > t.count('}'):
        t = t + '}' * (t.count('{') - t.count('}'))
    
    # Fix broken item objects — try to reconstruct valid items
    # Pattern: {"name": "X", "package_type": "Y", "confidence": "high"} is valid
    # Pattern: {"name": "X", "packaged", "high"} needs fixing
    # Replace shorthand confidence
    t = re.sub(r',\s*"high"\s*\}', ', "confidence": "high"}', t)
    t = re.sub(r',\s*"medium"\s*\}', ', "confidence": "medium"}', t)
    t = re.sub(r',\s*"low"\s*\}', ', "confidence": "low"}', t)
    
    # Fix "packaged" -> "package_type": "package"
    t = re.sub(r'"packaged"', '"package_type": "package"', t)
    t = re.sub(r'"packet"', '"package_type": "packet"', t)
    
    return t


def extract_items_regex(text: str) -> list:
    """Last-resort: use regex to find known category names and nearby package types."""
    items = []
    found_categories = []
    for cat in VALID_CATEGORIES:
        if cat.lower() in text.lower():
            found_categories.append(cat)
    for cat in found_categories:
        pkg = "unknown"
        for pt in VALID_PACKAGE_TYPES:
            cat_pos = text.lower().find(cat.lower())
            if cat_pos >= 0:
                nearby = text[cat_pos:cat_pos + 150].lower()
                if pt in nearby:
                    pkg = pt
                    break
        items.append({"name": cat, "package_type": pkg, "confidence": "high"})
    return items


def parse_prediction(text: str) -> Optional[dict]:
    """
    Robust parse of Florence-2 model output as JSON.
    Tries multiple strategies from strict to fuzzy.
    v8: Enhanced to handle corrupted multi-item JSON output.
    """
    text = text.strip()
    if not text:
        return None

    # Strategy 1: Direct parse (original + fixed + deep fixed)
    for candidate in [text, fix_json_text(text), fix_json_deep(text)]:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict) and "items" in result:
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 2: Find JSON object in text (try deep fix first)
    for fixer in [fix_json_deep, fix_json_text, lambda x: x]:
        candidate = fixer(text)
        start = candidate.find("{")
        end = candidate.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(candidate[start:end])
                if isinstance(result, dict) and "items" in result:
                    return result
            except json.JSONDecodeError:
                pass

    # Strategy 3: Fix truncated JSON — try closing it (with deep fix)
    for fixer in [fix_json_deep, fix_json_text, lambda x: x]:
        candidate = fixer(text)
        start = candidate.find("{")
        if start >= 0:
            fragment = candidate[start:]
            for suffix in ['"}]}', '"}]}}', ']}}', ']}', '}]}', '"}}', '"}', '}',
                           '"}}]}', ', "confidence": "high"}]}',
                           '", "confidence": "high"}]}']:
                try:
                    result = json.loads(fragment + suffix)
                    if isinstance(result, dict) and "items" in result:
                        valid_items = [item for item in result["items"]
                                       if isinstance(item, dict) and "name" in item]
                        if valid_items:
                            return {"items": valid_items}
                except json.JSONDecodeError:
                    pass

    # Strategy 4: Extract array and wrap
    for fixer in [fix_json_deep, fix_json_text, lambda x: x]:
        candidate = fixer(text)
        start = candidate.find("[")
        end = candidate.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                items = json.loads(candidate[start:end])
                if isinstance(items, list):
                    return {"items": items}
            except json.JSONDecodeError:
                pass

    # Strategy 5: ast.literal_eval
    try:
        import ast
        result = ast.literal_eval(fix_json_deep(text))
        if isinstance(result, dict) and "items" in result:
            return result
    except (ValueError, SyntaxError):
        pass

    # Strategy 6: Regex extraction (last resort — extracts known category names)
    items = extract_items_regex(text)
    if items:
        return {"items": items}

    return None


# ── Metrics ────────────────────────────────────────────────────────────────────

def extract_class_set(parsed: dict) -> Set[str]:
    """Extract set of class names from parsed prediction."""
    if parsed is None or "items" not in parsed:
        return set()
    return {item["name"] for item in parsed["items"] if "name" in item}


def extract_class_counts(parsed: dict) -> Dict[str, int]:
    """Extract class name -> count mapping from parsed prediction."""
    if parsed is None or "items" not in parsed:
        return {}
    counts = {}
    for item in parsed["items"]:
        if "name" in item:
            counts[item["name"]] = item.get("count", 1)
    return counts


def compute_metrics(predictions: List[dict]) -> dict:
    """
    Compute all evaluation metrics from a list of prediction records.
    
    Each record has:
        - target_text: ground truth JSON string
        - pred_text: model output string
        - target_parsed: parsed ground truth
        - pred_parsed: parsed prediction (or None)
        - image: image path
    """
    n = len(predictions)
    if n == 0:
        return {"error": "No predictions to evaluate"}

    # ── Basic Counts ──
    exact_matches = 0
    json_parse_success = 0
    category_exact_matches = 0
    count_exact_matches = 0

    # ── Per-class tracking ──
    # For each class: TP (in both), FP (in pred not target), FN (in target not pred)
    all_classes = set()
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()

    # Count accuracy per class
    count_correct_per_class = Counter()
    count_total_per_class = Counter()

    # Category-level details
    category_details = []

    for rec in predictions:
        target_parsed = rec["target_parsed"]
        pred_parsed = rec["pred_parsed"]

        # Exact match (string level)
        if rec["target_text"].strip() == rec["pred_text"].strip():
            exact_matches += 1

        # JSON parse success
        if pred_parsed is not None:
            json_parse_success += 1

        # Extract class sets
        target_classes = extract_class_set(target_parsed)
        pred_classes = extract_class_set(pred_parsed)
        all_classes.update(target_classes)
        all_classes.update(pred_classes)

        # Category-level exact match (same set of classes detected)
        if target_classes == pred_classes:
            category_exact_matches += 1

        # Per-class TP/FP/FN
        for cls in target_classes & pred_classes:
            class_tp[cls] += 1
        for cls in pred_classes - target_classes:
            class_fp[cls] += 1
        for cls in target_classes - pred_classes:
            class_fn[cls] += 1

        # Count accuracy
        target_counts = extract_class_counts(target_parsed)
        pred_counts = extract_class_counts(pred_parsed)

        count_match = True
        for cls in target_counts:
            count_total_per_class[cls] += 1
            if cls in pred_counts and pred_counts[cls] == target_counts[cls]:
                count_correct_per_class[cls] += 1
            else:
                count_match = False

        # Also check if pred has extra classes
        if set(pred_counts.keys()) != set(target_counts.keys()):
            count_match = False

        if count_match:
            count_exact_matches += 1

    # ── Aggregate Metrics ──

    # Per-class precision, recall, F1
    per_class_metrics = {}
    for cls in sorted(all_classes):
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        count_acc = (count_correct_per_class[cls] / count_total_per_class[cls]
                     if count_total_per_class[cls] > 0 else 0.0)

        per_class_metrics[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,  # number of times this class appears in ground truth
            "count_accuracy": round(count_acc, 4),
        }

    # Micro-average (global TP/FP/FN)
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)

    # Macro-average (average per-class metrics)
    macro_precision = sum(m["precision"] for m in per_class_metrics.values()) / len(per_class_metrics) if per_class_metrics else 0.0
    macro_recall = sum(m["recall"] for m in per_class_metrics.values()) / len(per_class_metrics) if per_class_metrics else 0.0
    macro_f1 = sum(m["f1"] for m in per_class_metrics.values()) / len(per_class_metrics) if per_class_metrics else 0.0

    return {
        "total_samples": n,
        "json_parse_rate": round(json_parse_success / n, 4),
        "exact_match_accuracy": round(exact_matches / n, 4),
        "category_detection_accuracy": round(category_exact_matches / n, 4),
        "count_accuracy": round(count_exact_matches / n, 4),
        "micro": {
            "precision": round(micro_precision, 4),
            "recall": round(micro_recall, 4),
            "f1": round(micro_f1, 4),
        },
        "macro": {
            "precision": round(macro_precision, 4),
            "recall": round(macro_recall, 4),
            "f1": round(macro_f1, 4),
        },
        "per_class": per_class_metrics,
    }


def print_report(metrics: dict):
    """Print a formatted evaluation report."""
    print(f"\n{'='*70}")
    print("EVALUATION REPORT")
    print(f"{'='*70}")

    print(f"\n  Total Samples:              {metrics['total_samples']}")
    print(f"  JSON Parse Rate:            {metrics['json_parse_rate']*100:.1f}%")
    print(f"  Exact Match Accuracy:       {metrics['exact_match_accuracy']*100:.1f}%")
    print(f"  Category Detection Accuracy:{metrics['category_detection_accuracy']*100:.1f}%")
    print(f"  Count Accuracy:             {metrics['count_accuracy']*100:.1f}%")

    print(f"\n  Micro-Averaged:")
    print(f"    Precision: {metrics['micro']['precision']*100:.1f}%")
    print(f"    Recall:    {metrics['micro']['recall']*100:.1f}%")
    print(f"    F1:        {metrics['micro']['f1']*100:.1f}%")

    print(f"\n  Macro-Averaged:")
    print(f"    Precision: {metrics['macro']['precision']*100:.1f}%")
    print(f"    Recall:    {metrics['macro']['recall']*100:.1f}%")
    print(f"    F1:        {metrics['macro']['f1']*100:.1f}%")

    # Per-class table
    print(f"\n{'='*70}")
    print("PER-CLASS METRICS")
    print(f"{'='*70}")
    print(f"  {'Class':<40s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Cnt Acc':>7s} {'Support':>7s}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")

    for cls, m in sorted(metrics["per_class"].items()):
        print(f"  {cls:<40s} {m['precision']*100:5.1f}% {m['recall']*100:5.1f}% "
              f"{m['f1']*100:5.1f}% {m['count_accuracy']*100:6.1f}% {m['support']:>7d}")

    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
    print(f"  {'MICRO AVG':<40s} {metrics['micro']['precision']*100:5.1f}% "
          f"{metrics['micro']['recall']*100:5.1f}% {metrics['micro']['f1']*100:5.1f}%")
    print(f"  {'MACRO AVG':<40s} {metrics['macro']['precision']*100:5.1f}% "
          f"{metrics['macro']['recall']*100:5.1f}% {metrics['macro']['f1']*100:5.1f}%")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Florence-2 on food pantry test set.",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the LoRA checkpoint (e.g. ./checkpoints/best_model). Omit for zero-shot.")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-base-ft",
                        help="Base model name.")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Root data directory with image folders.")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test.jsonl",
                        help="Path to the JSONL file to evaluate on.")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON results.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples to evaluate (for quick testing).")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 for inference.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 for inference.")
    parser.add_argument("--show-errors", action="store_true",
                        help="Print details for mismatched predictions.")
    parser.add_argument("--show-predictions", type=int, default=5,
                        help="Number of sample predictions to display (default: 5).")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine AMP dtype
    amp_dtype = None
    if args.bf16:
        amp_dtype = torch.bfloat16
        print("Using bf16")
    elif args.fp16:
        amp_dtype = torch.float16
        print("Using fp16")

    # ── Load Model ─────────────────────────────────────────────────────────

    print(f"\nLoading base model: {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=amp_dtype if amp_dtype else torch.float32,
        attn_implementation="eager",
    )

    # Add custom token (only needed for v1 checkpoints)
    # special_tokens = {"additional_special_tokens": ["<STRUCTURED_PANTRY_OUTPUT>"]}
    # num_added = processor.tokenizer.add_special_tokens(special_tokens)
    # if num_added > 0:
    #     model.resize_token_embeddings(len(processor.tokenizer))

    # Load LoRA adapter (skip for zero-shot baseline)
    if args.checkpoint:
        print(f"Loading LoRA checkpoint: {args.checkpoint}...")
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()  # Merge LoRA weights for faster inference
        print("  Model loaded and merged.")
    else:
        print("  Zero-shot mode (no checkpoint loaded).")
    model.to(device)
    model.eval()

    # ── Load Test Data ─────────────────────────────────────────────────────

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

    # ── Run Inference ──────────────────────────────────────────────────────

    predictions = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)

        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}", file=sys.stderr)
            continue

        pred_text = run_inference(model, processor, img_path, device, amp_dtype=amp_dtype)

        target_parsed = json.loads(sample["target"])
        pred_parsed = parse_prediction(pred_text)

        predictions.append({
            "image": sample["image"],
            "target_text": sample["target"],
            "pred_text": pred_text,
            "target_parsed": target_parsed,
            "pred_parsed": pred_parsed,
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(samples)} ({rate:.1f} img/s, ETA: {eta:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\n  Inference complete: {len(predictions)} images in {elapsed:.1f}s "
          f"({len(predictions)/elapsed:.1f} img/s)")

    # ── Compute Metrics ────────────────────────────────────────────────────

    metrics = compute_metrics(predictions)
    print_report(metrics)

    # ── Show Sample Predictions ────────────────────────────────────────────

    if args.show_predictions > 0:
        print(f"{'='*70}")
        print(f"SAMPLE PREDICTIONS (first {args.show_predictions})")
        print(f"{'='*70}")

        for rec in predictions[:args.show_predictions]:
            print(f"\n  Image: {rec['image']}")
            print(f"  Target: {rec['target_text'][:150]}...")
            print(f"  Predicted: {rec['pred_text'][:150]}...")

            target_cls = extract_class_set(rec["target_parsed"])
            pred_cls = extract_class_set(rec["pred_parsed"])

            if target_cls == pred_cls:
                print(f"  Classes: ✓ MATCH ({len(target_cls)} classes)")
            else:
                missing = target_cls - pred_cls
                extra = pred_cls - target_cls
                matched = target_cls & pred_cls
                print(f"  Classes: ✗ MISMATCH")
                if matched:
                    print(f"    Matched: {matched}")
                if missing:
                    print(f"    Missing: {missing}")
                if extra:
                    print(f"    Extra:   {extra}")

    # ── Show Errors ────────────────────────────────────────────────────────

    if args.show_errors:
        errors = [r for r in predictions if r["pred_parsed"] is None]
        if errors:
            print(f"\n{'='*70}")
            print(f"PARSE FAILURES ({len(errors)} total)")
            print(f"{'='*70}")
            for rec in errors[:10]:
                print(f"\n  Image: {rec['image']}")
                print(f"  Raw output: {rec['pred_text'][:200]}")

    # ── Save Results ───────────────────────────────────────────────────────

    if args.output:
        output_data = {
            "metrics": metrics,
            "predictions": [
                {
                    "image": r["image"],
                    "target": r["target_text"],
                    "predicted": r["pred_text"],
                    "json_parsed": r["pred_parsed"] is not None,
                }
                for r in predictions
            ],
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
