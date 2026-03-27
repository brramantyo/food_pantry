#!/usr/bin/env python3
"""
evaluate_ocr_input.py
=====================
Test Florence-2 <REGION_TO_OCR> as additional input signal for classification.

Strategy (per Prof Zheng's suggestion):
  1. Run Florence-2 OCR on the image to extract visible text
  2. Concatenate OCR text with the classification prompt
  3. Run fine-tuned classifier with the enriched prompt
  4. Compare performance vs image-only classification

This is different from OCR post-processing (which failed):
  - OCR post-processing: classify first → OCR second → override predictions
  - OCR input: OCR first → feed text AS INPUT to classifier → single prediction

Usage:
  python evaluate_ocr_input.py \
    --base-model microsoft/Florence-2-large-ft \
    --checkpoint ./checkpoints_v11/best_model \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_results_ocr_input.json \
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


def extract_ocr_text(model_vanilla, processor, image, device, amp_dtype=None):
    """
    Use Florence-2 to extract text from the image.
    Try <OCR> first, fall back to <MORE_DETAILED_CAPTION> for text cues.
    """
    # Try OCR task
    inputs = processor(text="<OCR>", images=image, return_tensors="pt").to(device)

    if amp_dtype:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            generated_ids = model_vanilla.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3,
                early_stopping=True,
            )
    else:
        generated_ids = model_vanilla.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3,
            early_stopping=True,
        )

    ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return ocr_text


def classify_with_ocr_context(model_finetuned, processor, image, ocr_text, device, amp_dtype=None):
    """
    Classify image with OCR text prepended to the prompt.

    Strategy: We modify the prompt to include OCR context.
    Since the model was trained with <OD> prompt, we keep that but
    add OCR text as a prefix that the model can use as additional signal.

    Alternative approaches to test:
    1. Prepend OCR to prompt: "Text on package: {ocr_text} <OD>"
    2. Append OCR after prompt: "<OD> Package text: {ocr_text}"
    3. Use a structured prompt: "<OD> [OCR: {ocr_text}]"
    """
    # Approach: prepend OCR context before task prompt
    if ocr_text:
        # Truncate OCR to avoid overwhelming the model
        ocr_truncated = ocr_text[:200]
        enhanced_prompt = f"Package text: {ocr_truncated} {TASK_PROMPT}"
    else:
        enhanced_prompt = TASK_PROMPT

    inputs = processor(text=enhanced_prompt, images=image, return_tensors="pt").to(device)

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
        return pred, True
    except json.JSONDecodeError:
        return {"items": []}, False


def classify_image_only(model_finetuned, processor, image, device, amp_dtype=None):
    """Baseline: classify with image only (no OCR)."""
    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)

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
        return pred, True
    except json.JSONDecodeError:
        return {"items": []}, False


def compute_metrics(all_targets, all_predictions):
    """Compute precision, recall, F1 (micro and macro)."""
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

    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

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
    return {
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro": {"precision": macro_p_sum / n_classes, "recall": macro_r_sum / n_classes,
                   "f1": macro_f1_sum / n_classes},
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="OCR-input classification experiment")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--output", type=str, default="./eval_results_ocr_input.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    amp_dtype = None
    if args.bf16 and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        print("Using bf16")
    elif args.fp16:
        amp_dtype = torch.float16

    # Load vanilla Florence-2 for OCR
    print(f"\nLoading vanilla Florence-2 for OCR: {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model_vanilla = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    print("  Vanilla model loaded ✓")

    # Load fine-tuned model for classification
    print(f"Loading fine-tuned model: {args.checkpoint}...")
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

    # Evaluate both: with OCR and without (for fair comparison)
    all_targets = []
    all_preds_ocr = []
    all_preds_baseline = []
    results = []
    ocr_parse_success = 0
    baseline_parse_success = 0

    start = time.time()
    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to load {img_path}: {e}")
            continue

        target = json.loads(sample["target"])
        target_classes = {normalize_category(item["name"]) for item in target.get("items", [])}

        with torch.no_grad():
            # Step 1: Extract OCR text
            ocr_text = extract_ocr_text(model_vanilla, processor, image, device, amp_dtype)

            # Step 2: Classify WITH OCR context
            pred_ocr, parsed_ocr = classify_with_ocr_context(
                model_finetuned, processor, image, ocr_text, device, amp_dtype
            )
            if parsed_ocr:
                ocr_parse_success += 1

            # Step 3: Classify WITHOUT OCR (baseline, same model)
            pred_baseline, parsed_baseline = classify_image_only(
                model_finetuned, processor, image, device, amp_dtype
            )
            if parsed_baseline:
                baseline_parse_success += 1

        # Extract predicted classes
        ocr_classes = {normalize_category(item["name"]) for item in pred_ocr.get("items", [])
                       if normalize_category(item["name"]) in VALID_CATEGORIES}
        baseline_classes = {normalize_category(item["name"]) for item in pred_baseline.get("items", [])
                           if normalize_category(item["name"]) in VALID_CATEGORIES}

        all_targets.append(target_classes)
        all_preds_ocr.append(ocr_classes)
        all_preds_baseline.append(baseline_classes)

        results.append({
            "image": img_rel,
            "ocr_text": ocr_text[:300],
            "target_classes": sorted(target_classes),
            "pred_ocr_classes": sorted(ocr_classes),
            "pred_baseline_classes": sorted(baseline_classes),
            "ocr_match": target_classes == ocr_classes,
            "baseline_match": target_classes == baseline_classes,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(samples)} ({rate:.2f} img/s, ETA: {eta:.0f}s)")

    elapsed = time.time() - start

    # Compute metrics for both
    metrics_ocr = compute_metrics(all_targets, all_preds_ocr)
    metrics_baseline = compute_metrics(all_targets, all_preds_baseline)

    # Print comparison report
    print(f"\n{'='*70}")
    print("OCR INPUT vs IMAGE-ONLY COMPARISON")
    print(f"{'='*70}")
    print(f"\n  Total Samples: {len(results)}")
    print(f"  Time: {elapsed:.1f}s")

    print(f"\n  {'Metric':<25} {'Image-Only':>12} {'OCR+Image':>12} {'Diff':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    b_f1 = metrics_baseline['micro']['f1']
    o_f1 = metrics_ocr['micro']['f1']
    print(f"  {'Micro Precision':<25} {100*metrics_baseline['micro']['precision']:>11.1f}% {100*metrics_ocr['micro']['precision']:>11.1f}% {100*(metrics_ocr['micro']['precision']-metrics_baseline['micro']['precision']):>+9.1f}%")
    print(f"  {'Micro Recall':<25} {100*metrics_baseline['micro']['recall']:>11.1f}% {100*metrics_ocr['micro']['recall']:>11.1f}% {100*(metrics_ocr['micro']['recall']-metrics_baseline['micro']['recall']):>+9.1f}%")
    print(f"  {'Micro F1':<25} {100*b_f1:>11.1f}% {100*o_f1:>11.1f}% {100*(o_f1-b_f1):>+9.1f}%")
    print(f"  {'Macro F1':<25} {100*metrics_baseline['macro']['f1']:>11.1f}% {100*metrics_ocr['macro']['f1']:>11.1f}% {100*(metrics_ocr['macro']['f1']-metrics_baseline['macro']['f1']):>+9.1f}%")
    print(f"  {'JSON Parse Rate':<25} {100*baseline_parse_success/len(results):>11.1f}% {100*ocr_parse_success/len(results):>11.1f}%")

    # Per-class comparison for classes with biggest differences
    print(f"\n{'='*70}")
    print("PER-CLASS F1 COMPARISON (sorted by difference)")
    print(f"{'='*70}")
    all_cls = sorted(set(list(metrics_baseline['per_class'].keys()) + list(metrics_ocr['per_class'].keys())))
    diffs = []
    for cls in all_cls:
        b = metrics_baseline['per_class'].get(cls, {"f1": 0, "support": 0})
        o = metrics_ocr['per_class'].get(cls, {"f1": 0, "support": 0})
        diff = o["f1"] - b["f1"]
        support = max(b["support"], o["support"])
        diffs.append((cls, b["f1"], o["f1"], diff, support))

    diffs.sort(key=lambda x: x[3], reverse=True)
    print(f"  {'Class':<45} {'Base F1':>8} {'OCR F1':>8} {'Diff':>8} {'Sup':>5}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8} {'-'*5}")
    for cls, bf1, of1, diff, sup in diffs:
        marker = "↑" if diff > 0.01 else ("↓" if diff < -0.01 else "=")
        print(f"  {cls:<45} {100*bf1:>7.1f}% {100*of1:>7.1f}% {100*diff:>+7.1f}% {sup:>4} {marker}")

    # Save
    output = {
        "experiment": "ocr_input_comparison",
        "checkpoint": args.checkpoint,
        "n_samples": len(results),
        "metrics_ocr": metrics_ocr,
        "metrics_baseline": metrics_baseline,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
