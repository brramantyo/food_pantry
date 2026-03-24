#!/usr/bin/env python3
"""
evaluate_ensemble.py
====================
Ensemble evaluation: combine predictions from two Florence-2 checkpoints.

Strategy:
  1. Run inference with both models on each image
  2. Aggregate predictions:
     - UNION mode: keep all classes from both models (boosts recall)
     - INTERSECTION mode: keep only classes both models agree on (boosts precision)
     - MAJORITY mode: keep classes that at least 1 model predicts, weighted by confidence
  3. Default: UNION with agreement bonus (if both agree → keep, if only 1 → keep but lower priority)

Usage:
  python evaluate_ensemble.py \
    --checkpoint1 ./checkpoints_v9/best_model \
    --checkpoint2 ./checkpoints_v11/best_model \
    --base-model microsoft/Florence-2-large-ft \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --mode union \
    --bf16
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Set

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_florence2 import (
    parse_prediction,
    extract_class_set,
    normalize_category_name,
    compute_metrics,
    print_report,
    VALID_CATEGORIES,
)


@torch.no_grad()
def run_inference(model, processor, image_path, device, prompt="<OD>", amp_dtype=None):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    if amp_dtype is not None:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024, num_beams=3, early_stopping=True,
            )
    else:
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, num_beams=3, early_stopping=True,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def load_model(base_model_name, checkpoint_path, device, amp_dtype=None):
    """Load a model with LoRA checkpoint."""
    print(f"  Loading {checkpoint_path}...")
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model, processor


def ensemble_predictions(parsed1, parsed2, mode="union"):
    """
    Combine predictions from two models.

    Modes:
      - union: keep all classes from both (maximize recall)
      - intersection: keep only classes both agree on (maximize precision)
      - smart: union, but prefer items from the model that has higher agreement
    """
    classes1 = extract_class_set(parsed1)
    classes2 = extract_class_set(parsed2)

    if mode == "union":
        combined_classes = classes1 | classes2
    elif mode == "intersection":
        combined_classes = classes1 & classes2
        # If intersection is empty but both had predictions, fall back to union
        if not combined_classes and (classes1 or classes2):
            combined_classes = classes1 | classes2
    elif mode == "smart":
        # Keep intersection (high confidence) + classes from model with better precision
        agreed = classes1 & classes2
        # For non-agreed: keep from both but could weight later
        combined_classes = classes1 | classes2
    else:
        combined_classes = classes1 | classes2

    # Build combined items list
    items_by_class = {}

    # First pass: items from model 1
    if parsed1 and "items" in parsed1:
        for item in parsed1["items"]:
            cls = normalize_category_name(item.get("name", ""))
            if cls in combined_classes and cls not in items_by_class:
                items_by_class[cls] = item

    # Second pass: items from model 2 (fill gaps)
    if parsed2 and "items" in parsed2:
        for item in parsed2["items"]:
            cls = normalize_category_name(item.get("name", ""))
            if cls in combined_classes and cls not in items_by_class:
                items_by_class[cls] = item

    if items_by_class:
        return {"items": list(items_by_class.values())}
    return {"items": []}


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation of two Florence-2 models")
    parser.add_argument("--checkpoint1", type=str, required=True, help="First model checkpoint")
    parser.add_argument("--checkpoint2", type=str, required=True, help="Second model checkpoint")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--output", type=str, default="./eval_results_ensemble.json")
    parser.add_argument("--mode", type=str, default="union", choices=["union", "intersection", "smart"])
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--show-predictions", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    amp_dtype = None
    if args.bf16 and torch.cuda.is_available():
        amp_dtype = torch.bfloat16
        print("Using bf16")

    # Load both models sequentially (can't fit both in VRAM simultaneously)
    # Strategy: run model1 on all images, save results, unload, load model2, run again

    # ── Model 1 ────────────────────────────────────────────────────────────
    print(f"\n=== Loading Model 1 ===")
    model1, processor1 = load_model(args.base_model, args.checkpoint1, device, amp_dtype)

    print(f"\nLoading test data from {args.jsonl}...")
    samples = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"  Running Model 1 on {len(samples)} samples...")
    preds1 = []
    start = time.time()
    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)
        if not os.path.exists(img_path):
            preds1.append(None)
            continue
        text = run_inference(model1, processor1, img_path, device, amp_dtype=amp_dtype)
        preds1.append(parse_prediction(text))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"    Progress: {i+1}/{len(samples)} ({rate:.1f} img/s)")

    elapsed = time.time() - start
    print(f"  Model 1 done: {elapsed:.1f}s")

    # Unload model 1
    del model1
    torch.cuda.empty_cache()

    # ── Model 2 ────────────────────────────────────────────────────────────
    print(f"\n=== Loading Model 2 ===")
    model2, processor2 = load_model(args.base_model, args.checkpoint2, device, amp_dtype)

    print(f"  Running Model 2 on {len(samples)} samples...")
    preds2 = []
    start = time.time()
    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)
        if not os.path.exists(img_path):
            preds2.append(None)
            continue
        text = run_inference(model2, processor2, img_path, device, amp_dtype=amp_dtype)
        preds2.append(parse_prediction(text))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"    Progress: {i+1}/{len(samples)} ({rate:.1f} img/s)")

    elapsed = time.time() - start
    print(f"  Model 2 done: {elapsed:.1f}s")

    del model2
    torch.cuda.empty_cache()

    # ── Ensemble ───────────────────────────────────────────────────────────
    print(f"\n=== Ensemble ({args.mode} mode) ===")

    predictions = []
    agree_count = 0
    for i, sample in enumerate(samples):
        target_parsed = json.loads(sample["target"])
        p1 = preds1[i]
        p2 = preds2[i]

        if p1 is None and p2 is None:
            continue

        # Check agreement
        c1 = extract_class_set(p1)
        c2 = extract_class_set(p2)
        if c1 == c2:
            agree_count += 1

        ensemble_parsed = ensemble_predictions(p1, p2, mode=args.mode)
        ensemble_text = json.dumps(ensemble_parsed)

        predictions.append({
            "image": sample["image"],
            "target_text": sample["target"],
            "pred_text": ensemble_text,
            "target_parsed": target_parsed,
            "pred_parsed": ensemble_parsed,
            "model1_classes": sorted(c1),
            "model2_classes": sorted(c2),
            "ensemble_classes": sorted(extract_class_set(ensemble_parsed)),
        })

    print(f"  Model agreement: {agree_count}/{len(predictions)} ({100*agree_count/len(predictions):.1f}%)")

    # Compute metrics
    metrics = compute_metrics(predictions)
    print_report(metrics)

    # Show predictions
    if args.show_predictions > 0:
        print(f"\n{'='*70}")
        print(f"SAMPLE PREDICTIONS (first {args.show_predictions})")
        print(f"{'='*70}")
        for rec in predictions[:args.show_predictions]:
            target_cls = extract_class_set(rec["target_parsed"])
            pred_cls = extract_class_set(rec["pred_parsed"])

            print(f"\n  Image: {rec['image']}")
            print(f"  Target:   {sorted(target_cls)}")
            print(f"  Model 1:  {rec['model1_classes']}")
            print(f"  Model 2:  {rec['model2_classes']}")
            print(f"  Ensemble: {rec['ensemble_classes']}")

            if target_cls == pred_cls:
                print(f"  Result: ✓ MATCH")
            else:
                missing = target_cls - pred_cls
                extra = pred_cls - target_cls
                if missing:
                    print(f"  Missing: {missing}")
                if extra:
                    print(f"  Extra: {extra}")

    # Save
    save_data = {
        "metrics": metrics,
        "config": {
            "checkpoint1": args.checkpoint1,
            "checkpoint2": args.checkpoint2,
            "mode": args.mode,
        },
    }
    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
