#!/usr/bin/env python3
"""
evaluate_tta.py
===============
Evaluate a fine-tuned Florence-2 model with Test-Time Augmentation (TTA).

For each image, runs inference on multiple augmented versions:
  1. Original image
  2. Horizontally flipped
  3. Slight brightness increase (+20%)
  4. Slight brightness decrease (-20%)
  5. Center-cropped (90%)

Then aggregates predictions via majority voting on detected categories.
Items that appear in >= threshold (default 2/5 = 40%) of augmentations are kept.

Usage:
  python evaluate_tta.py \
    --checkpoint ./checkpoints_v11/best_model \
    --base-model microsoft/Florence-2-large-ft \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_results_v11_tta.json \
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
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

# Import from existing eval script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_florence2 import (
    parse_prediction,
    extract_class_set,
    normalize_category_name,
    compute_metrics,
    print_report,
    VALID_CATEGORIES,
)


# ── TTA Augmentations ─────────────────────────────────────────────────────────

def get_tta_images(image: Image.Image) -> List[tuple]:
    """Generate augmented versions of an image for TTA."""
    w, h = image.size
    augmented = []

    # 1. Original
    augmented.append(("original", image))

    # 2. Horizontal flip
    augmented.append(("hflip", TF.hflip(image)))

    # 3. Brightness +20%
    augmented.append(("bright+", ImageEnhance.Brightness(image).enhance(1.2)))

    # 4. Brightness -20%
    augmented.append(("bright-", ImageEnhance.Brightness(image).enhance(0.8)))

    # 5. Center crop 90%
    crop_w, crop_h = int(w * 0.9), int(h * 0.9)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.BILINEAR)
    augmented.append(("crop90", cropped))

    return augmented


# ── TTA Inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_on_image(model, processor, image, device, prompt="<OD>", amp_dtype=None):
    """Run inference on a PIL Image (not path)."""
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

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def tta_predict(model, processor, image_path, device, amp_dtype=None, threshold=2):
    """
    Run TTA on a single image and return aggregated prediction.

    threshold: minimum number of augmentations that must agree on a category
               for it to be included. Default 2 out of 5 (40%).
    """
    image = Image.open(image_path).convert("RGB")
    tta_images = get_tta_images(image)

    # Collect all predictions
    all_classes = Counter()
    all_predictions = []
    best_pred_text = None

    for aug_name, aug_image in tta_images:
        pred_text = run_inference_on_image(model, processor, aug_image, device, amp_dtype=amp_dtype)
        pred_parsed = parse_prediction(pred_text)
        classes = extract_class_set(pred_parsed)

        for cls in classes:
            all_classes[cls] += 1

        all_predictions.append({
            "aug": aug_name,
            "text": pred_text,
            "parsed": pred_parsed,
            "classes": classes,
        })

        # Keep original prediction as base
        if aug_name == "original":
            best_pred_text = pred_text

    # Aggregate: keep classes that appear in >= threshold augmentations
    agreed_classes = {cls for cls, count in all_classes.items() if count >= threshold}

    # Build aggregated prediction JSON
    # Use original prediction as template, but filter to agreed classes
    orig_parsed = all_predictions[0]["parsed"]
    if orig_parsed and "items" in orig_parsed:
        # Keep items from original that are in agreed set
        agg_items = [item for item in orig_parsed["items"]
                     if normalize_category_name(item.get("name", "")) in agreed_classes]

        # Add classes found by TTA but missing from original
        orig_classes = {normalize_category_name(item.get("name", ""))
                       for item in orig_parsed["items"]}
        missing = agreed_classes - orig_classes

        # Find these from other augmentations
        for cls in missing:
            for pred in all_predictions[1:]:
                if pred["parsed"] and "items" in pred["parsed"]:
                    for item in pred["parsed"]["items"]:
                        if normalize_category_name(item.get("name", "")) == cls:
                            agg_items.append(item)
                            break
                    else:
                        continue
                    break

        agg_parsed = {"items": agg_items}
        agg_text = json.dumps(agg_parsed)
    else:
        agg_parsed = orig_parsed
        agg_text = best_pred_text

    return agg_text, agg_parsed, all_predictions


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Florence-2 with Test-Time Augmentation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--output", type=str, default="./eval_results_tta.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--threshold", type=int, default=2,
                        help="Min augmentations agreeing on a class (default: 2 out of 5)")
    parser.add_argument("--show-predictions", type=int, default=10)
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    amp_dtype = None
    if args.bf16 and torch.cuda.is_available():
        amp_dtype = torch.bfloat16
        print("Using bf16")

    # Load model
    print(f"\nLoading base model: {args.base_model}...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32,
        attn_implementation="eager",
    )

    print(f"Loading LoRA checkpoint: {args.checkpoint}...")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model = model.merge_and_unload()
    print("  Model loaded and merged.")

    model.to(device)
    model.eval()

    # Load test data
    print(f"\nLoading test data from {args.jsonl}...")
    samples = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"  Evaluating {len(samples)} samples with TTA (5 augmentations, threshold={args.threshold})...")
    print(f"  Total inference runs: {len(samples) * 5}")

    # Run TTA inference
    predictions = []
    start_time = time.time()

    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)

        if not os.path.exists(img_path):
            print(f"  [SKIP] Image not found: {img_path}", file=sys.stderr)
            continue

        agg_text, agg_parsed, tta_details = tta_predict(
            model, processor, img_path, device,
            amp_dtype=amp_dtype, threshold=args.threshold
        )

        target_parsed = json.loads(sample["target"])

        predictions.append({
            "image": sample["image"],
            "target_text": sample["target"],
            "pred_text": agg_text,
            "target_parsed": target_parsed,
            "pred_parsed": agg_parsed,
            "tta_details": tta_details,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(samples)} ({rate:.2f} img/s, ETA: {eta:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\n  TTA inference complete: {len(predictions)} images in {elapsed:.1f}s "
          f"({len(predictions)/elapsed:.2f} img/s)")

    # Compute metrics
    metrics = compute_metrics(predictions)
    print_report(metrics)

    # Show sample predictions
    if args.show_predictions > 0:
        print(f"\n{'='*70}")
        print(f"SAMPLE PREDICTIONS (first {args.show_predictions})")
        print(f"{'='*70}")
        for rec in predictions[:args.show_predictions]:
            target_cls = extract_class_set(rec["target_parsed"])
            pred_cls = extract_class_set(rec["pred_parsed"])

            print(f"\n  Image: {rec['image']}")
            print(f"  Target: {rec['target_text'][:150]}...")
            print(f"  TTA Pred: {rec['pred_text'][:150]}...")

            if target_cls == pred_cls:
                print(f"  Classes: ✓ MATCH ({len(target_cls)} classes)")
            else:
                print(f"  Classes: ✗ MISMATCH")
                matched = target_cls & pred_cls
                missing = target_cls - pred_cls
                extra = pred_cls - target_cls
                if matched:
                    print(f"  Matched: {matched}")
                if missing:
                    print(f"  Missing: {missing}")
                if extra:
                    print(f"  Extra: {extra}")

            # Show per-augmentation breakdown
            if "tta_details" in rec:
                aug_summary = []
                for d in rec["tta_details"]:
                    aug_summary.append(f"{d['aug']}:{d['classes']}")
                print(f"  TTA breakdown: {len(rec['tta_details'])} augmentations")

    # Save results
    save_data = {
        "metrics": metrics,
        "predictions": [
            {
                "image": r["image"],
                "target": r["target_text"],
                "predicted": r["pred_text"],
            }
            for r in predictions
        ],
        "config": {
            "checkpoint": args.checkpoint,
            "threshold": args.threshold,
            "n_augmentations": 5,
            "augmentations": ["original", "hflip", "bright+", "bright-", "crop90"],
        },
    }
    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
