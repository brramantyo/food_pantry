#!/usr/bin/env python3
"""
generate_crop_dataset.py
========================
Generate a single-item crop dataset from training images using the fine-tuned OD model.

Pipeline:
  1. Load fine-tuned OD model
  2. For each training image, run OD to get bounding boxes
  3. Also use COCO ground-truth bboxes (guaranteed correct labels)
  4. Crop each bbox with padding → save as individual image
  5. Assign label: match GT bbox by IoU, or use v11 classifier as fallback
  6. Output: JSONL file with crop image paths + category labels

This creates training data so a classifier can learn from BOTH full shelf images
AND individual item crops, fixing the precision problem in the OD→crop→classify pipeline.

Usage:
  python generate_crop_dataset.py \
    --data-dir . \
    --od-checkpoint ./checkpoints_od_v1/best_model \
    --cls-checkpoint ./checkpoints_v11/best_model \
    --base-model microsoft/Florence-2-large-ft \
    --output-dir ./crop_data \
    --bf16
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel


# ── Constants ──────────────────────────────────────────────────────────────────

OD_PROMPT = "<OD>"
CLS_PROMPT = "<OD>"

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

# COCO category ID → name (same as OD training)
PANTRY_CATEGORIES = {
    1: "Baby Food",
    2: "Beans and Legumes - Canned or Dried",
    3: "Bread and Bakery Products",
    4: "Canned Tomato Products",
    5: "Carbohydrate Meal",
    7: "Condiments and Sauces",
    8: "Dairy and Dairy Alternatives",
    9: "Desserts and Sweets",
    10: "Drinks",
    11: "Fresh Fruit",
    13: "Fruits - Canned or Processed",
    14: "Granola Products",
    15: "Meat and Poultry - Canned",
    16: "Meat and Poultry - Fresh",
    17: "Nut Butters and Nuts",
    19: "Ready Meals",
    20: "Savory Snacks and Crackers",
    21: "Seafood - Canned",
    22: "Soup",
    24: "Vegetables - Canned",
    25: "Vegetables - Fresh",
}

EXCLUDED_CATEGORY_IDS = {6, 12, 18, 23}


def normalize_category(name):
    return CANONICAL_MAP.get(name.lower().strip(), name)


# ── Bbox Utils ─────────────────────────────────────────────────────────────────

def parse_od_bboxes(text, img_width, img_height):
    """Parse Florence-2 OD output → list of [x1, y1, x2, y2] in pixel coords."""
    bboxes = []
    pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    for m in re.finditer(pattern, text):
        x1 = int(m.group(1)) / 999 * img_width
        y1 = int(m.group(2)) / 999 * img_height
        x2 = int(m.group(3)) / 999 * img_width
        y2 = int(m.group(4)) / 999 * img_height
        if x2 > x1 + 5 and y2 > y1 + 5:  # min 5px
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def coco_bbox_to_xyxy(bbox):
    """Convert COCO [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def deduplicate_bboxes(bboxes, iou_threshold=0.5):
    if not bboxes:
        return []
    bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    for bbox in bboxes:
        is_dup = False
        for kept in keep:
            if compute_iou(bbox, kept) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(bbox)
    return keep


def crop_with_padding(image, bbox, padding_ratio=0.15):
    """Crop image with padding. Returns None if too small."""
    x1, y1, x2, y2 = bbox
    w, h = image.size
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    if crop.width < 32 or crop.height < 32:
        return None
    return crop


# ── Label Assignment ───────────────────────────────────────────────────────────

def match_bbox_to_gt(pred_bbox, gt_annotations, iou_threshold=0.3):
    """
    Match a predicted bbox to ground-truth annotations by IoU.
    Returns the GT category name if match found, else None.
    """
    best_iou = 0
    best_cat = None
    for ann in gt_annotations:
        if ann["category_id"] in EXCLUDED_CATEGORY_IDS:
            continue
        gt_bbox = coco_bbox_to_xyxy(ann["bbox"])
        iou = compute_iou(pred_bbox, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_cat = PANTRY_CATEGORIES.get(ann["category_id"])
    if best_iou >= iou_threshold and best_cat:
        return best_cat
    return None


def parse_classification(text):
    """Parse v11 classifier output (JSON format) → set of category names."""
    text = text.strip()
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
    return set()


# ── Main ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_model(model, processor, image, prompt, device, amp_dtype, max_tokens=1024):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_tokens,
        num_beams=3,
        early_stopping=True,
    )
    if amp_dtype:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            generated_ids = model.generate(**gen_kwargs)
    else:
        generated_ids = model.generate(**gen_kwargs)
    return processor.batch_decode(generated_ids, skip_special_tokens=False)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--od-checkpoint", type=str, default="./checkpoints_od_v1/best_model")
    parser.add_argument("--cls-checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--output-dir", type=str, default="./crop_data")
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="IoU threshold for matching predicted bbox to GT")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use-gt-only", action="store_true",
                        help="Only use GT bboxes (skip OD model entirely)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    os.makedirs(args.output_dir, exist_ok=True)
    crops_dir = os.path.join(args.output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # ── Load COCO annotations ──────────────────────────────────────────────
    print("Loading COCO annotations...")
    splits = {}
    for split in ["train", "valid"]:
        coco_path = os.path.join(args.data_dir, split, "_annotations.coco.json")
        if os.path.exists(coco_path):
            with open(coco_path) as f:
                coco = json.load(f)
            # Build image_id → annotations
            img_anns = {}
            for ann in coco["annotations"]:
                img_id = ann["image_id"]
                if img_id not in img_anns:
                    img_anns[img_id] = []
                img_anns[img_id].append(ann)
            img_lookup = {img["id"]: img for img in coco["images"]}
            splits[split] = {"img_anns": img_anns, "img_lookup": img_lookup}
            print(f"  {split}: {len(img_lookup)} images, {sum(len(v) for v in img_anns.values())} annotations")

    if not splits:
        print("ERROR: No COCO annotations found!")
        sys.exit(1)

    # ── Strategy 1: GT-only crops (always do this) ─────────────────────────
    print("\n=== Generating crops from GROUND-TRUTH bboxes ===")
    gt_crops = []
    crop_idx = 0

    for split_name, split_data in splits.items():
        for img_id, anns in split_data["img_anns"].items():
            if img_id not in split_data["img_lookup"]:
                continue
            img_info = split_data["img_lookup"][img_id]
            img_path = os.path.join(args.data_dir, split_name, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  [WARN] Failed to load {img_path}: {e}")
                continue

            for ann in anns:
                if ann["category_id"] in EXCLUDED_CATEGORY_IDS:
                    continue
                cat_name = PANTRY_CATEGORIES.get(ann["category_id"])
                if not cat_name:
                    continue

                bbox = coco_bbox_to_xyxy(ann["bbox"])
                crop = crop_with_padding(image, bbox, args.padding)
                if crop is None:
                    continue

                crop_filename = f"gt_crop_{crop_idx:05d}.jpg"
                crop_path = os.path.join(crops_dir, crop_filename)
                crop.save(crop_path, quality=90)

                gt_crops.append({
                    "image": os.path.join("crops", crop_filename),
                    "source": "gt",
                    "split": split_name,
                    "original_image": img_info["file_name"],
                    "category": cat_name,
                })
                crop_idx += 1

    print(f"  Generated {len(gt_crops)} GT crops")
    cat_counts = Counter(c["category"] for c in gt_crops)
    for cat in sorted(cat_counts.keys()):
        print(f"    {cat:45s} {cat_counts[cat]:>5}")

    # ── Strategy 2: OD-detected crops (optional) ──────────────────────────
    od_crops = []
    if not args.use_gt_only:
        print(f"\n=== Generating crops from OD model detections ===")
        print(f"Loading OD model: {args.od_checkpoint}")
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, trust_remote_code=True, torch_dtype=torch.float32
        )
        od_model = PeftModel.from_pretrained(base_model, args.od_checkpoint)
        od_model = od_model.to(device).eval()

        # Also load classifier for labeling OD crops that don't match GT
        print(f"Loading classifier: {args.cls_checkpoint}")
        cls_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, trust_remote_code=True, torch_dtype=torch.float32
        )
        cls_model = PeftModel.from_pretrained(cls_base, args.cls_checkpoint)
        cls_model = cls_model.to(device).eval()

        n_gt_matched = 0
        n_cls_labeled = 0
        n_skipped = 0

        for split_name, split_data in splits.items():
            print(f"\n  Processing {split_name} split...")
            for i, (img_id, anns) in enumerate(split_data["img_anns"].items()):
                if img_id not in split_data["img_lookup"]:
                    continue
                img_info = split_data["img_lookup"][img_id]
                img_path = os.path.join(args.data_dir, split_name, img_info["file_name"])
                if not os.path.exists(img_path):
                    continue

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                # Run OD
                od_text = run_model(od_model, processor, image, OD_PROMPT, device, amp_dtype)
                pred_bboxes = parse_od_bboxes(od_text, image.width, image.height)
                pred_bboxes = deduplicate_bboxes(pred_bboxes)

                for bbox in pred_bboxes:
                    crop = crop_with_padding(image, bbox, args.padding)
                    if crop is None:
                        continue

                    # Try to match to GT first
                    cat_name = match_bbox_to_gt(bbox, anns, args.iou_threshold)

                    if cat_name:
                        n_gt_matched += 1
                        source = "od_gt"
                    else:
                        # Use v11 classifier as fallback
                        cls_text = run_model(cls_model, processor, crop, CLS_PROMPT, device, amp_dtype, max_tokens=512)
                        preds = parse_classification(cls_text)
                        if len(preds) == 1:
                            cat_name = list(preds)[0]
                            n_cls_labeled += 1
                            source = "od_cls"
                        else:
                            # Ambiguous — skip
                            n_skipped += 1
                            continue

                    crop_filename = f"od_crop_{crop_idx:05d}.jpg"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    crop.save(crop_path, quality=90)

                    od_crops.append({
                        "image": os.path.join("crops", crop_filename),
                        "source": source,
                        "split": split_name,
                        "original_image": img_info["file_name"],
                        "category": cat_name,
                    })
                    crop_idx += 1

                if (i + 1) % 50 == 0:
                    print(f"    Processed {i+1} images... ({len(od_crops)} OD crops so far)")

        print(f"\n  OD crops: {len(od_crops)} total")
        print(f"    GT-matched: {n_gt_matched}")
        print(f"    Classifier-labeled: {n_cls_labeled}")
        print(f"    Skipped (ambiguous): {n_skipped}")

        # Free GPU memory
        del od_model, cls_model, base_model, cls_base
        torch.cuda.empty_cache()

    # ── Save combined dataset ──────────────────────────────────────────────
    all_crops = gt_crops + od_crops
    print(f"\n=== TOTAL: {len(all_crops)} crops ===")

    # Save as JSONL (Florence-2 training format)
    train_crops = [c for c in all_crops if c["split"] == "train"]
    val_crops = [c for c in all_crops if c["split"] == "valid"]

    # Convert to Florence-2 JSONL format
    def to_florence2_jsonl(crops, jsonl_path):
        with open(jsonl_path, "w") as f:
            for c in crops:
                entry = {
                    "image": c["image"],
                    "prefix": "<OD>",
                    "target": json.dumps({
                        "items": [{"name": c["category"], "confidence": "high"}]
                    }),
                    "source": c["source"],
                }
                f.write(json.dumps(entry) + "\n")
        print(f"  Saved {len(crops)} entries to {jsonl_path}")

    to_florence2_jsonl(train_crops, os.path.join(args.output_dir, "train_crops.jsonl"))
    to_florence2_jsonl(val_crops, os.path.join(args.output_dir, "val_crops.jsonl"))

    # Also save raw metadata
    with open(os.path.join(args.output_dir, "crop_metadata.json"), "w") as f:
        json.dump({
            "total_crops": len(all_crops),
            "gt_crops": len(gt_crops),
            "od_crops": len(od_crops),
            "train_crops": len(train_crops),
            "val_crops": len(val_crops),
            "category_distribution": dict(Counter(c["category"] for c in all_crops)),
        }, f, indent=2)

    print("\nDone! Next step: train v13 classifier on mixed full-image + crop data.")


if __name__ == "__main__":
    main()
