#!/usr/bin/env python3
"""
Quick script to generate JSONL from existing crop files.
Run this if generate_crop_dataset.py was interrupted after crops were saved.

Reads the COCO annotations to match gt_crop_XXXXX.jpg files to their labels.
For od_crop_XXXXX.jpg files, skips them (no reliable labels without the full pipeline).

Usage:
  python fix_crop_jsonl.py --data-dir . --crop-dir ./crop_data
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


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


def coco_bbox_to_xyxy(bbox):
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--crop-dir", type=str, default="./crop_data")
    args = parser.parse_args()

    crops_dir = os.path.join(args.crop_dir, "crops")

    # Count existing crops
    gt_crops = sorted([f for f in os.listdir(crops_dir) if f.startswith("gt_crop_")])
    od_crops = sorted([f for f in os.listdir(crops_dir) if f.startswith("od_crop_")])
    print(f"Found {len(gt_crops)} GT crops, {len(od_crops)} OD crops")

    # We need to regenerate labels for GT crops from COCO annotations
    # The GT crops were generated in order: iterate through splits → images → annotations
    # We'll reconstruct the same order

    print("\nReconstructing GT crop labels from COCO annotations...")
    gt_entries = []
    crop_idx = 0

    for split in ["train", "valid"]:
        coco_path = os.path.join(args.data_dir, split, "_annotations.coco.json")
        if not os.path.exists(coco_path):
            print(f"  WARNING: {coco_path} not found, skipping {split}")
            continue

        with open(coco_path) as f:
            coco = json.load(f)

        # Build image_id → annotations mapping (same order as original script)
        img_anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)

        img_lookup = {img["id"]: img for img in coco["images"]}

        for img_id, anns in img_anns.items():
            if img_id not in img_lookup:
                continue
            img_info = img_lookup[img_id]
            img_path = os.path.join(args.data_dir, split, img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            for ann in anns:
                if ann["category_id"] in EXCLUDED_CATEGORY_IDS:
                    continue
                cat_name = PANTRY_CATEGORIES.get(ann["category_id"])
                if not cat_name:
                    continue

                # Check if the corresponding crop file exists
                crop_filename = f"gt_crop_{crop_idx:05d}.jpg"
                crop_path = os.path.join(crops_dir, crop_filename)

                if os.path.exists(crop_path):
                    gt_entries.append({
                        "image": os.path.join("crops", crop_filename),
                        "prefix": "<OD>",
                        "target": json.dumps({
                            "items": [{"name": cat_name, "confidence": "high"}]
                        }),
                        "source": "gt",
                        "split": split,
                    })
                crop_idx += 1

    print(f"  Matched {len(gt_entries)} GT crops to labels")

    # Split into train/valid
    train_entries = [e for e in gt_entries if e["split"] == "train"]
    val_entries = [e for e in gt_entries if e["split"] == "valid"]

    # Save JSONL
    train_path = os.path.join(args.crop_dir, "train_crops.jsonl")
    val_path = os.path.join(args.crop_dir, "val_crops.jsonl")

    for entries, path in [(train_entries, train_path), (val_entries, val_path)]:
        with open(path, "w") as f:
            for e in entries:
                row = {k: v for k, v in e.items() if k != "split"}
                f.write(json.dumps(row) + "\n")
        print(f"  Saved {len(entries)} entries to {path}")

    # Category distribution
    cat_counts = Counter()
    for e in gt_entries:
        target = json.loads(e["target"])
        for item in target["items"]:
            cat_counts[item["name"]] += 1

    print(f"\nCategory distribution ({len(gt_entries)} total):")
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat:45s} {cat_counts[cat]:>5}")

    # Save metadata
    meta = {
        "total_crops": len(gt_entries),
        "gt_crops": len(gt_entries),
        "od_crops_available": len(od_crops),
        "od_crops_used": 0,
        "train_crops": len(train_entries),
        "val_crops": len(val_entries),
        "note": "Generated from interrupted run. GT crops only (reliable labels).",
        "category_distribution": dict(cat_counts),
    }
    meta_path = os.path.join(args.crop_dir, "crop_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    print(f"\nDone! Ready for: sbatch run_v13.sh")


if __name__ == "__main__":
    main()
