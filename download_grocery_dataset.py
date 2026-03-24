#!/usr/bin/env python3
"""
download_grocery_dataset.py
===========================
Download the Grocery Store Dataset and convert it to Florence-2 JSONL format
with food pantry category mapping.

Grocery Store Dataset: 5125 images, 81 fine-grained classes → mapped to our 21 categories.

Usage:
  python download_grocery_dataset.py --output-dir ./grocery_data

Steps:
  1. Clone GroceryStoreDataset repo
  2. Map 81 grocery classes → 21 food pantry categories
  3. Generate Florence-2 compatible JSONL (same format as train_v5.jsonl)
  4. Only use classes that map to our categories (skip unmappable ones)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from collections import Counter


# ── Mapping: Grocery Store coarse class → Food Pantry category ─────────────────
# Only map classes that clearly correspond to our 21 categories.
# Classes without a good match are skipped.

COARSE_TO_PANTRY = {
    # Fruits → Fresh Fruit
    "Apple": "Fresh Fruit",
    "Avocado": "Fresh Fruit",
    "Banana": "Fresh Fruit",
    "Kiwi": "Fresh Fruit",
    "Lemon": "Fresh Fruit",
    "Lime": "Fresh Fruit",
    "Mango": "Fresh Fruit",
    "Melon": "Fresh Fruit",
    "Nectarine": "Fresh Fruit",
    "Orange": "Fresh Fruit",
    "Papaya": "Fresh Fruit",
    "Passion-Fruit": "Fresh Fruit",
    "Peach": "Fresh Fruit",
    "Pear": "Fresh Fruit",
    "Pineapple": "Fresh Fruit",
    "Plum": "Fresh Fruit",
    "Pomegranate": "Fresh Fruit",
    "Red-Grapefruit": "Fresh Fruit",
    "Satsumas": "Fresh Fruit",

    # Packages - Juice → Drinks
    "Juice": "Drinks",

    # Packages - Milk/Dairy → Dairy and Dairy Alternatives
    "Milk": "Dairy and Dairy Alternatives",
    "Oatghurt": "Dairy and Dairy Alternatives",
    "Oat-Milk": "Dairy and Dairy Alternatives",
    "Sour-Cream": "Dairy and Dairy Alternatives",
    "Sour-Milk": "Dairy and Dairy Alternatives",
    "Soyghurt": "Dairy and Dairy Alternatives",
    "Soy-Milk": "Dairy and Dairy Alternatives",
    "Yoghurt": "Dairy and Dairy Alternatives",

    # Vegetables → Vegetables - Fresh
    "Asparagus": "Vegetables - Fresh",
    "Aubergine": "Vegetables - Fresh",
    "Cabbage": "Vegetables - Fresh",
    "Carrots": "Vegetables - Fresh",
    "Cucumber": "Vegetables - Fresh",
    "Garlic": "Vegetables - Fresh",
    "Ginger": "Vegetables - Fresh",
    "Leek": "Vegetables - Fresh",
    "Mushroom": "Vegetables - Fresh",
    "Onion": "Vegetables - Fresh",
    "Pepper": "Vegetables - Fresh",
    "Potato": "Vegetables - Fresh",
    "Red-Beet": "Vegetables - Fresh",
    "Tomato": "Vegetables - Fresh",
    "Zucchini": "Vegetables - Fresh",
}

# Package type mapping based on grocery item type
PACKAGE_TYPE_MAP = {
    "Fresh Fruit": "loose",
    "Vegetables - Fresh": "loose",
    "Drinks": "carton",
    "Dairy and Dairy Alternatives": "carton",
}


def guess_package_type(pantry_class, fine_class_name):
    """Guess package type from class name and pantry category."""
    name_lower = fine_class_name.lower()

    # Bottles
    if "juice" in name_lower or "milk" in name_lower:
        return "carton"
    if "yoghurt" in name_lower or "soyghurt" in name_lower or "oatghurt" in name_lower:
        return "container"
    if "cream" in name_lower:
        return "container"

    return PACKAGE_TYPE_MAP.get(pantry_class, "package")


def main():
    parser = argparse.ArgumentParser(description="Download and convert Grocery Store Dataset")
    parser.add_argument("--output-dir", type=str, default="./grocery_data",
                        help="Output directory for converted data")
    parser.add_argument("--repo-dir", type=str, default="./GroceryStoreDataset",
                        help="Where to clone the dataset repo")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip git clone (if already downloaded)")
    args = parser.parse_args()

    # ── Step 1: Clone dataset ──────────────────────────────────────────────
    if not args.skip_download:
        if os.path.exists(args.repo_dir):
            print(f"Directory {args.repo_dir} already exists, pulling latest...")
            subprocess.run(["git", "-C", args.repo_dir, "pull"], check=True)
        else:
            print("Cloning GroceryStoreDataset...")
            subprocess.run([
                "git", "clone",
                "https://github.com/marcusklasson/GroceryStoreDataset.git",
                args.repo_dir
            ], check=True)
        print(f"Dataset at: {args.repo_dir}")
    else:
        print(f"Skipping download, using existing: {args.repo_dir}")

    # ── Step 2: Parse classes.csv ──────────────────────────────────────────
    classes_csv = os.path.join(args.repo_dir, "dataset", "classes.csv")
    if not os.path.exists(classes_csv):
        print(f"ERROR: {classes_csv} not found!")
        sys.exit(1)

    # Build class_id → (fine_name, coarse_name) mapping
    class_map = {}
    with open(classes_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row["Class ID (int)"])
            fine_name = row["Class Name (str)"]
            coarse_name = row["Coarse Class Name (str)"]
            class_map[class_id] = (fine_name, coarse_name)

    print(f"Loaded {len(class_map)} fine-grained classes")

    # ── Step 3: Process train/val/test splits ──────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    total_mapped = 0
    total_skipped = 0
    category_counts = Counter()

    for split_name, split_file in [("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")]:
        split_path = os.path.join(args.repo_dir, "dataset", split_file)
        if not os.path.exists(split_path):
            print(f"  SKIP: {split_path} not found")
            continue

        jsonl_path = os.path.join(args.output_dir, f"grocery_{split_name}.jsonl")
        mapped_count = 0
        skipped_count = 0

        with open(split_path, "r") as f_in, open(jsonl_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue

                img_rel_path = parts[0].strip()
                fine_label = int(parts[1].strip())
                coarse_label = int(parts[2].strip())

                if fine_label not in class_map:
                    skipped_count += 1
                    continue

                fine_name, coarse_name = class_map[fine_label]

                # Map to pantry category
                pantry_category = COARSE_TO_PANTRY.get(coarse_name)
                if pantry_category is None:
                    skipped_count += 1
                    continue

                # Build full image path (relative to food_pantry dir)
                img_full = os.path.join(args.repo_dir, "dataset", img_rel_path.lstrip("/"))
                if not os.path.exists(img_full):
                    skipped_count += 1
                    continue

                # Create Florence-2 format target
                pkg_type = guess_package_type(pantry_category, fine_name)
                target = {
                    "items": [{
                        "name": pantry_category,
                        "package_type": pkg_type,
                        "confidence": "high"
                    }]
                }

                # Use path relative to food_pantry working dir
                record = {
                    "image": img_full,
                    "target": json.dumps(target, ensure_ascii=False)
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                mapped_count += 1
                category_counts[pantry_category] += 1

        total_mapped += mapped_count
        total_skipped += skipped_count
        print(f"  {split_name}: {mapped_count} mapped, {skipped_count} skipped → {jsonl_path}")

    # ── Step 4: Print summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GROCERY STORE DATASET CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total mapped:  {total_mapped}")
    print(f"  Total skipped: {total_skipped}")
    print(f"\n  Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {count:5d}  {cat}")

    # ── Step 5: Create merged training JSONL ───────────────────────────────
    # Merge grocery train + food pantry train for two-stage training
    grocery_train = os.path.join(args.output_dir, "grocery_train.jsonl")
    pantry_train = "./florence2_data/train_v5.jsonl"
    merged_path = os.path.join(args.output_dir, "merged_train.jsonl")

    if os.path.exists(grocery_train) and os.path.exists(pantry_train):
        grocery_count = 0
        pantry_count = 0
        with open(merged_path, "w", encoding="utf-8") as f_out:
            # Write all food pantry data first
            with open(pantry_train, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if line.strip():
                        f_out.write(line)
                        pantry_count += 1
            # Append grocery data
            with open(grocery_train, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if line.strip():
                        f_out.write(line)
                        grocery_count += 1

        print(f"\n  Merged training set: {merged_path}")
        print(f"    Food pantry: {pantry_count}")
        print(f"    Grocery:     {grocery_count}")
        print(f"    Total:       {pantry_count + grocery_count}")

    print(f"\nDone! Use merged_train.jsonl for two-stage training.")


if __name__ == "__main__":
    main()
