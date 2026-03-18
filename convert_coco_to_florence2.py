#!/usr/bin/env python3
"""
convert_coco_to_florence2.py
============================
Convert COCO-format pantry food annotations into Florence-2 fine-tuning JSONL format,
enriched with USDA FoodData Central metadata.

Usage:
    python convert_coco_to_florence2.py \
        --data-dir ./Food-Items-4/ \
        --mapping usda_mapping.json \
        --output-dir ./florence2_data/

Expected directory structure for --data-dir:
    Food-Items-4/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    └── test/
        ├── _annotations.coco.json
        └── *.jpg

Each output JSONL line has the format:
    {
        "image": "train/image001.jpg",
        "prompt": "<STRUCTURED_PANTRY_OUTPUT>",
        "target": "{\"items\": [...]}"
    }

The structured target JSON per image groups bounding boxes by class and enriches
each entry with USDA category group, package type, and search terms from the
mapping table.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Constants ──────────────────────────────────────────────────────────────────

PROMPT_TOKEN = "<STRUCTURED_PANTRY_OUTPUT>"
SPLITS = ["train", "valid", "test"]
COCO_ANNOTATION_FILENAME = "_annotations.coco.json"

# The dummy class exported by Roboflow — always id 0, zero annotations
DUMMY_CLASS_NAME = "Food-Items-Food-Items-4Fxl"
DUMMY_CLASS_ID = 0


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    """Load and return a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_usda_lookup(mapping_path: str) -> Dict[str, dict]:
    """
    Build a lookup dict from class_name -> USDA mapping entry.

    Returns:
        dict keyed by class_name with values containing category_group,
        typical_package_types, usda_search_terms, usda_food_group, and notes.
    """
    mapping = load_json(mapping_path)
    lookup = {}
    for entry in mapping["categories"]:
        lookup[entry["class_name"]] = entry
    return lookup


def build_coco_lookups(coco: dict) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, List[dict]]]:
    """
    Parse a COCO annotation dict and build convenient lookup structures.

    Returns:
        - cat_lookup: category_id -> category name
        - img_lookup: image_id -> file_name
        - img_annotations: image_id -> list of annotation dicts
    """
    # Category id → name (skip the dummy class)
    cat_lookup = {}
    for cat in coco.get("categories", []):
        if cat["id"] == DUMMY_CLASS_ID or cat["name"] == DUMMY_CLASS_NAME:
            continue
        cat_lookup[cat["id"]] = cat["name"]

    # Image id → file_name
    img_lookup = {}
    for img in coco.get("images", []):
        img_lookup[img["id"]] = img["file_name"]

    # Image id → list of annotations (filtering out dummy class)
    img_annotations: Dict[int, List[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        cat_id = ann["category_id"]
        if cat_id == DUMMY_CLASS_ID or cat_id not in cat_lookup:
            continue
        img_annotations[ann["image_id"]].append(ann)

    return cat_lookup, img_lookup, img_annotations


def build_structured_target(
    annotations: List[dict],
    cat_lookup: Dict[int, str],
    usda_lookup: Dict[str, dict],
) -> dict:
    """
    Given a list of COCO annotations for a single image, group by class,
    count occurrences, and enrich with USDA metadata.

    Returns:
        A dict with an "items" key containing the structured list.
    """
    # Count annotations per category id
    class_counts: Counter = Counter()
    for ann in annotations:
        class_counts[ann["category_id"]] += 1

    items = []
    # Sort by category id for deterministic output
    for cat_id, count in sorted(class_counts.items()):
        class_name = cat_lookup.get(cat_id)
        if class_name is None:
            continue

        usda_info = usda_lookup.get(class_name)

        if usda_info:
            category_group = usda_info["category_group"]
            # Default package type: first (most common) in the list
            package_type = usda_info["typical_package_types"][0]
            # Primary USDA search term: first in the list
            usda_search_term = usda_info["usda_search_terms"][0]
        else:
            # Fallback if class not found in mapping (shouldn't happen with correct mapping)
            category_group = "Unknown"
            package_type = "unknown"
            usda_search_term = class_name.lower()

        items.append({
            "name": class_name,
            "category": category_group,
            "package_type": package_type,
            "count": count,
            "usda_search_term": usda_search_term,
            "confidence": "high",
        })

    return {"items": items}


def process_split(
    split_name: str,
    data_dir: str,
    usda_lookup: Dict[str, dict],
    output_dir: str,
    image_path_prefix: Optional[str],
) -> dict:
    """
    Process one dataset split (train/valid/test).

    Reads the COCO annotation file, builds structured targets for each image,
    and writes a JSONL output file.

    Args:
        split_name: One of 'train', 'valid', 'test'.
        data_dir: Root data directory containing split subdirectories.
        usda_lookup: USDA mapping lookup dict.
        output_dir: Directory to write the output JSONL file.
        image_path_prefix: Optional prefix for image paths in the output.
                          If None, uses "{split_name}/{file_name}".

    Returns:
        A dict with statistics about the conversion.
    """
    split_dir = os.path.join(data_dir, split_name)
    ann_path = os.path.join(split_dir, COCO_ANNOTATION_FILENAME)

    if not os.path.isfile(ann_path):
        print(f"  [SKIP] {ann_path} not found", file=sys.stderr)
        return {"split": split_name, "status": "skipped", "reason": "annotation file not found"}

    print(f"  Processing {split_name}...")
    coco = load_json(ann_path)

    cat_lookup, img_lookup, img_annotations = build_coco_lookups(coco)

    # Stats tracking
    stats = {
        "split": split_name,
        "status": "ok",
        "total_images": len(img_lookup),
        "images_with_annotations": 0,
        "images_without_annotations": 0,
        "total_annotations_used": 0,
        "unique_classes_seen": set(),
    }

    output_path = os.path.join(output_dir, f"{split_name}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        # Iterate over ALL images (including those with no annotations)
        for img_id, file_name in sorted(img_lookup.items()):
            annotations = img_annotations.get(img_id, [])

            # Build image path for the output
            if image_path_prefix:
                img_path = os.path.join(image_path_prefix, split_name, file_name)
            else:
                img_path = os.path.join(split_name, file_name)

            if annotations:
                stats["images_with_annotations"] += 1
                stats["total_annotations_used"] += len(annotations)

                # Track unique classes
                for ann in annotations:
                    cat_name = cat_lookup.get(ann["category_id"])
                    if cat_name:
                        stats["unique_classes_seen"].add(cat_name)

                target = build_structured_target(annotations, cat_lookup, usda_lookup)
            else:
                # Image with no annotations → empty items list
                stats["images_without_annotations"] += 1
                target = {"items": []}

            record = {
                "image": img_path,
                "prompt": PROMPT_TOKEN,
                "target": json.dumps(target, ensure_ascii=False),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Convert set to count for JSON-friendly stats
    stats["unique_classes_seen"] = len(stats["unique_classes_seen"])

    print(f"    → {stats['total_images']} images "
          f"({stats['images_with_annotations']} annotated, "
          f"{stats['images_without_annotations']} empty)")
    print(f"    → {stats['total_annotations_used']} annotations across "
          f"{stats['unique_classes_seen']} classes")
    print(f"    → Written to {output_path}")

    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO pantry food annotations to Florence-2 fine-tuning JSONL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default paths
  python convert_coco_to_florence2.py \\
      --data-dir ./Food-Items-4 \\
      --mapping ./usda_mapping.json \\
      --output-dir ./florence2_data

  # With custom image path prefix (useful if images will be stored elsewhere)
  python convert_coco_to_florence2.py \\
      --data-dir ./Food-Items-4 \\
      --mapping ./usda_mapping.json \\
      --output-dir ./florence2_data \\
      --image-prefix data/pantry

  # Process only specific splits
  python convert_coco_to_florence2.py \\
      --data-dir ./Food-Items-4 \\
      --mapping ./usda_mapping.json \\
      --output-dir ./florence2_data \\
      --splits train valid
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing train/valid/test subdirectories with COCO annotations.",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        required=True,
        help="Path to usda_mapping.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write output JSONL files (train.jsonl, valid.jsonl, test.jsonl).",
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default=None,
        help="Optional prefix for image paths in the output JSONL. "
             "If not set, paths are relative like 'train/image.jpg'.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=SPLITS,
        choices=SPLITS,
        help=f"Which splits to process (default: {' '.join(SPLITS)}).",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.data_dir):
        print(f"Error: data directory '{args.data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.mapping):
        print(f"Error: mapping file '{args.mapping}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load USDA mapping
    print(f"Loading USDA mapping from {args.mapping}...")
    usda_lookup = build_usda_lookup(args.mapping)
    print(f"  Loaded {len(usda_lookup)} category mappings")

    # Process each split
    print(f"\nProcessing splits from {args.data_dir}...")
    all_stats = []
    for split in args.splits:
        stats = process_split(
            split_name=split,
            data_dir=args.data_dir,
            usda_lookup=usda_lookup,
            output_dir=args.output_dir,
            image_path_prefix=args.image_prefix,
        )
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    total_images = 0
    total_annotated = 0
    total_annotations = 0
    for s in all_stats:
        if s["status"] == "ok":
            total_images += s["total_images"]
            total_annotated += s["images_with_annotations"]
            total_annotations += s["total_annotations_used"]
            print(f"  {s['split']:>6s}: {s['total_images']:>5d} images, "
                  f"{s['total_annotations_used']:>6d} annotations, "
                  f"{s['unique_classes_seen']:>2d} classes")
        else:
            print(f"  {s['split']:>6s}: SKIPPED ({s.get('reason', 'unknown')})")

    print(f"  {'TOTAL':>6s}: {total_images:>5d} images, "
          f"{total_annotations:>6d} annotations")
    print(f"\nOutput written to: {args.output_dir}/")
    print("Files: " + ", ".join(f"{s}.jsonl" for s in args.splits))


if __name__ == "__main__":
    main()
