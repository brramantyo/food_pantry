"""
stratified_split.py
====================
Split a single COCO annotation file into stratified train/valid/test sets.

Ensures every category appears in every split proportionally,
which is critical for underrepresented classes (e.g., Oil=34, Frozen Mix Vegetable=22).

Usage:
    python stratified_split.py --data-dir ./Dataset --output-dir ./Dataset_Split --ratios 70 15 15

Output structure:
    Dataset_Split/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    └── test/
        ├── _annotations.coco.json
        └── *.jpg
"""

import json
import os
import shutil
import argparse
import random
from collections import defaultdict


def load_coco(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_image_categories(coco_data):
    """For each image, find which category ids appear (for stratification)."""
    img_cats = defaultdict(set)
    for ann in coco_data["annotations"]:
        # Skip dummy class id 0
        if ann["category_id"] == 0:
            continue
        img_cats[ann["image_id"]].add(ann["category_id"])
    return img_cats


def stratified_split(coco_data, ratios, seed=42):
    """
    Stratified split based on rarest category per image.
    
    Strategy: assign each image to a "primary category" (its rarest class),
    then split within each primary category group proportionally.
    This ensures rare classes are distributed across all splits.
    """
    random.seed(seed)
    
    img_cats = get_image_categories(coco_data)
    
    # Count total annotations per category (for rarity ranking)
    cat_counts = defaultdict(int)
    for ann in coco_data["annotations"]:
        if ann["category_id"] != 0:
            cat_counts[ann["category_id"]] += 1
    
    # Assign each image to its rarest category
    img_primary_cat = {}
    for img_id, cats in img_cats.items():
        if cats:
            # Pick the rarest category in this image
            rarest = min(cats, key=lambda c: cat_counts.get(c, float('inf')))
            img_primary_cat[img_id] = rarest
    
    # Images with no annotations (after filtering dummy class)
    all_img_ids = {img["id"] for img in coco_data["images"]}
    annotated_ids = set(img_cats.keys())
    unannotated_ids = all_img_ids - annotated_ids
    
    # Group images by primary category
    cat_groups = defaultdict(list)
    for img_id, cat_id in img_primary_cat.items():
        cat_groups[cat_id].append(img_id)
    
    # Normalize ratios
    total_ratio = sum(ratios)
    norm_ratios = [r / total_ratio for r in ratios]
    
    # Split each category group proportionally
    train_ids, valid_ids, test_ids = set(), set(), set()
    
    for cat_id, img_ids in sorted(cat_groups.items()):
        random.shuffle(img_ids)
        n = len(img_ids)
        n_train = max(1, round(n * norm_ratios[0]))
        n_valid = max(1, round(n * norm_ratios[1])) if n > 2 else 0
        n_test = n - n_train - n_valid
        
        # Ensure at least 1 in test if we have enough images
        if n_test <= 0 and n > 3:
            n_train -= 1
            n_test = 1
        elif n_test < 0:
            n_test = 0
            n_valid = n - n_train
        
        train_ids.update(img_ids[:n_train])
        valid_ids.update(img_ids[n_train:n_train + n_valid])
        test_ids.update(img_ids[n_train + n_valid:])
    
    # Distribute unannotated images into train
    train_ids.update(unannotated_ids)
    
    return train_ids, valid_ids, test_ids


def build_coco_subset(coco_data, image_ids):
    """Build a COCO dict for a subset of image ids."""
    image_ids = set(image_ids)
    
    images = [img for img in coco_data["images"] if img["id"] in image_ids]
    annotations = [ann for ann in coco_data["annotations"] 
                   if ann["image_id"] in image_ids and ann["category_id"] != 0]
    
    # Only include categories that actually appear
    present_cats = {ann["category_id"] for ann in annotations}
    categories = [cat for cat in coco_data["categories"] 
                  if cat["id"] in present_cats]
    
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }


def copy_images(coco_data, image_ids, src_dir, dst_dir):
    """Copy image files for given image ids."""
    os.makedirs(dst_dir, exist_ok=True)
    
    id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    copied = 0
    missing = 0
    
    for img_id in image_ids:
        fname = id_to_filename.get(img_id)
        if fname:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1
            else:
                missing += 1
    
    return copied, missing


def print_split_stats(coco_data, train_ids, valid_ids, test_ids):
    """Print detailed statistics about the split."""
    cat_names = {cat["id"]: cat["name"] for cat in coco_data["categories"] if cat["id"] != 0}
    
    # Count annotations per category per split
    splits = {"train": train_ids, "valid": valid_ids, "test": test_ids}
    
    print("\n" + "=" * 70)
    print(f"{'Category':<45} {'Train':>7} {'Valid':>7} {'Test':>7}")
    print("=" * 70)
    
    for cat_id in sorted(cat_names.keys()):
        counts = {}
        for split_name, split_ids in splits.items():
            count = sum(1 for ann in coco_data["annotations"] 
                       if ann["category_id"] == cat_id and ann["image_id"] in split_ids)
            counts[split_name] = count
        
        name = cat_names[cat_id]
        if len(name) > 44:
            name = name[:41] + "..."
        print(f"  {name:<43} {counts['train']:>7} {counts['valid']:>7} {counts['test']:>7}")
    
    print("-" * 70)
    print(f"  {'TOTAL ANNOTATIONS':<43} ", end="")
    for split_name, split_ids in splits.items():
        total = sum(1 for ann in coco_data["annotations"] 
                   if ann["image_id"] in split_ids and ann["category_id"] != 0)
        print(f"{total:>7} ", end="")
    print()
    
    print(f"  {'TOTAL IMAGES':<43} {len(train_ids):>7} {len(valid_ids):>7} {len(test_ids):>7}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Stratified COCO dataset splitter")
    parser.add_argument("--data-dir", required=True, 
                       help="Directory containing _annotations.coco.json and images")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for split data")
    parser.add_argument("--ratios", nargs=3, type=int, default=[70, 15, 15],
                       help="Train/valid/test ratio (default: 70 15 15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--ann-file", default="_annotations.coco.json",
                       help="Annotation filename (default: _annotations.coco.json)")
    args = parser.parse_args()
    
    # Load COCO data
    ann_path = os.path.join(args.data_dir, args.ann_file)
    if not os.path.exists(ann_path):
        # Try without .json extension
        ann_path = os.path.join(args.data_dir, "_annotations.coco")
        if not os.path.exists(ann_path):
            print(f"ERROR: Annotation file not found in {args.data_dir}")
            print("  Looked for: _annotations.coco.json and _annotations.coco")
            return
    
    print(f"Loading annotations from: {ann_path}")
    coco_data = load_coco(ann_path)
    
    print(f"Images: {len(coco_data['images'])}")
    print(f"Annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {len(coco_data['categories'])}")
    print(f"Split ratios: {args.ratios[0]}/{args.ratios[1]}/{args.ratios[2]}")
    print(f"Random seed: {args.seed}")
    
    # Perform stratified split
    train_ids, valid_ids, test_ids = stratified_split(coco_data, args.ratios, args.seed)
    
    # Print stats
    print_split_stats(coco_data, train_ids, valid_ids, test_ids)
    
    # Save splits
    for split_name, split_ids in [("train", train_ids), ("valid", valid_ids), ("test", test_ids)]:
        split_dir = os.path.join(args.output_dir, split_name)
        
        # Build COCO subset
        subset = build_coco_subset(coco_data, split_ids)
        
        # Copy images
        copied, missing = copy_images(coco_data, split_ids, args.data_dir, split_dir)
        
        # Save annotation file
        ann_out = os.path.join(split_dir, "_annotations.coco.json")
        with open(ann_out, 'w') as f:
            json.dump(subset, f)
        
        print(f"\n{split_name}: {len(split_ids)} images ({copied} copied, {missing} missing)")
        print(f"  Annotations: {len(subset['annotations'])}")
        print(f"  Saved to: {split_dir}")
    
    print(f"\nDone! Dataset split saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
