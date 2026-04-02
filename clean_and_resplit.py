#!/usr/bin/env python3
"""
clean_and_resplit.py
====================
Clean the Food Pantry dataset and create a proper train/valid/test split.

Steps:
1. Load all images and annotations from all splits
2. Remove near-duplicate images (keep one copy)
3. Remove blurry images (below blur threshold)
4. Optionally merge/drop small categories
5. Re-split with stratified sampling (no data leak)
6. Write new COCO annotation files + symlink images

Usage:
    python3 clean_and_resplit.py \
        --data-dir . \
        --output-dir cleaned_data \
        --duplicates data_quality_report/duplicates.json \
        --bad-images data_quality_report/bad_images.json \
        --blur-threshold 50 \
        --drop-categories "Frozen Mix Vegetable,Oil,Spices Seasonings and Mixes,Carton of Eggs" \
        --merge "Meat and Poultry - Canned+Seafood - Canned=Canned Protein" \
        --merge "Vegetables - Fresh+Fresh Fruit=Fresh Produce" \
        --split-ratio 0.7 0.15 0.15 \
        --seed 42
"""

import argparse
import json
import os
import sys
import shutil
import hashlib
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def compute_image_hash(img_path):
    """Compute MD5 hash of image file for exact duplicate detection."""
    h = hashlib.md5()
    try:
        with open(img_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except:
        return None


def compute_laplacian_variance(img_path):
    """Compute blur score for an image."""
    try:
        img = Image.open(img_path).convert('L')
        arr = np.array(img, dtype=np.float64)
        if arr.shape[0] < 3 or arr.shape[1] < 3:
            return 0.0
        lap = 4*arr[1:-1,1:-1] - arr[:-2,1:-1] - arr[2:,1:-1] - arr[1:-1,:-2] - arr[1:-1,2:]
        return float(np.var(lap))
    except:
        return 0.0


def load_all_data(data_dir):
    """Load all images and annotations from all splits into a unified pool."""
    all_images = {}  # keyed by file content hash to deduplicate
    all_annotations = []  # list of annotations with unified image refs
    category_map = {}  # id -> name (use largest split's mapping)
    
    # First pass: collect category mappings
    for split in ['train', 'valid', 'test']:
        ann_path = os.path.join(data_dir, split, '_annotations.coco.json')
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            coco = json.load(f)
        for cat in coco.get('categories', []):
            if cat['name'] != 'Food-Items-Food-Items-4Fxl':
                category_map[cat['id']] = cat['name']
    
    # Second pass: collect all unique images + annotations
    img_id_counter = 0
    ann_id_counter = 0
    filename_to_unified = {}  # (split, original_filename) -> unified image entry
    original_id_to_unified = {}  # (split, original_image_id) -> unified_image_id
    
    seen_hashes = {}  # content_hash -> unified_image_id
    
    for split in ['train', 'valid', 'test']:
        ann_path = os.path.join(data_dir, split, '_annotations.coco.json')
        if not os.path.exists(ann_path):
            continue
        with open(ann_path) as f:
            coco = json.load(f)
        
        # Map original image IDs to image info
        orig_images = {img['id']: img for img in coco.get('images', [])}
        
        # Group annotations by image
        img_annotations = defaultdict(list)
        for ann in coco.get('annotations', []):
            if ann['category_id'] in category_map:
                img_annotations[ann['image_id']].append(ann)
        
        for orig_img_id, img_info in orig_images.items():
            img_path = os.path.join(data_dir, split, img_info['file_name'])
            if not os.path.exists(img_path):
                continue
            
            # Compute content hash for dedup
            content_hash = compute_image_hash(img_path)
            if content_hash is None:
                continue
            
            if content_hash in seen_hashes:
                # Duplicate — map to existing unified ID but keep annotations
                # (merge annotations from both copies)
                unified_id = seen_hashes[content_hash]
                original_id_to_unified[(split, orig_img_id)] = unified_id
            else:
                # New unique image
                img_id_counter += 1
                unified_id = img_id_counter
                seen_hashes[content_hash] = unified_id
                original_id_to_unified[(split, orig_img_id)] = unified_id
                
                all_images[unified_id] = {
                    'id': unified_id,
                    'file_name': img_info['file_name'],
                    'source_path': img_path,
                    'source_split': split,
                    'width': img_info.get('width', 0),
                    'height': img_info.get('height', 0),
                    'content_hash': content_hash,
                }
            
            # Add annotations (use unified image ID)
            for ann in img_annotations.get(orig_img_id, []):
                ann_id_counter += 1
                all_annotations.append({
                    'id': ann_id_counter,
                    'image_id': unified_id,
                    'category_id': ann['category_id'],
                    'category_name': category_map.get(ann['category_id'], ''),
                    'bbox': ann['bbox'],
                    'area': ann.get('area', 0),
                    'iscrowd': ann.get('iscrowd', 0),
                })
    
    return all_images, all_annotations, category_map


def remove_blurry(all_images, blur_threshold):
    """Remove images below blur threshold."""
    removed = []
    kept = {}
    
    for img_id, img_info in all_images.items():
        blur_score = compute_laplacian_variance(img_info['source_path'])
        img_info['blur_score'] = blur_score
        
        if blur_score < blur_threshold:
            removed.append({
                'file_name': img_info['file_name'],
                'blur_score': round(blur_score, 2),
                'source_split': img_info['source_split'],
            })
        else:
            kept[img_id] = img_info
    
    return kept, removed


def apply_category_changes(all_annotations, category_map, drop_categories, merge_rules):
    """Drop and merge categories as specified."""
    # Parse merge rules: "A+B=NewName"
    merges = {}  # old_name -> new_name
    if merge_rules:
        for rule in merge_rules:
            parts = rule.split('=')
            if len(parts) != 2:
                logger.warning(f"Invalid merge rule: {rule}")
                continue
            new_name = parts[1].strip()
            old_names = [n.strip() for n in parts[0].split('+')]
            for old in old_names:
                merges[old] = new_name
    
    # Build new category mapping
    drop_set = set(c.strip() for c in drop_categories) if drop_categories else set()
    
    new_cat_names = set()
    filtered_annotations = []
    
    for ann in all_annotations:
        cat_name = ann['category_name']
        
        # Drop?
        if cat_name in drop_set:
            continue
        
        # Merge?
        if cat_name in merges:
            cat_name = merges[cat_name]
        
        ann['category_name'] = cat_name
        new_cat_names.add(cat_name)
        filtered_annotations.append(ann)
    
    # Build new category ID mapping
    new_category_map = {}
    for i, name in enumerate(sorted(new_cat_names), start=1):
        new_category_map[i] = name
    
    # Reverse lookup
    name_to_id = {v: k for k, v in new_category_map.items()}
    
    # Update annotation category IDs
    for ann in filtered_annotations:
        ann['category_id'] = name_to_id[ann['category_name']]
    
    return filtered_annotations, new_category_map


def stratified_split(all_images, all_annotations, split_ratio, seed):
    """
    Stratified split ensuring:
    - No image appears in multiple splits
    - Category proportions preserved
    - Proper randomization
    """
    rng = random.Random(seed)
    
    train_ratio, valid_ratio, test_ratio = split_ratio
    
    # Get primary category for each image (most common, or first)
    img_categories = defaultdict(list)
    for ann in all_annotations:
        img_categories[ann['image_id']].append(ann['category_name'])
    
    # Assign primary category (most frequent in that image)
    img_primary = {}
    for img_id, cats in img_categories.items():
        if img_id in all_images:
            counter = Counter(cats)
            img_primary[img_id] = counter.most_common(1)[0][0]
    
    # Group images by primary category
    cat_images = defaultdict(list)
    for img_id, cat in img_primary.items():
        cat_images[cat].append(img_id)
    
    # Images without annotations
    no_ann_images = [img_id for img_id in all_images if img_id not in img_primary]
    
    train_ids, valid_ids, test_ids = [], [], []
    
    for cat, img_ids in sorted(cat_images.items()):
        rng.shuffle(img_ids)
        n = len(img_ids)
        n_train = max(1, round(n * train_ratio))
        n_valid = max(1, round(n * valid_ratio)) if n > 2 else 0
        n_test = n - n_train - n_valid
        
        if n_test < 0:
            n_valid = n - n_train
            n_test = 0
        if n_valid < 0:
            n_valid = 0
            n_test = n - n_train
        
        train_ids.extend(img_ids[:n_train])
        valid_ids.extend(img_ids[n_train:n_train+n_valid])
        test_ids.extend(img_ids[n_train+n_valid:])
    
    # Distribute no-annotation images
    rng.shuffle(no_ann_images)
    n = len(no_ann_images)
    n_train = round(n * train_ratio)
    n_valid = round(n * valid_ratio)
    train_ids.extend(no_ann_images[:n_train])
    valid_ids.extend(no_ann_images[n_train:n_train+n_valid])
    test_ids.extend(no_ann_images[n_train+n_valid:])
    
    return set(train_ids), set(valid_ids), set(test_ids)


def write_coco_split(split_name, img_ids, all_images, all_annotations, category_map, output_dir):
    """Write a COCO annotation file and copy/symlink images for a split."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Filter images and annotations
    split_images = []
    split_annotations = []
    
    img_id_set = set(img_ids)
    
    for img_id in sorted(img_ids):
        if img_id not in all_images:
            continue
        img_info = all_images[img_id]
        split_images.append({
            'id': img_info['id'],
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height'],
        })
        
        # Symlink or copy image
        src = img_info['source_path']
        dst = os.path.join(split_dir, img_info['file_name'])
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.abspath(src), dst)
            except OSError:
                shutil.copy2(src, dst)
    
    for ann in all_annotations:
        if ann['image_id'] in img_id_set:
            split_annotations.append({
                'id': ann['id'],
                'image_id': ann['image_id'],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann.get('iscrowd', 0),
            })
    
    # Build categories list
    categories = [{'id': k, 'name': v} for k, v in sorted(category_map.items())]
    
    coco_output = {
        'images': split_images,
        'annotations': split_annotations,
        'categories': categories,
    }
    
    ann_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(ann_path, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    return len(split_images), len(split_annotations)


def main():
    parser = argparse.ArgumentParser(description='Clean and re-split Food Pantry dataset')
    parser.add_argument('--data-dir', default='.', help='Root data directory')
    parser.add_argument('--output-dir', default='cleaned_data', help='Output directory')
    parser.add_argument('--blur-threshold', type=float, default=50.0, help='Blur threshold')
    parser.add_argument('--drop-categories', type=str, default=None,
                        help='Comma-separated categories to drop')
    parser.add_argument('--merge', action='append', default=None,
                        help='Merge rule: "CatA+CatB=NewName" (can repeat)')
    parser.add_argument('--split-ratio', nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        help='Train/valid/test ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-blur-filter', action='store_true', help='Skip blur filtering')
    args = parser.parse_args()
    
    print("=" * 70)
    print("FOOD PANTRY DATASET CLEANING & RE-SPLIT")
    print("=" * 70)
    
    # ── Step 1: Load all data ─────────────────────────────────────────────
    print("\n[1/6] Loading all data from all splits...")
    all_images, all_annotations, category_map = load_all_data(args.data_dir)
    
    total_original = sum(1 for split in ['train','valid','test'] 
                        for f in os.listdir(os.path.join(args.data_dir, split))
                        if f.endswith(('.jpg','.jpeg','.png'))
                        ) if os.path.isdir(os.path.join(args.data_dir, 'train')) else 0
    
    print(f"   Original images across all splits: {total_original}")
    print(f"   Unique images (after dedup): {len(all_images)}")
    print(f"   Duplicates removed: {total_original - len(all_images)}")
    print(f"   Total annotations: {len(all_annotations)}")
    print(f"   Categories: {len(category_map)}")
    
    # ── Step 2: Remove blurry images ──────────────────────────────────────
    if not args.no_blur_filter:
        print(f"\n[2/6] Removing blurry images (threshold={args.blur_threshold})...")
        all_images, removed_blurry = remove_blurry(all_images, args.blur_threshold)
        print(f"   Removed {len(removed_blurry)} blurry images")
        print(f"   Remaining: {len(all_images)} images")
        
        if removed_blurry:
            print(f"   Worst 5:")
            for r in sorted(removed_blurry, key=lambda x: x['blur_score'])[:5]:
                print(f"      {r['file_name'][:60]} blur={r['blur_score']}")
    else:
        print("\n[2/6] Blur filtering skipped")
        removed_blurry = []
    
    # ── Step 3: Drop/merge categories ─────────────────────────────────────
    drop_cats = [c.strip() for c in args.drop_categories.split(',')] if args.drop_categories else []
    
    print(f"\n[3/6] Applying category changes...")
    if drop_cats:
        print(f"   Dropping: {drop_cats}")
    if args.merge:
        print(f"   Merging: {args.merge}")
    
    all_annotations, new_category_map = apply_category_changes(
        all_annotations, category_map, drop_cats, args.merge or []
    )
    
    # Remove images that lost all annotations
    annotated_img_ids = set(ann['image_id'] for ann in all_annotations)
    before = len(all_images)
    all_images = {k: v for k, v in all_images.items() if k in annotated_img_ids}
    orphaned = before - len(all_images)
    
    print(f"   New categories: {len(new_category_map)}")
    for cid, cname in sorted(new_category_map.items()):
        count = sum(1 for a in all_annotations if a['category_id'] == cid)
        # Count unique images
        img_count = len(set(a['image_id'] for a in all_annotations if a['category_id'] == cid))
        print(f"      [{cid:>2}] {cname:<40} {count:>4} annotations, {img_count:>4} images")
    if orphaned:
        print(f"   Removed {orphaned} images with no remaining annotations")
    
    # ── Step 4: Stratified split ──────────────────────────────────────────
    print(f"\n[4/6] Stratified split (ratio={args.split_ratio}, seed={args.seed})...")
    train_ids, valid_ids, test_ids = stratified_split(
        all_images, all_annotations, args.split_ratio, args.seed
    )
    
    # Verify no overlap
    assert len(train_ids & valid_ids) == 0, "Train/valid overlap!"
    assert len(train_ids & test_ids) == 0, "Train/test overlap!"
    assert len(valid_ids & test_ids) == 0, "Valid/test overlap!"
    print(f"   ✓ No overlap between splits")
    print(f"   Train: {len(train_ids)} | Valid: {len(valid_ids)} | Test: {len(test_ids)}")
    
    # ── Step 5: Write output ──────────────────────────────────────────────
    print(f"\n[5/6] Writing cleaned dataset to {args.output_dir}/...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    for split_name, split_ids in [('train', train_ids), ('valid', valid_ids), ('test', test_ids)]:
        n_imgs, n_anns = write_coco_split(
            split_name, split_ids, all_images, all_annotations, new_category_map, args.output_dir
        )
        print(f"   {split_name}: {n_imgs} images, {n_anns} annotations")
    
    # ── Step 6: Verification ──────────────────────────────────────────────
    print(f"\n[6/6] Verifying output...")
    
    # Check per-category distribution in each split
    print(f"\n   {'Category':<40} {'Train':>6} {'Valid':>6} {'Test':>6} {'Total':>6}")
    print("   " + "-" * 64)
    
    split_map = {}
    for img_id in train_ids: split_map[img_id] = 'train'
    for img_id in valid_ids: split_map[img_id] = 'valid'
    for img_id in test_ids: split_map[img_id] = 'test'
    
    cat_split_counts = defaultdict(lambda: defaultdict(int))
    for ann in all_annotations:
        if ann['image_id'] in split_map:
            split = split_map[ann['image_id']]
            cat_split_counts[ann['category_name']][split] += 1
    
    for cname in sorted(cat_split_counts.keys()):
        c = cat_split_counts[cname]
        t = sum(c.values())
        print(f"   {cname:<40} {c.get('train',0):>6} {c.get('valid',0):>6} {c.get('test',0):>6} {t:>6}")
    
    # ── Save cleaning report ──────────────────────────────────────────────
    report = {
        'original_total_images': total_original,
        'unique_images': total_original - (total_original - len(all_images) - len(removed_blurry)),
        'duplicates_removed': total_original - len(all_images) - len(removed_blurry),
        'blurry_removed': len(removed_blurry),
        'categories_dropped': drop_cats,
        'categories_merged': args.merge or [],
        'final_categories': len(new_category_map),
        'final_images': len(all_images),
        'split': {
            'train': len(train_ids),
            'valid': len(valid_ids),
            'test': len(test_ids),
        },
        'split_ratio': args.split_ratio,
        'seed': args.seed,
        'blur_threshold': args.blur_threshold,
        'removed_blurry_images': removed_blurry,
    }
    
    report_path = os.path.join(args.output_dir, 'cleaning_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n   ✓ Cleaning report saved to {report_path}")
    
    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"   Original: {total_original} images, {len(category_map)} categories")
    print(f"   Cleaned:  {len(all_images)} images, {len(new_category_map)} categories")
    print(f"   Removed:  {total_original - len(all_images)} total")
    print(f"     - Exact duplicates (data leak fix): {total_original - len(all_images) - len(removed_blurry)}")
    print(f"     - Blurry images: {len(removed_blurry)}")
    if drop_cats:
        print(f"     - Dropped categories: {', '.join(drop_cats)}")
    if args.merge:
        print(f"     - Merged: {', '.join(args.merge)}")
    print(f"   Split: {len(train_ids)} train / {len(valid_ids)} valid / {len(test_ids)} test")
    print(f"   Output: {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
