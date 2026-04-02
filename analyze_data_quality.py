#!/usr/bin/env python3
"""
analyze_data_quality.py
========================
Analyze image quality and identify problematic samples in the Food Pantry dataset.

Checks:
1. Blur detection (Laplacian variance)
2. Brightness (too dark / too bright)
3. Low contrast
4. Very small images
5. Near-duplicate detection (perceptual hashing)
6. Per-class quality distribution
7. Potential mislabels (using model predictions vs ground truth)

Usage:
    python3 analyze_data_quality.py \
        --data-dir . \
        --output-dir data_quality_report

    # With model predictions for mislabel detection:
    python3 analyze_data_quality.py \
        --data-dir . \
        --predictions eval_results_v11.json \
        --output-dir data_quality_report
"""

import argparse
import json
import os
import sys
import hashlib
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
BLUR_THRESHOLD = 50.0        # Below = blurry (Laplacian variance)
DARK_THRESHOLD = 40.0        # Mean brightness below = too dark
BRIGHT_THRESHOLD = 220.0     # Mean brightness above = too bright
LOW_CONTRAST_THRESHOLD = 30.0  # Std dev below = low contrast
MIN_RESOLUTION = 100         # Width or height below = too small
PHASH_BITS = 64              # Perceptual hash size
DUPLICATE_THRESHOLD = 5      # Hamming distance below = near-duplicate


def compute_laplacian_variance(img_gray):
    """Compute Laplacian variance as a blur metric (higher = sharper)."""
    # Manual Laplacian using convolution-like approach with PIL
    # We'll use numpy for this
    arr = np.array(img_gray, dtype=np.float64)
    # Laplacian kernel approximation: difference from neighbors
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0
    # Simple Laplacian: pixel - mean of 4 neighbors
    laplacian = (
        4 * arr[1:-1, 1:-1]
        - arr[:-2, 1:-1]   # top
        - arr[2:, 1:-1]    # bottom
        - arr[1:-1, :-2]   # left
        - arr[1:-1, 2:]    # right
    )
    return float(np.var(laplacian))


def compute_phash(img, hash_size=8):
    """Compute perceptual hash of an image."""
    # Resize to hash_size+1 x hash_size
    img_small = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    img_gray = img_small.convert('L')
    pixels = np.array(img_gray, dtype=np.float64)
    # Compute difference hash (dHash)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()


def hamming_distance(hash1, hash2):
    """Compute Hamming distance between two hashes."""
    return int(np.sum(hash1 != hash2))


def analyze_single_image(img_path):
    """Analyze a single image and return quality metrics."""
    try:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
    except Exception as e:
        return {'error': str(e), 'path': img_path}

    width, height = img.size
    img_gray = img.convert('L')

    # Brightness and contrast
    stat = ImageStat.Stat(img_gray)
    mean_brightness = stat.mean[0]
    std_brightness = stat.stddev[0]

    # Blur detection
    blur_score = compute_laplacian_variance(img_gray)

    # Perceptual hash for duplicate detection
    phash = compute_phash(img)

    # File size
    file_size = os.path.getsize(img_path) if os.path.exists(img_path) else 0

    # Issues
    issues = []
    if blur_score < BLUR_THRESHOLD:
        issues.append('blurry')
    if mean_brightness < DARK_THRESHOLD:
        issues.append('too_dark')
    if mean_brightness > BRIGHT_THRESHOLD:
        issues.append('too_bright')
    if std_brightness < LOW_CONTRAST_THRESHOLD:
        issues.append('low_contrast')
    if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
        issues.append('too_small')

    return {
        'path': img_path,
        'width': width,
        'height': height,
        'blur_score': round(blur_score, 2),
        'mean_brightness': round(mean_brightness, 2),
        'std_brightness': round(std_brightness, 2),
        'file_size_kb': round(file_size / 1024, 1),
        'issues': issues,
        'phash': phash,
    }


def load_coco_annotations(data_dir):
    """Load COCO annotations from train/valid/test splits."""
    all_images = []  # list of {split, file_name, categories, image_id, annotations}
    category_map = {}

    for split in ['train', 'valid', 'test']:
        ann_path = os.path.join(data_dir, split, '_annotations.coco.json')
        if not os.path.exists(ann_path):
            logger.warning(f"No annotations found at {ann_path}")
            continue

        with open(ann_path) as f:
            coco = json.load(f)

        # Build category map
        for cat in coco.get('categories', []):
            if cat['name'] != 'Food-Items-Food-Items-4Fxl':  # Skip dummy class
                category_map[cat['id']] = cat['name']

        # Build image_id -> annotations
        img_anns = defaultdict(list)
        for ann in coco.get('annotations', []):
            cat_id = ann['category_id']
            if cat_id in category_map:
                img_anns[ann['image_id']].append({
                    'category_id': cat_id,
                    'category_name': category_map[cat_id],
                    'bbox': ann['bbox'],
                    'area': ann.get('area', 0),
                })

        for img_info in coco.get('images', []):
            img_id = img_info['id']
            anns = img_anns.get(img_id, [])
            cats = list(set(a['category_name'] for a in anns))
            all_images.append({
                'split': split,
                'file_name': img_info['file_name'],
                'image_path': os.path.join(data_dir, split, img_info['file_name']),
                'image_id': img_id,
                'categories': cats,
                'num_annotations': len(anns),
                'annotations': anns,
                'width': img_info.get('width', 0),
                'height': img_info.get('height', 0),
            })

    return all_images, category_map


def load_predictions(pred_path):
    """Load model predictions for mislabel analysis."""
    with open(pred_path) as f:
        data = json.load(f)

    preds = data.get('predictions', data.get('per_sample_results', []))
    return preds


def parse_prediction_categories(pred_text):
    """Parse categories from a prediction text."""
    try:
        if isinstance(pred_text, str):
            data = json.loads(pred_text)
        else:
            data = pred_text

        items = data.get('items', [])
        cats = []
        for item in items:
            name = item.get('name', item.get('category', ''))
            if name:
                cats.append(name)
        return cats
    except (json.JSONDecodeError, AttributeError):
        return []


def find_mislabel_candidates(images, predictions):
    """Find images where model predictions consistently differ from labels."""
    candidates = []

    # Build a lookup from image filename to prediction
    pred_lookup = {}
    for pred in predictions:
        img = pred.get('image', '')
        # Normalize path
        img = img.replace('\\', '/').split('/')[-1]
        pred_lookup[img] = pred

    for img_info in images:
        if img_info['split'] != 'test':
            continue

        fname = img_info['file_name']
        pred = pred_lookup.get(fname)
        if not pred:
            continue

        gt_cats = set(img_info['categories'])
        pred_text = pred.get('prediction', pred.get('predicted', ''))
        pred_cats = set(parse_prediction_categories(pred_text))

        if not pred_cats:
            continue

        # Categories in prediction but not in GT
        false_positives = pred_cats - gt_cats
        # Categories in GT but not in prediction
        false_negatives = gt_cats - pred_cats

        if false_positives or false_negatives:
            candidates.append({
                'file_name': fname,
                'split': img_info['split'],
                'gt_categories': sorted(gt_cats),
                'pred_categories': sorted(pred_cats),
                'false_positives': sorted(false_positives),
                'false_negatives': sorted(false_negatives),
            })

    return candidates


def main():
    parser = argparse.ArgumentParser(description='Analyze food pantry dataset quality')
    parser.add_argument('--data-dir', default='.', help='Root data directory with train/valid/test')
    parser.add_argument('--predictions', default=None, help='Predictions JSON for mislabel detection')
    parser.add_argument('--output-dir', default='data_quality_report', help='Output directory')
    parser.add_argument('--max-images', type=int, default=None, help='Max images to analyze (for testing)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load annotations ──────────────────────────────────────────────────
    print("=" * 70)
    print("FOOD PANTRY DATA QUALITY ANALYSIS")
    print("=" * 70)

    print("\n📂 Loading COCO annotations...")
    images, category_map = load_coco_annotations(args.data_dir)
    print(f"   Found {len(images)} images across all splits")

    # ── Category distribution ─────────────────────────────────────────────
    print("\n📊 Category Distribution:")
    print("-" * 70)
    cat_counts = defaultdict(lambda: defaultdict(int))
    total_by_split = defaultdict(int)
    for img in images:
        total_by_split[img['split']] += 1
        for cat in img['categories']:
            cat_counts[cat][img['split']] += 1

    print(f"{'Category':<40} {'Train':>6} {'Valid':>6} {'Test':>6} {'Total':>6}")
    print("-" * 70)
    sorted_cats = sorted(cat_counts.keys())
    for cat in sorted_cats:
        counts = cat_counts[cat]
        total = sum(counts.values())
        print(f"{cat:<40} {counts.get('train',0):>6} {counts.get('valid',0):>6} {counts.get('test',0):>6} {total:>6}")

    print("-" * 70)
    print(f"{'Total images':<40} {total_by_split.get('train',0):>6} {total_by_split.get('valid',0):>6} {total_by_split.get('test',0):>6} {sum(total_by_split.values()):>6}")

    # ── Analyze image quality ─────────────────────────────────────────────
    print("\n🔍 Analyzing image quality...")
    results = []
    errors = []
    count = 0

    analyze_list = images[:args.max_images] if args.max_images else images

    for i, img_info in enumerate(analyze_list):
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(analyze_list)} images...")

        result = analyze_single_image(img_info['image_path'])

        if 'error' in result:
            errors.append({
                'file': img_info['file_name'],
                'split': img_info['split'],
                'error': result['error'],
            })
            continue

        result['split'] = img_info['split']
        result['file_name'] = img_info['file_name']
        result['categories'] = img_info['categories']
        result['num_annotations'] = img_info['num_annotations']
        results.append(result)
        count += 1

    print(f"   ✓ Analyzed {count} images ({len(errors)} errors)")

    # ── Summary of issues ─────────────────────────────────────────────────
    print("\n⚠️  Quality Issues Found:")
    print("-" * 70)

    issue_counts = Counter()
    images_with_issues = []
    for r in results:
        for issue in r['issues']:
            issue_counts[issue] += 1
        if r['issues']:
            images_with_issues.append(r)

    for issue, count in issue_counts.most_common():
        print(f"   {issue:<20} {count:>5} images")

    print(f"\n   Total images with issues: {len(images_with_issues)} / {len(results)}")

    # ── Per-class quality breakdown ───────────────────────────────────────
    print("\n📋 Per-Class Quality Breakdown:")
    print("-" * 70)
    print(f"{'Category':<40} {'Total':>5} {'Blurry':>6} {'Dark':>5} {'Bright':>6} {'LoCon':>5} {'Small':>5} {'Avg Blur':>8}")
    print("-" * 70)

    class_stats = defaultdict(lambda: {
        'count': 0, 'blurry': 0, 'too_dark': 0, 'too_bright': 0,
        'low_contrast': 0, 'too_small': 0, 'blur_scores': []
    })

    for r in results:
        for cat in r['categories']:
            stats = class_stats[cat]
            stats['count'] += 1
            stats['blur_scores'].append(r['blur_score'])
            for issue in r['issues']:
                if issue in stats:
                    stats[issue] += 1

    for cat in sorted_cats:
        if cat not in class_stats:
            continue
        s = class_stats[cat]
        avg_blur = np.mean(s['blur_scores']) if s['blur_scores'] else 0
        print(f"{cat:<40} {s['count']:>5} {s['blurry']:>6} {s['too_dark']:>5} {s['too_bright']:>6} {s['low_contrast']:>5} {s['too_small']:>5} {avg_blur:>8.1f}")

    # ── Near-duplicate detection ──────────────────────────────────────────
    print("\n🔄 Checking for near-duplicates...")
    hashes = [(r['file_name'], r['split'], r['phash'], r['categories']) for r in results if 'phash' in r]

    duplicates = []
    # Only check within reasonable bounds to avoid O(n^2) explosion
    if len(hashes) > 5000:
        print(f"   ⚠ {len(hashes)} images — sampling for duplicate check")
        # Check within same split only
        by_split = defaultdict(list)
        for h in hashes:
            by_split[h[1]].append(h)
        for split, split_hashes in by_split.items():
            for i in range(len(split_hashes)):
                for j in range(i + 1, min(i + 200, len(split_hashes))):
                    dist = hamming_distance(split_hashes[i][2], split_hashes[j][2])
                    if dist <= DUPLICATE_THRESHOLD:
                        duplicates.append({
                            'image_a': split_hashes[i][0],
                            'image_b': split_hashes[j][0],
                            'split_a': split_hashes[i][1],
                            'split_b': split_hashes[j][1],
                            'distance': dist,
                            'cats_a': split_hashes[i][3],
                            'cats_b': split_hashes[j][3],
                        })
    else:
        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                dist = hamming_distance(hashes[i][2], hashes[j][2])
                if dist <= DUPLICATE_THRESHOLD:
                    duplicates.append({
                        'image_a': hashes[i][0],
                        'image_b': hashes[j][0],
                        'split_a': hashes[i][1],
                        'split_b': hashes[j][1],
                        'distance': dist,
                        'cats_a': hashes[i][3],
                        'cats_b': hashes[j][3],
                    })

    print(f"   Found {len(duplicates)} near-duplicate pairs")

    # Check cross-split duplicates (train image appears in test = data leak!)
    cross_split = [d for d in duplicates if d['split_a'] != d['split_b']]
    if cross_split:
        print(f"   ⚠️  {len(cross_split)} CROSS-SPLIT duplicates (potential data leak!)")
        for d in cross_split[:10]:
            print(f"      {d['split_a']}/{d['image_a']} ↔ {d['split_b']}/{d['image_b']} (dist={d['distance']})")

    # ── Mislabel analysis ─────────────────────────────────────────────────
    mislabel_candidates = []
    if args.predictions and os.path.exists(args.predictions):
        print("\n🏷️  Analyzing potential mislabels...")
        predictions = load_predictions(args.predictions)
        mislabel_candidates = find_mislabel_candidates(images, predictions)
        print(f"   Found {len(mislabel_candidates)} prediction-label mismatches")

        # Show most common confusion patterns
        confusion_patterns = Counter()
        for mc in mislabel_candidates:
            for fp in mc['false_positives']:
                for gt in mc['gt_categories']:
                    confusion_patterns[(gt, fp)] += 1

        if confusion_patterns:
            print(f"\n   Top confusion patterns (GT → Predicted):")
            for (gt, pred), count in confusion_patterns.most_common(15):
                print(f"      {gt} → {pred}: {count}x")

    # ── Worst images ──────────────────────────────────────────────────────
    print("\n📸 Worst Quality Images (lowest blur scores):")
    print("-" * 70)
    sorted_by_blur = sorted(results, key=lambda x: x['blur_score'])
    for r in sorted_by_blur[:20]:
        cats = ', '.join(r['categories'][:2])
        issues = ', '.join(r['issues']) if r['issues'] else 'ok'
        print(f"   {r['split']}/{r['file_name'][:50]:<50} blur={r['blur_score']:>8.1f} bright={r['mean_brightness']:>5.1f} [{cats}] ({issues})")

    # ── Save detailed report ──────────────────────────────────────────────
    print(f"\n💾 Saving reports to {args.output_dir}/")

    # 1. Summary JSON
    summary = {
        'total_images': len(results),
        'errors': len(errors),
        'issue_counts': dict(issue_counts),
        'images_with_issues': len(images_with_issues),
        'duplicates_found': len(duplicates),
        'cross_split_duplicates': len(cross_split),
        'category_distribution': {
            cat: {
                'total': sum(cat_counts[cat].values()),
                'train': cat_counts[cat].get('train', 0),
                'valid': cat_counts[cat].get('valid', 0),
                'test': cat_counts[cat].get('test', 0),
            }
            for cat in sorted_cats
        },
        'per_class_quality': {
            cat: {
                'count': class_stats[cat]['count'],
                'blurry': class_stats[cat]['blurry'],
                'too_dark': class_stats[cat]['too_dark'],
                'too_bright': class_stats[cat]['too_bright'],
                'low_contrast': class_stats[cat]['low_contrast'],
                'avg_blur_score': round(np.mean(class_stats[cat]['blur_scores']), 2) if class_stats[cat]['blur_scores'] else 0,
            }
            for cat in sorted_cats if cat in class_stats
        },
    }
    with open(os.path.join(args.output_dir, 'quality_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("   ✓ quality_summary.json")

    # 2. Bad images list (for potential deletion)
    bad_images = []
    for r in results:
        if r['issues']:
            bad_images.append({
                'file_name': r['file_name'],
                'split': r['split'],
                'categories': r['categories'],
                'issues': r['issues'],
                'blur_score': r['blur_score'],
                'mean_brightness': r['mean_brightness'],
                'std_brightness': r['std_brightness'],
                'resolution': f"{r['width']}x{r['height']}",
            })
    bad_images.sort(key=lambda x: x['blur_score'])

    with open(os.path.join(args.output_dir, 'bad_images.json'), 'w') as f:
        json.dump(bad_images, f, indent=2)
    print(f"   ✓ bad_images.json ({len(bad_images)} images)")

    # 3. Duplicates list
    # Remove phash (not JSON serializable) from duplicates
    dup_clean = []
    for d in duplicates:
        dup_clean.append({k: v for k, v in d.items() if k != 'phash'})
    with open(os.path.join(args.output_dir, 'duplicates.json'), 'w') as f:
        json.dump(dup_clean, f, indent=2)
    print(f"   ✓ duplicates.json ({len(duplicates)} pairs)")

    # 4. Mislabel candidates
    if mislabel_candidates:
        with open(os.path.join(args.output_dir, 'mislabel_candidates.json'), 'w') as f:
            json.dump(mislabel_candidates, f, indent=2)
        print(f"   ✓ mislabel_candidates.json ({len(mislabel_candidates)} candidates)")

    # 5. Category merge recommendations
    print("\n" + "=" * 70)
    print("📝 RECOMMENDATIONS")
    print("=" * 70)

    # Small categories
    small_cats = [(cat, sum(cat_counts[cat].values())) for cat in sorted_cats
                  if sum(cat_counts[cat].values()) < 30]
    if small_cats:
        print("\n🔸 Small categories (< 30 total samples):")
        for cat, count in sorted(small_cats, key=lambda x: x[1]):
            print(f"   {cat}: {count} samples → consider merging or collecting more data")

    # Low performance categories (from YOLO results)
    print("\n🔸 Low-performing categories (from YOLO mAP50):")
    yolo_results = {
        'Vegetables - Fresh': 24.8,
        'Nut Butters and Nuts': 28.8,
        'Meat and Poultry - Canned': 33.7,
        'Granola Products': 46.6,
        'Seafood - Canned': 46.2,
    }
    for cat, map50 in sorted(yolo_results.items(), key=lambda x: x[1]):
        sample_count = sum(cat_counts.get(cat, {}).values())
        quality = class_stats.get(cat, {})
        blurry = quality.get('blurry', 0) if isinstance(quality, dict) else 0
        print(f"   {cat}: mAP50={map50}%, {sample_count} samples, {blurry} blurry images")

    # Merge suggestions
    print("\n🔸 Merge candidates based on confusion analysis:")
    print("   1. Meat Canned + Seafood Canned → 'Canned Protein'")
    print("      (bidirectional confusion, both protein in cans)")
    print("   2. Vegetables Fresh + Fresh Fruit → 'Fresh Produce'")
    print("      (few samples each, visually similar)")
    print("\n🔸 Keep separate (despite low accuracy):")
    print("   - Granola Products: distinct food type (breakfast cereal/bars)")
    print("   - Nut Butters: distinct food type (spread/jar)")
    print("   → Improve via data quality, not merging")

    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
