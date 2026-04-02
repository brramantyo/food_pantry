#!/usr/bin/env python3
"""
eval_fuzzy.py — Re-evaluate model predictions with fuzzy category name matching.
Fixes mismatches like "Vegetables - Canned or Dried" → "Vegetables - Canned".

Usage:
    python3 eval_fuzzy.py --input eval_results_v11_clean.json --output eval_results_v11_clean_fuzzy.json
"""

import argparse
import json
from collections import Counter

CLEAN_CATEGORIES = {
    'Beans and Legumes - Canned or Dried',
    'Bread and Bakery Products',
    'Canned Protein',
    'Canned Tomato Products',
    'Carbohydrate Meal',
    'Condiments and Sauces',
    'Dairy and Dairy Alternatives',
    'Desserts and Sweets',
    'Drinks',
    'Fresh Produce',
    'Fruits - Canned or Processed',
    'Granola Products',
    'Meat and Poultry - Fresh',
    'Nut Butters and Nuts',
    'Ready Meals',
    'Savory Snacks and Crackers',
    'Soup',
    'Vegetables - Canned',
}

# Model output → canonical name
ALIASES = {
    'vegetables - canned or dried': 'Vegetables - Canned',
    'vegetables - canned and dried': 'Vegetables - Canned',
    'vegetables canned': 'Vegetables - Canned',
    'fruits - canned or dried': 'Fruits - Canned or Processed',
    'fruits - canned and dried': 'Fruits - Canned or Processed',
    'fruits canned': 'Fruits - Canned or Processed',
    'beans and legumes': 'Beans and Legumes - Canned or Dried',
    'beans - canned or dried': 'Beans and Legumes - Canned or Dried',
    'canned protein products': 'Canned Protein',
    'canned proteins': 'Canned Protein',
    'canned meat': 'Canned Protein',
    'canned seafood': 'Canned Protein',
    'meat and poultry - canned': 'Canned Protein',
    'seafood - canned': 'Canned Protein',
    'fresh produce items': 'Fresh Produce',
    'fresh fruit': 'Fresh Produce',
    'fresh fruits': 'Fresh Produce',
    'vegetables - fresh': 'Fresh Produce',
    'fresh vegetables': 'Fresh Produce',
    'snacks and crackers': 'Savory Snacks and Crackers',
    'savory snacks': 'Savory Snacks and Crackers',
    'nut butters': 'Nut Butters and Nuts',
    'granola': 'Granola Products',
    'ready meals and entrees': 'Ready Meals',
    'dairy': 'Dairy and Dairy Alternatives',
    'desserts': 'Desserts and Sweets',
    'sweets and desserts': 'Desserts and Sweets',
    'carbohydrate meals': 'Carbohydrate Meal',
    'bread and bakery': 'Bread and Bakery Products',
    'canned tomato': 'Canned Tomato Products',
    'condiments': 'Condiments and Sauces',
}

# Build lookup
_canonical = {c.lower(): c for c in CLEAN_CATEGORIES}
for alias, canon in ALIASES.items():
    _canonical[alias.lower()] = canon


def normalize(name):
    n = name.strip().lower()
    if n in _canonical:
        return _canonical[n]
    # Substring match
    for key, val in sorted(_canonical.items(), key=lambda x: -len(x[0])):
        if key in n or n in key:
            return val
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    preds = data.get('predictions', [])

    tp, fp, fn = 0, 0, 0
    per_class_tp = Counter()
    per_class_fp = Counter()
    per_class_fn = Counter()
    per_class_support = Counter()
    mismatches = []
    exact_match = 0

    for pred in preds:
        # Parse GT
        try:
            gt_data = json.loads(pred.get('target', '{}'))
            gt_cats = set(normalize(item['name']) for item in gt_data.get('items', []) if 'name' in item)
        except:
            gt_cats = set()

        # Parse prediction
        try:
            pred_text = pred.get('prediction', pred.get('predicted', ''))
            if isinstance(pred_text, str):
                pred_data = json.loads(pred_text)
            else:
                pred_data = pred_text
            pred_cats = set(normalize(item['name']) for item in pred_data.get('items', []) if 'name' in item)
        except:
            pred_cats = set()

        matched = gt_cats & pred_cats
        missed = gt_cats - pred_cats
        extra = pred_cats - gt_cats

        if not missed and not extra:
            exact_match += 1

        tp += len(matched)
        fn += len(missed)
        fp += len(extra)

        for c in gt_cats:
            per_class_support[c] += 1
        for c in matched:
            per_class_tp[c] += 1
        for c in missed:
            per_class_fn[c] += 1
        for c in extra:
            per_class_fp[c] += 1

        if missed or extra:
            mismatches.append({
                'image': pred.get('image', ''),
                'gt': sorted(gt_cats),
                'pred': sorted(pred_cats),
                'missed': sorted(missed),
                'extra': sorted(extra),
            })

    # Metrics
    micro_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    print("=" * 70)
    print("EVALUATION WITH FUZZY CATEGORY MATCHING")
    print("=" * 70)
    print(f"")
    print(f"  Micro Precision: {micro_p*100:.1f}%")
    print(f"  Micro Recall:    {micro_r*100:.1f}%")
    print(f"  Micro F1:        {micro_f1*100:.1f}%")
    print(f"  Exact Match:     {exact_match}/{len(preds)} ({100*exact_match/len(preds):.1f}%)")
    print(f"")
    print(f"  TP={tp}, FP={fp}, FN={fn}")
    print(f"  Mismatched images: {len(mismatches)} / {len(preds)}")
    print(f"")

    # Per-class
    all_cats = sorted(set(list(per_class_support.keys()) + list(per_class_fp.keys())))
    print(f"  {'Category':<40} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>5}")
    print("  " + "-" * 64)
    for cat in all_cats:
        t = per_class_tp[cat]
        f_p = per_class_fp[cat]
        f_n = per_class_fn[cat]
        sup = per_class_support[cat]
        p = t / (t + f_p) if (t + f_p) > 0 else 0
        r = t / (t + f_n) if (t + f_n) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  {cat:<40} {p*100:>5.1f}% {r*100:>5.1f}% {f1*100:>5.1f}% {sup:>5}")

    # Mismatches
    if mismatches:
        print(f"")
        print("  Sample mismatches:")
        for m in mismatches[:10]:
            print(f"    {m['image'][-60:]}")
            print(f"      GT:   {m['gt']}")
            print(f"      Pred: {m['pred']}")
            if m['missed']:
                print(f"      Miss: {m['missed']}")
            if m['extra']:
                print(f"      Extra:{m['extra']}")

    # Save
    output_path = args.output or args.input.replace('.json', '_fuzzy.json')
    result = {
        'micro_precision': round(micro_p, 4),
        'micro_recall': round(micro_r, 4),
        'micro_f1': round(micro_f1, 4),
        'exact_match_rate': round(exact_match / len(preds), 4),
        'tp': tp, 'fp': fp, 'fn': fn,
        'total_samples': len(preds),
        'mismatches': len(mismatches),
        'per_class': {cat: {
            'precision': round(per_class_tp[cat] / (per_class_tp[cat] + per_class_fp[cat]), 4) if (per_class_tp[cat] + per_class_fp[cat]) > 0 else 0,
            'recall': round(per_class_tp[cat] / (per_class_tp[cat] + per_class_fn[cat]), 4) if (per_class_tp[cat] + per_class_fn[cat]) > 0 else 0,
            'support': per_class_support[cat],
        } for cat in all_cats},
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
