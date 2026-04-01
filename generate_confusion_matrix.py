#!/usr/bin/env python3
"""
generate_confusion_matrix.py
=============================
Generate confusion matrices for the best models (ensemble, v11, contrastive).
Produces:
  1. NxN confusion matrix heatmaps (saved as PNG)
  2. Per-class P/R/F1 table
  3. Most common confusions summary

Usage:
  python generate_confusion_matrix.py [--json eval_pipeline_ensemble.json]
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

CATEGORIES = [
    "Baby Food", "Beans and Legumes - Canned or Dried", "Bread and Bakery Products",
    "Canned Tomato Products", "Carbohydrate Meal", "Condiments and Sauces",
    "Dairy and Dairy Alternatives", "Desserts and Sweets", "Drinks", "Fresh Fruit",
    "Fruits - Canned or Processed", "Granola Products", "Meat and Poultry - Canned",
    "Meat and Poultry - Fresh", "Nut Butters and Nuts", "Ready Meals",
    "Savory Snacks and Crackers", "Seafood - Canned", "Soup",
    "Vegetables - Canned", "Vegetables - Fresh",
]

# Short names for display
SHORT_NAMES = [
    "Baby", "Beans", "Bread", "CannTom", "CarbMeal", "Condim",
    "Dairy", "Dessert", "Drinks", "FreshFr", "CannFr", "Granola",
    "MeatCan", "MeatFr", "NutBut", "ReadyMl", "Snacks", "Seafood",
    "Soup", "VegCan", "VegFr",
]

NUM_CLASSES = len(CATEGORIES)
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}


def build_confusion_matrix(results, pred_key="ensemble_preds"):
    """
    Build multi-label confusion matrix.
    For multi-label: count (true_class, pred_class) co-occurrences.
    
    Approach:
    - For each image, we have target set and predicted set
    - TP: category in both target and pred
    - FP: category in pred but not target → confused WITH what?
    - FN: category in target but not pred → missed
    
    For the NxN matrix:
    - cm[i][j] = number of times true class i was predicted as class j
    - Diagonal = correct predictions
    - Off-diagonal = confusions
    """
    # Multi-label approach: for each target category, 
    # track what was predicted for images containing that category
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    
    # Also track: for each FP prediction, what was the actual target?
    fp_pairs = []  # (predicted_wrong, actual_target)
    fn_list = []   # missed categories
    
    tp = Counter()
    fp = Counter()
    fn = Counter()
    
    for r in results:
        target = set(r.get("target", []))
        
        # Get predictions based on available keys
        if pred_key in r:
            preds = set(r[pred_key])
        elif "ensemble" in r:
            preds = set(r["ensemble"])
        elif "v11_preds" in r:
            # Reconstruct ensemble from v11 + contrastive
            v11 = set(r.get("v11_preds", []))
            con = set(r.get("contrastive_preds", []))
            preds = v11 | con
        else:
            continue
        
        # Filter to valid categories
        target = {c for c in target if c in CAT_TO_IDX}
        preds = {c for c in preds if c in CAT_TO_IDX}
        
        # Count TP/FP/FN
        for c in target & preds:
            tp[c] += 1
            cm[CAT_TO_IDX[c]][CAT_TO_IDX[c]] += 1
        
        for c in preds - target:
            fp[c] += 1
            # This was a false positive — attribute it to each target class
            for t in target:
                fp_pairs.append((c, t))
                cm[CAT_TO_IDX[t]][CAT_TO_IDX[c]] += 1
        
        for c in target - preds:
            fn[c] += 1
            fn_list.append(c)
    
    return cm, tp, fp, fn, fp_pairs, fn_list


def print_per_class_metrics(tp, fp, fn):
    """Print per-class P/R/F1 table."""
    print(f"\n{'='*80}")
    print(f"  PER-CLASS METRICS")
    print(f"{'='*80}")
    print(f"  {'Class':<40} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*5}")
    
    f1s = []
    for cls in CATEGORIES:
        t, f, n = tp[cls], fp[cls], fn[cls]
        p = t / max(t + f, 1)
        r = t / max(t + n, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        f1s.append(f1)
        print(f"  {cls:<40} {p:>6.1%} {r:>6.1%} {f1:>6.1%} {t:>5} {f:>5} {n:>5}")
    
    # Micro
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / max(total_tp + total_fp, 1)
    micro_r = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    macro_f1 = sum(f1s) / len(f1s)
    
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*7}")
    print(f"  {'MICRO':<40} {micro_p:>6.1%} {micro_r:>6.1%} {micro_f1:>6.1%}")
    print(f"  {'MACRO':<40} {'':>7} {'':>7} {macro_f1:>6.1%}")


def print_confusion_matrix(cm):
    """Print the confusion matrix as a formatted table."""
    print(f"\n{'='*80}")
    print(f"  CONFUSION MATRIX (rows=true, cols=predicted)")
    print(f"  Diagonal = correct, off-diagonal = errors")
    print(f"{'='*80}")
    
    # Header
    label = 'True \\ Pred'
    header = f"  {label:<12}"
    for sn in SHORT_NAMES:
        header += f"{sn:>7}"
    header += f"{'Total':>7}"
    print(header)
    print(f"  {'-'*12}" + "-"*7*len(SHORT_NAMES) + "-"*7)
    
    for i, (cat, sn) in enumerate(zip(CATEGORIES, SHORT_NAMES)):
        row = f"  {sn:<12}"
        row_sum = cm[i].sum()
        for j in range(NUM_CLASSES):
            val = cm[i][j]
            if val == 0:
                row += f"{'·':>7}"
            elif i == j:
                row += f"{'['+str(val)+']':>7}"
            else:
                row += f"{val:>7}"
        row += f"{row_sum:>7}"
        print(row)


def print_top_confusions(fp_pairs, fn_list, top_n=15):
    """Print most common confusion pairs and most missed classes."""
    print(f"\n{'='*80}")
    print(f"  TOP CONFUSION PAIRS (false positive → what was actually there)")
    print(f"{'='*80}")
    
    pair_counts = Counter()
    for pred, actual in fp_pairs:
        pair_counts[(actual, pred)] += 1
    
    print(f"  {'Actual (true)':<35} {'Predicted (wrong)':<35} {'Count':>6}")
    print(f"  {'-'*35} {'-'*35} {'-'*6}")
    for (actual, pred), count in pair_counts.most_common(top_n):
        print(f"  {actual:<35} {pred:<35} {count:>6}")
    
    print(f"\n{'='*80}")
    print(f"  MOST MISSED CLASSES (false negatives)")
    print(f"{'='*80}")
    fn_counts = Counter(fn_list)
    for cls, count in fn_counts.most_common(10):
        print(f"  {cls:<40} {count:>5} missed")


def save_heatmap(cm, output_path):
    """Save confusion matrix as PNG heatmap."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 14))
        
        # Normalize by row (true class) for percentages
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums * 100
        
        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=100)
        
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(SHORT_NAMES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(SHORT_NAMES, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix (row-normalized %)', fontsize=14)
        
        # Add text annotations
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                val = cm_norm[i][j]
                raw = cm[i][j]
                if raw > 0:
                    color = 'white' if val > 50 else 'black'
                    ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                            fontsize=7, color=color, fontweight='bold' if i==j else 'normal')
        
        plt.colorbar(im, ax=ax, shrink=0.8, label='Row %')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Heatmap saved to: {output_path}")
    except ImportError:
        print("\n  matplotlib not available — skipping heatmap PNG")


def save_latex_table(cm, tp, fp, fn, output_path):
    """Save per-class metrics as LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-class metrics for ensemble (v11 $\cup$ contrastive)}")
    lines.append(r"\label{tab:perclass_ensemble}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Class & Precision & Recall & F1 & Support \\")
    lines.append(r"\midrule")
    
    for cls in CATEGORIES:
        t, f, n = tp[cls], fp[cls], fn[cls]
        p = t / max(t + f, 1)
        r = t / max(t + n, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        support = t + n
        cls_escaped = cls.replace("&", r"\&").replace("--", " -- ")
        lines.append(f"{cls_escaped} & {p:.1%} & {r:.1%} & {f1:.1%} & {support} \\\\")
    
    lines.append(r"\midrule")
    total_tp = sum(tp.values())
    total_fp = sum(fp.values())
    total_fn = sum(fn.values())
    micro_p = total_tp / max(total_tp + total_fp, 1)
    micro_r = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    lines.append(f"\\textbf{{Micro avg}} & \\textbf{{{micro_p:.1%}}} & \\textbf{{{micro_r:.1%}}} & \\textbf{{{micro_f1:.1%}}} & {total_tp + total_fn} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="eval_pipeline_ensemble.json")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    
    with open(args.json) as f:
        data = json.load(f)
    
    results = data.get("results", data.get("samples", []))
    print(f"Loaded {len(results)} results from {args.json}")
    
    # Detect available prediction keys
    if results:
        sample = results[0]
        print(f"Available keys: {list(sample.keys())}")
    
    # Try different prediction key patterns
    methods = {}
    if results:
        sample = results[0]
        if "ensemble" in sample or "v11_preds" in sample:
            methods["ensemble"] = "ensemble"
        if "v11_preds" in sample:
            methods["v11"] = "v11_preds"
        if "contrastive_preds" in sample:
            methods["contrastive"] = "contrastive_preds"
    
    if not methods:
        # Fallback: try to find any prediction key
        print("WARNING: Could not detect prediction keys. Trying generic approach...")
        methods["predictions"] = "predictions"
    
    for method_name, pred_key in methods.items():
        print(f"\n{'#'*80}")
        print(f"  ANALYSIS: {method_name.upper()}")
        print(f"{'#'*80}")
        
        cm, tp, fp, fn, fp_pairs, fn_list = build_confusion_matrix(results, pred_key)
        
        print_per_class_metrics(tp, fp, fn)
        print_confusion_matrix(cm)
        print_top_confusions(fp_pairs, fn_list)
        
        # Save outputs
        heatmap_path = os.path.join(args.output_dir, f"confusion_matrix_{method_name}.png")
        save_heatmap(cm, heatmap_path)
        
        latex_path = os.path.join(args.output_dir, f"perclass_{method_name}.tex")
        save_latex_table(cm, tp, fp, fn, latex_path)


if __name__ == "__main__":
    main()
