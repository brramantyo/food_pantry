#!/usr/bin/env python3
"""
dataset_visualizations.py
=========================
Generate dataset distribution charts for the Food Pantry report.

Usage (on cluster):
    python3 dataset_visualizations.py --jsonl-dir ./florence2_data --output-dir ./charts

Generates:
    1. Per-class distribution bar chart (train/val/test stacked)
    2. Single vs multi-label pie chart
    3. Classes-per-image histogram
    4. Top multi-class co-occurrence heatmap
    5. Class imbalance ratio chart
"""

import json
import argparse
import os
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Nicer style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'train': '#2196F3',   # blue
    'valid': '#FF9800',   # orange
    'test': '#4CAF50',    # green
}

# Short names for readability
SHORT_NAMES = {
    "Beans and Legumes - Canned or Dried": "Beans/Legumes",
    "Bread and Bakery Products": "Bread/Bakery",
    "Canned Tomato Products": "Canned Tomato",
    "Carbohydrate Meal": "Carb Meal",
    "Condiments and Sauces": "Condiments",
    "Dairy and Dairy Alternatives": "Dairy/Alt",
    "Desserts and Sweets": "Desserts",
    "Fresh Fruit": "Fresh Fruit",
    "Fruits - Canned or Processed": "Fruits Canned",
    "Granola Products": "Granola",
    "Meat and Poultry - Canned": "Meat Canned",
    "Meat and Poultry - Fresh": "Meat Fresh",
    "Nut Butters and Nuts": "Nut Butters",
    "Ready Meals": "Ready Meals",
    "Savory Snacks and Crackers": "Snacks/Crackers",
    "Seafood - Canned": "Seafood Canned",
    "Vegetables - Canned": "Veg Canned",
    "Vegetables - Fresh": "Veg Fresh",
    "Baby Food": "Baby Food",
    "Drinks": "Drinks",
    "Soup": "Soup",
}


def load_split(jsonl_path):
    """Load a JSONL split and return per-image class lists."""
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            target = json.loads(rec['target'])
            items = target.get('items', [])
            names = sorted(set(item['name'] for item in items))
            records.append(names)
    return records


def get_class_counts(records):
    """Count occurrences of each class."""
    counter = Counter()
    for names in records:
        for n in names:
            counter[n] += 1
    return counter


def get_all_classes(splits_data):
    """Get sorted list of all classes by total count (descending)."""
    total = Counter()
    for records in splits_data.values():
        total += get_class_counts(records)
    return [cls for cls, _ in total.most_common()]


# ── Chart 1: Stacked bar chart per-class ───────────────────────────────────────

def plot_class_distribution(splits_data, all_classes, output_path):
    """Stacked horizontal bar chart showing train/val/test per class."""
    fig, ax = plt.subplots(figsize=(12, 9))

    short = [SHORT_NAMES.get(c, c) for c in reversed(all_classes)]
    train_counts = [get_class_counts(splits_data['train']).get(c, 0) for c in reversed(all_classes)]
    valid_counts = [get_class_counts(splits_data['valid']).get(c, 0) for c in reversed(all_classes)]
    test_counts = [get_class_counts(splits_data['test']).get(c, 0) for c in reversed(all_classes)]

    y = np.arange(len(short))
    h = 0.7

    bars_train = ax.barh(y, train_counts, h, label='Train', color=COLORS['train'], alpha=0.85)
    bars_valid = ax.barh(y, valid_counts, h, left=train_counts, label='Valid', color=COLORS['valid'], alpha=0.85)
    bars_test = ax.barh(y, test_counts, h,
                        left=[t + v for t, v in zip(train_counts, valid_counts)],
                        label='Test', color=COLORS['test'], alpha=0.85)

    # Add total count labels
    for i, (t, v, te) in enumerate(zip(train_counts, valid_counts, test_counts)):
        total = t + v + te
        ax.text(total + 2, i, str(total), va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_title('Per-Class Distribution Across Splits', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(0, max(t + v + te for t, v, te in zip(train_counts, valid_counts, test_counts)) * 1.12)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Chart 2: Pie chart single vs multi-label ──────────────────────────────────

def plot_label_type_pie(splits_data, output_path):
    """Pie chart: single-label vs multi-label images per split."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (split_name, records) in zip(axes, splits_data.items()):
        single = sum(1 for r in records if len(r) == 1)
        multi = sum(1 for r in records if len(r) > 1)
        empty = sum(1 for r in records if len(r) == 0)

        sizes = [single, multi]
        labels = [f'Single-label\n({single})', f'Multi-label\n({multi})']
        colors = ['#64B5F6', '#FF8A65']
        if empty > 0:
            sizes.append(empty)
            labels.append(f'Empty\n({empty})')
            colors.append('#E0E0E0')

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90,
            textprops={'fontsize': 10},
            pctdistance=0.6,
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')
        ax.set_title(f'{split_name.upper()} ({len(records)} images)',
                     fontsize=12, fontweight='bold')

    fig.suptitle('Single-Label vs Multi-Label Images', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Chart 3: Classes-per-image histogram ───────────────────────────────────────

def plot_classes_per_image(splits_data, output_path):
    """Histogram of how many classes appear per image."""
    fig, ax = plt.subplots(figsize=(8, 5))

    all_counts = []
    for split_name, records in splits_data.items():
        counts = [len(r) for r in records]
        all_counts.extend(counts)

    max_classes = max(all_counts) if all_counts else 1
    bins = np.arange(0.5, max_classes + 1.5, 1)

    for split_name, records in splits_data.items():
        counts = [len(r) for r in records]
        ax.hist(counts, bins=bins, alpha=0.6, label=f'{split_name} ({len(records)})',
                color=COLORS[split_name], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Number of Classes per Image', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Distribution of Classes per Image', fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, max_classes + 1))
    ax.legend(fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Chart 4: Co-occurrence heatmap ─────────────────────────────────────────────

def plot_cooccurrence(splits_data, all_classes, output_path):
    """Heatmap of class co-occurrence (how often two classes appear together)."""
    # Build co-occurrence matrix from ALL splits combined
    n = len(all_classes)
    cooccur = np.zeros((n, n), dtype=int)
    cls_idx = {c: i for i, c in enumerate(all_classes)}

    for records in splits_data.values():
        for names in records:
            if len(names) > 1:
                for i_name in names:
                    for j_name in names:
                        if i_name != j_name:
                            ii = cls_idx.get(i_name)
                            jj = cls_idx.get(j_name)
                            if ii is not None and jj is not None:
                                cooccur[ii][jj] += 1

    # Only upper triangle (symmetric)
    mask = np.triu(np.ones_like(cooccur, dtype=bool), k=0)
    cooccur_masked = np.where(mask, cooccur, np.nan)

    short = [SHORT_NAMES.get(c, c) for c in all_classes]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cooccur, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(short, fontsize=9)

    # Add text annotations for non-zero values
    for i in range(n):
        for j in range(n):
            if cooccur[i][j] > 0:
                color = 'white' if cooccur[i][j] > cooccur.max() * 0.6 else 'black'
                ax.text(j, i, str(cooccur[i][j]), ha='center', va='center',
                        fontsize=7, color=color, fontweight='bold')

    ax.set_title('Class Co-occurrence Matrix (All Splits)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Co-occurrence Count', fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Chart 5: Imbalance ratio ──────────────────────────────────────────────────

def plot_imbalance(splits_data, all_classes, output_path):
    """Bar chart showing how imbalanced each class is (ratio to largest class)."""
    total = Counter()
    for records in splits_data.values():
        total += get_class_counts(records)

    max_count = max(total.values())
    ratios = [(total[c] / max_count, c) for c in all_classes]
    ratios.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(10, 8))

    short = [SHORT_NAMES.get(c, c) for c in [r[1] for r in ratios]]
    vals = [r[0] for r in ratios]
    colors_list = ['#EF5350' if v < 0.2 else '#FF9800' if v < 0.4 else '#4CAF50' for v in vals]

    bars = ax.barh(range(len(short)), vals, color=colors_list, alpha=0.85, edgecolor='white')

    for i, (v, (_, c)) in enumerate(zip(vals, ratios)):
        count = total[c]
        ax.text(v + 0.01, i, f'{count} ({v:.0%})', va='center', fontsize=9)

    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel('Ratio to Largest Class', fontsize=12)
    ax.set_title('Class Imbalance Ratio\n(Red = severely underrepresented)', fontsize=14, fontweight='bold')
    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.5, label='Critical threshold (20%)')
    ax.axvline(x=0.4, color='orange', linestyle='--', alpha=0.5, label='Warning threshold (40%)')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 1.2)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Chart 6: Split proportion per class ────────────────────────────────────────

def plot_split_proportions(splits_data, all_classes, output_path):
    """Show train/valid/test proportion for each class (should be ~70/15/15)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    counts = {}
    for split_name, records in splits_data.items():
        counts[split_name] = get_class_counts(records)

    short = [SHORT_NAMES.get(c, c) for c in reversed(all_classes)]
    y = np.arange(len(short))

    for cls_list in [reversed(all_classes)]:
        train_pct, valid_pct, test_pct = [], [], []
        for c in reversed(all_classes):
            total = sum(counts[s].get(c, 0) for s in ['train', 'valid', 'test'])
            if total == 0:
                train_pct.append(0); valid_pct.append(0); test_pct.append(0)
            else:
                train_pct.append(counts['train'].get(c, 0) / total * 100)
                valid_pct.append(counts['valid'].get(c, 0) / total * 100)
                test_pct.append(counts['test'].get(c, 0) / total * 100)

    h = 0.7
    ax.barh(y, train_pct, h, label='Train', color=COLORS['train'], alpha=0.85)
    ax.barh(y, valid_pct, h, left=train_pct, label='Valid', color=COLORS['valid'], alpha=0.85)
    ax.barh(y, test_pct, h,
            left=[t + v for t, v in zip(train_pct, valid_pct)],
            label='Test', color=COLORS['test'], alpha=0.85)

    ax.axvline(x=70, color='blue', linestyle=':', alpha=0.3)
    ax.axvline(x=85, color='orange', linestyle=':', alpha=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_title('Split Proportion per Class (ideal: 70/15/15)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(0, 105)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate dataset distribution charts")
    parser.add_argument("--jsonl-dir", default="./florence2_data",
                        help="Directory with train_v5.jsonl, valid_v5.jsonl, test_v5.jsonl")
    parser.add_argument("--output-dir", default="./charts",
                        help="Output directory for chart images")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output format")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ext = args.format

    # Load data
    print("Loading dataset splits...")
    splits_data = {}
    for split_name, filename in [('train', 'train_v5.jsonl'), ('valid', 'valid_v5.jsonl'), ('test', 'test_v5.jsonl')]:
        path = os.path.join(args.jsonl_dir, filename)
        if os.path.exists(path):
            splits_data[split_name] = load_split(path)
            print(f"  {split_name}: {len(splits_data[split_name])} images")
        else:
            print(f"  [SKIP] {path} not found")

    if not splits_data:
        print("ERROR: No data loaded!")
        return

    all_classes = get_all_classes(splits_data)
    print(f"  Total classes: {len(all_classes)}")

    # Generate charts
    print("\nGenerating charts...")

    plot_class_distribution(splits_data, all_classes,
                           os.path.join(args.output_dir, f'1_class_distribution.{ext}'))

    plot_label_type_pie(splits_data,
                        os.path.join(args.output_dir, f'2_label_type_pie.{ext}'))

    plot_classes_per_image(splits_data,
                          os.path.join(args.output_dir, f'3_classes_per_image.{ext}'))

    plot_cooccurrence(splits_data, all_classes,
                      os.path.join(args.output_dir, f'4_cooccurrence_heatmap.{ext}'))

    plot_imbalance(splits_data, all_classes,
                   os.path.join(args.output_dir, f'5_class_imbalance.{ext}'))

    plot_split_proportions(splits_data, all_classes,
                           os.path.join(args.output_dir, f'6_split_proportions.{ext}'))

    print(f"\nDone! All charts saved to: {args.output_dir}/")
    print("Files:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(f'.{ext}'):
            print(f"  {f}")


if __name__ == "__main__":
    main()
