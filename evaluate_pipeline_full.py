#!/usr/bin/env python3
"""
evaluate_pipeline_full.py
=========================
Full evaluation of end-to-end pipeline on the ENTIRE test set.

Generates:
  1. Classification metrics (Task 1) — same as evaluate_florence2.py
  2. USDA matching quality analysis (Task 2):
     - Per-category: top USDA matches, nutrition ranges, match diversity
     - Retrieval relevance: cosine similarity stats (mean, min, max)
     - Cross-category nutrition sanity check
  3. Error propagation analysis:
     - When Task 1 is wrong, how far off is the nutrition estimate?
     - Nutrition distance between correct vs misclassified categories
  4. Full pipeline stats:
     - Latency (per-image, per-stage)
     - Example outputs for report (LaTeX-ready tables)

Output: JSON results + human-readable report + LaTeX snippet

Usage:
  python evaluate_pipeline_full.py \
    --jsonl ./florence2_data/test_v5.jsonl \
    --data-dir . \
    --base-model microsoft/Florence-2-large-ft \
    --checkpoint ./checkpoints_v11/best_model \
    --usda-dir ./usda_data \
    --output ./eval_pipeline_full.json \
    --bf16
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

from usda_matcher import USDAMatcher


# ── Constants ──────────────────────────────────────────────────────────────────

TASK_PROMPT = "<OD>"

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


def normalize_category(name):
    return CANONICAL_MAP.get(name.lower().strip(), name)


# ── Parse helpers (same robust parser as evaluate_florence2.py) ────────────────

def parse_prediction(text):
    """Parse Florence-2 classifier output."""
    text = text.strip()
    if not text:
        return None
    
    # Try direct JSON
    for candidate in [text, text.replace("'", '"')]:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict) and "items" in result:
                return result
        except json.JSONDecodeError:
            pass
    
    # Find JSON in text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict) and "items" in result:
                return result
        except json.JSONDecodeError:
            pass
    
    # Try closing truncated JSON
    if start >= 0:
        fragment = text[start:]
        for suffix in ['"}]}', '"}]}}', ']}}', ']}', '}]}', '"}}', '"}', '}',
                       ', "confidence": "high"}]}']:
            try:
                result = json.loads(fragment + suffix)
                if isinstance(result, dict) and "items" in result:
                    valid = [i for i in result["items"] if isinstance(i, dict) and "name" in i]
                    if valid:
                        return {"items": valid}
            except json.JSONDecodeError:
                pass
    
    # Regex fallback
    items = []
    for cat in VALID_CATEGORIES:
        if cat.lower() in text.lower():
            items.append({"name": cat, "package_type": "unknown", "confidence": "high"})
    if items:
        return {"items": items}
    
    return None


def extract_categories(parsed):
    if not parsed or "items" not in parsed:
        return set()
    return {normalize_category(item["name"]) for item in parsed["items"]
            if normalize_category(item.get("name", "")) in VALID_CATEGORIES}


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full pipeline evaluation")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--usda-dir", type=str, default="./usda_data")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default="./eval_pipeline_full.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Device: {device}")
    
    # ── Load classifier ────────────────────────────────────────────────────
    print(f"\nLoading classifier: {args.checkpoint}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model = model.merge_and_unload().to(device).eval()
    print("  Classifier loaded ✓")
    
    # ── Load USDA matcher ──────────────────────────────────────────────────
    matcher = USDAMatcher(usda_dir=args.usda_dir)
    print("  USDA matcher loaded ✓")
    
    # ── Load test data ─────────────────────────────────────────────────────
    samples = []
    with open(args.jsonl, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"\n  Evaluating {len(samples)} samples...\n")
    
    # ── Pre-compute category nutrition profiles ────────────────────────────
    print("  Building per-category nutrition profiles...")
    category_nutrition = {}
    for cat in sorted(VALID_CATEGORIES):
        summary = matcher.get_category_nutrition_summary(cat, top_k=50)
        category_nutrition[cat] = summary
    print("  Done.\n")
    
    # ── Process all images ─────────────────────────────────────────────────
    results = []
    
    # Task 1 tracking
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()
    class_support = Counter()
    exact_match = 0
    
    # Task 2 tracking
    usda_scores_per_cat = defaultdict(list)  # category → list of top-1 cosine scores
    
    # Error propagation
    correct_nutrition = []   # nutrition when classification correct
    wrong_nutrition = []     # nutrition when classification wrong
    
    # Timing
    classify_times = []
    match_times = []
    total_times = []
    
    t0_all = time.time()
    
    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)
        
        if not os.path.exists(img_path):
            continue
        
        image = Image.open(img_path).convert("RGB")
        
        # Ground truth
        target_parsed = json.loads(sample["target"])
        target_cats = {normalize_category(item["name"]) for item in target_parsed.get("items", [])
                       if normalize_category(item.get("name", "")) in VALID_CATEGORIES}
        
        # ── Task 1: Classify ──────────────────────────────────────────
        t1 = time.time()
        
        inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    gen_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512, num_beams=3, early_stopping=True,
                    )
            else:
                gen_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512, num_beams=3, early_stopping=True,
                )
        
        pred_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        pred_parsed = parse_prediction(pred_text)
        pred_cats = extract_categories(pred_parsed)
        
        t2 = time.time()
        classify_time = t2 - t1
        classify_times.append(classify_time)
        
        # Task 1 metrics
        for cls in target_cats | pred_cats:
            if cls in target_cats and cls in pred_cats:
                class_tp[cls] += 1
            elif cls in pred_cats:
                class_fp[cls] += 1
            elif cls in target_cats:
                class_fn[cls] += 1
        for cls in target_cats:
            class_support[cls] += 1
        if target_cats == pred_cats:
            exact_match += 1
        
        # ── Task 2: USDA Match ────────────────────────────────────────
        t3 = time.time()
        
        usda_results = {}
        pred_items = pred_parsed.get("items", []) if pred_parsed else []
        
        for item in pred_items:
            name = normalize_category(item.get("name", ""))
            if name not in VALID_CATEGORIES:
                continue
            pkg = item.get("package_type", "unknown")
            
            matches = matcher.match_pantry_prediction(name, package_type=pkg, top_k=args.top_k)
            usda_results[name] = matches
            
            # Track retrieval scores
            if matches:
                usda_scores_per_cat[name].append(matches[0].get("score", 0))
        
        t4 = time.time()
        match_time = t4 - t3
        match_times.append(match_time)
        total_times.append(t4 - t1)
        
        # ── Error propagation analysis ────────────────────────────────
        is_correct = (target_cats == pred_cats)
        
        # Get nutrition from predicted categories
        pred_nutrition = {}
        for cat in pred_cats:
            if cat in usda_results and usda_results[cat]:
                top_match = usda_results[cat][0]
                nuts = top_match.get("nutrients", {})
                for k, v in nuts.items():
                    if v is not None:
                        pred_nutrition[k] = pred_nutrition.get(k, 0) + v
        
        if is_correct:
            correct_nutrition.append(pred_nutrition)
        else:
            wrong_nutrition.append(pred_nutrition)
        
        # Store result
        result_entry = {
            "image": img_rel,
            "target_categories": sorted(target_cats),
            "predicted_categories": sorted(pred_cats),
            "classification_correct": is_correct,
            "classify_time_s": round(classify_time, 3),
            "match_time_s": round(match_time, 3),
            "usda_top_matches": {
                cat: [{
                    "description": m.get("description", ""),
                    "brand": m.get("brand_owner", ""),
                    "score": round(m.get("score", 0), 4),
                    "nutrients": m.get("nutrients", {}),
                    "serving": f"{m.get('serving_size', '?')} {m.get('serving_size_unit', '')}".strip(),
                } for m in matches[:3]]
                for cat, matches in usda_results.items()
            },
            "predicted_nutrition": pred_nutrition,
        }
        results.append(result_entry)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0_all
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{len(samples)} ({rate:.1f} img/s, ETA: {(len(samples)-i-1)/rate:.0f}s)")
    
    total_elapsed = time.time() - t0_all
    n = len(results)
    
    # ══════════════════════════════════════════════════════════════════
    #  COMPUTE ALL METRICS
    # ══════════════════════════════════════════════════════════════════
    
    # ── Task 1 Metrics ─────────────────────────────────────────────────
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    micro_p = total_tp / max(total_tp + total_fp, 1)
    micro_r = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    
    per_class = {}
    all_classes = sorted(set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_fn.keys())))
    for cls in all_classes:
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        per_class[cls] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                          "support": class_support[cls]}
    
    macro_f1 = sum(m["f1"] for m in per_class.values()) / max(len(per_class), 1)
    
    task1_metrics = {
        "exact_match": round(exact_match / max(n, 1), 4),
        "micro": {"precision": round(micro_p, 4), "recall": round(micro_r, 4), "f1": round(micro_f1, 4)},
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }
    
    # ── Task 2 Metrics ─────────────────────────────────────────────────
    usda_retrieval_stats = {}
    for cat in sorted(usda_scores_per_cat.keys()):
        scores = usda_scores_per_cat[cat]
        usda_retrieval_stats[cat] = {
            "mean_score": round(float(np.mean(scores)), 4),
            "min_score": round(float(np.min(scores)), 4),
            "max_score": round(float(np.max(scores)), 4),
            "n_queries": len(scores),
        }
    
    overall_scores = [s for scores in usda_scores_per_cat.values() for s in scores]
    
    task2_metrics = {
        "overall_retrieval_score": {
            "mean": round(float(np.mean(overall_scores)), 4) if overall_scores else 0,
            "std": round(float(np.std(overall_scores)), 4) if overall_scores else 0,
            "min": round(float(np.min(overall_scores)), 4) if overall_scores else 0,
            "max": round(float(np.max(overall_scores)), 4) if overall_scores else 0,
        },
        "per_category_retrieval": usda_retrieval_stats,
        "category_nutrition_profiles": {
            cat: {
                "top_3_matches": summary.get("top_matches", [])[:3],
                "nutrition_range": {
                    k: {"mean": v["mean"], "min": v["min"], "max": v["max"]}
                    for k, v in summary.get("nutrition_summary", {}).items()
                    if k in ["energy_kcal", "protein_g", "carbohydrate_g", "total_fat_g",
                             "fiber_g", "sugar_g", "sodium_mg"]
                }
            }
            for cat, summary in category_nutrition.items()
        },
    }
    
    # ── Error Propagation Analysis ─────────────────────────────────────
    def avg_nutrition(nutrition_list, key):
        vals = [n.get(key, 0) for n in nutrition_list if key in n]
        return round(float(np.mean(vals)), 1) if vals else 0
    
    error_analysis = {
        "n_correct": len(correct_nutrition),
        "n_wrong": len(wrong_nutrition),
        "correct_pct": round(len(correct_nutrition) / max(n, 1) * 100, 1),
    }
    
    for key, label in [("energy_kcal", "Calories"), ("protein_g", "Protein"),
                        ("carbohydrate_g", "Carbs"), ("total_fat_g", "Fat")]:
        correct_avg = avg_nutrition(correct_nutrition, key)
        wrong_avg = avg_nutrition(wrong_nutrition, key)
        error_analysis[f"{label}_correct_avg"] = correct_avg
        error_analysis[f"{label}_wrong_avg"] = wrong_avg
        error_analysis[f"{label}_delta"] = round(wrong_avg - correct_avg, 1)
    
    # ── Timing ─────────────────────────────────────────────────────────
    timing = {
        "total_time_s": round(total_elapsed, 1),
        "images_per_second": round(n / total_elapsed, 2),
        "avg_classify_ms": round(float(np.mean(classify_times)) * 1000, 1),
        "avg_match_ms": round(float(np.mean(match_times)) * 1000, 1),
        "avg_total_ms": round(float(np.mean(total_times)) * 1000, 1),
        "p95_total_ms": round(float(np.percentile(total_times, 95)) * 1000, 1),
    }
    
    # ══════════════════════════════════════════════════════════════════
    #  PRINT REPORT
    # ══════════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"  END-TO-END PIPELINE EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"  Samples: {n}")
    print(f"  Total time: {total_elapsed:.1f}s ({n/total_elapsed:.1f} img/s)")
    
    # Task 1
    print(f"\n{'='*70}")
    print(f"  TASK 1: CLASSIFICATION")
    print(f"{'='*70}")
    print(f"  Exact Match: {task1_metrics['exact_match']:.1%}")
    print(f"  Micro:  P={micro_p:.1%}  R={micro_r:.1%}  F1={micro_f1:.1%}")
    print(f"  Macro F1: {macro_f1:.1%}")
    
    print(f"\n  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
    for cls in sorted(per_class.keys()):
        m = per_class[cls]
        print(f"  {cls:<45} {m['precision']:>5.1%} {m['recall']:>5.1%} {m['f1']:>5.1%} {m['support']:>5}")
    
    # Task 2
    print(f"\n{'='*70}")
    print(f"  TASK 2: USDA RETRIEVAL")
    print(f"{'='*70}")
    if overall_scores:
        os_stats = task2_metrics["overall_retrieval_score"]
        print(f"  Overall cosine similarity: {os_stats['mean']:.3f} ± {os_stats['std']:.3f} "
              f"(range: {os_stats['min']:.3f} — {os_stats['max']:.3f})")
    
    print(f"\n  {'Category':<40} {'Mean Score':>10} {'Queries':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*8}")
    for cat, stats in sorted(usda_retrieval_stats.items()):
        print(f"  {cat:<40} {stats['mean_score']:>9.3f} {stats['n_queries']:>8}")
    
    # Nutrition sanity check
    print(f"\n{'='*70}")
    print(f"  NUTRITION PROFILES BY CATEGORY (top-50 USDA matches)")
    print(f"{'='*70}")
    print(f"  {'Category':<35} {'Kcal':>8} {'Protein':>8} {'Carbs':>8} {'Fat':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cat in sorted(VALID_CATEGORIES):
        profile = task2_metrics["category_nutrition_profiles"][cat]
        nuts = profile.get("nutrition_range", {})
        kcal = nuts.get("energy_kcal", {}).get("mean", "-")
        prot = nuts.get("protein_g", {}).get("mean", "-")
        carb = nuts.get("carbohydrate_g", {}).get("mean", "-")
        fat = nuts.get("total_fat_g", {}).get("mean", "-")
        kcal_s = f"{kcal}" if isinstance(kcal, (int, float)) else "-"
        prot_s = f"{prot}g" if isinstance(prot, (int, float)) else "-"
        carb_s = f"{carb}g" if isinstance(carb, (int, float)) else "-"
        fat_s = f"{fat}g" if isinstance(fat, (int, float)) else "-"
        print(f"  {cat:<35} {kcal_s:>8} {prot_s:>8} {carb_s:>8} {fat_s:>8}")
    
    # Error propagation
    print(f"\n{'='*70}")
    print(f"  ERROR PROPAGATION: Classification → Nutrition")
    print(f"{'='*70}")
    print(f"  Correct classifications: {error_analysis['n_correct']}/{n} ({error_analysis['correct_pct']}%)")
    print(f"  Wrong classifications:   {error_analysis['n_wrong']}/{n}")
    print(f"\n  {'Nutrient':<20} {'Correct Avg':>12} {'Wrong Avg':>12} {'Delta':>10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    for key, label in [("Calories", "kcal"), ("Protein", "g"), ("Carbs", "g"), ("Fat", "g")]:
        c_avg = error_analysis[f"{key}_correct_avg"]
        w_avg = error_analysis[f"{key}_wrong_avg"]
        delta = error_analysis[f"{key}_delta"]
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<20} {c_avg:>10.1f}{label:>2} {w_avg:>10.1f}{label:>2} {sign}{delta:>7.1f}{label:>2}")
    
    # Timing
    print(f"\n{'='*70}")
    print(f"  LATENCY")
    print(f"{'='*70}")
    print(f"  Classification: {timing['avg_classify_ms']:.0f} ms/image")
    print(f"  USDA matching:  {timing['avg_match_ms']:.0f} ms/image")
    print(f"  Total:          {timing['avg_total_ms']:.0f} ms/image (p95: {timing['p95_total_ms']:.0f} ms)")
    print(f"  Throughput:     {timing['images_per_second']:.1f} img/s")
    
    # ── Sample outputs (first 5) ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SAMPLE OUTPUTS (first 5)")
    print(f"{'='*70}")
    for r in results[:5]:
        print(f"\n  📷 {os.path.basename(r['image'])}")
        print(f"  Target:    {r['target_categories']}")
        print(f"  Predicted: {r['predicted_categories']}")
        match = "✓" if r['classification_correct'] else "✗"
        print(f"  Task 1: {match}")
        for cat, matches in r.get("usda_top_matches", {}).items():
            if matches:
                top = matches[0]
                nuts = top.get("nutrients", {})
                kcal = nuts.get("energy_kcal", "?")
                print(f"  → {cat}: {top['description'][:60]} [{top['score']:.3f}] ({kcal} kcal)")
    
    # ── Save ───────────────────────────────────────────────────────────
    output = {
        "pipeline": "end_to_end_v11_usda",
        "n_samples": n,
        "task1_metrics": task1_metrics,
        "task2_metrics": task2_metrics,
        "error_propagation": error_analysis,
        "timing": timing,
        "per_image_results": results[:100],  # first 100 for analysis
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")
    
    # ── LaTeX snippet ──────────────────────────────────────────────────
    latex_path = args.output.replace(".json", "_table.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated pipeline evaluation table\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{End-to-End Pipeline Performance}\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Metric & Precision & Recall & F1 & \\\\\n\\hline\n")
        f.write(f"Task 1 (Micro) & {micro_p:.1%} & {micro_r:.1%} & {micro_f1:.1%} & \\\\\n")
        f.write(f"Task 1 (Macro F1) & — & — & {macro_f1:.1%} & \\\\\n")
        if overall_scores:
            f.write(f"Task 2 (Avg Retrieval Score) & \\multicolumn{{3}}{{c}}{{{os_stats['mean']:.3f} ± {os_stats['std']:.3f}}} & \\\\\n")
        f.write(f"Latency & \\multicolumn{{3}}{{c}}{{{timing['avg_total_ms']:.0f} ms/image}} & \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
        
        # Nutrition profile table
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Average Nutritional Profile per Category (USDA Top-50 Matches)}\n")
        f.write("\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write("Category & Kcal & Protein (g) & Carbs (g) & Fat (g) \\\\\n\\hline\n")
        for cat in sorted(VALID_CATEGORIES):
            profile = task2_metrics["category_nutrition_profiles"][cat]
            nuts = profile.get("nutrition_range", {})
            kcal = nuts.get("energy_kcal", {}).get("mean", "—")
            prot = nuts.get("protein_g", {}).get("mean", "—")
            carb = nuts.get("carbohydrate_g", {}).get("mean", "—")
            fat = nuts.get("total_fat_g", {}).get("mean", "—")
            # Shorten category name for LaTeX
            short = cat.replace(" and ", " \\& ").replace(" - ", "—")
            f.write(f"{short} & {kcal} & {prot} & {carb} & {fat} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    
    print(f"LaTeX tables saved to {latex_path}")


if __name__ == "__main__":
    main()
