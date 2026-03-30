#!/usr/bin/env python3
"""
evaluate_pipeline_ensemble.py
=============================
Full end-to-end pipeline evaluation using ENSEMBLE (v11 + Contrastive union)
as the Task 1 classifier, then USDA matching for Task 2.

Same output format as evaluate_pipeline_full.py but with better Task 1.

Usage:
  python evaluate_pipeline_ensemble.py \
    --jsonl ./florence2_data/test_v5.jsonl \
    --data-dir . \
    --base-model microsoft/Florence-2-large-ft \
    --v11-checkpoint ./checkpoints_v11/best_model \
    --contrastive-checkpoint ./checkpoints_contrastive/best_model.pt \
    --usda-dir ./usda_data \
    --output ./eval_pipeline_ensemble.json \
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel

from usda_matcher import USDAMatcher


# ── Categories ─────────────────────────────────────────────────────────────────

CATEGORIES = [
    "Baby Food", "Beans and Legumes - Canned or Dried", "Bread and Bakery Products",
    "Canned Tomato Products", "Carbohydrate Meal", "Condiments and Sauces",
    "Dairy and Dairy Alternatives", "Desserts and Sweets", "Drinks", "Fresh Fruit",
    "Fruits - Canned or Processed", "Granola Products", "Meat and Poultry - Canned",
    "Meat and Poultry - Fresh", "Nut Butters and Nuts", "Ready Meals",
    "Savory Snacks and Crackers", "Seafood - Canned", "Soup",
    "Vegetables - Canned", "Vegetables - Fresh",
]

CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
NUM_CLASSES = len(CATEGORIES)
VALID_CATEGORIES = set(CATEGORIES)
CANONICAL_MAP = {c.lower(): c for c in CATEGORIES}
TASK_PROMPT = "<OD>"


def normalize_category(name):
    return CANONICAL_MAP.get(name.lower().strip(), name)


# ── Contrastive Model Architecture ─────────────────────────────────────────────

class ContrastiveClassifier(nn.Module):
    def __init__(self, florence_model, feature_dim, proj_dim=128, num_classes=21):
        super().__init__()
        self.florence = florence_model
        for param in self.florence.parameters():
            param.requires_grad = False
        self.feature_dim = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def extract_features(self, pixel_values, input_ids):
        with torch.no_grad():
            dummy_decoder_ids = torch.zeros(
                (pixel_values.shape[0], 1), dtype=torch.long, device=pixel_values.device
            )
            outputs = self.florence(
                input_ids=input_ids, pixel_values=pixel_values,
                decoder_input_ids=dummy_decoder_ids,
                output_hidden_states=True, return_dict=True,
            )
            if hasattr(outputs, 'encoder_last_hidden_state') and outputs.encoder_last_hidden_state is not None:
                encoder_hidden = outputs.encoder_last_hidden_state
            elif hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
                encoder_hidden = outputs.encoder_hidden_states[-1]
            else:
                raise RuntimeError("Cannot find encoder hidden states")
            features = encoder_hidden.mean(dim=1)
        return features

    def forward(self, pixel_values, input_ids):
        features = self.extract_features(pixel_values, input_ids)
        proj = F.normalize(self.projection(features), dim=1)
        logits = self.classifier(features)
        return proj, logits


# ── v11 Parser ─────────────────────────────────────────────────────────────────

def parse_v11_prediction(text):
    """Parse v11 generative output -> set of categories."""
    text = text.strip()
    categories = set()

    for candidate in [text, text.replace("'", '"')]:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict) and "items" in result:
                for item in result["items"]:
                    name = normalize_category(item.get("name", ""))
                    if name in VALID_CATEGORIES:
                        categories.add(name)
                return categories
        except (json.JSONDecodeError, TypeError):
            pass

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict) and "items" in result:
                for item in result["items"]:
                    name = normalize_category(item.get("name", ""))
                    if name in VALID_CATEGORIES:
                        categories.add(name)
                return categories
        except (json.JSONDecodeError, TypeError):
            pass

    if start >= 0:
        fragment = text[start:]
        for suffix in ['"}]}', '"}]}}', ']}}', ']}', '}]}', '"}}', '"}', '}',
                       ', "confidence": "high"}]}']:
            try:
                result = json.loads(fragment + suffix)
                if isinstance(result, dict) and "items" in result:
                    for item in result["items"]:
                        if isinstance(item, dict) and "name" in item:
                            name = normalize_category(item["name"])
                            if name in VALID_CATEGORIES:
                                categories.add(name)
                    if categories:
                        return categories
            except (json.JSONDecodeError, TypeError):
                pass

    for cat in VALID_CATEGORIES:
        if cat.lower() in text.lower():
            categories.add(cat)
    return categories


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ensemble pipeline evaluation")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--v11-checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--contrastive-checkpoint", type=str, default="./checkpoints_contrastive/best_model.pt")
    parser.add_argument("--usda-dir", type=str, default="./usda_data")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default="./eval_pipeline_ensemble.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Device: {device}")

    # ── Load v11 generative model ──────────────────────────────────────
    print(f"\nLoading v11: {args.v11_checkpoint}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    v11_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32,
        attn_implementation="eager",
    )
    v11_model = PeftModel.from_pretrained(v11_model, args.v11_checkpoint)
    v11_model = v11_model.merge_and_unload().to(device).eval()
    print("  v11 loaded")

    # ── Load contrastive model ─────────────────────────────────────────
    print(f"Loading contrastive: {args.contrastive_checkpoint}")
    florence_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=torch.float32, attn_implementation="eager",
    )
    florence_base.eval()

    config = florence_base.config
    feature_dim = getattr(config, 'd_model', None) or getattr(config, 'hidden_size', 1024)

    con_model = ContrastiveClassifier(
        florence_model=florence_base, feature_dim=feature_dim,
        proj_dim=128, num_classes=NUM_CLASSES,
    ).to(device)

    ckpt = torch.load(args.contrastive_checkpoint, map_location=device, weights_only=False)
    current_state = con_model.state_dict()
    saved_state = ckpt["model_state_dict"]
    current_state.update(saved_state)
    con_model.load_state_dict(current_state)
    con_model.eval()
    print("  Contrastive loaded")

    # ── Load USDA matcher ──────────────────────────────────────────────
    matcher = USDAMatcher(usda_dir=args.usda_dir)
    print("  USDA matcher loaded\n")

    # ── Load test data ─────────────────────────────────────────────────
    samples = []
    with open(args.jsonl, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  Evaluating {len(samples)} samples (ensemble union)...\n")

    # ── Pre-compute category nutrition profiles ────────────────────────
    print("  Building per-category nutrition profiles...")
    category_nutrition = {}
    for cat in sorted(VALID_CATEGORIES):
        summary = matcher.get_category_nutrition_summary(cat, top_k=50)
        category_nutrition[cat] = summary
    print("  Done.\n")

    # ── Tracking ───────────────────────────────────────────────────────
    results = []
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()
    class_support = Counter()
    exact_match = 0

    # Also track v11-only and contrastive-only for comparison
    v11_all_targets = []
    v11_all_preds = []
    con_all_targets = []
    con_all_preds = []

    usda_scores_per_cat = defaultdict(list)
    correct_nutrition = []
    wrong_nutrition = []
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

        # ── Task 1A: v11 prediction ───────────────────────────────────
        t1 = time.time()

        inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    gen_ids = v11_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512, num_beams=3, early_stopping=True,
                    )
            else:
                gen_ids = v11_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512, num_beams=3, early_stopping=True,
                )
        pred_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        v11_preds = parse_v11_prediction(pred_text)

        # ── Task 1B: contrastive prediction ───────────────────────────
        con_inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _, logits = con_model(con_inputs["pixel_values"], con_inputs["input_ids"])
            else:
                _, logits = con_model(con_inputs["pixel_values"], con_inputs["input_ids"])
        probs = torch.sigmoid(logits).cpu().squeeze()
        con_preds = {CATEGORIES[j] for j in range(NUM_CLASSES) if probs[j] >= args.threshold}

        # ── Ensemble: union ───────────────────────────────────────────
        pred_cats = v11_preds | con_preds

        t2 = time.time()
        classify_time = t2 - t1
        classify_times.append(classify_time)

        # Track individual model preds for comparison
        v11_all_targets.append(target_cats)
        v11_all_preds.append(v11_preds)
        con_all_targets.append(target_cats)
        con_all_preds.append(con_preds)

        # Task 1 metrics (ensemble)
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
        for cat in pred_cats:
            matches = matcher.match_pantry_prediction(cat, top_k=args.top_k)
            usda_results[cat] = matches
            if matches:
                usda_scores_per_cat[cat].append(matches[0].get("score", 0))

        t4 = time.time()
        match_time = t4 - t3
        match_times.append(match_time)
        total_times.append(t4 - t1)

        # ── Error propagation ─────────────────────────────────────────
        is_correct = (target_cats == pred_cats)
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
        results.append({
            "image": img_rel,
            "target_categories": sorted(target_cats),
            "v11_predictions": sorted(v11_preds),
            "contrastive_predictions": sorted(con_preds),
            "ensemble_predictions": sorted(pred_cats),
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
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0_all
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{len(samples)} ({rate:.1f} img/s, ETA: {(len(samples)-i-1)/rate:.0f}s)")

    total_elapsed = time.time() - t0_all
    n = len(results)

    # ══════════════════════════════════════════════════════════════════
    #  COMPUTE METRICS
    # ══════════════════════════════════════════════════════════════════

    # ── Task 1 (ensemble) ──────────────────────────────────────────────
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    micro_p = total_tp / max(total_tp + total_fp, 1)
    micro_r = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)

    per_class = {}
    all_classes = sorted(set(list(class_tp) + list(class_fp) + list(class_fn)))
    for cls in all_classes:
        tp, fp, fn = class_tp[cls], class_fp[cls], class_fn[cls]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        per_class[cls] = {"precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(f1, 4), "support": class_support[cls]}

    macro_f1 = sum(m["f1"] for m in per_class.values()) / max(len(per_class), 1)

    # ── v11-only metrics for comparison ────────────────────────────────
    def compute_simple_metrics(targets, preds):
        tp_total = fp_total = fn_total = 0
        for t, p in zip(targets, preds):
            for c in t | p:
                if c in t and c in p:
                    tp_total += 1
                elif c in p:
                    fp_total += 1
                else:
                    fn_total += 1
        mp = tp_total / max(tp_total + fp_total, 1)
        mr = tp_total / max(tp_total + fn_total, 1)
        mf1 = 2 * mp * mr / max(mp + mr, 1e-8)
        return {"micro_p": round(mp, 4), "micro_r": round(mr, 4), "micro_f1": round(mf1, 4)}

    v11_metrics = compute_simple_metrics(v11_all_targets, v11_all_preds)
    con_metrics = compute_simple_metrics(con_all_targets, con_all_preds)

    # ── Task 2 ─────────────────────────────────────────────────────────
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

    # ── Error propagation ──────────────────────────────────────────────
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
        c_avg = avg_nutrition(correct_nutrition, key)
        w_avg = avg_nutrition(wrong_nutrition, key)
        error_analysis[f"{label}_correct_avg"] = c_avg
        error_analysis[f"{label}_wrong_avg"] = w_avg
        error_analysis[f"{label}_delta"] = round(w_avg - c_avg, 1)

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
    print(f"  END-TO-END PIPELINE (ENSEMBLE UNION) EVALUATION")
    print(f"{'='*70}")
    print(f"  Samples: {n}")
    print(f"  Total time: {total_elapsed:.1f}s ({n/total_elapsed:.1f} img/s)")

    # Comparison table
    print(f"\n{'='*70}")
    print(f"  TASK 1: CLASSIFICATION COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Method':<30} {'Prec':>8} {'Recall':>8} {'F1':>9}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*9}")
    print(f"  {'v11 only':<30} {v11_metrics['micro_p']:>7.1%} {v11_metrics['micro_r']:>7.1%} {v11_metrics['micro_f1']:>8.1%}")
    print(f"  {'Contrastive only':<30} {con_metrics['micro_p']:>7.1%} {con_metrics['micro_r']:>7.1%} {con_metrics['micro_f1']:>8.1%}")
    print(f"  {'Ensemble Union (USED)':<30} {micro_p:>7.1%} {micro_r:>7.1%} {micro_f1:>8.1%}")

    # Per-class
    print(f"\n  Ensemble Union — Per Class:")
    print(f"  Exact Match: {exact_match}/{n} ({exact_match/max(n,1):.1%})")
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
        print(f"  Overall cosine similarity: {np.mean(overall_scores):.3f} +/- {np.std(overall_scores):.3f} "
              f"(range: {np.min(overall_scores):.3f} - {np.max(overall_scores):.3f})")

    print(f"\n  {'Category':<40} {'Mean Score':>10} {'Queries':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*8}")
    for cat, stats in sorted(usda_retrieval_stats.items()):
        print(f"  {cat:<40} {stats['mean_score']:>9.3f} {stats['n_queries']:>8}")

    # Nutrition profiles
    print(f"\n{'='*70}")
    print(f"  NUTRITION PROFILES BY CATEGORY (top-50 USDA matches)")
    print(f"{'='*70}")
    print(f"  {'Category':<35} {'Kcal':>8} {'Protein':>8} {'Carbs':>8} {'Fat':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for cat in sorted(VALID_CATEGORIES):
        profile = category_nutrition.get(cat, {})
        nuts = profile.get("nutrition_summary", {})
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
    print(f"  ERROR PROPAGATION: Classification -> Nutrition")
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
    print(f"  Classification (ensemble): {timing['avg_classify_ms']:.0f} ms/image")
    print(f"  USDA matching:             {timing['avg_match_ms']:.0f} ms/image")
    print(f"  Total:                     {timing['avg_total_ms']:.0f} ms/image (p95: {timing['p95_total_ms']:.0f} ms)")
    print(f"  Throughput:                {timing['images_per_second']:.1f} img/s")

    # Sample outputs
    print(f"\n{'='*70}")
    print(f"  SAMPLE OUTPUTS (first 5)")
    print(f"{'='*70}")
    for r in results[:5]:
        print(f"\n  Image: {os.path.basename(r['image'])}")
        print(f"  Target:      {r['target_categories']}")
        print(f"  v11:         {r['v11_predictions']}")
        print(f"  Contrastive: {r['contrastive_predictions']}")
        print(f"  Ensemble:    {r['ensemble_predictions']}")
        match = "CORRECT" if r['classification_correct'] else "WRONG"
        print(f"  Task 1: {match}")
        for cat, matches in r.get("usda_top_matches", {}).items():
            if matches:
                top = matches[0]
                nuts = top.get("nutrients", {})
                kcal = nuts.get("energy_kcal", "?")
                print(f"  -> {cat}: {top['description'][:60]} [{top['score']:.3f}] ({kcal} kcal)")

    # ── Save ───────────────────────────────────────────────────────────
    output = {
        "pipeline": "ensemble_union_v11_contrastive_usda",
        "n_samples": n,
        "task1_comparison": {
            "v11_only": v11_metrics,
            "contrastive_only": con_metrics,
            "ensemble_union": {
                "micro_p": round(micro_p, 4), "micro_r": round(micro_r, 4),
                "micro_f1": round(micro_f1, 4), "macro_f1": round(macro_f1, 4),
                "exact_match": round(exact_match / max(n, 1), 4),
                "per_class": per_class,
            },
        },
        "task2_retrieval": {
            "overall": {
                "mean": round(float(np.mean(overall_scores)), 4) if overall_scores else 0,
                "std": round(float(np.std(overall_scores)), 4) if overall_scores else 0,
            },
            "per_category": usda_retrieval_stats,
        },
        "error_propagation": error_analysis,
        "timing": timing,
        "category_nutrition_profiles": {
            cat: {
                "nutrition_range": {
                    k: {"mean": v["mean"], "min": v["min"], "max": v["max"]}
                    for k, v in summary.get("nutrition_summary", {}).items()
                    if k in ["energy_kcal", "protein_g", "carbohydrate_g", "total_fat_g"]
                }
            }
            for cat, summary in category_nutrition.items()
        },
        "per_image_results": results[:100],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")

    # ── LaTeX tables ───────────────────────────────────────────────────
    latex_path = args.output.replace(".json", "_table.tex")
    with open(latex_path, "w") as f:
        f.write("% Auto-generated ensemble pipeline evaluation tables\n\n")

        # Comparison table
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Task 1 Classification: Single Models vs.\\ Ensemble}\n")
        f.write("\\label{tab:ensemble_comparison}\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Method & Precision & Recall & Micro F1 \\\\\n\\midrule\n")
        f.write(f"Florence-2 v11 (generative) & {v11_metrics['micro_p']:.1%} & {v11_metrics['micro_r']:.1%} & {v11_metrics['micro_f1']:.1%} \\\\\n")
        f.write(f"Contrastive (discriminative) & {con_metrics['micro_p']:.1%} & {con_metrics['micro_r']:.1%} & {con_metrics['micro_f1']:.1%} \\\\\n")
        f.write(f"\\textbf{{Ensemble Union}} & \\textbf{{{micro_p:.1%}}} & \\textbf{{{micro_r:.1%}}} & \\textbf{{{micro_f1:.1%}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Per-class table
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Ensemble Union Per-Class F1 Scores}\n")
        f.write("\\label{tab:ensemble_perclass}\n")
        f.write("\\small\n\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Category & Precision & Recall & F1 & Support \\\\\n\\midrule\n")
        for cls in sorted(per_class.keys()):
            m = per_class[cls]
            short = cls.replace(" and ", " \\& ").replace(" - ", "---")
            f.write(f"{short} & {m['precision']:.1%} & {m['recall']:.1%} & {m['f1']:.1%} & {m['support']} \\\\\n")
        f.write(f"\\midrule\n")
        f.write(f"Micro Average & {micro_p:.1%} & {micro_r:.1%} & {micro_f1:.1%} & {n} \\\\\n")
        f.write(f"Macro Average & --- & --- & {macro_f1:.1%} & \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Error propagation table
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Error Propagation: Nutrition Estimate When Classification Is Wrong}\n")
        f.write("\\label{tab:error_propagation}\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("Nutrient & Correct Avg & Wrong Avg & Delta \\\\\n\\midrule\n")
        for key, unit in [("Calories", "kcal"), ("Protein", "g"), ("Carbs", "g"), ("Fat", "g")]:
            c = error_analysis[f"{key}_correct_avg"]
            w = error_analysis[f"{key}_wrong_avg"]
            d = error_analysis[f"{key}_delta"]
            sign = "+" if d >= 0 else ""
            f.write(f"{key} & {c:.1f} {unit} & {w:.1f} {unit} & {sign}{d:.1f} {unit} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"LaTeX tables saved to {latex_path}")


if __name__ == "__main__":
    main()
