#!/usr/bin/env python3
"""
evaluate_pipeline_cot.py
========================
Enhanced end-to-end pipeline with VLM Chain-of-Thought for Task 2 USDA matching.

Improvement over evaluate_pipeline_full.py:
  - Task 1: Same Florence-2 v11 classifier (or any checkpoint)
  - Task 2: INSTEAD of just category→USDA text search, uses Qwen2.5-VL-7B to:
    1. Read product labels/packaging text from the image
    2. Identify specific product details (brand, flavor, type)
    3. Generate a detailed USDA search query
    4. Rerank USDA candidates based on VLM reasoning

This should improve USDA matching precision significantly:
  Old: "Soup" → generic soup entries
  New: "Campbell's Chicken Noodle Soup, condensed, 10.75 oz can" → exact match

Usage:
  python evaluate_pipeline_cot.py \
    --jsonl ./florence2_data/test_v5.jsonl \
    --data-dir . \
    --cls-model microsoft/Florence-2-large-ft \
    --cls-checkpoint ./checkpoints_v11/best_model \
    --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
    --usda-dir ./usda_data \
    --output ./eval_pipeline_cot.json \
    --bf16
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
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


# ── VLM Product Identification Prompt ──────────────────────────────────────────

PRODUCT_ID_PROMPT = """You are analyzing a food pantry image. The classifier has identified these food categories in the image: {categories}

For EACH identified category, look at the image carefully and:
1. Read any visible text on packaging (brand name, product name, flavor, size)
2. Identify the specific product as precisely as possible
3. Generate a USDA search query that would find this exact product

Respond in JSON format:
```json
{{
  "products": [
    {{
      "category": "<pantry category>",
      "brand": "<brand name if visible, else 'unknown'>",
      "product_name": "<specific product name>",
      "details": "<flavor, variety, size, form (canned/fresh/frozen)>",
      "usda_query": "<optimized search query for USDA database>"
    }}
  ]
}}
```

Be specific! "Del Monte Cut Green Beans, 14.5 oz can" is much better than just "canned vegetables".
If you can't read the label clearly, make your best guess based on visual appearance."""


RERANK_PROMPT = """You are matching a food item to USDA database entries.

The food item from the pantry image is:
- Category: {category}
- Product: {product_description}

Here are the top USDA candidate matches:
{candidates}

Which candidate is the BEST match for this specific product? Consider:
1. Product type match (canned vs fresh vs frozen)
2. Brand match (if known)
3. Specific variety/flavor match
4. Serving size reasonableness

Respond in JSON:
```json
{{
  "best_match_index": <0-based index of best candidate>,
  "confidence": "<high/medium/low>",
  "reasoning": "<brief explanation>"
}}
```"""


# ── Parse Helpers ──────────────────────────────────────────────────────────────

def parse_prediction(text):
    """Parse Florence-2 classifier output."""
    text = text.strip()
    if not text:
        return None
    for candidate in [text, text.replace("'", '"')]:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict) and "items" in result:
                return result
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end])
            if isinstance(result, dict) and "items" in result:
                return result
        except json.JSONDecodeError:
            pass
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


def parse_json_from_text(text):
    """Extract JSON from VLM output (may have markdown code blocks)."""
    text = text.strip()
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline with CoT-enhanced USDA matching")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--cls-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--cls-checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--usda-dir", type=str, default="./usda_data")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default="./eval_pipeline_cot.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-rerank", action="store_true",
                        help="Skip VLM reranking step (only do product identification)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None

    # ── Load Florence-2 classifier ─────────────────────────────────────
    print(f"Loading classifier: {args.cls_checkpoint}")
    cls_processor = AutoProcessor.from_pretrained(args.cls_model, trust_remote_code=True)
    cls_model = AutoModelForCausalLM.from_pretrained(
        args.cls_model, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32,
        attn_implementation="eager",
    )
    cls_model = PeftModel.from_pretrained(cls_model, args.cls_checkpoint)
    cls_model = cls_model.merge_and_unload().to(device).eval()
    print("  Classifier loaded ✓")

    # ── Load VLM (Qwen2.5-VL) ─────────────────────────────────────────
    print(f"\nLoading VLM: {args.vlm_model}")
    vlm_processor = AutoProcessor.from_pretrained(
        args.vlm_model, trust_remote_code=True,
        min_pixels=128*28*28,    # ~100K
        max_pixels=256*28*28,    # ~200K (keep VRAM reasonable)
    )

    # Try flash_attention_2 first, fallback to eager
    try:
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            args.vlm_model,
            torch_dtype=amp_dtype or torch.float32,
            attn_implementation="flash_attention_2",
        ).to(device).eval()
        print("  VLM loaded with flash_attention_2 ✓")
    except Exception:
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            args.vlm_model,
            torch_dtype=amp_dtype or torch.float32,
            attn_implementation="eager",
        ).to(device).eval()
        print("  VLM loaded with eager ✓")

    # ── Load USDA matcher ──────────────────────────────────────────────
    matcher = USDAMatcher(usda_dir=args.usda_dir)
    print("  USDA matcher loaded ✓")

    # ── Load test data ─────────────────────────────────────────────────
    samples = []
    with open(args.jsonl, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"\n  Evaluating {len(samples)} samples...\n")

    # ── Process all images ─────────────────────────────────────────────
    results = []
    class_tp, class_fp, class_fn, class_support = Counter(), Counter(), Counter(), Counter()
    exact_match = 0

    # Comparison tracking
    baseline_scores = []  # category-only matching scores
    cot_scores = []       # CoT-enhanced matching scores
    rerank_improvements = 0
    total_reranks = 0

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

        # ── Task 1: Classify with Florence-2 ──────────────────────────
        inputs = cls_processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    gen_ids = cls_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512, num_beams=3, early_stopping=True,
                    )
            else:
                gen_ids = cls_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512, num_beams=3, early_stopping=True,
                )
        pred_text = cls_processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        pred_parsed = parse_prediction(pred_text)
        pred_cats = extract_categories(pred_parsed)

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

        # ── Task 2a: Baseline USDA match (category only) ─────────────
        baseline_usda = {}
        for cat in pred_cats:
            matches = matcher.match_pantry_prediction(cat, top_k=args.top_k)
            baseline_usda[cat] = matches
            if matches:
                baseline_scores.append(matches[0].get("score", 0))

        # ── Task 2b: VLM Product Identification ──────────────────────
        cot_usda = {}
        vlm_products = {}

        if pred_cats:
            cats_str = ", ".join(sorted(pred_cats))
            prompt_text = PRODUCT_ID_PROMPT.format(categories=cats_str)

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ]}
            ]

            text_input = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vlm_inputs = vlm_processor(
                text=[text_input], images=[image],
                padding=True, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                if amp_dtype:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        vlm_ids = vlm_model.generate(
                            **vlm_inputs,
                            max_new_tokens=1024,
                            temperature=0.3,
                            do_sample=True,
                        )
                else:
                    vlm_ids = vlm_model.generate(
                        **vlm_inputs,
                        max_new_tokens=1024,
                        temperature=0.3,
                        do_sample=True,
                    )

            # Trim input tokens from output
            generated_ids = vlm_ids[:, vlm_inputs.input_ids.shape[1]:]
            vlm_text = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Parse VLM product identification
            vlm_parsed = parse_json_from_text(vlm_text)

            if vlm_parsed and "products" in vlm_parsed:
                for prod in vlm_parsed["products"]:
                    cat = normalize_category(prod.get("category", ""))
                    if cat not in VALID_CATEGORIES:
                        continue

                    usda_query = prod.get("usda_query", cat)
                    vlm_products[cat] = prod

                    # Search USDA with the VLM-generated specific query
                    cot_matches = matcher.search_hybrid(
                        usda_query, pantry_category=cat, top_k=args.top_k
                    )
                    cot_usda[cat] = cot_matches

                    if cot_matches:
                        cot_scores.append(cot_matches[0].get("score", 0))

            # For categories not identified by VLM, fall back to baseline
            for cat in pred_cats:
                if cat not in cot_usda:
                    cot_usda[cat] = baseline_usda.get(cat, [])

        # ── Task 2c: VLM Reranking (optional) ────────────────────────
        reranked_usda = dict(cot_usda)  # start with CoT results

        if not args.skip_rerank and vlm_products:
            for cat, matches in cot_usda.items():
                if cat not in vlm_products or len(matches) < 2:
                    continue

                prod = vlm_products[cat]
                prod_desc = f"{prod.get('brand', 'unknown')} {prod.get('product_name', cat)} {prod.get('details', '')}".strip()

                candidates_text = ""
                for j, m in enumerate(matches[:5]):
                    nutrients = m.get("nutrients", {})
                    kcal = nutrients.get("energy_kcal", "?")
                    candidates_text += f"  {j}. {m.get('description', '?')} — {m.get('brand_owner', 'N/A')} ({kcal} kcal)\n"

                rerank_prompt = RERANK_PROMPT.format(
                    category=cat,
                    product_description=prod_desc,
                    candidates=candidates_text,
                )

                messages = [{"role": "user", "content": [{"type": "text", "text": rerank_prompt}]}]
                text_input = vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                rr_inputs = vlm_processor(
                    text=[text_input], padding=True, return_tensors="pt"
                ).to(device)

                with torch.no_grad():
                    if amp_dtype:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            rr_ids = vlm_model.generate(
                                **rr_inputs, max_new_tokens=256,
                                temperature=0.1, do_sample=True,
                            )
                    else:
                        rr_ids = vlm_model.generate(
                            **rr_inputs, max_new_tokens=256,
                            temperature=0.1, do_sample=True,
                        )

                rr_generated = rr_ids[:, rr_inputs.input_ids.shape[1]:]
                rr_text = vlm_processor.batch_decode(rr_generated, skip_special_tokens=True)[0]
                rr_parsed = parse_json_from_text(rr_text)

                if rr_parsed and "best_match_index" in rr_parsed:
                    best_idx = rr_parsed["best_match_index"]
                    total_reranks += 1
                    if isinstance(best_idx, int) and 0 <= best_idx < len(matches) and best_idx != 0:
                        # Rerank: move the VLM-selected candidate to top
                        reranked = list(matches)
                        selected = reranked.pop(best_idx)
                        reranked.insert(0, selected)
                        reranked_usda[cat] = reranked
                        rerank_improvements += 1

        # ── Store result ──────────────────────────────────────────────
        result_entry = {
            "image": img_rel,
            "target_categories": sorted(target_cats),
            "predicted_categories": sorted(pred_cats),
            "classification_correct": target_cats == pred_cats,
            "vlm_products": vlm_products,
            "baseline_top_match": {
                cat: {"description": ms[0]["description"], "score": round(ms[0].get("score", 0), 4),
                       "nutrients": ms[0].get("nutrients", {})}
                for cat, ms in baseline_usda.items() if ms
            },
            "cot_top_match": {
                cat: {"description": ms[0]["description"], "score": round(ms[0].get("score", 0), 4),
                       "nutrients": ms[0].get("nutrients", {})}
                for cat, ms in reranked_usda.items() if ms
            },
        }
        results.append(result_entry)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0_all
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(samples)}] {rate:.2f} img/s | "
                  f"ETA: {(len(samples)-i-1)/rate:.0f}s | "
                  f"Baseline avg: {np.mean(baseline_scores):.4f} | "
                  f"CoT avg: {np.mean(cot_scores):.4f}" if cot_scores else "")

    total_elapsed = time.time() - t0_all
    n = len(results)

    # ══════════════════════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════════════════════
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    micro_p = total_tp / max(total_tp + total_fp, 1)
    micro_r = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)

    print(f"\n{'='*70}")
    print(f"  END-TO-END PIPELINE WITH CoT EVALUATION")
    print(f"{'='*70}")
    print(f"  Samples: {n}")
    print(f"  Time: {total_elapsed:.1f}s ({n/total_elapsed:.2f} img/s)")

    print(f"\n{'='*70}")
    print(f"  TASK 1: CLASSIFICATION (unchanged)")
    print(f"{'='*70}")
    print(f"  Micro F1: {micro_f1:.1%} (P={micro_p:.1%}, R={micro_r:.1%})")
    print(f"  Exact Match: {exact_match/max(n,1):.1%}")

    print(f"\n{'='*70}")
    print(f"  TASK 2: USDA MATCHING COMPARISON")
    print(f"{'='*70}")
    if baseline_scores:
        print(f"  Baseline (category-only):")
        print(f"    Mean retrieval score: {np.mean(baseline_scores):.4f}")
        print(f"    Min: {np.min(baseline_scores):.4f} | Max: {np.max(baseline_scores):.4f}")
    if cot_scores:
        print(f"  CoT-enhanced (VLM product ID):")
        print(f"    Mean retrieval score: {np.mean(cot_scores):.4f}")
        print(f"    Min: {np.min(cot_scores):.4f} | Max: {np.max(cot_scores):.4f}")
        improvement = np.mean(cot_scores) - np.mean(baseline_scores)
        print(f"    Improvement: {improvement:+.4f} ({improvement/max(np.mean(baseline_scores), 0.001)*100:+.1f}%)")

    if total_reranks > 0:
        print(f"\n  Reranking:")
        print(f"    Total rerank attempts: {total_reranks}")
        print(f"    Reranked (changed top-1): {rerank_improvements}")
        print(f"    Rerank rate: {rerank_improvements/total_reranks:.1%}")

    # ── Sample comparisons ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SAMPLE COMPARISONS (first 5)")
    print(f"{'='*70}")
    for r in results[:5]:
        print(f"\n  📷 {os.path.basename(r['image'])}")
        print(f"  Target: {r['target_categories']}")
        print(f"  Predicted: {r['predicted_categories']}")
        for cat in r["predicted_categories"]:
            if cat in r.get("vlm_products", {}):
                prod = r["vlm_products"][cat]
                print(f"    VLM identified: {prod.get('brand', '?')} {prod.get('product_name', '?')} ({prod.get('details', '')})")
            bl = r.get("baseline_top_match", {}).get(cat, {})
            ct = r.get("cot_top_match", {}).get(cat, {})
            if bl:
                print(f"    Baseline USDA: {bl.get('description', '?')} (score={bl.get('score', 0):.4f})")
            if ct:
                print(f"    CoT USDA:      {ct.get('description', '?')} (score={ct.get('score', 0):.4f})")

    # ── Save ───────────────────────────────────────────────────────────
    output = {
        "config": vars(args),
        "task1_metrics": {
            "micro_f1": round(micro_f1, 4),
            "micro_p": round(micro_p, 4),
            "micro_r": round(micro_r, 4),
            "exact_match": round(exact_match / max(n, 1), 4),
        },
        "task2_comparison": {
            "baseline_mean_score": round(float(np.mean(baseline_scores)), 4) if baseline_scores else 0,
            "cot_mean_score": round(float(np.mean(cot_scores)), 4) if cot_scores else 0,
            "improvement": round(float(np.mean(cot_scores) - np.mean(baseline_scores)), 4) if cot_scores and baseline_scores else 0,
            "rerank_attempts": total_reranks,
            "rerank_changes": rerank_improvements,
        },
        "timing": {
            "total_s": round(total_elapsed, 1),
            "per_image_s": round(total_elapsed / max(n, 1), 2),
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
