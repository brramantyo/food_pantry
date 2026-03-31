#!/usr/bin/env python3
"""
evaluate_pipeline_full_cot.py
=============================
Best-of-everything full pipeline:
  Task 1: Ensemble (v11 generative ∪ contrastive discriminative) = 80.3% F1
  Task 2: CoT-enhanced USDA matching (VLM reads labels → specific query → rerank)

This combines:
  - evaluate_pipeline_ensemble.py (ensemble Task 1)
  - evaluate_pipeline_cot.py (CoT Task 2)

Usage:
  python evaluate_pipeline_full_cot.py \
    --jsonl ./florence2_data/test_v5.jsonl \
    --data-dir . \
    --base-model microsoft/Florence-2-large-ft \
    --v11-checkpoint ./checkpoints_v11/best_model \
    --contrastive-checkpoint ./checkpoints_contrastive/best_model.pt \
    --vlm-model Qwen/Qwen2.5-VL-7B-Instruct \
    --usda-dir ./usda_data \
    --output ./eval_pipeline_full_cot.json \
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
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


# ── Parsers ────────────────────────────────────────────────────────────────────

def parse_v11_prediction(text):
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


def parse_json_from_text(text):
    text = text.strip()
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


# ── VLM Prompts ────────────────────────────────────────────────────────────────

PRODUCT_ID_PROMPT = """You are analyzing a food pantry image. The classifier has identified these food categories: {categories}

For EACH category, look at the image and:
1. Read any visible text on packaging (brand, product name, flavor, size)
2. Identify the specific product as precisely as possible
3. Generate a USDA search query for this exact product

Respond in JSON:
```json
{{
  "products": [
    {{
      "category": "<pantry category>",
      "brand": "<brand if visible, else 'unknown'>",
      "product_name": "<specific product name>",
      "details": "<flavor, variety, size, form>",
      "usda_query": "<optimized USDA search query>"
    }}
  ]
}}
```

Be specific! "Del Monte Cut Green Beans, 14.5 oz can" >> "canned vegetables"."""


RERANK_PROMPT = """Match this food item to the best USDA entry.

Item: {product_description}
Category: {category}

Candidates:
{candidates}

Which is the BEST match? Respond in JSON:
```json
{{"best_match_index": <0-based>, "confidence": "<high/medium/low>", "reasoning": "<brief>"}}
```"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full pipeline: Ensemble Task 1 + CoT Task 2")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--v11-checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--contrastive-checkpoint", type=str, default="./checkpoints_contrastive/best_model.pt")
    parser.add_argument("--vlm-model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--usda-dir", type=str, default="./usda_data")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=str, default="./eval_pipeline_full_cot.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Contrastive classifier threshold")
    parser.add_argument("--skip-rerank", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Device: {device}")

    # ── Load v11 ───────────────────────────────────────────────────────
    print(f"\nLoading v11: {args.v11_checkpoint}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    v11_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32, attn_implementation="eager",
    )
    v11_model = PeftModel.from_pretrained(v11_model, args.v11_checkpoint)
    v11_model = v11_model.merge_and_unload().to(device).eval()
    print("  v11 loaded ✓")

    # ── Load contrastive ───────────────────────────────────────────────
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
    current_state.update(ckpt["model_state_dict"])
    con_model.load_state_dict(current_state)
    con_model.eval()
    print("  Contrastive loaded ✓")

    # ── Load VLM ───────────────────────────────────────────────────────
    print(f"Loading VLM: {args.vlm_model}")
    vlm_processor = AutoProcessor.from_pretrained(
        args.vlm_model, trust_remote_code=True,
        min_pixels=128*28*28,    # ~100K
        max_pixels=256*28*28,    # ~200K (keep VRAM reasonable)
    )
    try:
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            args.vlm_model, torch_dtype=amp_dtype or torch.float32,
            attn_implementation="flash_attention_2",
        ).to(device).eval()
        print("  VLM loaded (flash_attention_2) ✓")
    except Exception:
        vlm_model = AutoModelForImageTextToText.from_pretrained(
            args.vlm_model, torch_dtype=amp_dtype or torch.float32,
            attn_implementation="eager",
        ).to(device).eval()
        print("  VLM loaded (eager) ✓")

    # ── Load USDA ──────────────────────────────────────────────────────
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
    print(f"\n  Evaluating {len(samples)} samples...")
    print(f"  Pipeline: Ensemble (v11 ∪ contrastive) → CoT USDA matching\n")

    # ── Tracking ───────────────────────────────────────────────────────
    results = []
    class_tp, class_fp, class_fn, class_support = Counter(), Counter(), Counter(), Counter()
    exact_match = 0
    baseline_scores, cot_scores = [], []
    rerank_improvements, total_reranks = 0, 0
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

        # ══ TASK 1: ENSEMBLE CLASSIFICATION ═══════════════════════════

        # v11 prediction
        inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    gen_ids = v11_model.generate(
                        input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                        max_new_tokens=512, num_beams=3, early_stopping=True,
                    )
            else:
                gen_ids = v11_model.generate(
                    input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                    max_new_tokens=512, num_beams=3, early_stopping=True,
                )
        pred_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        v11_preds = parse_v11_prediction(pred_text)

        # Contrastive prediction
        con_inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _, logits = con_model(con_inputs["pixel_values"], con_inputs["input_ids"])
            else:
                _, logits = con_model(con_inputs["pixel_values"], con_inputs["input_ids"])
        probs = torch.sigmoid(logits).cpu().squeeze()
        con_preds = {CATEGORIES[j] for j in range(NUM_CLASSES) if probs[j] >= args.threshold}

        # Union ensemble
        pred_cats = v11_preds | con_preds

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

        # ══ TASK 2a: BASELINE USDA MATCH ═════════════════════════════
        baseline_usda = {}
        for cat in pred_cats:
            matches = matcher.match_pantry_prediction(cat, top_k=args.top_k)
            baseline_usda[cat] = matches
            if matches:
                baseline_scores.append(matches[0].get("score", 0))

        # ══ TASK 2b: CoT PRODUCT IDENTIFICATION ═════════════════════
        cot_usda = {}
        vlm_products = {}

        if pred_cats:
            cats_str = ", ".join(sorted(pred_cats))
            prompt_text = PRODUCT_ID_PROMPT.format(categories=cats_str)

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ]}]

            text_input = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vlm_inputs = vlm_processor(
                text=[text_input], images=[image], padding=True, return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                if amp_dtype:
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        vlm_ids = vlm_model.generate(
                            **vlm_inputs, max_new_tokens=1024,
                            temperature=0.3, do_sample=True,
                        )
                else:
                    vlm_ids = vlm_model.generate(
                        **vlm_inputs, max_new_tokens=1024,
                        temperature=0.3, do_sample=True,
                    )

            generated_ids = vlm_ids[:, vlm_inputs.input_ids.shape[1]:]
            vlm_text = vlm_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            vlm_parsed = parse_json_from_text(vlm_text)

            if vlm_parsed and "products" in vlm_parsed:
                for prod in vlm_parsed["products"]:
                    cat = normalize_category(prod.get("category", ""))
                    if cat not in VALID_CATEGORIES:
                        continue
                    usda_query = prod.get("usda_query", cat)
                    vlm_products[cat] = prod
                    cot_matches = matcher.search_hybrid(
                        usda_query, pantry_category=cat, top_k=args.top_k
                    )
                    cot_usda[cat] = cot_matches
                    if cot_matches:
                        cot_scores.append(cot_matches[0].get("score", 0))

            for cat in pred_cats:
                if cat not in cot_usda:
                    cot_usda[cat] = baseline_usda.get(cat, [])

        # ══ TASK 2c: VLM RERANKING ═══════════════════════════════════
        reranked_usda = dict(cot_usda)

        if not args.skip_rerank and vlm_products:
            for cat, matches in cot_usda.items():
                if cat not in vlm_products or len(matches) < 2:
                    continue
                prod = vlm_products[cat]
                prod_desc = f"{prod.get('brand', 'unknown')} {prod.get('product_name', cat)} {prod.get('details', '')}".strip()

                candidates_text = ""
                for j, m in enumerate(matches[:5]):
                    kcal = m.get("nutrients", {}).get("energy_kcal", "?")
                    candidates_text += f"  {j}. {m.get('description', '?')} — {m.get('brand_owner', 'N/A')} ({kcal} kcal)\n"

                rerank_prompt = RERANK_PROMPT.format(
                    category=cat, product_description=prod_desc, candidates=candidates_text,
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
                        reranked = list(matches)
                        selected = reranked.pop(best_idx)
                        reranked.insert(0, selected)
                        reranked_usda[cat] = reranked
                        rerank_improvements += 1

        # ── Store result ──────────────────────────────────────────────
        results.append({
            "image": img_rel,
            "target_categories": sorted(target_cats),
            "v11_predictions": sorted(v11_preds),
            "contrastive_predictions": sorted(con_preds),
            "ensemble_predictions": sorted(pred_cats),
            "classification_correct": target_cats == pred_cats,
            "vlm_products": vlm_products,
            "baseline_top_match": {
                cat: {"description": ms[0]["description"], "score": round(ms[0].get("score", 0), 4)}
                for cat, ms in baseline_usda.items() if ms
            },
            "cot_top_match": {
                cat: {"description": ms[0]["description"], "score": round(ms[0].get("score", 0), 4)}
                for cat, ms in reranked_usda.items() if ms
            },
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0_all
            rate = (i + 1) / elapsed
            bl_avg = f"{np.mean(baseline_scores):.4f}" if baseline_scores else "N/A"
            ct_avg = f"{np.mean(cot_scores):.4f}" if cot_scores else "N/A"
            print(f"  [{i+1}/{len(samples)}] {rate:.2f} img/s | "
                  f"Baseline: {bl_avg} | CoT: {ct_avg}")

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
    print(f"  FULL PIPELINE: ENSEMBLE + CoT USDA MATCHING")
    print(f"{'='*70}")
    print(f"  Samples: {n} | Time: {total_elapsed:.1f}s ({n/total_elapsed:.2f} img/s)")

    print(f"\n  TASK 1 (Ensemble v11 ∪ Contrastive):")
    print(f"    Micro F1: {micro_f1:.1%} (P={micro_p:.1%}, R={micro_r:.1%})")
    print(f"    Exact Match: {exact_match/max(n,1):.1%}")

    print(f"\n  TASK 2 (USDA Matching):")
    if baseline_scores:
        print(f"    Baseline (category-only):  mean={np.mean(baseline_scores):.4f}")
    if cot_scores:
        print(f"    CoT (VLM product ID):      mean={np.mean(cot_scores):.4f}")
        imp = np.mean(cot_scores) - np.mean(baseline_scores)
        print(f"    Improvement:               {imp:+.4f} ({imp/max(np.mean(baseline_scores),0.001)*100:+.1f}%)")
    if total_reranks > 0:
        print(f"    Reranked: {rerank_improvements}/{total_reranks} ({rerank_improvements/total_reranks:.0%})")

    # Sample comparisons
    print(f"\n{'='*70}")
    print(f"  SAMPLE COMPARISONS (first 5)")
    print(f"{'='*70}")
    for r in results[:5]:
        print(f"\n  📷 {os.path.basename(r['image'])}")
        print(f"     Target: {r['target_categories']}")
        print(f"     Ensemble: {r['ensemble_predictions']}")
        for cat in r["ensemble_predictions"]:
            if cat in r.get("vlm_products", {}):
                prod = r["vlm_products"][cat]
                print(f"       VLM: {prod.get('brand', '?')} {prod.get('product_name', '?')} ({prod.get('details', '')})")
            bl = r.get("baseline_top_match", {}).get(cat, {})
            ct = r.get("cot_top_match", {}).get(cat, {})
            if bl:
                print(f"       Baseline USDA: {bl.get('description', '?')} ({bl.get('score', 0):.4f})")
            if ct:
                print(f"       CoT USDA:      {ct.get('description', '?')} ({ct.get('score', 0):.4f})")

    # ── Save ───────────────────────────────────────────────────────────
    output = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "task1": {
            "method": "ensemble_union (v11 + contrastive)",
            "micro_f1": round(micro_f1, 4),
            "micro_p": round(micro_p, 4),
            "micro_r": round(micro_r, 4),
            "exact_match": round(exact_match / max(n, 1), 4),
        },
        "task2": {
            "baseline_mean_score": round(float(np.mean(baseline_scores)), 4) if baseline_scores else 0,
            "cot_mean_score": round(float(np.mean(cot_scores)), 4) if cot_scores else 0,
            "improvement": round(float(np.mean(cot_scores) - np.mean(baseline_scores)), 4) if cot_scores and baseline_scores else 0,
            "rerank_attempts": total_reranks,
            "rerank_changes": rerank_improvements,
        },
        "timing": {"total_s": round(total_elapsed, 1), "per_image_s": round(total_elapsed / max(n, 1), 2)},
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
