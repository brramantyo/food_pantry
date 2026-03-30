#!/usr/bin/env python3
"""
evaluate_ensemble.py
====================
Ensemble evaluation: Contrastive classifier + v11 generative classifier.

Two fundamentally different models:
  - v11: Generative (seq2seq, Florence-2 with LoRA)
  - Contrastive: Discriminative (frozen encoder + classification head)

Ensemble strategies:
  1. Union: predict category if EITHER model predicts it
  2. Intersection: predict category if BOTH models predict it
  3. Average: average sigmoid probabilities from contrastive + v11 binary predictions

Usage:
  python evaluate_ensemble.py \
    --base-model microsoft/Florence-2-large-ft \
    --v11-checkpoint ./checkpoints_v11/best_model \
    --contrastive-checkpoint ./checkpoints_contrastive/best_model.pt \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_ensemble_v11_contrastive.json \
    --bf16
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel


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

def normalize_category(name):
    return CANONICAL_MAP.get(name.lower().strip(), name)


# ── Contrastive Model (same architecture as training) ──────────────────────────

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


# ── v11 Prediction Parser ─────────────────────────────────────────────────────

def parse_v11_prediction(text):
    """Parse v11 generative output → set of categories."""
    text = text.strip()
    categories = set()
    
    # Try JSON parse
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
    
    # Find JSON in text
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
    
    # Try closing truncated JSON
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
    
    # Regex fallback
    for cat in VALID_CATEGORIES:
        if cat.lower() in text.lower():
            categories.add(cat)
    return categories


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(all_targets, all_preds):
    class_tp = Counter()
    class_fp = Counter()
    class_fn = Counter()
    exact = 0
    
    for t, p in zip(all_targets, all_preds):
        if t == p:
            exact += 1
        for c in t | p:
            if c in t and c in p:
                class_tp[c] += 1
            elif c in p:
                class_fp[c] += 1
            else:
                class_fn[c] += 1
    
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
        per_class[cls] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}
    
    macro_f1 = sum(m["f1"] for m in per_class.values()) / max(len(per_class), 1)
    
    return {
        "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
        "macro_f1": macro_f1, "exact_match": exact / max(len(all_targets), 1),
        "per_class": per_class,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ensemble: v11 + Contrastive")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--v11-checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--contrastive-checkpoint", type=str, default="./checkpoints_contrastive/best_model.pt")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl", type=str, default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--output", type=str, default="./eval_ensemble_v11_contrastive.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    
    # ── Load v11 generative model ──────────────────────────────────────
    print(f"Loading v11 generative model: {args.v11_checkpoint}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    
    v11_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True,
        torch_dtype=amp_dtype or torch.float32,
        attn_implementation="eager",
    )
    v11_model = PeftModel.from_pretrained(v11_model, args.v11_checkpoint)
    v11_model = v11_model.merge_and_unload().to(device).eval()
    print("  v11 loaded ✓")
    
    # ── Load contrastive model ─────────────────────────────────────────
    print(f"Loading contrastive model: {args.contrastive_checkpoint}")
    
    # Need a separate Florence-2 instance for contrastive
    florence_con = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    config = florence_con.config
    feature_dim = getattr(config, 'd_model', None) or getattr(config, 'hidden_size', 1024)
    
    con_model = ContrastiveClassifier(
        florence_model=florence_con,
        feature_dim=feature_dim,
        proj_dim=128,
        num_classes=NUM_CLASSES,
    ).to(device)
    
    # Load trained heads
    checkpoint = torch.load(args.contrastive_checkpoint, map_location=device, weights_only=False)
    current_state = con_model.state_dict()
    current_state.update(checkpoint["model_state_dict"])
    con_model.load_state_dict(current_state)
    con_model.eval()
    print("  Contrastive loaded ✓")
    
    # ── Load test data ─────────────────────────────────────────────────
    samples = []
    with open(args.jsonl, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"\n  Evaluating {len(samples)} samples...\n")
    
    # ── Evaluate ───────────────────────────────────────────────────────
    all_targets = []
    v11_preds_all = []
    con_preds_all = []
    union_preds_all = []
    inter_preds_all = []
    
    t0 = time.time()
    
    for i, sample in enumerate(samples):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel)
        
        if not os.path.exists(img_path):
            continue
        
        image = Image.open(img_path).convert("RGB")
        
        # Ground truth
        target = json.loads(sample["target"])
        target_cats = {normalize_category(item["name"]) for item in target.get("items", [])
                       if normalize_category(item.get("name", "")) in VALID_CATEGORIES}
        
        # v11 prediction
        inputs = processor(text="<OD>", images=image, return_tensors="pt").to(device)
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
        v11_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        v11_cats = parse_v11_prediction(v11_text)
        
        # Contrastive prediction
        with torch.no_grad():
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _, logits = con_model(inputs["pixel_values"], inputs["input_ids"])
            else:
                _, logits = con_model(inputs["pixel_values"], inputs["input_ids"])
        
        probs = torch.sigmoid(logits).cpu().squeeze()
        con_cats = {CATEGORIES[j] for j in range(NUM_CLASSES) if probs[j] >= args.threshold}
        
        # Ensemble strategies
        union_cats = v11_cats | con_cats
        inter_cats = v11_cats & con_cats
        # If intersection is empty, fall back to v11
        if not inter_cats:
            inter_cats = v11_cats
        
        all_targets.append(target_cats)
        v11_preds_all.append(v11_cats)
        con_preds_all.append(con_cats)
        union_preds_all.append(union_cats)
        inter_preds_all.append(inter_cats)
        
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{len(samples)} ({rate:.1f} img/s)")
    
    elapsed = time.time() - t0
    n = len(all_targets)
    
    # ── Compute metrics for all strategies ─────────────────────────────
    v11_metrics = compute_metrics(all_targets, v11_preds_all)
    con_metrics = compute_metrics(all_targets, con_preds_all)
    union_metrics = compute_metrics(all_targets, union_preds_all)
    inter_metrics = compute_metrics(all_targets, inter_preds_all)
    
    # ── Print report ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE EVALUATION: v11 + Contrastive")
    print(f"{'='*70}")
    print(f"  Samples: {n}, Time: {elapsed:.1f}s")
    
    print(f"\n  {'Method':<30} {'Micro P':>8} {'Micro R':>8} {'Micro F1':>9} {'Macro F1':>9} {'Exact':>7}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*7}")
    for name, m in [("v11 only", v11_metrics), ("Contrastive only", con_metrics),
                     ("Union (v11 ∪ con)", union_metrics), ("Intersection (v11 ∩ con)", inter_metrics)]:
        print(f"  {name:<30} {m['micro_p']:>7.1%} {m['micro_r']:>7.1%} "
              f"{m['micro_f1']:>8.1%} {m['macro_f1']:>8.1%} {m['exact_match']:>6.1%}")
    
    # Per-class comparison for best ensemble vs v11
    best_ens = max([("union", union_metrics), ("intersection", inter_metrics)], 
                   key=lambda x: x[1]["micro_f1"])
    best_name, best_m = best_ens
    
    print(f"\n  Best ensemble: {best_name} (Micro F1={best_m['micro_f1']:.1%})")
    delta = best_m["micro_f1"] - v11_metrics["micro_f1"]
    print(f"  Δ vs v11 only: {delta:+.1%}")
    
    print(f"\n  {'Class':<40} {'v11':>6} {'Con':>6} {best_name:>6} {'Δ':>6}")
    print(f"  {'-'*40} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for cls in sorted(VALID_CATEGORIES):
        v11_f1 = v11_metrics["per_class"].get(cls, {}).get("f1", 0)
        con_f1 = con_metrics["per_class"].get(cls, {}).get("f1", 0)
        ens_f1 = best_m["per_class"].get(cls, {}).get("f1", 0)
        d = ens_f1 - v11_f1
        print(f"  {cls:<40} {v11_f1:>5.1%} {con_f1:>5.1%} {ens_f1:>5.1%} {d:>+5.1%}")
    
    # ── Save ───────────────────────────────────────────────────────────
    results = {
        "ensemble": "v11_contrastive",
        "n_samples": n,
        "v11_metrics": v11_metrics,
        "contrastive_metrics": con_metrics,
        "union_metrics": union_metrics,
        "intersection_metrics": inter_metrics,
        "best_ensemble": best_name,
        "best_ensemble_metrics": best_m,
    }
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
