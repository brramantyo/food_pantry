#!/usr/bin/env python3
"""
evaluate_od_contrastive.py
==========================
OD → Crop → Contrastive Classifier Pipeline.

Uses EXISTING fine-tuned OD model (Florence-2 LoRA) for detection,
then EXISTING contrastive model for classification of each crop.

Key difference from evaluate_od_classify.py:
  - Uses contrastive discriminative classifier (sigmoid + threshold)
  - NOT Florence-2 generative (which tends to over-predict on crops)
  - Separate threshold for crops (higher) vs full images

Pipeline:
  1. Run fine-tuned OD → bounding boxes (ignore labels)
  2. Crop each bbox with padding
  3. Run contrastive classifier on each crop (sigmoid, crop_threshold)
  4. Run contrastive classifier on full image (sigmoid, threshold)
  5. Union of all predictions → final

Usage:
  python evaluate_od_contrastive.py \
    --base-model microsoft/Florence-2-large-ft \
    --od-checkpoint ./checkpoints_od_v1/best_model \
    --contrastive-checkpoint ./checkpoints_contrastive/best_model.pt \
    --data-dir . \
    --jsonl ./florence2_data/test_v5.jsonl \
    --output ./eval_od_contrastive.json \
    --threshold 0.5 \
    --crop-threshold 0.6 \
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


# ── Constants ──────────────────────────────────────────────────────────────────

TASK_PROMPT = "<OD>"

CATEGORIES = [
    "Baby Food", "Beans and Legumes - Canned or Dried", "Bread and Bakery Products",
    "Canned Tomato Products", "Carbohydrate Meal", "Condiments and Sauces",
    "Dairy and Dairy Alternatives", "Desserts and Sweets", "Drinks", "Fresh Fruit",
    "Fruits - Canned or Processed", "Granola Products", "Meat and Poultry - Canned",
    "Meat and Poultry - Fresh", "Nut Butters and Nuts", "Ready Meals",
    "Savory Snacks and Crackers", "Seafood - Canned", "Soup",
    "Vegetables - Canned", "Vegetables - Fresh",
]

NUM_CLASSES = len(CATEGORIES)
VALID_CATEGORIES = set(CATEGORIES)
CANONICAL_MAP = {c.lower(): c for c in CATEGORIES}


def normalize_category(name):
    return CANONICAL_MAP.get(name.lower().strip(), name)


# ── Contrastive Model (same as evaluate_pipeline_ensemble.py) ──────────────────

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


# ── OD Parsing ─────────────────────────────────────────────────────────────────

def parse_od_bboxes(text, img_width, img_height):
    bboxes = []
    pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    for m in re.finditer(pattern, text):
        x1 = int(m.group(1)) / 999 * img_width
        y1 = int(m.group(2)) / 999 * img_height
        x2 = int(m.group(3)) / 999 * img_width
        y2 = int(m.group(4)) / 999 * img_height
        if x2 > x1 + 5 and y2 > y1 + 5:
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def deduplicate_bboxes(bboxes, iou_threshold=0.5):
    if not bboxes:
        return []
    bboxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    for bbox in bboxes:
        is_dup = False
        for kept in keep:
            if compute_iou(bbox, kept) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(bbox)
    return keep


def crop_with_padding(image, bbox, padding_ratio=0.15):
    x1, y1, x2, y2 = bbox
    w, h = image.size
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    if crop.width < 32 or crop.height < 32:
        return None
    return crop


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_od(model, processor, image, device, amp_dtype=None):
    """Run OD model, return bounding boxes."""
    inputs = processor(text="<OD>", images=image, return_tensors="pt").to(device)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        early_stopping=True,
    )
    if amp_dtype:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            gen_ids = model.generate(**gen_kwargs)
    else:
        gen_ids = model.generate(**gen_kwargs)
    text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    bboxes = parse_od_bboxes(text, image.width, image.height)
    return deduplicate_bboxes(bboxes)


@torch.no_grad()
def run_contrastive(con_model, processor, image, device, threshold=0.5, amp_dtype=None):
    """Run contrastive classifier, return set of predicted categories."""
    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
    if amp_dtype:
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            _, logits = con_model(inputs["pixel_values"], inputs["input_ids"])
    else:
        _, logits = con_model(inputs["pixel_values"], inputs["input_ids"])
    probs = torch.sigmoid(logits).cpu().squeeze()
    preds = {CATEGORIES[j] for j in range(NUM_CLASSES) if probs[j] >= threshold}
    return preds


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OD → Crop → Contrastive eval")
    parser.add_argument("--base-model", default="microsoft/Florence-2-large-ft")
    parser.add_argument("--od-checkpoint", default="./checkpoints_od_v1/best_model")
    parser.add_argument("--contrastive-checkpoint", default="./checkpoints_contrastive/best_model.pt")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--jsonl", default="./florence2_data/test_v5.jsonl")
    parser.add_argument("--output", default="./eval_od_contrastive.json")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for full-image classification")
    parser.add_argument("--crop-threshold", type=float, default=0.6,
                        help="Higher threshold for crop classification (reduce FP)")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Device: {device}")

    # ── Load OD model ──────────────────────────────────────────────────
    print(f"\nLoading OD model: {args.od_checkpoint}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    od_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32,
    )
    od_model = PeftModel.from_pretrained(od_base, args.od_checkpoint)
    od_model = od_model.to(device).eval()
    print("  OD model loaded ✓")

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
    current_state.update(ckpt["model_state_dict"])
    con_model.load_state_dict(current_state)
    con_model.eval()
    print("  Contrastive model loaded ✓")

    # ── Load test data ─────────────────────────────────────────────────
    samples = []
    with open(args.jsonl, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"\n  Evaluating {len(samples)} samples...")
    print(f"  Full-image threshold: {args.threshold}")
    print(f"  Crop threshold: {args.crop_threshold}\n")

    # ── Tracking ───────────────────────────────────────────────────────
    results = []

    # Metrics for 3 strategies
    strategies = {"full_only": {}, "crops_only": {}, "union": {}}
    for s in strategies:
        strategies[s] = {"tp": Counter(), "fp": Counter(), "fn": Counter()}

    t0 = time.time()

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

        # ── OD: detect bboxes ─────────────────────────────────────────
        bboxes = run_od(od_model, processor, image, device, amp_dtype)

        # ── Classify full image ───────────────────────────────────────
        full_preds = run_contrastive(con_model, processor, image, device, args.threshold, amp_dtype)

        # ── Classify each crop ────────────────────────────────────────
        crop_preds = set()
        valid_crops = 0
        for bbox in bboxes:
            crop = crop_with_padding(image, bbox)
            if crop is None:
                continue
            valid_crops += 1
            crop_cats = run_contrastive(con_model, processor, crop, device, args.crop_threshold, amp_dtype)
            crop_preds |= crop_cats

        # ── Strategies ────────────────────────────────────────────────
        union_preds = full_preds | crop_preds

        preds_map = {
            "full_only": full_preds,
            "crops_only": crop_preds,
            "union": union_preds,
        }

        for strat_name, pred_cats in preds_map.items():
            s = strategies[strat_name]
            for cls in target_cats | pred_cats:
                if cls in target_cats and cls in pred_cats:
                    s["tp"][cls] += 1
                elif cls in pred_cats:
                    s["fp"][cls] += 1
                elif cls in target_cats:
                    s["fn"][cls] += 1

        # ── Store result ──────────────────────────────────────────────
        is_match = target_cats == union_preds
        results.append({
            "image": img_rel,
            "target": sorted(target_cats),
            "full_image": sorted(full_preds),
            "crops": sorted(crop_preds),
            "union": sorted(union_preds),
            "bboxes": len(bboxes),
            "valid_crops": valid_crops,
            "match": is_match,
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(samples)}] {rate:.1f} img/s")

    total_elapsed = time.time() - t0
    n = len(results)

    # ══════════════════════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  OD → CROP → CONTRASTIVE CLASSIFIER PIPELINE")
    print(f"{'='*70}")
    print(f"  Samples: {n} | Time: {total_elapsed:.1f}s ({n/total_elapsed:.1f} img/s)")
    print(f"  Full threshold: {args.threshold} | Crop threshold: {args.crop_threshold}")

    print(f"\n  {'Method':<30} {'Micro P':>8} {'Micro R':>8} {'Micro F1':>9} {'Macro F1':>9}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    summary = {}
    for strat_name in ["full_only", "crops_only", "union"]:
        s = strategies[strat_name]
        total_tp = sum(s["tp"].values())
        total_fp = sum(s["fp"].values())
        total_fn = sum(s["fn"].values())
        micro_p = total_tp / max(total_tp + total_fp, 1)
        micro_r = total_tp / max(total_tp + total_fn, 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)

        # Macro
        all_cls = sorted(set(list(s["tp"]) + list(s["fp"]) + list(s["fn"])))
        f1s = []
        for cls in all_cls:
            tp, fp, fn = s["tp"][cls], s["fp"][cls], s["fn"][cls]
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-8)
            f1s.append(f1)
        macro_f1 = sum(f1s) / max(len(f1s), 1)

        summary[strat_name] = {
            "micro_p": round(micro_p, 4), "micro_r": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4), "macro_f1": round(macro_f1, 4),
        }
        print(f"  {strat_name:<30} {micro_p:>7.1%} {micro_r:>7.1%} {micro_f1:>8.1%} {macro_f1:>8.1%}")

    # Comparison
    print(f"\n{'='*70}")
    print(f"  COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Method':<45} {'Micro F1':>9}")
    print(f"  {'-'*45} {'-'*9}")
    print(f"  {'v11 direct classification':<45} {'76.5%':>9}")
    print(f"  {'Contrastive direct (full image)':<45} {'77.4%':>9}")
    print(f"  {'OD→crop→v11 (generative)':<45} {'56.4%':>9}")
    print(f"  {'OD→crop→v13 (trained on crops)':<45} {'58.8%':>9}")
    print(f"  {'OD→crop→contrastive (full only)':<45} {summary['full_only']['micro_f1']:>8.1%}")
    print(f"  {'OD→crop→contrastive (crops only)':<45} {summary['crops_only']['micro_f1']:>8.1%}")
    print(f"  {'OD→crop→contrastive (union)':<45} {summary['union']['micro_f1']:>8.1%}")
    print(f"  {'Ensemble v11∪contrastive':<45} {'80.3%':>9}")

    # Sample outputs
    print(f"\n{'='*70}")
    print(f"  SAMPLE OUTPUTS (first 10)")
    print(f"{'='*70}")
    for r in results[:10]:
        match_str = "✓" if r["match"] else "✗"
        print(f"\n  {match_str} {os.path.basename(r['image'])}")
        print(f"    Target: {r['target']}")
        print(f"    Full:   {r['full_image']}")
        print(f"    Crops:  {r['crops']} ({r['valid_crops']} crops from {r['bboxes']} bboxes)")
        print(f"    Union:  {r['union']}")

    # Save
    output = {
        "config": vars(args),
        "summary": summary,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
