#!/usr/bin/env python3
"""
train_florence2_v12.py
======================
Florence-2-large fine-tuning v12 — Multi-label recall focus.

Key findings from v11 error analysis:
  - Main problem is UNDER-DETECTION (not confusion)
  - Model predicts 1 class when 2-3 are present (34.8% images are multi-label)
  - Granola missed 100%, Dairy 89%, Condiments 80%, Ready Meals 76%
  - Very few direct confusion pairs (most <2 occurrences)

V12 Strategy (from v11 checkpoint):
  1. Much stronger multi-item loss weighting (3x for multi-label samples)
  2. Per-sample class-count aware weighting (more items = higher weight)
  3. More aggressive oversampling for multi-label images
  4. Bottom-7 recall classes get aggressive boost
  5. Slightly higher LR than v11 (2e-5 vs 1e-5) to allow more plasticity
  6. Focal loss gamma=0.5 (light) to help with hard multi-label examples
  7. Continue from v11 checkpoint (which already has v9 knowledge)

Usage:
  python train_florence2_v12.py \\
    --data-dir . \\
    --jsonl-dir ./florence2_data \\
    --checkpoint ./checkpoints_v11/best_model \\
    --output-dir ./checkpoints_v12 \\
    --epochs 15 \\
    --bf16
"""

import argparse
import json
import os
import sys
import time
import random
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


TASK_PROMPT = "<OD>"

# ── Case Normalization ─────────────────────────────────────────────────────────

CANONICAL_CATEGORIES = {
    "baby food": "Baby Food",
    "beans and legumes - canned or dried": "Beans and Legumes - Canned or Dried",
    "bread and bakery products": "Bread and Bakery Products",
    "canned tomato products": "Canned Tomato Products",
    "carbohydrate meal": "Carbohydrate Meal",
    "condiments and sauces": "Condiments and Sauces",
    "dairy and dairy alternatives": "Dairy and Dairy Alternatives",
    "desserts and sweets": "Desserts and Sweets",
    "drinks": "Drinks",
    "fresh fruit": "Fresh Fruit",
    "fruits - canned or processed": "Fruits - Canned or Processed",
    "granola products": "Granola Products",
    "meat and poultry - canned": "Meat and Poultry - Canned",
    "meat and poultry - fresh": "Meat and Poultry - Fresh",
    "nut butters and nuts": "Nut Butters and Nuts",
    "ready meals": "Ready Meals",
    "savory snacks and crackers": "Savory Snacks and Crackers",
    "seafood - canned": "Seafood - Canned",
    "soup": "Soup",
    "vegetables - canned": "Vegetables - Canned",
    "vegetables - fresh": "Vegetables - Fresh",
}

def normalize_category(name):
    """Normalize category name to canonical casing."""
    return CANONICAL_CATEGORIES.get(name.lower().strip(), name)


# ── Focal Loss ─────────────────────────────────────────────────────────────────

class FocalCrossEntropyLoss(torch.nn.Module):
    """Focal Loss — focuses on hard (poorly-classified) tokens."""

    def __init__(self, gamma=0.5, label_smoothing=0.03, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        mask = targets_flat != self.ignore_index
        logits_flat = logits_flat[mask]
        targets_flat = targets_flat[mask]

        if logits_flat.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = F.log_softmax(logits_flat, dim=-1)
        target_log_probs = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        target_probs = target_log_probs.exp()

        focal_weight = (1.0 - target_probs) ** self.gamma

        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean(dim=-1)
            nll_loss = -target_log_probs
            loss_per_token = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
        else:
            loss_per_token = -target_log_probs

        loss = (focal_weight * loss_per_token).mean()
        return loss


# ── Data Augmentation ──────────────────────────────────────────────────────────

class PantryAugmentationV12:
    """Augmentation with emphasis on multi-item image variety."""

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image

        if random.random() < 0.5:
            image = TF.hflip(image)

        if random.random() < 0.35:
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle, fill=128)

        if random.random() < 0.5:
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() < 0.5:
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if random.random() < 0.4:
            factor = random.uniform(0.6, 1.4)
            image = ImageEnhance.Color(image).enhance(factor)

        if random.random() < 0.3:
            factor = random.uniform(0.5, 2.0)
            image = ImageEnhance.Sharpness(image).enhance(factor)

        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

        if random.random() < 0.25:
            w, h = image.size
            crop_frac = random.uniform(0.80, 0.95)
            new_w, new_h = int(w * crop_frac), int(h * crop_frac)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            image = image.crop((left, top, left + new_w, top + new_h))
            image = image.resize((w, h), Image.BILINEAR)

        if random.random() < 0.15:
            w, h = image.size
            max_shift = int(min(w, h) * 0.08)
            if max_shift > 0:
                startpoints = [(0, 0), (w, 0), (w, h), (0, h)]
                endpoints = [
                    (random.randint(0, max_shift), random.randint(0, max_shift)),
                    (w - random.randint(0, max_shift), random.randint(0, max_shift)),
                    (w - random.randint(0, max_shift), h - random.randint(0, max_shift)),
                    (random.randint(0, max_shift), h - random.randint(0, max_shift)),
                ]
                image = TF.perspective(image, startpoints, endpoints, fill=128)

        return image


# ── Dataset with Multi-Label Aware Oversampling ────────────────────────────────

class Florence2PantryDatasetV12(Dataset):
    """
    V12 Dataset with multi-label recall focus.

    Key changes from v11:
    - 3x weight for multi-label samples (up from 1x in v11)
    - Per-class weighting based on inverse frequency
    - Bottom-7 classes get aggressive boost
    - Track per-sample item count for loss weighting
    """

    # Bottom recall classes from v11 confusion analysis
    BOOST_CLASSES = {
        "Granola Products",             # 50.0% recall, 100% missed in multi-label
        "Dairy and Dairy Alternatives", # 55.6% recall, 89% missed
        "Condiments and Sauces",        # 60.0% recall, 80% missed
        "Ready Meals",                  # 35.3% recall, 76% missed
        "Vegetables - Fresh",           # 33.3% recall, 67% missed
        "Meat and Poultry - Canned",    # 57.1% recall, 71% missed
        "Savory Snacks and Crackers",   # 58.8% recall, 65% missed
    }

    def __init__(self, jsonl_path, data_dir, processor, max_length=1024,
                 augment=False, oversample=False, min_samples_per_class=50,
                 multi_item_boost=3, boost_intensity=0.6):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.augment = PantryAugmentationV12(p=0.7) if augment else None
        self.sample_weights = []  # for loss weighting

        raw_samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_samples.append(json.loads(line))

        print(f"  Loaded {len(raw_samples)} raw samples from {jsonl_path}")

        if oversample:
            self.samples = self._oversample_v12(
                raw_samples, min_samples_per_class, multi_item_boost, boost_intensity
            )
            print(f"  After oversampling: {len(self.samples)} samples")
        else:
            self.samples = raw_samples

        # Compute per-sample weights for loss weighting
        self._compute_sample_weights()

    def _oversample_v12(self, samples, min_samples_per_class, multi_item_boost, boost_intensity):
        class_samples = {}
        multi_item_samples = []
        multi_item_3plus = []  # images with 3+ classes (hardest)

        for s in samples:
            try:
                target = json.loads(s["target"])
                items = target.get("items", [])
                classes_in_sample = set()
                for item in items:
                    cls = normalize_category(item.get("name", "__unknown__"))
                    classes_in_sample.add(cls)

                num_classes = len(classes_in_sample)
                if num_classes > 1:
                    multi_item_samples.append(s)
                if num_classes >= 3:
                    multi_item_3plus.append(s)

                if not classes_in_sample:
                    classes_in_sample = {"__empty__"}

                for cls in classes_in_sample:
                    if cls not in class_samples:
                        class_samples[cls] = []
                    class_samples[cls].append(s)

            except (json.JSONDecodeError, KeyError):
                pass

        print(f"  Class distribution (before oversampling):")
        for cls in sorted(class_samples.keys()):
            count = len(class_samples[cls])
            markers = []
            if count < min_samples_per_class:
                markers.append("UNDERSAMPLE")
            if cls in self.BOOST_CLASSES:
                markers.append("LOW-RECALL-BOOST")
            marker = f" <- {', '.join(markers)}" if markers else ""
            print(f"    {cls}: {count}{marker}")

        print(f"  Multi-item samples (2+): {len(multi_item_samples)}")
        print(f"  Multi-item samples (3+): {len(multi_item_3plus)}")

        result = list(samples)

        # 1. Standard class oversampling to minimum
        for cls, cls_samps in class_samples.items():
            if cls in ("__empty__", "__unknown__"):
                continue
            if len(cls_samps) < min_samples_per_class:
                needed = min_samples_per_class - len(cls_samps)
                extras = [cls_samps[i % len(cls_samps)] for i in range(needed)]
                result.extend(extras)
                print(f"    Oversampled {cls}: +{needed} (total: {len(cls_samps) + needed})")

        # 2. Multi-item boost (3x for 2+ items, 5x for 3+ items)
        if multi_item_samples:
            for _ in range(multi_item_boost):
                result.extend(multi_item_samples)
            print(f"    Multi-item boost (2+): +{len(multi_item_samples) * multi_item_boost} ({multi_item_boost}x)")

        if multi_item_3plus:
            extra_boost = 2  # additional 2x on top
            for _ in range(extra_boost):
                result.extend(multi_item_3plus)
            print(f"    Multi-item boost (3+): +{len(multi_item_3plus) * extra_boost} ({extra_boost}x extra)")

        # 3. Aggressive boost for low-recall classes
        for cls in self.BOOST_CLASSES:
            if cls in class_samples:
                boost_count = max(10, int(len(class_samples[cls]) * boost_intensity))
                extras = [class_samples[cls][i % len(class_samples[cls])]
                          for i in range(boost_count)]
                result.extend(extras)
                print(f"    Low-recall boost {cls}: +{boost_count}")

        random.shuffle(result)
        return result

    def _compute_sample_weights(self):
        """Compute per-sample loss weights based on number of items."""
        self.sample_weights = []
        for s in self.samples:
            try:
                target = json.loads(s["target"])
                num_items = len(target.get("items", []))
                # Multi-label samples get higher weight
                if num_items >= 3:
                    weight = 3.0
                elif num_items == 2:
                    weight = 2.0
                else:
                    weight = 1.0
                self.sample_weights.append(weight)
            except (json.JSONDecodeError, KeyError):
                self.sample_weights.append(1.0)

    def __len__(self):
        return len(self.samples)

    def get_weight(self, idx):
        """Get loss weight for sample at index."""
        if idx < len(self.sample_weights):
            return self.sample_weights[idx]
        return 1.0

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(self.data_dir, img_rel)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to load {img_path}: {e}", file=sys.stderr)
            image = Image.new("RGB", (640, 480), (0, 0, 0))

        if self.augment is not None:
            image = self.augment(image)

        target = sample["target"]

        # Normalize category casing
        try:
            target_obj = json.loads(target)
            if "items" in target_obj:
                for item in target_obj["items"]:
                    if "name" in item:
                        item["name"] = normalize_category(item["name"])
                target = json.dumps(target_obj)
        except (json.JSONDecodeError, KeyError):
            pass

        inputs = self.processor(
            text=TASK_PROMPT,
            images=image,
            return_tensors="pt",
        )

        labels = self.processor.tokenizer(
            text=target,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": label_ids,
            "weight": torch.tensor(self.get_weight(idx), dtype=torch.float32),
        }


def collate_fn(batch):
    max_input_len = max(x["input_ids"].shape[0] for x in batch)
    padded_input_ids = []
    for x in batch:
        ids = x["input_ids"]
        pad_len = max_input_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_len), value=1)
        padded_input_ids.append(ids)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "weights": torch.stack([x["weight"] for x in batch]),
    }


# ── Training Functions ─────────────────────────────────────────────────────────

def weighted_ce_loss(logits, targets, weights, label_smoothing=0.03, ignore_index=-100):
    """Cross-entropy with per-sample weighting for multi-label emphasis."""
    batch_size = logits.size(0)
    vocab_size = logits.size(-1)
    total_loss = torch.tensor(0.0, device=logits.device)
    valid_count = 0

    for i in range(batch_size):
        logits_i = logits[i].view(-1, vocab_size)
        targets_i = targets[i].view(-1)
        mask = targets_i != ignore_index

        if mask.sum() == 0:
            continue

        loss_i = F.cross_entropy(
            logits_i[mask], targets_i[mask],
            label_smoothing=label_smoothing,
        )
        total_loss = total_loss + loss_i * weights[i]
        valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return total_loss / valid_count


@torch.no_grad()
def evaluate(model, dataloader, device, label_smoothing=0.03, amp_dtype=None):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if amp_dtype is not None:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                # Use standard CE for validation (no weighting)
                logits = outputs.logits
                vocab_size = logits.size(-1)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size), labels.view(-1),
                    ignore_index=-100, label_smoothing=label_smoothing,
                )
        else:
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            vocab_size = logits.size(-1)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), labels.view(-1),
                ignore_index=-100, label_smoothing=label_smoothing,
            )
        total_loss += loss.item()
    return total_loss / num_batches


@torch.no_grad()
def run_inference_samples(model, processor, samples, data_dir, device, amp_dtype=None, n=5):
    model.eval()
    correct = 0
    total = 0
    multi_correct = 0
    multi_total = 0

    for i, sample in enumerate(samples[:n]):
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(data_dir, img_rel)
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)
        if amp_dtype is not None:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                    max_new_tokens=512, num_beams=3, early_stopping=True,
                )
        else:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=512, num_beams=3, early_stopping=True,
            )
        clean_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        target = json.loads(sample["target"])
        target_classes = {normalize_category(item["name"]) for item in target.get("items", [])}
        try:
            pred = json.loads(clean_text)
            pred_classes = {normalize_category(item["name"]) for item in pred.get("items", [])}
        except (json.JSONDecodeError, AttributeError):
            pred_classes = set()

        is_multi = len(target_classes) > 1
        match_str = "✓" if target_classes == pred_classes else "✗"
        if target_classes == pred_classes:
            correct += 1
            if is_multi:
                multi_correct += 1
        total += 1
        if is_multi:
            multi_total += 1

        print(f"\n    Sample {i+1}: {os.path.basename(img_rel)}")
        print(f"    Target ({len(target_classes)} classes): {target_classes}")
        print(f"    Pred   ({len(pred_classes)} classes): {pred_classes}")
        print(f"    {match_str} {'[MULTI]' if is_multi else '[SINGLE]'}")

    if total > 0:
        print(f"\n    Quick accuracy: {correct}/{total} ({100*correct/total:.0f}%)")
    if multi_total > 0:
        print(f"    Multi-label accuracy: {multi_correct}/{multi_total} ({100*multi_correct/multi_total:.0f}%)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Florence-2 fine-tuning v12 — multi-label recall focus")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl-dir", type=str, default="./florence2_data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_v12")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_v11/best_model",
                        help="Starting checkpoint (v11 best_model)")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Slightly higher LR for more plasticity")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=48)
    parser.add_argument("--lora-alpha", type=int, default=96)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--patience", type=int, default=6,
                        help="Slightly longer patience for more epochs")
    parser.add_argument("--min-samples-per-class", type=int, default=60,
                        help="Higher minimum for v12")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--multi-item-boost", type=int, default=3,
                        help="How many times to duplicate multi-label samples")
    parser.add_argument("--boost-intensity", type=float, default=0.6,
                        help="Fraction of class samples to add as boost for low-recall classes")
    parser.add_argument("--train-jsonl", type=str, default=None)
    parser.add_argument("--valid-jsonl", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    use_amp = False
    amp_dtype = torch.float32
    if args.bf16 and torch.cuda.is_bf16_supported():
        use_amp = True
        amp_dtype = torch.bfloat16
        print("Using bf16 mixed precision")
    elif args.fp16:
        use_amp = True
        amp_dtype = torch.float16
        print("Using fp16 mixed precision")

    effective_batch = args.batch_size * args.gradient_accumulation
    print(f"Effective batch size: {args.batch_size} x {args.gradient_accumulation} = {effective_batch}")

    # ── Load Model ────────────────────────────────────────────────────────

    print(f"\nLoading base model: {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=amp_dtype if use_amp else torch.float32,
        attn_implementation="eager",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    # ── Load Checkpoint ───────────────────────────────────────────────────

    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        print(f"\nLoading LoRA checkpoint from: {checkpoint_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        print("  ✓ Checkpoint loaded — continuing training")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")
    else:
        print(f"\n  WARNING: Checkpoint not found at {checkpoint_path}")
        print("  Falling back to fresh LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "qkv", "proj",
                "q_proj", "v_proj", "k_proj", "o_proj", "out_proj",
                "fc1", "fc2",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.to(device)

    # ── Datasets ──────────────────────────────────────────────────────────

    print("\nLoading datasets...")
    train_jsonl = args.train_jsonl or os.path.join(args.jsonl_dir, "train_v5.jsonl")
    valid_jsonl = args.valid_jsonl or os.path.join(args.jsonl_dir, "valid_v5.jsonl")

    train_dataset = Florence2PantryDatasetV12(
        jsonl_path=train_jsonl,
        data_dir=args.data_dir,
        processor=processor,
        max_length=args.max_length,
        augment=True,
        oversample=True,
        min_samples_per_class=args.min_samples_per_class,
        multi_item_boost=args.multi_item_boost,
        boost_intensity=args.boost_intensity,
    )
    valid_dataset = Florence2PantryDatasetV12(
        jsonl_path=valid_jsonl,
        data_dir=args.data_dir,
        processor=processor,
        max_length=args.max_length,
        augment=False,
        oversample=False,
    )

    valid_samples_raw = []
    with open(valid_jsonl, "r") as f:
        for line in f:
            if line.strip():
                valid_samples_raw.append(json.loads(line.strip()))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Optimizer & Scheduler ──────────────────────────────────────────────

    total_steps = (len(train_loader) // args.gradient_accumulation) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\nTraining config (v12 — Multi-label recall focus):")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Multi-item boost: {args.multi_item_boost}x")
    print(f"  Low-recall boost intensity: {args.boost_intensity}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)}")
    print(f"  Min samples per class: {args.min_samples_per_class}")

    # ── Training Loop ──────────────────────────────────────────────────────

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print("STARTING TRAINING (v12 — Multi-label Recall Focus)")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            weights = batch["weights"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                    loss = weighted_ce_loss(
                        outputs.logits, labels, weights,
                        label_smoothing=args.label_smoothing,
                    ) / args.gradient_accumulation
                loss.backward()
            else:
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = weighted_ce_loss(
                    outputs.logits, labels, weights,
                    label_smoothing=args.label_smoothing,
                ) / args.gradient_accumulation
                loss.backward()

            total_loss += (loss.item() * args.gradient_accumulation)

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                avg_loss = total_loss / (batch_idx + 1)
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch [{epoch+1}/{args.epochs}] "
                      f"Step [{batch_idx+1}/{num_batches}] "
                      f"Loss: {loss.item()*args.gradient_accumulation:.4f} (avg: {avg_loss:.4f}) "
                      f"LR: {current_lr:.2e}")

        train_loss = total_loss / num_batches

        val_loss = evaluate(model, valid_loader, device,
                            label_smoothing=args.label_smoothing,
                            amp_dtype=amp_dtype if use_amp else None)
        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Valid Loss: {val_loss:.4f}")
        print(f"    Time: {epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "best_model")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    ✓ New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{args.patience})")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n    Inference check (epoch {epoch+1}):")
            run_inference_samples(
                model, processor, valid_samples_raw, args.data_dir,
                device, amp_dtype=amp_dtype if use_amp else None, n=8,
            )

        if patience_counter >= args.patience:
            print(f"\n  ⚠ Early stopping at epoch {epoch+1}")
            break

        print()

    # ── Final Save ─────────────────────────────────────────────────────────

    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE (v12 — Multi-label Recall Focus)")
    print(f"{'='*60}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model: {os.path.join(args.output_dir, 'best_model')}")
    print(f"  Final model: {final_path}")

    print(f"\n{'='*60}")
    print("FINAL INFERENCE TEST")
    print(f"{'='*60}")
    run_inference_samples(
        model, processor, valid_samples_raw, args.data_dir,
        device, amp_dtype=amp_dtype if use_amp else None, n=10,
    )


if __name__ == "__main__":
    main()