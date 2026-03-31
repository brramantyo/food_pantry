#!/usr/bin/env python3
"""
train_florence2_v13.py
======================
Florence-2 v13 — Continue from v11 checkpoint, trained on MIXED data:
  - Original full shelf images (from florence2_data/)
  - Single-item crop images (from crop_data/)

This addresses the precision problem in the OD→crop→classify pipeline:
v11 was only trained on full shelf images, so it over-predicts when given
individual item crops. v13 sees both, learning to output single categories
for crops and multiple categories for full images.

Key changes from v11:
  - Mixed dataset: full images + crop images in same training loop
  - Crop images have single-category targets
  - Full images have multi-category targets (unchanged)
  - Crop mixing ratio configurable (default 50% crops per batch)

Usage:
  python train_florence2_v13.py \\
    --data-dir . \\
    --jsonl-dir ./florence2_data \\
    --crop-jsonl ./crop_data/train_crops.jsonl \\
    --crop-val-jsonl ./crop_data/val_crops.jsonl \\
    --checkpoint ./checkpoints_v11/best_model \\
    --output-dir ./checkpoints_v13 \\
    --epochs 12 \\
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
    return CANONICAL_CATEGORIES.get(name.lower().strip(), name)


# ── Data Augmentation ──────────────────────────────────────────────────────────

class PantryAugmentation:
    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, image):
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
        return image


# ── Dataset ────────────────────────────────────────────────────────────────────

class Florence2MixedDataset(Dataset):
    """
    Dataset that loads from a JSONL file (works for both full images and crops).
    Supports oversampling for class balance.
    """

    CONFUSION_BOOST_CLASSES = {
        "Dairy and Dairy Alternatives",
        "Vegetables - Fresh",
        "Nut Butters and Nuts",
        "Granola Products",
        "Ready Meals",
    }

    def __init__(self, jsonl_path, data_dir, processor, max_length=512,
                 augment=False, oversample=False, min_samples_per_class=25,
                 confusion_boost=True, label=""):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.augment = PantryAugmentation(p=0.7) if augment else None
        self.label = label

        raw_samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_samples.append(json.loads(line))

        print(f"  [{label}] Loaded {len(raw_samples)} samples from {jsonl_path}")

        if oversample and raw_samples:
            self.samples = self._oversample(raw_samples, min_samples_per_class, confusion_boost)
            print(f"  [{label}] After oversampling: {len(self.samples)} samples")
        else:
            self.samples = raw_samples

    def _oversample(self, samples, min_samples_per_class, confusion_boost):
        class_samples = {}
        multi_item_samples = []

        for s in samples:
            try:
                target = json.loads(s["target"])
                items = target.get("items", [])
                if len(items) > 1:
                    multi_item_samples.append(s)
                classes_in_sample = set()
                for item in items:
                    cls = item.get("name", "__unknown__")
                    classes_in_sample.add(cls)
                if not classes_in_sample:
                    classes_in_sample = {"__empty__"}
                for cls in classes_in_sample:
                    if cls not in class_samples:
                        class_samples[cls] = []
                    class_samples[cls].append(s)
            except (json.JSONDecodeError, KeyError):
                pass

        result = list(samples)

        # Standard class oversampling
        for cls, cls_samps in class_samples.items():
            if cls in ("__empty__", "__unknown__"):
                continue
            if len(cls_samps) < min_samples_per_class:
                needed = min_samples_per_class - len(cls_samps)
                extras = [cls_samps[i % len(cls_samps)] for i in range(needed)]
                result.extend(extras)

        # Multi-item boost
        if multi_item_samples:
            result.extend(multi_item_samples)

        # Confusion-prone class boost
        if confusion_boost:
            for cls in self.CONFUSION_BOOST_CLASSES:
                if cls in class_samples:
                    boost_count = max(5, len(class_samples[cls]) // 2)
                    extras = [class_samples[cls][i % len(class_samples[cls])]
                              for i in range(boost_count)]
                    result.extend(extras)

        random.shuffle(result)
        return result

    def __len__(self):
        return len(self.samples)

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
        try:
            target_obj = json.loads(target)
            if "items" in target_obj:
                for item in target_obj["items"]:
                    if "name" in item:
                        item["name"] = normalize_category(item["name"])
                target = json.dumps(target_obj)
        except (json.JSONDecodeError, KeyError):
            pass

        inputs = self.processor(text=TASK_PROMPT, images=image, return_tensors="pt")
        labels = self.processor.tokenizer(
            text=target, return_tensors="pt",
            padding="max_length", max_length=self.max_length, truncation=True,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": label_ids,
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
    }


# ── Training Functions ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, amp_dtype=None, label_smoothing=0.03):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        if amp_dtype:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1),
                    ignore_index=-100, label_smoothing=label_smoothing,
                )
        else:
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
                ignore_index=-100, label_smoothing=label_smoothing,
            )
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Florence-2 v13: Mixed full-image + crop training")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl-dir", type=str, default="./florence2_data")
    parser.add_argument("--crop-jsonl", type=str, default="./crop_data/train_crops.jsonl")
    parser.add_argument("--crop-val-jsonl", type=str, default="./crop_data/val_crops.jsonl")
    parser.add_argument("--crop-data-dir", type=str, default="./crop_data",
                        help="Base dir for crop images (default: ./crop_data)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_v13")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--min-samples-per-class", type=int, default=40)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if args.bf16 else None

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load Model from v11 checkpoint ─────────────────────────────────────

    print(f"\nLoading base model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=amp_dtype if amp_dtype else torch.float32,
        attn_implementation="eager",
    )

    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading LoRA checkpoint: {checkpoint_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
        print("  ✓ v11 checkpoint loaded — continuing training")
    else:
        print(f"  WARNING: Checkpoint not found at {checkpoint_path}, using fresh LoRA")
        lora_config = LoraConfig(
            r=48, lora_alpha=96, lora_dropout=0.05,
            target_modules=["qkv", "proj", "q_proj", "v_proj", "k_proj", "o_proj", "out_proj", "fc1", "fc2"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Datasets ───────────────────────────────────────────────────────────

    print("\nLoading datasets...")

    # Full-image dataset (original)
    train_jsonl = os.path.join(args.jsonl_dir, "train_v5.jsonl")
    valid_jsonl = os.path.join(args.jsonl_dir, "valid_v5.jsonl")

    full_train = Florence2MixedDataset(
        train_jsonl, args.data_dir, processor, args.max_length,
        augment=True, oversample=True,
        min_samples_per_class=args.min_samples_per_class,
        confusion_boost=True, label="full-train",
    )

    full_valid = Florence2MixedDataset(
        valid_jsonl, args.data_dir, processor, args.max_length,
        augment=False, oversample=False, label="full-valid",
    )

    # Crop dataset
    if os.path.exists(args.crop_jsonl):
        crop_train = Florence2MixedDataset(
            args.crop_jsonl, args.crop_data_dir, processor, args.max_length,
            augment=True, oversample=True,
            min_samples_per_class=args.min_samples_per_class,
            confusion_boost=True, label="crop-train",
        )
        print(f"\n  Mixing: {len(full_train)} full + {len(crop_train)} crops")
        train_dataset = ConcatDataset([full_train, crop_train])
    else:
        print(f"\n  WARNING: No crop JSONL at {args.crop_jsonl}, training on full images only")
        train_dataset = full_train

    if os.path.exists(args.crop_val_jsonl):
        crop_valid = Florence2MixedDataset(
            args.crop_val_jsonl, args.crop_data_dir, processor, args.max_length,
            augment=False, oversample=False, label="crop-valid",
        )
        valid_dataset = ConcatDataset([full_valid, crop_valid])
    else:
        valid_dataset = full_valid

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Optimizer & Scheduler ──────────────────────────────────────────────

    total_steps = (len(train_loader) // args.gradient_accumulation) * args.epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    effective_batch = args.batch_size * args.gradient_accumulation
    print(f"\nTraining config (v13 — Mixed full + crop):")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Train total: {len(train_dataset)}")
    print(f"  Valid total: {len(valid_dataset)}")

    # ── Training Loop ──────────────────────────────────────────────────────

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print("STARTING TRAINING (v13 — Mixed full-image + crop)")
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

            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                    logits = outputs.logits
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1),
                        ignore_index=-100, label_smoothing=args.label_smoothing,
                    ) / args.gradient_accumulation
                loss.backward()
            else:
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), labels.view(-1),
                    ignore_index=-100, label_smoothing=args.label_smoothing,
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
                      f"Loss: {loss.item()*args.gradient_accumulation:.4f} "
                      f"(avg: {avg_loss:.4f}) LR: {current_lr:.2e}")

        train_loss = total_loss / num_batches
        val_loss = evaluate(model, valid_loader, device, amp_dtype, args.label_smoothing)
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

        if patience_counter >= args.patience:
            print(f"\n  ⚠ Early stopping at epoch {epoch+1}")
            break
        print()

    # ── Final Save ─────────────────────────────────────────────────────────

    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE (v13 — Mixed full-image + crop)")
    print(f"{'='*60}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model: {os.path.join(args.output_dir, 'best_model')}")
    print(f"  Final model: {final_path}")
    print(f"\nNext: evaluate with run_eval_v13.sh and run_od_classify_v13.sh")


if __name__ == "__main__":
    main()
