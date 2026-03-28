#!/usr/bin/env python3
"""
train_florence2_od.py
=====================
Fine-tune Florence-2 for Object Detection (<OD>) on pantry COCO annotations.

This trains Florence-2 to DETECT and LOCATE individual food items with bounding boxes,
instead of just classifying the whole image.

Detection-first pipeline:
  1. Fine-tuned <OD> → detect individual items with bboxes
  2. Each detected item already has a category label
  3. Map to USDA for nutrition

Input: COCO format annotations with bboxes
Output: Florence-2 model that can detect pantry items

Florence-2 OD format:
  Input prompt: "<OD>"
  Output: "category_1<loc_x1><loc_y1><loc_x2><loc_y2>category_2<loc_x1>..."

  Where loc values are quantized to 0-999 range relative to image dimensions.

Usage:
  python train_florence2_od.py \
    --data-dir . \
    --output-dir ./checkpoints_od_v1 \
    --epochs 15 \
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
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


TASK_PROMPT = "<OD>"

# 21 pantry categories (exclude Carton of Eggs, Frozen Mix Vegetable, Oil, Spices)
PANTRY_CATEGORIES = {
    1: "Baby Food",
    2: "Beans and Legumes - Canned or Dried",
    3: "Bread and Bakery Products",
    4: "Canned Tomato Products",
    5: "Carbohydrate Meal",
    7: "Condiments and Sauces",
    8: "Dairy and Dairy Alternatives",
    9: "Desserts and Sweets",
    10: "Drinks",
    11: "Fresh Fruit",
    13: "Fruits - Canned or Processed",
    14: "Granola Products",
    15: "Meat and Poultry - Canned",
    16: "Meat and Poultry - Fresh",
    17: "Nut Butters and Nuts",
    19: "Ready Meals",
    20: "Savory Snacks and Crackers",
    21: "Seafood - Canned",
    22: "Soup",
    24: "Vegetables - Canned",
    25: "Vegetables - Fresh",
}

EXCLUDED_CATEGORY_IDS = {6, 12, 18, 23}  # Eggs, Frozen Mix Veg, Oil, Spices


def quantize_bbox(bbox, img_w, img_h, num_bins=1000):
    """
    Convert COCO bbox [x, y, w, h] to Florence-2 format [x1, y1, x2, y2] quantized to 0-999.
    """
    x, y, w, h = [float(v) for v in bbox]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    
    # Clamp to image bounds
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    
    # Quantize to 0-(num_bins-1)
    qx1 = int(round(x1 / img_w * (num_bins - 1)))
    qy1 = int(round(y1 / img_h * (num_bins - 1)))
    qx2 = int(round(x2 / img_w * (num_bins - 1)))
    qy2 = int(round(y2 / img_h * (num_bins - 1)))
    
    return qx1, qy1, qx2, qy2


def format_od_target(annotations, img_w, img_h, cat_names):
    """
    Format annotations into Florence-2 <OD> target string.
    
    Florence-2 OD output format:
      "category_name<loc_x1><loc_y1><loc_x2><loc_y2>category_name<loc_x1>..."
    """
    parts = []
    for ann in annotations:
        cat_id = ann["category_id"]
        if cat_id in EXCLUDED_CATEGORY_IDS:
            continue
        if cat_id not in cat_names:
            continue
        
        name = cat_names[cat_id]
        qx1, qy1, qx2, qy2 = quantize_bbox(ann["bbox"], img_w, img_h)
        parts.append(f"{name}<loc_{qx1}><loc_{qy1}><loc_{qx2}><loc_{qy2}>")
    
    return "".join(parts)


# ── Data Augmentation ──────────────────────────────────────────────────────────

class ODDataAugmentation:
    """Augmentation for object detection (image-level only, no bbox transform needed
    since we re-quantize from COCO annotations each time)."""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image):
        if random.random() > self.p:
            return image
        
        # Only safe augmentations that don't change geometry
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)
        
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Contrast(image).enhance(factor)
        
        if random.random() < 0.4:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Color(image).enhance(factor)
        
        if random.random() < 0.2:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        
        return image


# ── Dataset ────────────────────────────────────────────────────────────────────

class Florence2ODDataset(Dataset):
    """
    Dataset for Florence-2 Object Detection fine-tuning.
    Loads COCO annotations and formats them for <OD> task.
    """
    
    def __init__(self, coco_json_path, image_dir, processor, max_length=1024,
                 augment=False):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.augment = ODDataAugmentation(p=0.5) if augment else None
        
        with open(coco_json_path) as f:
            coco = json.load(f)
        
        self.cat_names = {}
        for cat in coco["categories"]:
            if cat["id"] not in EXCLUDED_CATEGORY_IDS:
                self.cat_names[cat["id"]] = cat["name"]
        
        # Build image_id → annotations mapping
        img_anns = {}
        for ann in coco["annotations"]:
            if ann["category_id"] in EXCLUDED_CATEGORY_IDS:
                continue
            img_id = ann["image_id"]
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)
        
        # Build samples: only images with at least 1 valid annotation
        self.samples = []
        img_lookup = {img["id"]: img for img in coco["images"]}
        
        for img_id, anns in img_anns.items():
            if img_id not in img_lookup:
                continue
            img_info = img_lookup[img_id]
            self.samples.append({
                "file_name": img_info["file_name"],
                "width": img_info["width"],
                "height": img_info["height"],
                "annotations": anns,
            })
        
        # Stats
        total_anns = sum(len(s["annotations"]) for s in self.samples)
        cat_dist = Counter()
        for s in self.samples:
            for ann in s["annotations"]:
                cat_dist[self.cat_names.get(ann["category_id"], "?")] += 1
        
        print(f"  Loaded {len(self.samples)} images with {total_anns} annotations")
        print(f"  Categories: {len(self.cat_names)}")
        print(f"  Annotations per category:")
        for name in sorted(cat_dist.keys()):
            print(f"    {name:45s} {cat_dist[name]:>5}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample["file_name"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to load {img_path}: {e}", file=sys.stderr)
            image = Image.new("RGB", (640, 480), (0, 0, 0))
        
        if self.augment:
            image = self.augment(image)
        
        img_w = sample["width"]
        img_h = sample["height"]
        
        # Format target string
        target_str = format_od_target(sample["annotations"], img_w, img_h, self.cat_names)
        
        # Tokenize
        inputs = self.processor(
            text=TASK_PROMPT,
            images=image,
            return_tensors="pt",
        )
        
        # Tokenize target
        target_tokens = self.processor.tokenizer(
            target_str,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        labels = target_tokens["input_ids"].squeeze(0)
        
        # Mask padding tokens in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": labels,
        }


def collate_fn(batch):
    """Custom collate that pads input_ids and labels."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "labels": labels,
    }


# ── Training ───────────────────────────────────────────────────────────────────

def evaluate(model, val_loader, device, amp_dtype=None):
    """Compute validation loss."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=labels,
                )
            
            total_loss += outputs.loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Florence-2 OD fine-tuning on pantry COCO data")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_od_v1")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Longer than classification — OD outputs can be verbose")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    
    print(f"Device: {device}")
    print(f"AMP dtype: {amp_dtype}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model + processor
    print(f"\nLoading model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    )
    
    # Apply LoRA to both vision encoder and language decoder
    target_modules = []
    for name, _ in model.named_modules():
        if any(t in name for t in ["q_proj", "v_proj", "k_proj", "out_proj",
                                     "fc1", "fc2", "qkv", "proj"]):
            if "." in name:
                target_modules.append(name)
    
    # Deduplicate and filter
    target_modules = list(set(target_modules))
    print(f"  LoRA targets: {len(target_modules)} modules")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset = Florence2ODDataset(
        os.path.join(args.data_dir, "train", "_annotations.coco.json"),
        os.path.join(args.data_dir, "train"),
        processor,
        max_length=args.max_length,
        augment=True,
    )
    
    val_dataset = Florence2ODDataset(
        os.path.join(args.data_dir, "valid", "_annotations.coco.json"),
        os.path.join(args.data_dir, "valid"),
        processor,
        max_length=args.max_length,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = min(total_steps // 10, 100)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\nTraining config (OD fine-tuning):")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Max sequence length: {args.max_length}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING OD TRAINING")
    print(f"{'='*60}")
    
    best_val_loss = float("inf")
    patience_counter = 0
    scaler = torch.amp.GradScaler("cuda") if amp_dtype else None
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_steps = 0
        t0 = time.time()
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = outputs.loss / args.gradient_accumulation
                
                loss.backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss / args.gradient_accumulation
                loss.backward()
            
            epoch_loss += outputs.loss.item()
            n_steps += 1
            
            if (step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        avg_train_loss = epoch_loss / max(n_steps, 1)
        
        # Validation
        val_loss = evaluate(model, val_loader, device, amp_dtype)
        
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        
        print(f"  Epoch {epoch+1:>2}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {lr_now:.2e} | "
              f"Time: {elapsed:.0f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            save_path = os.path.join(args.output_dir, "best_model")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    ✓ New best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stopping (patience={args.patience})")
                break
    
    # Save final model
    save_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
    print(f"\n{'='*60}")
    print(f"OD TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best model: {args.output_dir}/best_model")
    print(f"  Final model: {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
