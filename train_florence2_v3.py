#!/usr/bin/env python3
"""
train_florence2_v3.py
=====================
Florence-2-large fine-tuning with all improvements:
  1. Florence-2-large-ft (770M params vs 230M)
  2. Higher LoRA rank (r=64, alpha=128) + more target modules
  3. Data augmentation for underrepresented classes (oversampling + image transforms)
  4. 25 epochs with early stopping patience
  5. Better learning rate scheduling

Usage:
    python train_florence2_v3.py \
        --data-dir . \
        --jsonl-dir ./florence2_data \
        --output-dir ./checkpoints_v3 \
        --epochs 25 \
        --bf16

Requirements:
    pip install torch torchvision transformers peft pillow accelerate einops timm
"""

import argparse
import json
import os
import sys
import time
import random
from collections import Counter
from pathlib import Path

import torch
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


# ── Data Augmentation ──────────────────────────────────────────────────────────

class PantryAugmentation:
    """Image augmentations for food pantry images."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image

        if random.random() < 0.5:
            image = TF.hflip(image)

        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, fill=128)

        if random.random() < 0.4:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() < 0.4:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Contrast(image).enhance(factor)

        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Color(image).enhance(factor)

        if random.random() < 0.15:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        if random.random() < 0.2:
            w, h = image.size
            crop_frac = random.uniform(0.85, 0.95)
            new_w, new_h = int(w * crop_frac), int(h * crop_frac)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            image = image.crop((left, top, left + new_w, top + new_h))
            image = image.resize((w, h), Image.BILINEAR)

        return image


# ── Dataset with Oversampling ──────────────────────────────────────────────────

class Florence2PantryDatasetV3(Dataset):
    """
    Dataset with class-balanced oversampling and augmentation.
    Underrepresented classes are repeated more often.
    """

    def __init__(self, jsonl_path, data_dir, processor, max_length=1024,
                 augment=False, oversample=False, min_samples_per_class=15):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.augment = PantryAugmentation(p=0.6) if augment else None

        # Load raw samples
        raw_samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_samples.append(json.loads(line))

        print(f"  Loaded {len(raw_samples)} raw samples from {jsonl_path}")

        if oversample:
            self.samples = self._oversample(raw_samples, min_samples_per_class)
            print(f"  After oversampling: {len(self.samples)} samples")
        else:
            self.samples = raw_samples

    def _oversample(self, samples, min_samples_per_class):
        """Oversample underrepresented classes to at least min_samples_per_class."""
        # Count primary class per sample (first item in the items list)
        class_samples = {}
        for s in samples:
            try:
                target = json.loads(s["target"])
                items = target.get("items", [])
                if items:
                    primary_class = items[0]["name"]
                else:
                    primary_class = "__empty__"
            except (json.JSONDecodeError, KeyError):
                primary_class = "__unknown__"

            if primary_class not in class_samples:
                class_samples[primary_class] = []
            class_samples[primary_class].append(s)

        # Print class distribution
        print(f"  Class distribution (before oversampling):")
        for cls in sorted(class_samples.keys()):
            count = len(class_samples[cls])
            marker = " ← UNDERREPRESENTED" if count < min_samples_per_class else ""
            print(f"    {cls}: {count}{marker}")

        # Oversample
        result = list(samples)  # Keep all originals
        for cls, cls_samples in class_samples.items():
            if cls in ("__empty__", "__unknown__"):
                continue
            if len(cls_samples) < min_samples_per_class:
                needed = min_samples_per_class - len(cls_samples)
                # Repeat samples with slight randomization
                extras = []
                for i in range(needed):
                    extras.append(cls_samples[i % len(cls_samples)])
                result.extend(extras)
                print(f"    Oversampled {cls}: +{needed} (total: {len(cls_samples) + needed})")

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

        # Apply augmentation during training
        if self.augment is not None:
            image = self.augment(image)

        target = sample["target"]

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
def evaluate(model, dataloader, device, amp_dtype=None):
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
        else:
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()
    return total_loss / num_batches


@torch.no_grad()
def run_inference_samples(model, processor, samples, data_dir, device, amp_dtype=None, n=5):
    model.eval()
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
                    max_new_tokens=1024, num_beams=3, early_stopping=True,
                )
        else:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=1024, num_beams=3, early_stopping=True,
            )
        clean_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Parse and compare
        target = json.loads(sample["target"])
        target_classes = {item["name"] for item in target.get("items", [])}
        try:
            pred = json.loads(clean_text)
            pred_classes = {item["name"] for item in pred.get("items", [])}
        except (json.JSONDecodeError, AttributeError):
            pred_classes = set()

        match = "✓" if target_classes == pred_classes else "✗"

        print(f"\n  Sample {i+1}: {img_rel}")
        print(f"    Target:    {sample['target'][:100]}...")
        print(f"    Predicted: {clean_text[:100]}...")
        print(f"    Classes:   {match} Target={target_classes} | Pred={pred_classes}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Florence-2-large fine-tuning v3")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl-dir", type=str, default="./florence2_data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_v3")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size (smaller for large model)")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Peak learning rate (lower for large model)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min-samples-per-class", type=int, default=15,
                        help="Minimum samples per class after oversampling")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * accum)")

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

    # ── Load Model ─────────────────────────────────────────────────────────

    print(f"\nLoading model: {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=amp_dtype if use_amp else torch.float32,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    # ── Apply LoRA ─────────────────────────────────────────────────────────

    print(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj", "out_proj",
            "fc1", "fc2",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # ── Datasets ───────────────────────────────────────────────────────────

    print("\nLoading datasets...")
    train_dataset = Florence2PantryDatasetV3(
        jsonl_path=os.path.join(args.jsonl_dir, "train.jsonl"),
        data_dir=args.data_dir,
        processor=processor,
        max_length=args.max_length,
        augment=True,
        oversample=True,
        min_samples_per_class=args.min_samples_per_class,
    )
    valid_dataset = Florence2PantryDatasetV3(
        jsonl_path=os.path.join(args.jsonl_dir, "valid.jsonl"),
        data_dir=args.data_dir,
        processor=processor,
        max_length=args.max_length,
        augment=False,
        oversample=False,
    )

    valid_samples_raw = []
    with open(os.path.join(args.jsonl_dir, "valid.jsonl"), "r") as f:
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
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\nTraining config:")
    print(f"  Model: {args.model}")
    print(f"  LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Total optimizer steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)}")

    # ── Training Loop ──────────────────────────────────────────────────────

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print("STARTING TRAINING")
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

            if use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation
                loss.backward()
            else:
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / args.gradient_accumulation
                loss.backward()

            total_loss += outputs.loss.item()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                avg_loss = total_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch [{epoch+1}/{args.epochs}] "
                      f"Step [{batch_idx+1}/{num_batches}] "
                      f"Loss: {outputs.loss.item():.4f} (avg: {avg_loss:.4f}) "
                      f"LR: {lr:.2e}")

        train_loss = total_loss / num_batches

        # Validate
        val_loss = evaluate(model, valid_loader, device, amp_dtype=amp_dtype if use_amp else None)
        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Valid Loss: {val_loss:.4f}")
        print(f"    Time: {epoch_time:.1f}s")

        # Save best
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

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    ✓ Checkpoint saved")

        # Inference check every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\n  Inference check (epoch {epoch+1}):")
            run_inference_samples(
                model, processor, valid_samples_raw, args.data_dir,
                device, amp_dtype=amp_dtype if use_amp else None, n=5,
            )

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n  ⚠ Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

        print()

    # ── Final Save ─────────────────────────────────────────────────────────

    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model: {os.path.join(args.output_dir, 'best_model')}")
    print(f"  Final model: {final_path}")

    # Final inference
    print(f"\n{'='*60}")
    print("FINAL INFERENCE TEST")
    print(f"{'='*60}")
    run_inference_samples(
        model, processor, valid_samples_raw, args.data_dir,
        device, amp_dtype=amp_dtype if use_amp else None, n=8,
    )


if __name__ == "__main__":
    main()
