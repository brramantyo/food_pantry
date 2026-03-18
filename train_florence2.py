#!/usr/bin/env python3
"""
train_florence2.py
==================
Fine-tune Florence-2 for structured pantry food detection using LoRA.

Usage (single GPU):
    python train_florence2.py \
        --data-dir . \
        --jsonl-dir ./florence2_data \
        --output-dir ./checkpoints \
        --model microsoft/Florence-2-base-ft \
        --epochs 10 \
        --batch-size 4 \
        --lr 1e-4

Usage (SLURM - Auburn cluster example):
    srun --partition=general --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=04:00:00 --pty bash
    source ~/food_pantry/venv/bin/activate
    cd ~/food_pantry
    python train_florence2.py --data-dir . --jsonl-dir ./florence2_data --output-dir ./checkpoints

Requirements:
    pip install torch torchvision transformers peft pillow accelerate datasets einops timm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


# ── Dataset ────────────────────────────────────────────────────────────────────

class Florence2PantryDataset(Dataset):
    """
    Dataset for Florence-2 fine-tuning from JSONL files.
    
    Each JSONL line has:
        {"image": "train/img.jpg", "prompt": "<STRUCTURED_PANTRY_OUTPUT>", "target": "{...}"}
    """

    def __init__(self, jsonl_path: str, data_dir: str, processor, max_length: int = 1024):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_dir, sample["image"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to load {img_path}: {e}", file=sys.stderr)
            # Return a blank image as fallback
            image = Image.new("RGB", (640, 480), (0, 0, 0))

        prompt = sample["prompt"]
        target = sample["target"]

        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Process labels (target text)
        labels = self.processor.tokenizer(
            target,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Squeeze batch dimension
        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)

        # Set padding tokens to -100 so they're ignored in loss
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": label_ids,
        }


def collate_fn(batch):
    """Custom collate function to stack batch items."""
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, total_epochs):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            avg_loss = total_loss / (batch_idx + 1)
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch [{epoch+1}/{total_epochs}] "
                  f"Step [{batch_idx+1}/{num_batches}] "
                  f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                  f"LR: {lr:.2e}")

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels,
        )

        total_loss += outputs.loss.item()

    return total_loss / num_batches


@torch.no_grad()
def run_inference_sample(model, processor, image_path, device, prompt="<STRUCTURED_PANTRY_OUTPUT>"):
    """Run inference on a single image and print the result."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        early_stopping=True,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Florence-2 for food pantry structured detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data-dir", type=str, default=".",
                        help="Root directory containing train/valid/test image folders.")
    parser.add_argument("--jsonl-dir", type=str, default="./florence2_data",
                        help="Directory containing train.jsonl, valid.jsonl, test.jsonl.")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-base-ft",
                        help="HuggingFace model name. Options: Florence-2-base-ft, Florence-2-large-ft")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size per GPU.")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Evaluation batch size.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Fraction of total steps for LR warmup.")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max token sequence length.")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader num_workers.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--save-every", type=int, default=2,
                        help="Save checkpoint every N epochs.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision (fp16) training.")
    parser.add_argument("--bf16", action="store_true",
                        help="Use mixed precision (bf16) training. Preferred on A100.")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save VRAM.")

    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Determine mixed precision dtype
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

    # ── Load Model & Processor ─────────────────────────────────────────────

    print(f"\nLoading model: {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=amp_dtype if use_amp else torch.float32,
    )

    # Add the custom prompt token to the tokenizer
    special_tokens = {"additional_special_tokens": ["<STRUCTURED_PANTRY_OUTPUT>"]}
    num_added = processor.tokenizer.add_special_tokens(special_tokens)
    if num_added > 0:
        model.resize_token_embeddings(len(processor.tokenizer))
        print(f"  Added {num_added} special token(s) to tokenizer")

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    # ── Apply LoRA ─────────────────────────────────────────────────────────

    print(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")

    # Find linear layers to apply LoRA to
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "out_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # ── Datasets ───────────────────────────────────────────────────────────

    print("\nLoading datasets...")
    train_dataset = Florence2PantryDataset(
        jsonl_path=os.path.join(args.jsonl_dir, "train.jsonl"),
        data_dir=args.data_dir,
        processor=processor,
        max_length=args.max_length,
    )
    valid_dataset = Florence2PantryDataset(
        jsonl_path=os.path.join(args.jsonl_dir, "valid.jsonl"),
        data_dir=args.data_dir,
        processor=processor,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Optimizer & Scheduler ──────────────────────────────────────────────

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    print(f"\nTraining config:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Valid samples: {len(valid_dataset)}")

    # ── Training Loop ──────────────────────────────────────────────────────

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # Train
        if use_amp:
            # AMP training
            model.train()
            total_loss = 0.0
            num_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    outputs = model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = outputs.loss

                if amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # bf16 doesn't need scaler
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                    avg_loss = total_loss / (batch_idx + 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch [{epoch+1}/{args.epochs}] "
                          f"Step [{batch_idx+1}/{num_batches}] "
                          f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                          f"LR: {lr:.2e}")

            train_loss = total_loss / num_batches
        else:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, epoch, args.epochs
            )

        # Validate
        val_loss = evaluate(model, valid_loader, device)
        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Valid Loss: {val_loss:.4f}")
        print(f"    Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_model")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    ✓ New best model saved to {save_path}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    ✓ Checkpoint saved to {save_path}")

        print()

    # ── Save Final Model ───────────────────────────────────────────────────

    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    print(f"{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model saved to: {os.path.join(args.output_dir, 'best_model')}")
    print(f"  Final model saved to: {final_path}")

    # ── Quick Inference Test ───────────────────────────────────────────────

    print(f"\n{'='*60}")
    print("INFERENCE TEST (sample from validation set)")
    print(f"{'='*60}")

    # Grab first sample from validation JSONL
    test_jsonl = os.path.join(args.jsonl_dir, "valid.jsonl")
    with open(test_jsonl, "r") as f:
        test_sample = json.loads(f.readline())

    test_img = os.path.join(args.data_dir, test_sample["image"])
    if os.path.exists(test_img):
        result = run_inference_sample(model, processor, test_img, device)
        print(f"  Image: {test_sample['image']}")
        print(f"  Expected: {test_sample['target'][:200]}...")
        print(f"  Predicted: {result[:200]}...")
    else:
        print(f"  [SKIP] Test image not found: {test_img}")


if __name__ == "__main__":
    main()
