#!/usr/bin/env python3
"""
train_florence2_v2.py
=====================
Fine-tune Florence-2 for structured pantry food detection using LoRA.

KEY DIFFERENCE from v1: Uses Florence-2's native processor._construct_prompts
and generates labels properly using the processor's built-in methods.

Usage:
    python train_florence2_v2.py \
        --data-dir . \
        --jsonl-dir ./florence2_data \
        --output-dir ./checkpoints_v2 \
        --model microsoft/Florence-2-base-ft \
        --epochs 10 \
        --batch-size 4 \
        --lr 1e-4 \
        --bf16

Requirements:
    pip install torch torchvision transformers peft pillow accelerate einops timm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model


# ── Constants ──────────────────────────────────────────────────────────────────

TASK_PROMPT = "<OD>"  # Use Florence-2's built-in Object Detection task token


# ── Dataset ────────────────────────────────────────────────────────────────────

class Florence2PantryDataset(Dataset):
    """
    Dataset for Florence-2 fine-tuning.
    
    Uses Florence-2's processor to properly encode both prompt and target,
    ensuring the model learns to generate structured output.
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
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(self.data_dir, img_rel)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [WARN] Failed to load {img_path}: {e}", file=sys.stderr)
            image = Image.new("RGB", (640, 480), (0, 0, 0))

        target = sample["target"]

        # Use processor to encode the prompt (image + task token)
        inputs = self.processor(
            text=TASK_PROMPT,
            images=image,
            return_tensors="pt",
        )

        # Encode the target text as labels
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

        # Mask padding tokens in labels
        label_ids[label_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "labels": label_ids,
        }


def collate_fn(batch):
    """Custom collate — pad input_ids to same length within batch."""
    # Find max input_ids length in this batch
    max_input_len = max(x["input_ids"].shape[0] for x in batch)
    
    padded_input_ids = []
    for x in batch:
        ids = x["input_ids"]
        pad_len = max_input_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_len), value=1)  # pad_token_id=1 for Florence-2
        padded_input_ids.append(ids)

    return {
        "input_ids": torch.stack(padded_input_ids),
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


# ── Training ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, device, amp_dtype=None):
    """Evaluate on validation set and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if amp_dtype is not None:
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

    return total_loss / num_batches


@torch.no_grad()
def run_inference_samples(model, processor, samples, data_dir, device, amp_dtype=None, n=5):
    """Run inference on a few samples and print results."""
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
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    early_stopping=True,
                )
        else:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                early_stopping=True,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        # Also decode skipping special tokens
        clean_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\n  Sample {i+1}: {img_rel}")
        print(f"    Target:    {sample['target'][:120]}...")
        print(f"    Predicted: {clean_text[:120]}...")
        print(f"    Raw:       {generated_text[:120]}...")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Florence-2 for food pantry detection (v2 - fixed prompting).",
    )
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl-dir", type=str, default="./florence2_data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_v2")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-base-ft")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
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

    # ── Load Model ─────────────────────────────────────────────────────────

    print(f"\nLoading model: {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=amp_dtype if use_amp else torch.float32,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    # ── Apply LoRA ─────────────────────────────────────────────────────────

    print(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")

    # Target all attention projections in both vision and language components
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "out_proj",
                        "fc1", "fc2"],  # Also target FFN layers
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

    # Load a few valid samples for inference checks
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

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    print(f"\nTraining config:")
    print(f"  Task prompt: {TASK_PROMPT}")
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
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
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
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            else:
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss
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

        # Run inference check every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"\n  Inference check (epoch {epoch+1}):")
            run_inference_samples(
                model, processor, valid_samples_raw, args.data_dir,
                device, amp_dtype=amp_dtype if use_amp else None, n=3,
            )

        print()

    # ── Final Save ─────────────────────────────────────────────────────────

    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    print(f"{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best model saved to: {os.path.join(args.output_dir, 'best_model')}")
    print(f"  Final model saved to: {final_path}")

    # Final inference test
    print(f"\n{'='*60}")
    print("FINAL INFERENCE TEST")
    print(f"{'='*60}")
    run_inference_samples(
        model, processor, valid_samples_raw, args.data_dir,
        device, amp_dtype=amp_dtype if use_amp else None, n=5,
    )


if __name__ == "__main__":
    main()
