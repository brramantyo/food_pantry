#!/usr/bin/env python3
"""
Fine-tune OpenAI CLIP for single-label classification of food pantry crops.

Trains classification head on top of frozen CLIP vision encoder.
Single-label classification: each crop = exactly 1 category.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CATEGORIES = [
    "Baby Food",
    "Beans and Legumes - Canned or Dried",
    "Bread and Bakery Products",
    "Canned Tomato Products",
    "Carbohydrate Meal",
    "Condiments and Sauces",
    "Dairy and Dairy Alternatives",
    "Desserts and Sweets",
    "Drinks",
    "Fresh Fruit",
    "Fruits - Canned or Processed",
    "Granola Products",
    "Meat and Poultry - Canned",
    "Meat and Poultry - Fresh",
    "Nut Butters and Nuts",
    "Ready Meals",
    "Savory Snacks and Crackers",
    "Seafood - Canned",
    "Soup",
    "Vegetables - Canned",
    "Vegetables - Fresh",
]


class CropDataset(Dataset):
    """Single-label crop classification dataset from JSONL."""

    def __init__(self, jsonl_path: str, image_dir: str, image_processor, augment: bool = False):
        self.image_dir = Path(image_dir)
        self.image_processor = image_processor
        self.samples = []

        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        # Build augmentation pipeline
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image_path = self.image_dir / sample["image"]

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = Image.open(image_path).convert("RGB")

        # Apply augmentation if specified
        if self.transform:
            image = self.transform(image)

        # Process image with CLIP processor
        image_inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].squeeze(0)

        # Extract category from target JSON
        target_text = sample.get("target", "{}")
        try:
            target_obj = json.loads(target_text)
            items = target_obj.get("items", [])
            if items:
                category_name = items[0]["name"]
                label = CATEGORIES.index(category_name)
            else:
                label = 0
        except:
            label = 0

        return pixel_values, label


class CLIPClassificationHead(nn.Module):
    """Classification head on top of CLIP vision encoder."""

    def __init__(self, vision_model, feature_dim: int = 768, num_classes: int = 21, freeze_encoder: bool = True):
        super().__init__()
        self.vision_model = vision_model
        if freeze_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns logits."""
        with torch.no_grad() if self.vision_model.training is False else torch.enable_grad():
            vision_outputs = self.vision_model(pixel_values=pixel_values, output_hidden_states=True)
            pooled_output = vision_outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion, 
    optimizer, 
    device: str
) -> float:
    """Train one epoch."""
    model.train()
    total_loss = 0
    for pixel_values, labels in tqdm(dataloader, desc="Training"):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_epoch(model: nn.Module, dataloader: DataLoader, device: str) -> Tuple[float, float]:
    """Validate one epoch. Returns accuracy and loss."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for pixel_values, labels in tqdm(dataloader, desc="Validating"):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            logits = model(pixel_values)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss


def train_clip_classifier(
    crop_dir: str,
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    model_name: str = "openai/clip-vit-base-patch32",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    freeze_encoder: bool = True,
    bf16: bool = False,
) -> None:
    """Main training pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP model
    logger.info(f"Loading CLIP model from {model_name}")
    image_processor = CLIPImageProcessor.from_pretrained(model_name)
    vision_model = CLIPVisionModel.from_pretrained(model_name)
    vision_model = vision_model.to(device)

    # Create classification head
    model = CLIPClassificationHead(
        vision_model, feature_dim=vision_model.config.hidden_size, num_classes=21, freeze_encoder=freeze_encoder
    )
    model = model.to(device)

    # Create datasets
    logger.info("Loading training dataset...")
    train_dataset = CropDataset(train_jsonl, crop_dir, image_processor, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    logger.info("Loading validation dataset...")
    val_dataset = CropDataset(val_jsonl, crop_dir, image_processor, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_accuracy = 0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_accuracy, val_loss = validate_epoch(model, val_loader, device)

        logger.info(f"Train Loss: {train_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step()

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model at epoch {epoch + 1} with accuracy {val_accuracy:.4f}")

    logger.info(f"\nTraining complete. Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CLIP for crop classification")
    parser.add_argument("--crop-dir", default="crop_data/", help="Crop image directory")
    parser.add_argument("--train-jsonl", default="crop_data/train_crops.jsonl", help="Training JSONL file")
    parser.add_argument("--val-jsonl", default="crop_data/val_crops.jsonl", help="Validation JSONL file")
    parser.add_argument("--output-dir", default="clip_output/", help="Output directory")
    parser.add_argument(
        "--model", 
        default="openai/clip-vit-base-patch32",
        help="CLIP model name (openai/clip-vit-base-patch32 or openai/clip-vit-large-patch14)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--freeze-encoder", action="store_true", default=True, help="Freeze CLIP encoder")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")

    args = parser.parse_args()

    train_clip_classifier(
        crop_dir=args.crop_dir,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        bf16=args.bf16,
    )
