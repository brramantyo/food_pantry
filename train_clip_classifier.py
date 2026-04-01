#!/usr/bin/env python3
"""
Fine-tune OpenAI CLIP for single-label classification of food pantry crops.

Trains classification head on top of frozen CLIP vision encoder.
Single-label classification: each crop = exactly 1 category.

v2 Improvements:
- Class-balanced sampling (WeightedRandomSampler)
- Focal loss for hard/rare examples
- Per-class accuracy logging
- Better augmentation
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma=0 → standard cross-entropy
    gamma=2 → strong focus on hard examples (recommended)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CLIPClassificationHead(nn.Module):
    """Classification head on top of CLIP vision encoder."""

    def __init__(self, vision_model, feature_dim: int = 768, num_classes: int = 21, 
                 freeze_encoder: bool = False, unfreeze_layers: int = -1):
        super().__init__()
        self.vision_model = vision_model
        
        if freeze_encoder or unfreeze_layers == 0:
            # Freeze everything
            for param in self.vision_model.parameters():
                param.requires_grad = False
            logger.info("Encoder: fully frozen (linear probe)")
        elif unfreeze_layers > 0:
            # Freeze all, then unfreeze last N layers
            for param in self.vision_model.parameters():
                param.requires_grad = False
            # Unfreeze last N transformer layers
            encoder_layers = self.vision_model.vision_model.encoder.layers
            total_layers = len(encoder_layers)
            for layer in encoder_layers[total_layers - unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            # Always unfreeze post_layernorm
            for param in self.vision_model.vision_model.post_layernorm.parameters():
                param.requires_grad = True
            logger.info(f"Encoder: unfreezing last {unfreeze_layers}/{total_layers} layers")
        else:
            # Unfreeze everything (full fine-tune)
            logger.info("Encoder: fully unfrozen (full fine-tune)")
        
        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
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
    freeze_encoder: bool = False,
    unfreeze_layers: int = -1,
    bf16: bool = False,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    use_balanced_sampling: bool = True,
) -> None:
    """Main training pipeline with class balancing and focal loss."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CLIP model
    logger.info(f"Loading CLIP model from {model_name}")
    image_processor = CLIPImageProcessor.from_pretrained(model_name)
    vision_model = CLIPVisionModel.from_pretrained(model_name, use_safetensors=True)
    vision_model = vision_model.to(device)

    # Create classification head
    model = CLIPClassificationHead(
        vision_model, feature_dim=vision_model.config.hidden_size, num_classes=21, 
        freeze_encoder=freeze_encoder, unfreeze_layers=unfreeze_layers
    )
    model = model.to(device)

    # Create datasets
    logger.info("Loading training dataset...")
    train_dataset = CropDataset(train_jsonl, crop_dir, image_processor, augment=True)
    
    # Class-balanced sampling
    if use_balanced_sampling:
        logger.info("Computing class-balanced sampling weights...")
        labels = []
        for i in range(len(train_dataset)):
            sample = train_dataset.samples[i]
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
            labels.append(label)
        
        class_counts = Counter(labels)
        total = len(labels)
        # Inverse frequency weighting
        class_weights = {cls: total / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
        
        logger.info("Class distribution:")
        for cls_id, count in sorted(class_counts.items()):
            cat_name = CATEGORIES[cls_id] if cls_id < len(CATEGORIES) else f"Unknown({cls_id})"
            logger.info(f"  {cat_name}: {count} samples (weight: {class_weights[cls_id]:.2f})")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    logger.info("Loading validation dataset...")
    val_dataset = CropDataset(val_jsonl, crop_dir, image_processor, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Loss function
    if use_focal_loss:
        logger.info(f"Using Focal Loss (gamma={focal_gamma})")
        # Compute class weights for focal loss alpha
        if use_balanced_sampling and class_counts:
            alpha = torch.zeros(21)
            for cls_id, count in class_counts.items():
                alpha[cls_id] = total / (21 * count)
            alpha = alpha.to(device)
            criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)
        else:
            criterion = FocalLoss(gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer with differential learning rates
    if freeze_encoder:
        # Only train classifier head
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
    else:
        # Differential LR: lower for encoder, higher for classifier
        encoder_params = [p for p in model.vision_model.parameters() if p.requires_grad]
        classifier_params = list(model.classifier.parameters())
        
        param_groups = [
            {"params": encoder_params, "lr": lr * 0.1},   # 10x lower for encoder
            {"params": classifier_params, "lr": lr},        # full LR for head
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        logger.info(f"Optimizer: encoder LR={lr*0.1:.6f}, head LR={lr:.6f}")
    
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
            logger.info(f"★ New best model at epoch {epoch + 1}: {val_accuracy:.4f}")

    # Final per-class evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Final Per-Class Evaluation:")
    logger.info("=" * 60)
    per_class_eval(model, val_loader, device)

    logger.info(f"\nTraining complete. Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Best model saved to {best_model_path}")


def per_class_eval(model, dataloader, device):
    """Evaluate per-class accuracy."""
    model.eval()
    class_correct = Counter()
    class_total = Counter()
    
    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values)
            preds = logits.argmax(dim=1)
            
            for pred, label in zip(preds, labels):
                class_total[label.item()] += 1
                if pred == label:
                    class_correct[label.item()] += 1
    
    for cls_id in sorted(class_total.keys()):
        cat_name = CATEGORIES[cls_id] if cls_id < len(CATEGORIES) else f"Unknown({cls_id})"
        correct = class_correct[cls_id]
        total = class_total[cls_id]
        acc = correct / total if total > 0 else 0
        logger.info(f"  {cat_name}: {acc:.1%} ({correct}/{total})")


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
    parser.add_argument("--freeze-encoder", action="store_true", default=False, help="Freeze CLIP encoder (default: fine-tune all)")
    parser.add_argument("--unfreeze-layers", type=int, default=-1, help="Number of encoder layers to unfreeze (-1 = all, 0 = none/frozen)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--focal-loss", action="store_true", default=True, help="Use focal loss")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma (higher = more focus on hard examples)")
    parser.add_argument("--balanced-sampling", action="store_true", default=True, help="Use class-balanced sampling")
    parser.add_argument("--method", type=str, default="finetune", choices=["finetune", "linear", "simclr"], help="Training method")

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
        unfreeze_layers=args.unfreeze_layers,
        bf16=args.bf16,
        use_focal_loss=args.focal_loss,
        focal_gamma=args.focal_gamma,
        use_balanced_sampling=args.balanced_sampling,
    )
