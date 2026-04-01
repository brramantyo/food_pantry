#!/usr/bin/env python3
"""
train_dinov2_classifier.py
==========================
Fine-tune DINOv2 for food pantry crop classification.

DINOv2 is a self-supervised vision model that produces excellent features
for fine-grained visual classification — better than CLIP for this task.

Features:
- DINOv2 ViT-L/14 backbone (frozen or fine-tuned)
- MixUp + CutMix augmentation
- Label smoothing
- Focal loss + class-balanced sampling
- Differential learning rates
- Per-class accuracy logging
- Gradient accumulation for large models

Usage:
    python train_dinov2_classifier.py \
        --crop-dir crop_data/ \
        --train-jsonl crop_data/train_crops.jsonl \
        --val-jsonl crop_data/val_crops.jsonl \
        --output-dir dinov2_output/ \
        --model dinov2_vitl14 \
        --epochs 30
"""

import argparse
import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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

NUM_CLASSES = len(CATEGORIES)


# ============================================================
# Augmentation: MixUp + CutMix
# ============================================================

def mixup_data(x, y, alpha=0.2):
    """MixUp: blend two images and their labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: cut and paste patches between images."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    
    # Adjust lambda based on actual area
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixed loss for MixUp/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.alpha, 
            label_smoothing=self.label_smoothing, reduction='none'
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


# ============================================================
# Dataset
# ============================================================

class CropDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, transform=None, img_size=224):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.img_size = img_size
        self.samples = []
        self.labels = []
        
        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                # Extract label
                target_text = sample.get("target", "{}")
                try:
                    target_obj = json.loads(target_text)
                    items = target_obj.get("items", [])
                    if items:
                        category_name = items[0]["name"]
                        if category_name in CATEGORIES:
                            label = CATEGORIES.index(category_name)
                            self.samples.append(sample)
                            self.labels.append(label)
                except:
                    pass
        
        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        
        image_path = self.image_dir / sample["image"]
        if not image_path.exists():
            # Try normalizing path
            image_path = self.image_dir / sample["image"].replace("\\", "/")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ============================================================
# DINOv2 Classification Model
# ============================================================

class DINOv2Classifier(nn.Module):
    def __init__(self, model_name="dinov2_vitl14", num_classes=21, 
                 freeze_encoder=False, unfreeze_layers=-1):
        super().__init__()
        
        # Load DINOv2 from torch hub
        logger.info(f"Loading DINOv2: {model_name}")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # Get feature dimension
        self.feature_dim = self.backbone.embed_dim
        logger.info(f"Feature dim: {self.feature_dim}")
        
        # Freeze/unfreeze logic
        if freeze_encoder or unfreeze_layers == 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone: fully frozen")
        elif unfreeze_layers > 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze last N blocks
            total_blocks = len(self.backbone.blocks)
            for block in self.backbone.blocks[total_blocks - unfreeze_layers:]:
                for param in block.parameters():
                    param.requires_grad = True
            # Unfreeze norm
            if hasattr(self.backbone, 'norm'):
                for param in self.backbone.norm.parameters():
                    param.requires_grad = True
            logger.info(f"Backbone: unfreezing last {unfreeze_layers}/{total_blocks} blocks")
        else:
            logger.info("Backbone: fully unfrozen")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim // 4, num_classes),
        )
    
    def forward(self, x):
        # DINOv2 returns CLS token features
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ============================================================
# Training
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device, 
                use_mixup=True, use_cutmix=True, grad_accum=1):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply MixUp or CutMix randomly
        use_mix = random.random() < 0.5 and (use_mixup or use_cutmix)
        
        if use_mix:
            if use_cutmix and random.random() < 0.5:
                images, labels_a, labels_b, lam = cutmix_data(images, labels)
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels)
            
            logits = model(images)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            
            # Accuracy (use original labels for tracking)
            preds = logits.argmax(dim=1)
            total_correct += (lam * (preds == labels_a).float() + 
                            (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
        
        total_samples += labels.size(0)
        
        # Gradient accumulation
        loss = loss / grad_accum
        loss.backward()
        
        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum
    
    # Final step if not aligned
    if (batch_idx + 1) % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def validate_epoch(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss, all_preds, all_labels


def per_class_eval(all_preds, all_labels):
    """Print per-class accuracy."""
    class_correct = Counter()
    class_total = Counter()
    
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    results = {}
    for cls_id in sorted(class_total.keys()):
        cat_name = CATEGORIES[cls_id] if cls_id < len(CATEGORIES) else f"Unknown({cls_id})"
        correct = class_correct[cls_id]
        total = class_total[cls_id]
        acc = correct / total if total > 0 else 0
        results[cat_name] = {"accuracy": acc, "correct": correct, "total": total}
        logger.info(f"  {cat_name}: {acc:.1%} ({correct}/{total})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Food Pantry Classifier")
    parser.add_argument("--crop-dir", default="crop_data/", help="Crop image directory")
    parser.add_argument("--train-jsonl", default="crop_data/train_crops.jsonl")
    parser.add_argument("--val-jsonl", default="crop_data/val_crops.jsonl")
    parser.add_argument("--output-dir", default="dinov2_output/")
    parser.add_argument("--model", default="dinov2_vitl14", 
                        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="DINOv2 model size (s=small, b=base, l=large, g=giant)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--freeze-encoder", action="store_true", default=False)
    parser.add_argument("--unfreeze-layers", type=int, default=6, help="Layers to unfreeze (-1=all)")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", action="store_true", default=True)
    parser.add_argument("--cutmix", action="store_true", default=True)
    parser.add_argument("--no-mixup", action="store_true", help="Disable MixUp")
    parser.add_argument("--no-cutmix", action="store_true", help="Disable CutMix")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--balanced-sampling", action="store_true", default=True)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_dataset = CropDataset(args.train_jsonl, args.crop_dir, train_transform, args.img_size)
    val_dataset = CropDataset(args.val_jsonl, args.crop_dir, val_transform, args.img_size)
    
    # Class-balanced sampling
    if args.balanced_sampling:
        class_counts = Counter(train_dataset.labels)
        total = len(train_dataset.labels)
        class_weights = {cls: total / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_dataset.labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, 
                                  num_workers=4, pin_memory=True, drop_last=True)
        
        logger.info("Class distribution:")
        for cls_id in sorted(class_counts.keys()):
            logger.info(f"  {CATEGORIES[cls_id]}: {class_counts[cls_id]} (weight: {class_weights[cls_id]:.2f})")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model
    model = DINOv2Classifier(
        model_name=args.model,
        num_classes=NUM_CLASSES,
        freeze_encoder=args.freeze_encoder,
        unfreeze_layers=args.unfreeze_layers,
    ).to(device)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    # Loss with label smoothing + focal
    if args.balanced_sampling:
        alpha = torch.zeros(NUM_CLASSES)
        class_counts_dict = Counter(train_dataset.labels)
        total = len(train_dataset.labels)
        for cls_id, count in class_counts_dict.items():
            alpha[cls_id] = total / (NUM_CLASSES * count)
        alpha = alpha.to(device)
    else:
        alpha = None
    
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    
    # Optimizer with differential LR
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    classifier_params = list(model.classifier.parameters())
    
    if backbone_params:
        param_groups = [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": classifier_params, "lr": args.lr},
        ]
    else:
        param_groups = [{"params": classifier_params, "lr": args.lr}]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    
    # Cosine annealing with warmup
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    best_accuracy = 0
    best_model_path = output_dir / "best_model.pt"
    patience = 10
    patience_counter = 0
    
    use_mixup = args.mixup and not args.no_mixup
    use_cutmix = args.cutmix and not args.no_cutmix
    logger.info(f"MixUp: {use_mixup}, CutMix: {use_cutmix}, Label Smoothing: {args.label_smoothing}")
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=use_mixup, use_cutmix=use_cutmix, 
            grad_accum=args.grad_accum
        )
        
        val_acc, val_loss, all_preds, all_labels = validate_epoch(model, val_loader, device)
        
        scheduler.step()
        
        current_lr = optimizer.param_groups[-1]['lr']
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'args': vars(args),
            }, best_model_path)
            logger.info(f"★ New best: {val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Final evaluation with best model
    logger.info("\n" + "=" * 60)
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_acc, val_loss, all_preds, all_labels = validate_epoch(model, val_loader, device)
    
    logger.info(f"\nBest Model Accuracy: {val_acc:.4f} (epoch {checkpoint['epoch'] + 1})")
    logger.info("\nPer-Class Results:")
    logger.info("=" * 60)
    per_class_results = per_class_eval(all_preds, all_labels)
    
    # Save results
    results = {
        "best_accuracy": best_accuracy,
        "best_epoch": checkpoint['epoch'] + 1,
        "model": args.model,
        "per_class": per_class_results,
        "args": vars(args),
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Best model saved to {best_model_path}")


if __name__ == "__main__":
    main()
