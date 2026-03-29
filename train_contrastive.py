#!/usr/bin/env python3
"""
train_contrastive.py
====================
Supervised Contrastive Learning on Florence-2 Vision Encoder features.

Approach:
  1. Use Florence-2-large-ft vision encoder as frozen feature extractor
  2. Add projection head (MLP) + linear classifier
  3. Train with SupCon loss (push same-class embeddings together, 
     different-class apart) + CE classification loss
  4. Multi-label: each image can have multiple categories → treat as
     multi-hot vector, SupCon per positive pair

Architecture:
  Florence-2 Vision Encoder (frozen)
    → Global Average Pool (feature dim)
    → Projection Head (feature_dim → 256 → 128)  [for SupCon loss]
    → Classification Head (feature_dim → 21)       [for CE loss]

Loss = α * SupCon_loss + (1-α) * BCE_loss

Usage:
  python train_contrastive.py \
    --base-model microsoft/Florence-2-large-ft \
    --data-dir . \
    --jsonl-dir ./florence2_data \
    --output-dir ./checkpoints_contrastive \
    --epochs 30 \
    --bf16
"""

import argparse
import json
import os
import random
import sys
import time
import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import get_cosine_schedule_with_warmup


# ── Categories ─────────────────────────────────────────────────────────────────

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

CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
IDX2CAT = {i: c for c, i in CAT2IDX.items()}
NUM_CLASSES = len(CATEGORIES)

CANONICAL_MAP = {c.lower(): c for c in CATEGORIES}


def normalize_category(name):
    return CANONICAL_MAP.get(name.lower().strip(), name)


# ── Augmentation ───────────────────────────────────────────────────────────────

class PantryAugmentation:
    """Data augmentation for contrastive learning — need strong augs for good representations."""
    
    def __init__(self, p=0.8):
        self.p = p
    
    def __call__(self, image):
        if random.random() > self.p:
            return image
        
        if random.random() < 0.5:
            image = TF.hflip(image)
        
        if random.random() < 0.4:
            angle = random.uniform(-25, 25)
            image = TF.rotate(image, angle, fill=128)
        
        if random.random() < 0.6:
            factor = random.uniform(0.5, 1.5)
            image = ImageEnhance.Brightness(image).enhance(factor)
        
        if random.random() < 0.6:
            factor = random.uniform(0.5, 1.5)
            image = ImageEnhance.Contrast(image).enhance(factor)
        
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            image = ImageEnhance.Color(image).enhance(factor)
        
        if random.random() < 0.3:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))
        
        # Random crop + resize
        if random.random() < 0.4:
            w, h = image.size
            crop_frac = random.uniform(0.7, 0.95)
            new_w, new_h = int(w * crop_frac), int(h * crop_frac)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            image = image.crop((left, top, left + new_w, top + new_h))
            image = image.resize((w, h), Image.BILINEAR)
        
        return image


# ── Dataset ────────────────────────────────────────────────────────────────────

class PantryContrastiveDataset(Dataset):
    """
    Dataset that returns 2 augmented views + multi-hot label vector.
    """
    
    def __init__(self, jsonl_path, data_dir, processor, augment=False, 
                 oversample=True, min_samples_per_class=20):
        self.data_dir = data_dir
        self.processor = processor
        self.augment = PantryAugmentation(p=0.8) if augment else None
        
        raw_samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_samples.append(json.loads(line))
        
        self.samples = []
        class_counts = Counter()
        
        for s in raw_samples:
            try:
                target = json.loads(s["target"])
                items = target.get("items", [])
                cats = set()
                for item in items:
                    name = normalize_category(item.get("name", ""))
                    if name in CAT2IDX:
                        cats.add(name)
                
                if cats:
                    self.samples.append({
                        "image": s["image"],
                        "categories": sorted(cats),
                    })
                    for c in cats:
                        class_counts[c] += 1
            except (json.JSONDecodeError, KeyError):
                pass
        
        print(f"  Loaded {len(self.samples)} samples from {jsonl_path}")
        
        if oversample and min_samples_per_class > 0:
            self.samples = self._oversample(self.samples, class_counts, min_samples_per_class)
            print(f"  After oversampling: {len(self.samples)} samples")
    
    def _oversample(self, samples, class_counts, min_samples):
        class_samples = {}
        for s in samples:
            for c in s["categories"]:
                if c not in class_samples:
                    class_samples[c] = []
                class_samples[c].append(s)
        
        extra = []
        for cls, count in class_counts.items():
            if count < min_samples and cls in class_samples:
                needed = min_samples - count
                pool = class_samples[cls]
                for _ in range(needed):
                    extra.append(random.choice(pool))
        
        return samples + extra
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_rel = sample["image"].replace("\\", "/")
        img_path = os.path.join(self.data_dir, img_rel)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.augment:
            view1 = self.augment(image)
            view2 = self.augment(image)
        else:
            view1 = image
            view2 = image
        
        inputs1 = self.processor(text="<OD>", images=view1, return_tensors="pt")
        inputs2 = self.processor(text="<OD>", images=view2, return_tensors="pt")
        
        label = torch.zeros(NUM_CLASSES)
        for c in sample["categories"]:
            label[CAT2IDX[c]] = 1.0
        
        return {
            "pixel_values_1": inputs1["pixel_values"].squeeze(0),
            "pixel_values_2": inputs2["pixel_values"].squeeze(0),
            "input_ids": inputs1["input_ids"].squeeze(0),
            "label": label,
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class ContrastiveClassifier(nn.Module):
    """
    Florence-2 Encoder (frozen) + Projection Head + Classification Head.
    
    Uses the full encoder (vision + text embedding) to get hidden states,
    then pools and classifies.
    """
    
    def __init__(self, florence_model, feature_dim, proj_dim=128, num_classes=21):
        super().__init__()
        
        # Keep full encoder for feature extraction
        self.encoder = florence_model.get_encoder()
        
        # Freeze entire encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.feature_dim = feature_dim
        
        # Projection head for contrastive loss
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes),
        )
    
    def extract_features(self, pixel_values, input_ids):
        """Extract pooled features from encoder."""
        with torch.no_grad():
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                pixel_values=pixel_values,
            )
            hidden = encoder_outputs.last_hidden_state  # (B, seq_len, D)
            features = hidden.mean(dim=1)  # Global average pool → (B, D)
        return features
    
    def forward(self, pixel_values, input_ids):
        features = self.extract_features(pixel_values, input_ids)
        proj = F.normalize(self.projection(features), dim=1)
        logits = self.classifier(features)
        return proj, logits


# ── Supervised Contrastive Loss ────────────────────────────────────────────────

class SupConLossMultiLabel(nn.Module):
    """
    SupCon for multi-label: positive pairs share at least one category.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        label_sim = torch.matmul(labels, labels.T)
        positive_mask = (label_sim > 0).float()
        
        identity = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - identity
        
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()
        
        exp_logits = torch.exp(logits) * (1 - identity)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        num_positives = positive_mask.sum(dim=1)
        valid = num_positives > 0
        
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)
        loss = -mean_log_prob_pos[valid].mean()
        
        return loss


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, supcon_loss_fn, 
                device, amp_dtype, alpha=0.5):
    model.train()
    model.encoder.eval()  # Keep encoder frozen/eval
    
    total_loss = 0
    total_supcon = 0
    total_bce = 0
    n_batches = 0
    
    for batch in dataloader:
        pv1 = batch["pixel_values_1"].to(device)
        pv2 = batch["pixel_values_2"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        
        if amp_dtype:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                proj1, logits1 = model(pv1, input_ids)
                proj2, logits2 = model(pv2, input_ids)
                
                projections = torch.cat([proj1, proj2], dim=0)
                labels_dup = torch.cat([labels, labels], dim=0)
                loss_supcon = supcon_loss_fn(projections, labels_dup)
                
                loss_bce = (F.binary_cross_entropy_with_logits(logits1, labels) +
                            F.binary_cross_entropy_with_logits(logits2, labels)) / 2
                
                loss = alpha * loss_supcon + (1 - alpha) * loss_bce
        else:
            proj1, logits1 = model(pv1, input_ids)
            proj2, logits2 = model(pv2, input_ids)
            
            projections = torch.cat([proj1, proj2], dim=0)
            labels_dup = torch.cat([labels, labels], dim=0)
            loss_supcon = supcon_loss_fn(projections, labels_dup)
            
            loss_bce = (F.binary_cross_entropy_with_logits(logits1, labels) +
                        F.binary_cross_entropy_with_logits(logits2, labels)) / 2
            
            loss = alpha * loss_supcon + (1 - alpha) * loss_bce
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_supcon += loss_supcon.item()
        total_bce += loss_bce.item()
        n_batches += 1
    
    return {
        "loss": total_loss / max(n_batches, 1),
        "supcon": total_supcon / max(n_batches, 1),
        "bce": total_bce / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, dataloader, device, amp_dtype, threshold=0.5):
    model.eval()
    
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        pv = batch["pixel_values_1"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        
        if amp_dtype:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                _, logits = model(pv, input_ids)
        else:
            _, logits = model(pv, input_ids)
        
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        
        all_preds.append(preds.cpu())
        all_targets.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    per_class = {}
    micro_tp = micro_fp = micro_fn = 0
    
    for i, cls in enumerate(CATEGORIES):
        tp = ((all_preds[:, i] == 1) & (all_targets[:, i] == 1)).sum().item()
        fp = ((all_preds[:, i] == 1) & (all_targets[:, i] == 0)).sum().item()
        fn = ((all_preds[:, i] == 0) & (all_targets[:, i] == 1)).sum().item()
        
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        support = int((all_targets[:, i] == 1).sum().item())
        
        per_class[cls] = {"precision": p, "recall": r, "f1": f1, "support": support}
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    
    micro_p = micro_tp / max(micro_tp + micro_fp, 1)
    micro_r = micro_tp / max(micro_tp + micro_fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / NUM_CLASSES
    exact = (all_preds == all_targets).all(dim=1).float().mean().item()
    
    return {
        "micro_p": micro_p,
        "micro_r": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "exact_match": exact,
        "per_class": per_class,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--jsonl-dir", type=str, default="./florence2_data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_contrastive")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--min-samples", type=int, default=20)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else None
    print(f"Device: {device}")
    print(f"Config: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, "
          f"alpha={args.alpha}, temp={args.temperature}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ── Load Florence-2 ────────────────────────────────────────────────────
    print(f"\nLoading Florence-2: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    florence = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    
    config = florence.config
    feature_dim = getattr(config, 'd_model', None) or getattr(config, 'hidden_size', 1024)
    print(f"  Feature dim: {feature_dim}")
    
    model = ContrastiveClassifier(
        florence_model=florence,
        feature_dim=feature_dim,
        proj_dim=args.proj_dim,
        num_classes=NUM_CLASSES,
    ).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {trainable:,} | Frozen: {frozen:,}")
    
    # ── Datasets ───────────────────────────────────────────────────────────
    train_jsonl = os.path.join(args.jsonl_dir, "train_v5.jsonl")
    valid_jsonl = os.path.join(args.jsonl_dir, "valid_v5.jsonl")
    test_jsonl = os.path.join(args.jsonl_dir, "test_v5.jsonl")
    
    print(f"\nLoading datasets...")
    train_ds = PantryContrastiveDataset(
        train_jsonl, args.data_dir, processor,
        augment=True, oversample=True, min_samples_per_class=args.min_samples,
    )
    valid_ds = PantryContrastiveDataset(
        valid_jsonl, args.data_dir, processor,
        augment=False, oversample=False,
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    
    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * 2
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    supcon_loss = SupConLossMultiLabel(temperature=args.temperature)
    
    # ── Training ───────────────────────────────────────────────────────────
    best_f1 = 0
    best_epoch = 0
    history = []
    
    print(f"\n{'='*70}")
    print(f"  TRAINING START — Supervised Contrastive Learning")
    print(f"{'='*70}")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, supcon_loss,
            device, amp_dtype, alpha=args.alpha,
        )
        
        val_metrics = evaluate(model, valid_loader, device, amp_dtype, args.threshold)
        
        elapsed = time.time() - t0
        
        print(f"\n  Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s)")
        print(f"    Train: loss={train_metrics['loss']:.4f} "
              f"(supcon={train_metrics['supcon']:.4f}, bce={train_metrics['bce']:.4f})")
        print(f"    Valid: Micro F1={val_metrics['micro_f1']:.1%}, "
              f"Macro F1={val_metrics['macro_f1']:.1%}, "
              f"Exact={val_metrics['exact_match']:.1%}")
        
        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "valid": {k: v for k, v in val_metrics.items() if k != "per_class"},
        })
        
        if val_metrics["micro_f1"] > best_f1:
            best_f1 = val_metrics["micro_f1"]
            best_epoch = epoch + 1
            
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": {k: v for k, v in model.state_dict().items() 
                                     if "encoder" not in k},
                "metrics": val_metrics,
                "args": vars(args),
                "feature_dim": feature_dim,
            }, save_path)
            print(f"    ★ New best! Saved to {save_path}")
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": {k: v for k, v in model.state_dict().items()
                                     if "encoder" not in k},
                "metrics": val_metrics,
                "feature_dim": feature_dim,
            }, save_path)
    
    # ── Final Report ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best Valid Micro F1: {best_f1:.1%}")
    
    # ── Test evaluation ────────────────────────────────────────────────────
    if os.path.exists(test_jsonl):
        print(f"\n  Loading best model for test evaluation...")
        
        checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"),
                                map_location=device, weights_only=False)
        
        model_eval = ContrastiveClassifier(
            florence_model=florence,
            feature_dim=feature_dim,
            proj_dim=args.proj_dim,
            num_classes=NUM_CLASSES,
        ).to(device)
        
        current_state = model_eval.state_dict()
        saved_state = checkpoint["model_state_dict"]
        current_state.update(saved_state)
        model_eval.load_state_dict(current_state)
        
        test_ds = PantryContrastiveDataset(
            test_jsonl, args.data_dir, processor,
            augment=False, oversample=False,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=2, pin_memory=True)
        
        test_metrics = evaluate(model_eval, test_loader, device, amp_dtype, args.threshold)
        
        print(f"\n{'='*70}")
        print(f"  TEST SET RESULTS")
        print(f"{'='*70}")
        print(f"  Micro P:     {test_metrics['micro_p']:.1%}")
        print(f"  Micro R:     {test_metrics['micro_r']:.1%}")
        print(f"  Micro F1:    {test_metrics['micro_f1']:.1%}")
        print(f"  Macro F1:    {test_metrics['macro_f1']:.1%}")
        print(f"  Exact Match: {test_metrics['exact_match']:.1%}")
        
        print(f"\n  {'Class':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}")
        print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
        for cls in sorted(test_metrics["per_class"].keys()):
            m = test_metrics["per_class"][cls]
            print(f"  {cls:<45} {m['precision']:>5.1%} {m['recall']:>5.1%} {m['f1']:>5.1%} {m['support']:>5}")
        
        # Save full results
        results = {
            "approach": "supervised_contrastive_learning",
            "base_model": args.base_model,
            "feature_dim": feature_dim,
            "proj_dim": args.proj_dim,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "best_epoch": best_epoch,
            "best_valid_f1": best_f1,
            "test_metrics": test_metrics,
            "training_history": history,
        }
        
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path}")
    
    # Save training history
    hist_path = os.path.join(args.output_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training history saved to {hist_path}")


if __name__ == "__main__":
    main()
