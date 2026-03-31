#!/usr/bin/env python3
"""
OD → Crop → Contrastive Classifier Pipeline

Uses EXISTING fine-tuned OD model for detection,
then EXISTING contrastive model for classification of each crop.

Key difference from evaluate_od_classify.py: uses contrastive discriminative classifier
instead of Florence-2 generative. Contrastive model outputs sigmoid probabilities per class
(threshold-based), reducing false positives on single-item crops.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score, precision_recall_fscore_support

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


class ContrastiveClassifier(nn.Module):
    """Contrastive discriminative classifier on top of Florence-2 encoder."""

    def __init__(self, florence_model, feature_dim, proj_dim=128, num_classes=21):
        super().__init__()
        self.florence = florence_model
        for param in self.florence.parameters():
            param.requires_grad = False
        self.feature_dim = feature_dim
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def extract_features(self, pixel_values, input_ids):
        """Extract features from Florence-2 encoder."""
        with torch.no_grad():
            dummy_decoder_ids = torch.zeros(
                (pixel_values.shape[0], 1), dtype=torch.long, device=pixel_values.device
            )
            outputs = self.florence(
                input_ids=input_ids,
                pixel_values=pixel_values,
                decoder_input_ids=dummy_decoder_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            if (
                hasattr(outputs, "encoder_last_hidden_state")
                and outputs.encoder_last_hidden_state is not None
            ):
                encoder_hidden = outputs.encoder_last_hidden_state
            elif (
                hasattr(outputs, "encoder_hidden_states")
                and outputs.encoder_hidden_states is not None
            ):
                encoder_hidden = outputs.encoder_hidden_states[-1]
            else:
                raise RuntimeError("Cannot find encoder hidden states")
            features = encoder_hidden.mean(dim=1)
        return features

    def forward(self, pixel_values, input_ids):
        """Forward pass: return normalized projection and logits."""
        features = self.extract_features(pixel_values, input_ids)
        proj = F.normalize(self.projection(features), dim=1)
        logits = self.classifier(features)
        return proj, logits


def parse_od_bboxes(text: str, img_width: int, img_height: int) -> List[List[float]]:
    """Parse OD bboxes from Florence-2 format."""
    bboxes = []
    pattern = r"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    for m in re.finditer(pattern, text):
        x1 = int(m.group(1)) / 999 * img_width
        y1 = int(m.group(2)) / 999 * img_height
        x2 = int(m.group(3)) / 999 * img_width
        y2 = int(m.group(4)) / 999 * img_height
        if x2 > x1 and y2 > y1:
            bboxes.append([x1, y1, x2, y2])
    return bboxes


def nms(bboxes: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Non-maximum suppression."""
    if len(bboxes) == 0:
        return []
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(areas)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def crop_bbox(image: Image.Image, bbox: List[float], padding: float = 0.15) -> Image.Image:
    """Crop image with bbox and padding."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = w * padding
    pad_y = h * padding
    x1_padded = max(0, x1 - pad_x)
    y1_padded = max(0, y1 - pad_y)
    x2_padded = min(image.width, x2 + pad_x)
    y2_padded = min(image.height, y2 + pad_y)
    return image.crop((x1_padded, y1_padded, x2_padded, y2_padded))


def classify_with_contrastive(
    image: Image.Image,
    contrastive_model: nn.Module,
    image_processor,
    tokenizer,
    threshold: float = 0.5,
    device: str = "cuda",
) -> Dict[str, float]:
    """Classify image with contrastive model. Returns dict of category -> probability."""
    image_inputs = image_processor(images=image, return_tensors="pt").to(device)
    text_inputs = tokenizer(CATEGORIES, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        _, logits = contrastive_model(
            pixel_values=image_inputs["pixel_values"],
            input_ids=text_inputs["input_ids"],
        )
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    results = {}
    for i, cat in enumerate(CATEGORIES):
        if probs[i] >= threshold:
            results[cat] = float(probs[i])
    return results


def run_od_pipeline(
    image: Image.Image,
    od_model,
    image_processor,
    tokenizer,
    device: str = "cuda",
) -> List[List[float]]:
    """Run OD on image, return deduplicated bboxes."""
    image_inputs = image_processor(images=image, return_tensors="pt").to(device)
    text_inputs = tokenizer(["<OD>"], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = od_model.generate(
            input_ids=text_inputs["input_ids"],
            pixel_values=image_inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    bboxes = parse_od_bboxes(text, image.width, image.height)
    if len(bboxes) == 0:
        return []

    bboxes_np = np.array(bboxes)
    keep_indices = nms(bboxes_np, iou_threshold=0.5)
    return [bboxes[i] for i in keep_indices]


def evaluate_od_contrastive(
    data_dir: str,
    jsonl_file: str,
    base_model: str,
    od_checkpoint: str,
    contrastive_checkpoint: str,
    output_file: str,
    threshold: float = 0.5,
    crop_threshold: float = 0.6,
    bf16: bool = False,
) -> None:
    """Main evaluation pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load OD model (Florence-2 LoRA)
    logger.info(f"Loading OD model from {base_model} with LoRA {od_checkpoint}")
    image_processor = AutoImageProcessor.from_pretrained(base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    od_model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=True, torch_dtype=torch.float16 if bf16 else torch.float32
    ).to(device)

    # Load LoRA for OD if provided
    if od_checkpoint and od_checkpoint != "none":
        try:
            from peft import PeftModel
            od_model = PeftModel.from_pretrained(od_model, od_checkpoint)
            logger.info(f"Loaded LoRA from {od_checkpoint}")
        except Exception as e:
            logger.warning(f"Could not load LoRA: {e}")

    # Load contrastive model
    logger.info(f"Loading contrastive model from {contrastive_checkpoint}")
    contrastive_model = ContrastiveClassifier(od_model, feature_dim=768, num_classes=21)
    ckpt = torch.load(contrastive_checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        contrastive_model.load_state_dict(ckpt["model_state_dict"])
    else:
        contrastive_model.load_state_dict(ckpt)
    contrastive_model = contrastive_model.to(device)
    contrastive_model.eval()

    # Load JSONL test data
    logger.info(f"Loading test data from {jsonl_file}")
    test_samples = []
    with open(jsonl_file) as f:
        for line in f:
            test_samples.append(json.loads(line))

    # Run evaluation
    results_full = []
    results_crops = []
    results_union = []
    all_true_labels = []
    all_pred_full = []
    all_pred_crops = []
    all_pred_union = []

    for sample in tqdm(test_samples, desc="Evaluating"):
        image_path = Path(data_dir) / sample["image"]
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")

        # Parse ground truth
        target_text = sample.get("target", "{}")
        try:
            target_obj = json.loads(target_text)
            items = target_obj.get("items", [])
            gt_categories = set(item["name"] for item in items)
        except:
            gt_categories = set()

        # Full image classification
        full_preds = classify_with_contrastive(
            image,
            contrastive_model,
            image_processor,
            tokenizer,
            threshold=threshold,
            device=device,
        )

        # OD + Crop classification
        bboxes = run_od_pipeline(image, od_model, image_processor, tokenizer, device)
        crop_preds_dict = {}
        for bbox in bboxes:
            crop_img = crop_bbox(image, bbox, padding=0.15)
            crop_preds = classify_with_contrastive(
                crop_img,
                contrastive_model,
                image_processor,
                tokenizer,
                threshold=crop_threshold,
                device=device,
            )
            crop_preds_dict.update(crop_preds)

        # Union of predictions
        union_preds = {**full_preds, **crop_preds_dict}

        results_full.append(full_preds)
        results_crops.append(crop_preds_dict)
        results_union.append(union_preds)

        # Convert to per-class predictions for metrics
        for cat in CATEGORIES:
            all_true_labels.append(1 if cat in gt_categories else 0)
            all_pred_full.append(1 if cat in full_preds else 0)
            all_pred_crops.append(1 if cat in crop_preds_dict else 0)
            all_pred_union.append(1 if cat in union_preds else 0)

    # Compute metrics
    logger.info("Computing metrics...")
    all_true_labels = np.array(all_true_labels).reshape(-1, 21)
    all_pred_full = np.array(all_pred_full).reshape(-1, 21)
    all_pred_crops = np.array(all_pred_crops).reshape(-1, 21)
    all_pred_union = np.array(all_pred_union).reshape(-1, 21)

    metrics = {}
    for name, preds in [("full_image", all_pred_full), ("crops", all_pred_crops), ("union", all_pred_union)]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels, preds, average="micro"
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_true_labels, preds, average="macro"
        )
        metrics[name] = {
            "micro_f1": float(f1),
            "micro_precision": float(precision),
            "micro_recall": float(recall),
            "macro_f1": float(f1_macro),
            "macro_precision": float(precision_macro),
            "macro_recall": float(recall_macro),
        }
        logger.info(f"{name} metrics: {metrics[name]}")

    # Save results
    output_data = {
        "metrics": metrics,
        "results_full": results_full,
        "results_crops": results_crops,
        "results_union": results_union,
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OD → Crop → Contrastive evaluation")
    parser.add_argument("--base-model", default="microsoft/Florence-2-large-ft", help="Florence-2 base model")
    parser.add_argument("--od-checkpoint", default="checkpoints_od_v1/best_model", help="OD LoRA checkpoint")
    parser.add_argument(
        "--contrastive-checkpoint",
        default="checkpoints_contrastive/best_model.pt",
        help="Contrastive model checkpoint",
    )
    parser.add_argument("--data-dir", default=".", help="Data directory")
    parser.add_argument("--jsonl", default=".test.jsonl", help="Test JSONL file")
    parser.add_argument("--output", default="eval_od_contrastive_results.json", help="Output file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for full image")
    parser.add_argument("--crop-threshold", type=float, default=0.6, help="Classification threshold for crops")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")

    args = parser.parse_args()

    evaluate_od_contrastive(
        data_dir=args.data_dir,
        jsonl_file=args.jsonl,
        base_model=args.base_model,
        od_checkpoint=args.od_checkpoint,
        contrastive_checkpoint=args.contrastive_checkpoint,
        output_file=args.output,
        threshold=args.threshold,
        crop_threshold=args.crop_threshold,
        bf16=args.bf16,
    )
