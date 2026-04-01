#!/usr/bin/env python3
"""
Fine-tune YOLOv11 on COCO pantry annotations for object detection.

Converts COCO annotations to YOLO format and trains YOLOv11m for 50 epochs.
Maps 21 valid categories (excluding COCO IDs 6, 12, 18, 23).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import yaml
from ultralytics import YOLO

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# COCO category IDs to exclude
EXCLUDED_COCO_IDS = {6, 12, 18, 23}

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


def load_coco_annotations(coco_json_path: str) -> Dict:
    """Load COCO annotations JSON."""
    with open(coco_json_path) as f:
        return json.load(f)


def build_category_mapping(coco_data: Dict) -> Dict[int, int]:
    """
    Build mapping from COCO category ID to sequential 0-20 ID.
    Skip excluded COCO IDs.
    """
    coco_to_seq = {}
    seq_id = 0
    for cat in coco_data["categories"]:
        coco_id = cat["id"]
        if coco_id not in EXCLUDED_COCO_IDS:
            coco_to_seq[coco_id] = seq_id
            seq_id += 1
    return coco_to_seq


def convert_coco_to_yolo(
    coco_data: Dict, coco_to_seq: Dict[int, int], image_dir: str, output_dir: str
) -> None:
    """Convert COCO annotations to YOLO format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build image ID to annotations mapping
    image_annots = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if cat_id not in EXCLUDED_COCO_IDS:
            if img_id not in image_annots:
                image_annots[img_id] = []
            image_annots[img_id].append(ann)

    # Convert each image
    for img_meta in coco_data["images"]:
        img_id = img_meta["id"]
        img_width = img_meta["width"]
        img_height = img_meta["height"]
        img_filename = img_meta["file_name"]

        annots = image_annots.get(img_id, [])
        if not annots:
            continue

        # Create YOLO annotations file
        txt_filename = output_dir / (Path(img_filename).stem + ".txt")
        with open(txt_filename, "w") as f:
            for ann in annots:
                cat_id = ann["category_id"]
                if cat_id not in EXCLUDED_COCO_IDS:
                    seq_id = coco_to_seq[cat_id]
                    bbox = ann["bbox"]  # [x, y, width, height]
                    x, y, w, h = [float(v) for v in bbox]
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    f.write(f"{seq_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def create_data_yaml(output_dir: str, train_images_dir: str, val_images_dir: str, yaml_path: str) -> None:
    """Create data.yaml for ultralytics."""
    data = {
        "path": str(Path(output_dir).resolve()),
        "train": str(Path(train_images_dir).resolve()),
        "val": str(Path(val_images_dir).resolve()),
        "nc": 21,
        "names": {i: cat for i, cat in enumerate(CATEGORIES)},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"Created data.yaml at {yaml_path}")


def train_yolo(
    data_yaml_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    model_name: str = "yolo11m.pt",
    patience: int = 15,
    mixup: float = 0.15,
    cutmix: float = 0.15,
    multi_scale: float = 0.0,
) -> None:
    """Fine-tune YOLOv11 with advanced augmentation."""
    logger.info(f"Loading {model_name}...")
    model = YOLO(model_name)

    logger.info(f"Training YOLOv11 for {epochs} epochs...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0,
        project=output_dir,
        name="yolo_detector",
        exist_ok=True,
        save=True,
        patience=patience,
        
        # Augmentation (boost for rare classes)
        mixup=mixup,
        cutmix=cutmix,
        mosaic=1.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.1,
        erasing=0.3,
        
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        cos_lr=True,
        
        # Multi-scale training
        multi_scale=multi_scale,
        
        # Close mosaic for last 15 epochs (fine-tune on clean images)
        close_mosaic=15,
    )

    logger.info(f"Training complete. Results: {results}")

    # Save best model
    best_model_path = Path(output_dir) / "yolo_detector" / "weights" / "best.pt"
    if best_model_path.exists():
        logger.info(f"Best model saved at {best_model_path}")
    else:
        logger.warning("Best model not found")


def main(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    model: str = "yolo11m.pt",
    patience: int = 15,
    mixup: float = 0.15,
    cutmix: float = 0.15,
    multi_scale: float = 0.0,
) -> None:
    """Main pipeline."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_coco_path = data_dir / "train" / "_annotations.coco.json"
    val_coco_path = data_dir / "valid" / "_annotations.coco.json"

    logger.info("Loading COCO annotations...")
    train_coco = load_coco_annotations(str(train_coco_path))
    val_coco = load_coco_annotations(str(val_coco_path))

    logger.info("Building category mapping...")
    coco_to_seq = build_category_mapping(train_coco)
    logger.info(f"Mapped {len(coco_to_seq)} COCO categories to 0-20")

    # YOLO expects: dataset_root/images/train/, dataset_root/labels/train/
    # It auto-maps images↔labels by replacing /images/ with /labels/ in path
    yolo_root = output_dir / "dataset"
    yolo_images_train = yolo_root / "images" / "train"
    yolo_images_val = yolo_root / "images" / "val"
    yolo_labels_train = yolo_root / "labels" / "train"
    yolo_labels_val = yolo_root / "labels" / "val"

    for d in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Symlink images into YOLO structure
    import glob
    logger.info("Symlinking train images...")
    for img_file in (data_dir / "train").glob("*"):
        if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            dst = yolo_images_train / img_file.name
            if not dst.exists():
                dst.symlink_to(img_file.resolve())

    logger.info("Symlinking val images...")
    for img_file in (data_dir / "valid").glob("*"):
        if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            dst = yolo_images_val / img_file.name
            if not dst.exists():
                dst.symlink_to(img_file.resolve())

    logger.info("Converting train annotations to YOLO format...")
    convert_coco_to_yolo(train_coco, coco_to_seq, str(data_dir / "train"), str(yolo_labels_train))

    logger.info("Converting val annotations to YOLO format...")
    convert_coco_to_yolo(val_coco, coco_to_seq, str(data_dir / "valid"), str(yolo_labels_val))

    # Count label files for verification
    n_train_labels = len(list(yolo_labels_train.glob("*.txt")))
    n_val_labels = len(list(yolo_labels_val.glob("*.txt")))
    n_train_images = len(list(yolo_images_train.glob("*")))
    n_val_images = len(list(yolo_images_val.glob("*")))
    logger.info(f"Train: {n_train_images} images, {n_train_labels} labels")
    logger.info(f"Val: {n_val_images} images, {n_val_labels} labels")

    logger.info("Creating data.yaml...")
    yaml_path = output_dir / "data.yaml"
    create_data_yaml(
        str(yolo_root),
        str(yolo_images_train),
        str(yolo_images_val),
        str(yaml_path),
    )

    logger.info("Training YOLOv11...")
    train_yolo(str(yaml_path), str(output_dir), epochs=epochs, batch_size=batch_size, imgsz=imgsz, 
               model_name=model, patience=patience, mixup=mixup, cutmix=cutmix, multi_scale=multi_scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 on COCO pantry data")
    parser.add_argument("--data-dir", default=".", help="COCO data directory with train/ and valid/")
    parser.add_argument("--output-dir", default="yolo_output/", help="Output directory for labels and model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", default="yolo11m.pt", help="YOLO model name")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--mixup", type=float, default=0.15, help="MixUp alpha")
    parser.add_argument("--cutmix", type=float, default=0.15, help="CutMix alpha")
    parser.add_argument("--multi-scale", type=float, default=0.0, help="Multi-scale training factor")

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        model=args.model,
        patience=args.patience,
        mixup=args.mixup,
        cutmix=args.cutmix,
        multi_scale=args.multi_scale,
    )
