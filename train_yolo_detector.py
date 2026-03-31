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
        "path": str(Path(output_dir).parent),
        "train": train_images_dir,
        "val": val_images_dir,
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
) -> None:
    """Fine-tune YOLOv11."""
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
        patience=5,
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
) -> None:
    """Main pipeline."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_coco_path = data_dir / "train" / "_annotations.coco.json"
    val_coco_path = data_dir / "valid" / "_annotations.coco.json"
    train_images_dir = data_dir / "train"
    val_images_dir = data_dir / "valid"

    logger.info("Loading COCO annotations...")
    train_coco = load_coco_annotations(str(train_coco_path))
    val_coco = load_coco_annotations(str(val_coco_path))

    logger.info("Building category mapping...")
    coco_to_seq = build_category_mapping(train_coco)
    logger.info(f"Mapped {len(coco_to_seq)} COCO categories to 0-20")

    logger.info("Converting train annotations to YOLO format...")
    train_labels_dir = output_dir / "labels" / "train"
    convert_coco_to_yolo(train_coco, coco_to_seq, str(train_images_dir), str(train_labels_dir))

    logger.info("Converting val annotations to YOLO format...")
    val_labels_dir = output_dir / "labels" / "val"
    convert_coco_to_yolo(val_coco, coco_to_seq, str(val_images_dir), str(val_labels_dir))

    logger.info("Creating data.yaml...")
    yaml_path = output_dir / "data.yaml"
    create_data_yaml(str(output_dir), str(train_images_dir), str(val_images_dir), str(yaml_path))

    logger.info("Training YOLOv11...")
    train_yolo(str(yaml_path), str(output_dir), epochs=epochs, batch_size=batch_size, imgsz=imgsz, model_name=model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 on COCO pantry data")
    parser.add_argument("--data-dir", default=".", help="COCO data directory with train/ and valid/")
    parser.add_argument("--output-dir", default="yolo_output/", help="Output directory for labels and model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", default="yolo11m.pt", help="YOLO model name")

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        model=args.model,
    )
