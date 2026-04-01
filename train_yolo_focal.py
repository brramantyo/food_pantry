#!/usr/bin/env python3
"""
train_yolo_focal.py
===================
Train YOLOv11 detector with focal loss for class imbalance.

Focal loss helps the model focus on hard examples and rare classes,
which is critical for pantry items where some categories (Baby Food,
Vegetables) are much rarer than others (Soup, Granola).

Usage:
    python train_yolo_focal.py \\
        --data-dir . \\
        --output-dir runs/detect/yolo_focal \\
        --epochs 50 \\
        --batch 16 \\
        --imgsz 640
"""

import argparse
import json
import os
import yaml
from pathlib import Path

try:
    from ultralytics import YOLO
    from ultralytics.utils.loss import FocalLoss
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("⚠ ultralytics not installed. Install: pip install ultralytics")


def create_yolo_dataset(data_dir, output_dir="yolo_dataset"):
    """
    Convert COCO annotations to YOLO format with proper directory structure.
    
    YOLO expects:
        dataset/
            images/
                train/
                val/
            labels/
                train/
                val/
            data.yaml
    """
    
    print("Creating YOLO dataset...")
    
    # Create directories
    for split in ["train", "valid"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)
    
    # Category mapping (exclude 4 categories: Eggs, Frozen Veg, Oil, Spices)
    excluded_ids = {6, 12, 18, 23}
    
    # Load COCO annotations and convert
    for split in ["train", "valid"]:
        coco_path = f"{data_dir}/{split}/_annotations.coco.json"
        
        with open(coco_path) as f:
            coco = json.load(f)
        
        # Build category mapping (COCO id → YOLO class index)
        categories = [c for c in coco["categories"] if c["id"] not in excluded_ids]
        cat_id_to_yolo = {c["id"]: i for i, c in enumerate(categories)}
        
        # Build image lookup
        img_lookup = {img["id"]: img for img in coco["images"]}
        
        # Group annotations by image
        img_annotations = {}
        for ann in coco["annotations"]:
            if ann["category_id"] in excluded_ids:
                continue
            img_id = ann["image_id"]
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
        
        # Convert each image
        for img_id, anns in img_annotations.items():
            img_info = img_lookup[img_id]
            img_filename = img_info["file_name"]
            img_width = img_info["width"]
            img_height = img_info["height"]
            
            # Create symlink to image
            src_img = f"{data_dir}/{split}/{img_filename}"
            dst_img = f"{output_dir}/images/{split}/{img_filename}"
            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(src_img), dst_img)
            
            # Write YOLO labels
            label_filename = img_filename.rsplit(".", 1)[0] + ".txt"
            label_path = f"{output_dir}/labels/{split}/{label_filename}"
            
            with open(label_path, "w") as f:
                for ann in anns:
                    cat_id = ann["category_id"]
                    yolo_class = cat_id_to_yolo[cat_id]
                    
                    # Convert COCO bbox (x, y, w, h) to YOLO (x_center, y_center, w, h) normalized
                    # Cast to float (COCO sometimes stores as strings)
                    x, y, w, h = [float(v) for v in ann["bbox"]]
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\\n")
        
        print(f"  {split}: {len(img_annotations)} images, {sum(len(a) for a in img_annotations.values())} boxes")
    
    # Create data.yaml
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/valid",
        "nc": len(categories),
        "names": [c["name"] for c in categories]
    }
    
    yaml_path = f"{output_dir}/data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✓ YOLO dataset created: {output_dir}")
    print(f"  Classes: {len(categories)}")
    print(f"  Config: {yaml_path}")
    
    return yaml_path


def train_with_focal_loss(data_yaml, output_dir, epochs=50, batch=16, imgsz=640):
    """
    Train YOLOv11 with focal loss.
    
    Focal loss: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    where γ (gamma) controls focus on hard examples.
    
    Higher γ → more focus on hard/rare examples.
    """
    
    if not HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics not installed")
    
    print("\\nTraining YOLOv11 with Focal Loss...")
    print(f"  Data: {data_yaml}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")
    
    # Load YOLOv11 medium (good balance of speed/accuracy)
    model = YOLO("yolo11m.pt")
    
    # Train with focal loss parameters
    # fl_gamma: focal loss gamma (default 0.0 = no focal loss, 2.0 = strong focus on hard examples)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=output_dir,
        name="yolo_focal",
        
        # Focal loss settings
        fl_gamma=2.0,  # Strong focus on hard examples
        
        # Class balancing
        cls_pw=1.0,  # Class weight power (1.0 = balanced)
        
        # Augmentation (help with rare classes)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        mixup=0.1,
        
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        
        # Early stopping
        patience=10,
        
        # Hardware
        device=0,  # GPU 0
        workers=8,
        
        # Logging
        verbose=True,
        plots=True,
    )
    
    print("\\n✓ Training complete!")
    print(f"  Best model: {output_dir}/yolo_focal/weights/best.pt")
    print(f"  Metrics: {output_dir}/yolo_focal/results.csv")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO with focal loss")
    parser.add_argument("--data-dir", type=str, default=".", help="Path to train/valid folders")
    parser.add_argument("--output-dir", type=str, default="runs/detect", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset creation (if already exists)")
    
    args = parser.parse_args()
    
    # Step 1: Create YOLO dataset
    if not args.skip_dataset:
        data_yaml = create_yolo_dataset(args.data_dir, "yolo_dataset")
    else:
        data_yaml = "yolo_dataset/data.yaml"
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Dataset not found: {data_yaml}. Remove --skip-dataset to create it.")
    
    # Step 2: Train with focal loss
    train_with_focal_loss(
        data_yaml=data_yaml,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
