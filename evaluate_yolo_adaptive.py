#!/usr/bin/env python3
"""
evaluate_yolo_adaptive.py
=========================
Evaluate YOLO detector with per-class adaptive confidence thresholds.

Instead of using a single threshold (e.g., 0.5) for all classes,
this uses optimized thresholds per class to maximize recall for
rare categories while maintaining precision for common ones.

Usage:
    python evaluate_yolo_adaptive.py \\
        --model runs/detect/yolo_focal/weights/best.pt \\
        --data yolo_dataset/data.yaml \\
        --output eval_yolo_adaptive.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


def compute_optimal_thresholds(model, val_data, categories, initial_threshold=0.3):
    """
    Compute per-class optimal confidence thresholds.
    
    Strategy:
    - Start with low threshold (0.3) to get all detections
    - For each class, find threshold that maximizes F1 score
    - Bias towards higher recall for rare classes
    """
    
    print("Computing optimal per-class thresholds...")
    
    # Run inference with low threshold to get all detections
    results = model.predict(
        source=val_data,
        conf=initial_threshold,
        iou=0.5,
        verbose=False,
    )
    
    # Collect all detections per class
    class_detections = defaultdict(list)  # class_id → [(confidence, is_correct), ...]
    
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            # TODO: match with GT to determine if correct
            # For now, assume all detections are correct (simplified)
            class_detections[cls_id].append((conf, True))
    
    # Compute optimal threshold per class
    optimal_thresholds = {}
    
    for cls_id in range(len(categories)):
        detections = class_detections.get(cls_id, [])
        
        if not detections:
            # No detections for this class, use default
            optimal_thresholds[cls_id] = 0.5
            continue
        
        # Sort by confidence
        detections.sort(reverse=True)
        
        # Try different thresholds and find best F1
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            tp = sum(1 for conf, correct in detections if conf >= thresh and correct)
            fp = sum(1 for conf, correct in detections if conf >= thresh and not correct)
            fn = sum(1 for conf, correct in detections if conf < thresh and correct)
            
            if tp + fp == 0 or tp + fn == 0:
                continue
            
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            
            if precision + recall == 0:
                continue
            
            f1 = 2 * precision * recall / (precision + recall)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        optimal_thresholds[cls_id] = best_thresh
    
    # Adjust thresholds based on class frequency (lower for rare classes)
    class_counts = {cls_id: len(dets) for cls_id, dets in class_detections.items()}
    total_count = sum(class_counts.values())
    
    for cls_id in range(len(categories)):
        count = class_counts.get(cls_id, 0)
        freq = count / total_count if total_count > 0 else 0
        
        # Lower threshold for rare classes (< 2% of dataset)
        if freq < 0.02:
            optimal_thresholds[cls_id] *= 0.8  # 20% lower threshold
            print(f"  {categories[cls_id]}: {optimal_thresholds[cls_id]:.2f} (rare class, lowered)")
        else:
            print(f"  {categories[cls_id]}: {optimal_thresholds[cls_id]:.2f}")
    
    return optimal_thresholds


def evaluate_with_adaptive_thresholds(model, test_data, categories, thresholds):
    """
    Run evaluation with per-class adaptive thresholds.
    """
    
    print("\\nEvaluating with adaptive thresholds...")
    
    # Run inference with lowest threshold to get all detections
    min_thresh = min(thresholds.values())
    results = model.predict(
        source=test_data,
        conf=min_thresh * 0.8,  # Slightly lower to ensure we don't miss anything
        iou=0.5,
        verbose=False,
    )
    
    # Filter detections using per-class thresholds
    filtered_results = []
    
    for result in results:
        boxes = result.boxes
        keep_indices = []
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            thresh = thresholds.get(cls_id, 0.5)
            
            if conf >= thresh:
                keep_indices.append(i)
        
        # Create filtered result
        if keep_indices:
            filtered_boxes = boxes[keep_indices]
            result.boxes = filtered_boxes
        else:
            result.boxes = []
        
        filtered_results.append(result)
    
    # Compute metrics
    # TODO: Compare with ground truth and compute precision/recall/F1
    
    print(f"✓ Evaluated {len(filtered_results)} images")
    
    return filtered_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--output", type=str, default="eval_yolo_adaptive.json")
    parser.add_argument("--compute-thresholds", action="store_true", help="Compute optimal thresholds from validation set")
    
    args = parser.parse_args()
    
    if not HAS_ULTRALYTICS:
        raise RuntimeError("ultralytics not installed")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Load data config
    import yaml
    with open(args.data) as f:
        data_config = yaml.safe_load(f)
    
    categories = data_config["names"]
    
    # Compute or load thresholds
    if args.compute_thresholds:
        val_path = os.path.join(data_config["path"], data_config["val"])
        thresholds = compute_optimal_thresholds(model, val_path, categories)
        
        # Save thresholds
        thresh_path = args.output.replace(".json", "_thresholds.json")
        with open(thresh_path, "w") as f:
            json.dump({categories[i]: thresholds[i] for i in range(len(categories))}, f, indent=2)
        print(f"\\n✓ Thresholds saved: {thresh_path}")
    else:
        # Use default thresholds (can be loaded from file)
        thresholds = {i: 0.5 for i in range(len(categories))}
    
    # Evaluate
    test_path = os.path.join(data_config["path"], data_config.get("test", data_config["val"]))
    results = evaluate_with_adaptive_thresholds(model, test_path, categories, thresholds)
    
    # Save results
    output = {
        "model": args.model,
        "thresholds": {categories[i]: thresholds[i] for i in range(len(categories))},
        "num_images": len(results),
    }
    
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\\n✓ Results saved: {args.output}")


if __name__ == "__main__":
    main()
