#!/usr/bin/env python3
"""
generate_failure_examples_cluster.py
=====================================
Generate 5 representative failure case visualizations for the report.
Designed to run on the cluster with full data access.

This script:
1. Loads test predictions from evaluation results
2. Identifies different types of failures (multi-item omission, confusion, etc.)
3. Generates annotated images showing ground truth vs prediction
4. Exports LaTeX code ready for the report

Usage:
    # If you have evaluation results with per-sample predictions:
    python3 generate_failure_examples_cluster.py \
        --predictions eval_results_v11.json \
        --test-jsonl florence2_data/test.jsonl \
        --data-dir /path/to/images \
        --output-dir figures/failure_examples

    # If you DON'T have predictions, run inference first:
    python3 generate_failure_examples_cluster.py \
        --checkpoint checkpoints/v11_best \
        --base-model microsoft/Florence-2-large-ft \
        --test-jsonl florence2_data/test.jsonl \
        --data-dir /path/to/images \
        --output-dir figures/failure_examples \
        --run-inference
"""

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for cluster
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib not available - will copy images without annotation")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def parse_florence_output(text: str) -> Dict[str, int]:
    """Parse Florence-2 output into category counts."""
    try:
        data = json.loads(text)
        counts = {}
        for item in data.get("pantry_items", []):
            cat = item.get("category", "")
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        return counts
    except:
        return {}


def load_test_data(jsonl_path: str) -> List[Dict]:
    """Load test set from JSONL."""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_predictions(pred_path: str) -> Dict:
    """Load evaluation results with predictions."""
    with open(pred_path, 'r') as f:
        return json.load(f)


def run_inference(checkpoint_path: str, base_model: str, test_data: List[Dict], 
                  data_dir: str, device: str = "cuda") -> List[str]:
    """Run inference on test set if predictions not available."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not available - cannot run inference")
    
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Resize embeddings to match checkpoint (if needed)
    # This handles cases where vocab was extended during training
    try:
        import json
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            # Check if we need to resize
            # Load one weight to check actual size
            import safetensors
            weights_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
            if os.path.exists(weights_path):
                from safetensors.torch import load_file
                weights = load_file(weights_path)
                # Check shared.weight size
                if "base_model.model.language_model.model.shared.weight" in weights:
                    checkpoint_vocab_size = weights["base_model.model.language_model.model.shared.weight"].shape[0]
                    current_vocab_size = base.language_model.model.shared.weight.shape[0]
                    if checkpoint_vocab_size != current_vocab_size:
                        print(f"  Resizing embeddings: {current_vocab_size} → {checkpoint_vocab_size}")
                        base.language_model.resize_token_embeddings(checkpoint_vocab_size)
    except Exception as e:
        print(f"  Warning: Could not check/resize embeddings: {e}")
    
    base = base.to(device)
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.eval()
    
    print(f"Running inference on {len(test_data)} samples...")
    predictions = []
    
    for i, sample in enumerate(test_data):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_data)}...")
        
        # Normalize path separators (handle Windows backslashes)
        img_rel_path = sample["image"].replace("\\", "/")
        img_path = os.path.join(data_dir, img_rel_path)
        image = Image.open(img_path).convert("RGB")
        
        prompt = "<STRUCTURED_PANTRY_OUTPUT>"
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Move to device and cast to bfloat16 to match model
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False
            )
        
        output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        pred_text = processor.post_process_generation(
            output, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        predictions.append(pred_text)
    
    print("✓ Inference complete")
    return predictions


def categorize_failures(test_data: List[Dict], predictions: List[str]) -> Dict[str, List[Dict]]:
    """Categorize failures into different types."""
    
    failures = {
        "multi_item_omission": [],
        "category_confusion": [],
        "count_error": [],
        "false_positive": []
    }
    
    for i, sample in enumerate(test_data):
        gt_text = sample.get("suffix", "")
        gt_counts = parse_florence_output(gt_text)
        
        if i >= len(predictions):
            continue
        
        pred_text = predictions[i]
        pred_counts = parse_florence_output(pred_text)
        
        # Skip correct predictions
        if gt_counts == pred_counts:
            continue
        
        failure_info = {
            "index": i,
            "image": sample.get("image", ""),
            "gt_counts": gt_counts,
            "pred_counts": pred_counts,
            "gt_text": gt_text,
            "pred_text": pred_text
        }
        
        # Categorize failure type
        gt_cats = set(gt_counts.keys())
        pred_cats = set(pred_counts.keys())
        
        # Multi-item omission: GT has multiple items, prediction misses some
        if len(gt_cats) > 1 and len(pred_cats) < len(gt_cats):
            failures["multi_item_omission"].append(failure_info)
        
        # Category confusion: predicted wrong category
        elif pred_cats - gt_cats:  # predicted categories not in GT
            failures["category_confusion"].append(failure_info)
        
        # Count error: right categories, wrong counts
        elif gt_cats == pred_cats:
            failures["count_error"].append(failure_info)
        
        # False positive: predicted items not in GT
        elif len(pred_cats) > len(gt_cats):
            failures["false_positive"].append(failure_info)
    
    return failures


def select_representative_examples(failures: Dict[str, List[Dict]], n: int = 5) -> List[Tuple[str, Dict]]:
    """Select n most representative failure examples."""
    
    selected = []
    
    # Priority order: multi-item omission > category confusion > count error
    priority = ["multi_item_omission", "category_confusion", "count_error", "false_positive"]
    
    for failure_type in priority:
        examples = failures[failure_type]
        if not examples:
            continue
        
        # For multi-item omission, prefer cases with 2+ GT items
        if failure_type == "multi_item_omission":
            examples = sorted(examples, key=lambda x: len(x["gt_counts"]), reverse=True)
        
        # Take up to 2 examples per type
        for ex in examples[:2]:
            if len(selected) >= n:
                break
            selected.append((failure_type, ex))
        
        if len(selected) >= n:
            break
    
    return selected[:n]


def create_failure_visualization(
    image_path: str,
    gt_counts: Dict[str, int],
    pred_counts: Dict[str, int],
    title: str,
    output_path: str
):
    """Create annotated visualization of a failure case."""
    
    if not HAS_MATPLOTLIB:
        # Fallback: just copy the image
        shutil.copy2(image_path, output_path)
        print(f"✓ Copied (no annotation): {output_path}")
        return
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img)
    ax.axis('off')
    
    # Format ground truth and prediction text
    gt_text = "Ground Truth:\n" + "\n".join([f"  • {cat}: {cnt}" for cat, cnt in sorted(gt_counts.items())])
    pred_text = "Prediction:\n" + "\n".join([f"  • {cat}: {cnt}" for cat, cnt in sorted(pred_counts.items())])
    
    if not pred_counts:
        pred_text = "Prediction:\n  (empty or unparseable)"
    
    # Highlight differences
    gt_cats = set(gt_counts.keys())
    pred_cats = set(pred_counts.keys())
    missed = gt_cats - pred_cats
    wrong = pred_cats - gt_cats
    
    if missed:
        pred_text += f"\n\n⚠ MISSED: {', '.join(missed)}"
    if wrong:
        pred_text += f"\n\n⚠ WRONG: {', '.join(wrong)}"
    
    # Add text boxes
    textstr = f"{gt_text}\n\n{pred_text}"
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    # Add title
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def get_failure_description(failure_type: str, example: Dict) -> str:
    """Generate human-readable description of the failure."""
    
    gt_cats = list(example["gt_counts"].keys())
    pred_cats = list(example["pred_counts"].keys())
    
    if failure_type == "multi_item_omission":
        missed = set(gt_cats) - set(pred_cats)
        predicted = pred_cats[0] if pred_cats else "nothing"
        return f"The model predicted only \\texttt{{{predicted}}} and missed \\texttt{{{', '.join(missed)}}}."
    
    elif failure_type == "category_confusion":
        wrong = set(pred_cats) - set(gt_cats)
        correct = ', '.join(gt_cats)
        return f"The model incorrectly predicted \\texttt{{{', '.join(wrong)}}} instead of \\texttt{{{correct}}}."
    
    elif failure_type == "count_error":
        return "The model predicted the correct categories but wrong counts."
    
    elif failure_type == "false_positive":
        extra = set(pred_cats) - set(gt_cats)
        return f"The model hallucinated \\texttt{{{', '.join(extra)}}} which are not present."
    
    return "Unknown failure type."


def generate_latex_code(descriptions: List[Dict], output_dir: str) -> str:
    """Generate LaTeX code for all failure examples."""
    
    latex_lines = [
        "\\subsubsection{Example Failure Cases}",
        "",
        "To better understand the remaining errors, we examine a few typical failure cases:",
        ""
    ]
    
    for desc in descriptions:
        example_num = desc["example_num"]
        filename = desc["output_filename"]
        failure_type = desc["failure_type"]
        description = desc["description"]
        
        gt_str = ", ".join([f"\\texttt{{{cat}}} ({cnt})" for cat, cnt in desc["gt_counts"].items()])
        pred_str = ", ".join([f"\\texttt{{{cat}}} ({cnt})" for cat, cnt in desc["pred_counts"].items()]) if desc["pred_counts"] else "(empty)"
        
        # Map failure type to readable title
        title_map = {
            "multi_item_omission": "Multi-Item Omission",
            "category_confusion": "Category Confusion",
            "count_error": "Count Error",
            "false_positive": "False Positive"
        }
        title = title_map.get(failure_type, failure_type)
        
        latex_lines.extend([
            "\\begin{figure}[H]",
            "    \\centering",
            f"    \\includegraphics[width=0.7\\textwidth]{{figures/failure_examples/{filename}}}",
            f"    \\caption{{\\textbf{{{title}.}} Ground truth: {gt_str}. Prediction: {pred_str}. {description}}}",
            f"    \\label{{fig:failure_example_{example_num}}}",
            "\\end{figure}",
            ""
        ])
    
    # Add summary paragraph
    latex_lines.extend([
        "These examples are consistent with the main pattern in the confusion analysis: ",
        "the dominant issue is \\textbf{omission in multi-item scenes} rather than ",
        "large-scale confusion across unrelated categories. The model tends to predict ",
        "the visually dominant item while missing secondary categories that occupy less ",
        "visual space or have lower contrast.",
        ""
    ])
    
    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str,
                        help="Path to evaluation results JSON with per-sample predictions")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to model checkpoint (if running inference)")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft",
                        help="Base model name")
    parser.add_argument("--test-jsonl", type=str, required=True,
                        help="Path to test.jsonl")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Base directory for image paths")
    parser.add_argument("--output-dir", type=str, default="figures/failure_examples",
                        help="Output directory for failure visualizations")
    parser.add_argument("--n-examples", type=int, default=5,
                        help="Number of failure examples to generate")
    parser.add_argument("--run-inference", action="store_true",
                        help="Run inference if predictions not provided")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.predictions and not (args.run_inference and args.checkpoint):
        parser.error("Either --predictions or (--run-inference + --checkpoint) required")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("FAILURE EXAMPLE GENERATOR")
    print("=" * 80)
    
    print("\nLoading test data...")
    test_data = load_test_data(args.test_jsonl)
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Get predictions
    if args.predictions:
        print("\nLoading predictions...")
        pred_data = load_predictions(args.predictions)
        
        # Extract per-sample predictions
        if "per_sample_results" in pred_data:
            predictions = [s.get("prediction", "") for s in pred_data["per_sample_results"]]
        else:
            raise ValueError("Predictions file must contain 'per_sample_results'")
        
        print(f"✓ Loaded {len(predictions)} predictions")
    
    elif args.run_inference:
        print("\nRunning inference...")
        predictions = run_inference(
            args.checkpoint,
            args.base_model,
            test_data,
            args.data_dir,
            args.device
        )
    
    print("\nCategorizing failures...")
    failures = categorize_failures(test_data, predictions)
    
    print("\nFailure statistics:")
    for ftype, examples in failures.items():
        print(f"  • {ftype}: {len(examples)} cases")
    
    print(f"\nSelecting {args.n_examples} representative examples...")
    selected = select_representative_examples(failures, args.n_examples)
    
    if len(selected) < args.n_examples:
        print(f"⚠ Only found {len(selected)} failure examples")
    
    print(f"\nGenerating visualizations...")
    
    descriptions = []
    
    for i, (failure_type, example) in enumerate(selected, 1):
        # Build image path (normalize Windows backslashes)
        img_rel_path = example["image"].replace("\\", "/")
        img_path = os.path.join(args.data_dir, img_rel_path)
        
        if not os.path.exists(img_path):
            print(f"⚠ Image not found: {img_path}")
            continue
        
        # Create title
        title_map = {
            "multi_item_omission": "Multi-Item Omission",
            "category_confusion": "Category Confusion",
            "count_error": "Count Error",
            "false_positive": "False Positive"
        }
        title = f"Failure Example {i}: {title_map.get(failure_type, failure_type)}"
        
        # Output filename
        output_filename = f"failure_example_{i}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Generate visualization
        create_failure_visualization(
            img_path,
            example["gt_counts"],
            example["pred_counts"],
            title,
            output_path
        )
        
        # Create description
        desc = {
            "example_num": i,
            "failure_type": failure_type,
            "image_path": img_rel_path,
            "output_filename": output_filename,
            "gt_counts": example["gt_counts"],
            "pred_counts": example["pred_counts"],
            "description": get_failure_description(failure_type, example)
        }
        descriptions.append(desc)
    
    # Save descriptions JSON
    desc_path = os.path.join(args.output_dir, "failure_descriptions.json")
    with open(desc_path, 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    # Generate and save LaTeX code
    latex_code = generate_latex_code(descriptions, args.output_dir)
    latex_path = os.path.join(args.output_dir, "latex_failure_examples.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    
    print(f"\n{'=' * 80}")
    print("✓ COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nGenerated {len(descriptions)} failure examples")
    print(f"Output directory: {args.output_dir}")
    print(f"\nFiles created:")
    print(f"  • {len(descriptions)} PNG images")
    print(f"  • {desc_path}")
    print(f"  • {latex_path}")
    print(f"\nNext steps:")
    print(f"  1. Review the images in {args.output_dir}")
    print(f"  2. Copy the LaTeX code from {latex_path} into your report")
    print(f"  3. Adjust the figure placement and captions as needed")


if __name__ == "__main__":
    main()
