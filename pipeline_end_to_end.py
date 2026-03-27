#!/usr/bin/env python3
"""
pipeline_end_to_end.py
======================
End-to-end pipeline: Image → Task 1 (classify) → Task 2 (USDA match) → Nutrition output.

This is the full demo pipeline that:
  1. Takes a pantry image as input
  2. Runs Florence-2 fine-tuned classifier (Task 1)
  3. Matches each predicted category to USDA foods (Task 2)
  4. Returns structured nutrition information

Usage:
  # Single image
  python pipeline_end_to_end.py --image path/to/image.jpg

  # Batch from test set
  python pipeline_end_to_end.py --jsonl ./florence2_data/test_v5.jsonl --max-samples 10

  # Full test set with evaluation
  python pipeline_end_to_end.py --jsonl ./florence2_data/test_v5.jsonl --evaluate
"""

import argparse
import json
import os
import sys
import time

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Import our USDA matcher
from usda_matcher import USDAMatcher

TASK_PROMPT = "<OD>"

CANONICAL_CATEGORIES = {
    "baby food": "Baby Food",
    "beans and legumes - canned or dried": "Beans and Legumes - Canned or Dried",
    "bread and bakery products": "Bread and Bakery Products",
    "canned tomato products": "Canned Tomato Products",
    "carbohydrate meal": "Carbohydrate Meal",
    "condiments and sauces": "Condiments and Sauces",
    "dairy and dairy alternatives": "Dairy and Dairy Alternatives",
    "desserts and sweets": "Desserts and Sweets",
    "drinks": "Drinks",
    "fresh fruit": "Fresh Fruit",
    "fruits - canned or processed": "Fruits - Canned or Processed",
    "granola products": "Granola Products",
    "meat and poultry - canned": "Meat and Poultry - Canned",
    "meat and poultry - fresh": "Meat and Poultry - Fresh",
    "nut butters and nuts": "Nut Butters and Nuts",
    "ready meals": "Ready Meals",
    "savory snacks and crackers": "Savory Snacks and Crackers",
    "seafood - canned": "Seafood - Canned",
    "soup": "Soup",
    "vegetables - canned": "Vegetables - Canned",
    "vegetables - fresh": "Vegetables - Fresh",
}

VALID_CATEGORIES = set(CANONICAL_CATEGORIES.values())


def normalize_category(name):
    return CANONICAL_CATEGORIES.get(name.lower().strip(), name)


class PantryNutritionPipeline:
    """End-to-end pipeline: Image → Classification → USDA Matching → Nutrition."""

    def __init__(self, base_model, checkpoint, usda_dir, device="cuda", amp_dtype=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.amp_dtype = amp_dtype

        # Load Florence-2 classifier
        print("Loading Florence-2 classifier...")
        self.processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=True,
            torch_dtype=amp_dtype or torch.float32,
            attn_implementation="eager",
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint)
        self.model = model.merge_and_unload().to(self.device).eval()
        print("  Classifier loaded ✓")

        # Load USDA matcher
        self.matcher = USDAMatcher(usda_dir=usda_dir)
        print("\nPipeline ready!\n")

    def classify_image(self, image):
        """Task 1: Classify pantry image into categories."""
        inputs = self.processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if self.amp_dtype:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    generated_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512,
                        num_beams=3,
                        early_stopping=True,
                    )
            else:
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=3,
                    early_stopping=True,
                )

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        try:
            pred = json.loads(text)
            items = []
            for item in pred.get("items", []):
                name = normalize_category(item.get("name", ""))
                if name in VALID_CATEGORIES:
                    items.append({
                        "name": name,
                        "package_type": item.get("package_type", "unknown"),
                        "confidence": item.get("confidence", "medium"),
                    })
            return items
        except json.JSONDecodeError:
            return []

    def match_to_usda(self, classified_items, top_k=5):
        """Task 2: Match each classified item to USDA entries."""
        matches = []
        for item in classified_items:
            usda_results = self.matcher.match_pantry_prediction(
                item["name"], package_type=item.get("package_type"), top_k=top_k
            )
            matches.append({
                "pantry_category": item["name"],
                "package_type": item.get("package_type", "unknown"),
                "confidence": item.get("confidence", "medium"),
                "usda_matches": [
                    {
                        "fdc_id": r.get("fdc_id"),
                        "description": r.get("description", ""),
                        "brand": r.get("brand_owner", ""),
                        "category": r.get("brand_category", ""),
                        "score": round(r.get("score", 0), 4),
                        "nutrients": r.get("nutrients", {}),
                        "serving": f"{r.get('serving_size', '?')} {r.get('serving_size_unit', '')}".strip(),
                    }
                    for r in usda_results
                ],
            })
        return matches

    def get_nutrition_summary(self, matches):
        """Summarize nutrition across all matched items."""
        total_nutrients = {}
        item_count = 0

        for match in matches:
            # Use top USDA match for each category
            if match["usda_matches"]:
                top = match["usda_matches"][0]
                nuts = top.get("nutrients", {})
                for key, val in nuts.items():
                    if val is not None:
                        total_nutrients[key] = total_nutrients.get(key, 0) + val
                item_count += 1

        return {
            "num_items": item_count,
            "estimated_total_nutrients": total_nutrients,
            "note": "Based on top USDA match per category. Values are per serving/100g depending on source.",
        }

    def process_image(self, image_path, top_k=5):
        """Full pipeline: image path → structured nutrition output."""
        image = Image.open(image_path).convert("RGB")

        # Task 1: Classify
        classified = self.classify_image(image)

        # Task 2: Match to USDA
        matches = self.match_to_usda(classified, top_k=top_k)

        # Nutrition summary
        summary = self.get_nutrition_summary(matches)

        return {
            "image": image_path,
            "classified_items": classified,
            "usda_matches": matches,
            "nutrition_summary": summary,
        }


def print_result(result, verbose=True):
    """Pretty-print pipeline result."""
    print(f"\n{'='*60}")
    print(f"📷 Image: {os.path.basename(result['image'])}")
    print(f"{'='*60}")

    items = result["classified_items"]
    print(f"\n🏷️  Task 1 — Classified {len(items)} item(s):")
    for item in items:
        print(f"  • {item['name']} ({item['package_type']}) [{item['confidence']}]")

    matches = result["usda_matches"]
    print(f"\n🔍 Task 2 — USDA Matches:")
    for match in matches:
        print(f"\n  Category: {match['pantry_category']}")
        for i, usda in enumerate(match["usda_matches"][:3]):
            score = usda["score"]
            desc = usda["description"]
            brand = usda.get("brand", "")
            nuts = usda.get("nutrients", {})

            print(f"    {i+1}. [{score:.3f}] {desc}")
            if brand:
                print(f"       Brand: {brand}")

            if verbose and nuts:
                parts = []
                if "energy_kcal" in nuts:
                    parts.append(f"{nuts['energy_kcal']}kcal")
                if "protein_g" in nuts:
                    parts.append(f"P:{nuts['protein_g']}g")
                if "carbohydrate_g" in nuts:
                    parts.append(f"C:{nuts['carbohydrate_g']}g")
                if "total_fat_g" in nuts:
                    parts.append(f"F:{nuts['total_fat_g']}g")
                if parts:
                    print(f"       Nutrition: {' | '.join(parts)}")

    summary = result["nutrition_summary"]
    total = summary["estimated_total_nutrients"]
    if total:
        print(f"\n📊 Estimated Total Nutrition (top match per item):")
        if "energy_kcal" in total:
            print(f"  Calories: {total['energy_kcal']:.0f} kcal")
        if "protein_g" in total:
            print(f"  Protein:  {total['protein_g']:.1f} g")
        if "carbohydrate_g" in total:
            print(f"  Carbs:    {total['carbohydrate_g']:.1f} g")
        if "total_fat_g" in total:
            print(f"  Fat:      {total['total_fat_g']:.1f} g")


def main():
    parser = argparse.ArgumentParser(description="End-to-end Pantry Nutrition Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--jsonl", type=str, default=None, help="Test JSONL for batch processing")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--base-model", type=str, default="microsoft/Florence-2-large-ft")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_v11/best_model")
    parser.add_argument("--usda-dir", type=str, default="./usda_data")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="./pipeline_results.json")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Run with evaluation metrics")
    args = parser.parse_args()

    amp_dtype = None
    if args.bf16 and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16

    pipeline = PantryNutritionPipeline(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        usda_dir=args.usda_dir,
        amp_dtype=amp_dtype,
    )

    if args.image:
        result = pipeline.process_image(args.image, top_k=args.top_k)
        print_result(result)

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")

    elif args.jsonl:
        samples = []
        with open(args.jsonl, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        if args.max_samples:
            samples = samples[:args.max_samples]

        print(f"Processing {len(samples)} images...\n")
        all_results = []
        start = time.time()

        for i, sample in enumerate(samples):
            img_rel = sample["image"].replace("\\", "/")
            img_path = os.path.join(args.data_dir, img_rel)

            if not os.path.exists(img_path):
                print(f"  [WARN] Image not found: {img_path}")
                continue

            result = pipeline.process_image(img_path, top_k=args.top_k)
            print_result(result, verbose=(i < 5))  # verbose for first 5 only
            all_results.append(result)

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"\n  --- Progress: {i+1}/{len(samples)} ({rate:.1f} img/s) ---\n")

        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"  Processed: {len(all_results)} images in {elapsed:.1f}s")
        print(f"  Rate: {len(all_results)/elapsed:.1f} img/s")

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
