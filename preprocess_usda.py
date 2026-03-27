#!/usr/bin/env python3
"""
preprocess_usda.py
==================
Parse USDA FoodData Central branded foods + SR Legacy into a flat, searchable format.

Output:
  - usda_data/branded_foods_flat.jsonl  (~400K+ items, key fields only)
  - usda_data/sr_legacy_flat.jsonl      (~7.8K items)
  - usda_data/usda_categories.json      (unique brandedFoodCategory values)
  - usda_data/category_mapping.json     (pantry 21 categories → USDA categories)

Usage:
  python preprocess_usda.py --usda-dir ./usda_data
"""

import argparse
import json
import os
import sys
from collections import Counter


# Key nutrients we care about (by USDA nutrient ID)
KEY_NUTRIENTS = {
    1003: "protein_g",
    1004: "total_fat_g",
    1005: "carbohydrate_g",
    1008: "energy_kcal",
    1079: "fiber_g",
    2000: "total_sugars_g",
    1093: "sodium_mg",
    1087: "calcium_mg",
    1089: "iron_mg",
    1104: "vitamin_a_iu",
    1162: "vitamin_c_mg",
    1253: "cholesterol_mg",
    1257: "trans_fat_g",
    1258: "saturated_fat_g",
}

# Also extract from labelNutrients (simpler structure, per 100g or per serving)
LABEL_NUTRIENT_MAP = {
    "protein": "protein_g",
    "fat": "total_fat_g",
    "carbohydrates": "carbohydrate_g",
    "calories": "energy_kcal",
    "fiber": "fiber_g",
    "sugars": "total_sugars_g",
    "sodium": "sodium_mg",
    "calcium": "calcium_mg",
    "iron": "iron_mg",
    "cholesterol": "cholesterol_mg",
    "transFat": "trans_fat_g",
    "saturatedFat": "saturated_fat_g",
}

# 21 Pantry categories → likely USDA brandedFoodCategory matches
PANTRY_TO_USDA_CATEGORY_HINTS = {
    "Baby Food": ["Baby Food"],
    "Beans and Legumes - Canned or Dried": ["Beans", "Canned Beans", "Dried Beans"],
    "Bread and Bakery Products": ["Bread", "Bakery", "Rolls", "Buns", "Tortillas"],
    "Canned Tomato Products": ["Tomato", "Canned Tomatoes", "Tomato Sauce", "Pasta Sauce"],
    "Carbohydrate Meal": ["Rice", "Pasta", "Noodles", "Macaroni", "Grains"],
    "Condiments and Sauces": ["Condiments", "Sauces", "Ketchup", "Mustard", "Dressing"],
    "Dairy and Dairy Alternatives": ["Dairy", "Milk", "Cheese", "Yogurt", "Butter"],
    "Desserts and Sweets": ["Candy", "Chocolate", "Cookies", "Cake", "Desserts", "Sweets"],
    "Drinks": ["Beverages", "Juice", "Water", "Soda", "Drink"],
    "Fresh Fruit": ["Fruit", "Fresh Fruit", "Apples", "Bananas", "Oranges"],
    "Fruits - Canned or Processed": ["Canned Fruit", "Fruit Cup", "Applesauce"],
    "Granola Products": ["Granola", "Cereal", "Breakfast Cereal", "Oats"],
    "Meat and Poultry - Canned": ["Canned Meat", "Canned Chicken", "Canned Tuna", "Spam"],
    "Meat and Poultry - Fresh": ["Meat", "Poultry", "Chicken", "Beef", "Pork", "Turkey"],
    "Nut Butters and Nuts": ["Nut Butter", "Peanut Butter", "Nuts", "Almonds"],
    "Ready Meals": ["Frozen Meals", "Ready Meals", "Frozen Dinner", "Frozen Entree"],
    "Savory Snacks and Crackers": ["Snacks", "Crackers", "Chips", "Pretzels", "Popcorn"],
    "Seafood - Canned": ["Canned Seafood", "Canned Fish", "Tuna", "Sardines", "Salmon"],
    "Soup": ["Soup", "Broth", "Stew", "Chili"],
    "Vegetables - Canned": ["Canned Vegetables", "Canned Corn", "Canned Peas", "Canned Green Beans"],
    "Vegetables - Fresh": ["Vegetables", "Fresh Vegetables", "Lettuce", "Tomatoes", "Carrots"],
}


def extract_nutrients_from_food_nutrients(food_nutrients):
    """Extract key nutrients from foodNutrients array."""
    nutrients = {}
    for fn in food_nutrients:
        nutrient = fn.get("nutrient", {})
        nutrient_id = nutrient.get("id")
        if nutrient_id in KEY_NUTRIENTS:
            key = KEY_NUTRIENTS[nutrient_id]
            amount = fn.get("amount")
            if amount is not None:
                nutrients[key] = amount
    return nutrients


def extract_nutrients_from_label(label_nutrients):
    """Extract key nutrients from labelNutrients dict."""
    nutrients = {}
    if not label_nutrients:
        return nutrients
    for label_key, our_key in LABEL_NUTRIENT_MAP.items():
        val = label_nutrients.get(label_key)
        if val is not None:
            if isinstance(val, dict):
                nutrients[our_key] = val.get("value")
            else:
                nutrients[our_key] = val
    return nutrients


def process_branded_foods(usda_dir):
    """Parse brandedDownload.json → flat JSONL."""
    input_path = os.path.join(usda_dir, "brandedDownload.json")
    output_path = os.path.join(usda_dir, "branded_foods_flat.jsonl")
    categories_path = os.path.join(usda_dir, "usda_categories.json")

    print(f"Loading branded foods from {input_path}...")
    print("  (this may take a minute for 3GB file)")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    foods = data.get("BrandedFoods", [])
    print(f"  Found {len(foods):,} branded food items")

    category_counter = Counter()
    written = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for food in foods:
            description = food.get("description", "").strip()
            if not description:
                skipped += 1
                continue

            # Extract nutrients — prefer labelNutrients (simpler), fallback to foodNutrients
            nutrients = extract_nutrients_from_label(food.get("labelNutrients"))
            if not nutrients:
                nutrients = extract_nutrients_from_food_nutrients(food.get("foodNutrients", []))

            category = food.get("brandedFoodCategory", "").strip()
            category_counter[category] += 1

            flat = {
                "fdc_id": food.get("fdcId"),
                "description": description,
                "brand_owner": food.get("brandOwner", "").strip(),
                "brand_category": category,
                "ingredients": food.get("ingredients", "").strip(),
                "serving_size": food.get("servingSize"),
                "serving_size_unit": food.get("servingSizeUnit", ""),
                "household_serving": food.get("householdServingFullText", "").strip(),
                "gtin_upc": food.get("gtinUpc", ""),
                "data_type": "branded",
                "nutrients": nutrients,
            }

            out.write(json.dumps(flat, ensure_ascii=False) + "\n")
            written += 1

            if written % 50000 == 0:
                print(f"    Written {written:,} items...")

    print(f"  Done: {written:,} items written, {skipped:,} skipped (no description)")
    print(f"  Output: {output_path}")

    # Save unique categories
    sorted_cats = sorted(category_counter.items(), key=lambda x: -x[1])
    with open(categories_path, "w", encoding="utf-8") as f:
        json.dump(sorted_cats, f, indent=2, ensure_ascii=False)
    print(f"  Unique categories: {len(sorted_cats)}")
    print(f"  Top 20 categories:")
    for cat, count in sorted_cats[:20]:
        print(f"    {cat}: {count:,}")

    return category_counter


def process_sr_legacy(usda_dir):
    """Parse SR Legacy JSON → flat JSONL."""
    input_path = os.path.join(usda_dir, "FoodData_Central_sr_legacy_food_json_2018-04.json")
    output_path = os.path.join(usda_dir, "sr_legacy_flat.jsonl")

    print(f"\nLoading SR Legacy from {input_path}...")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    foods = data.get("SRLegacyFoods", [])
    print(f"  Found {len(foods):,} SR Legacy items")

    written = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for food in foods:
            description = food.get("description", "").strip()
            if not description:
                continue

            nutrients = extract_nutrients_from_food_nutrients(food.get("foodNutrients", []))

            # SR Legacy has foodCategory
            category = ""
            fc = food.get("foodCategory", {})
            if isinstance(fc, dict):
                category = fc.get("description", "")

            flat = {
                "fdc_id": food.get("fdcId"),
                "description": description,
                "brand_owner": "",
                "brand_category": category,
                "ingredients": "",
                "serving_size": None,
                "serving_size_unit": "",
                "household_serving": "",
                "gtin_upc": "",
                "data_type": "sr_legacy",
                "nutrients": nutrients,
            }

            out.write(json.dumps(flat, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Done: {written:,} items written")
    print(f"  Output: {output_path}")


def build_category_mapping(usda_dir, category_counter):
    """Build initial mapping from 21 pantry categories to USDA categories."""
    output_path = os.path.join(usda_dir, "category_mapping.json")

    usda_cats = {cat.lower(): cat for cat, _ in category_counter.items() if cat}

    mapping = {}
    for pantry_cat, hints in PANTRY_TO_USDA_CATEGORY_HINTS.items():
        matched = []
        for hint in hints:
            hint_lower = hint.lower()
            for usda_lower, usda_original in usda_cats.items():
                if hint_lower in usda_lower or usda_lower in hint_lower:
                    if usda_original not in matched:
                        matched.append(usda_original)
        mapping[pantry_cat] = {
            "usda_categories": matched[:10],  # top 10 matches
            "hint_keywords": hints,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"\nCategory mapping saved to {output_path}")
    print(f"Pantry categories mapped:")
    for pantry_cat, info in mapping.items():
        n = len(info["usda_categories"])
        print(f"  {pantry_cat}: {n} USDA categories matched")
        if info["usda_categories"]:
            for uc in info["usda_categories"][:3]:
                print(f"    → {uc}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess USDA FoodData Central")
    parser.add_argument("--usda-dir", type=str, default="./usda_data")
    args = parser.parse_args()

    print("=" * 60)
    print("USDA FoodData Central Preprocessing")
    print("=" * 60)

    # 1. Process branded foods
    category_counter = process_branded_foods(args.usda_dir)

    # 2. Process SR Legacy
    process_sr_legacy(args.usda_dir)

    # 3. Build category mapping
    build_category_mapping(args.usda_dir, category_counter)

    # 4. Summary
    branded_path = os.path.join(args.usda_dir, "branded_foods_flat.jsonl")
    sr_path = os.path.join(args.usda_dir, "sr_legacy_flat.jsonl")

    branded_size = os.path.getsize(branded_path) / (1024 * 1024)
    sr_size = os.path.getsize(sr_path) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Branded foods: {branded_path} ({branded_size:.1f} MB)")
    print(f"  SR Legacy:     {sr_path} ({sr_size:.1f} MB)")
    print(f"  Categories:    {os.path.join(args.usda_dir, 'usda_categories.json')}")
    print(f"  Mapping:       {os.path.join(args.usda_dir, 'category_mapping.json')}")


if __name__ == "__main__":
    main()
