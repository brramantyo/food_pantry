#!/usr/bin/env python3
"""
fix_case_v9.py
==============
Fix case-mismatch in merged training data (grocery + food pantry).
Some labels from Grocery Store Dataset have lowercase variants:
  "Seafood - canned" vs "Seafood - Canned"
  "Vegetables - canned" vs "Vegetables - Canned"

This script normalizes all category names in JSONL files to the canonical casing.
"""

import json
import os
import sys
from pathlib import Path

# Canonical category names (correct casing)
CANONICAL = {
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


def normalize_name(name):
    """Normalize category name to canonical casing."""
    return CANONICAL.get(name.lower().strip(), name)


def fix_jsonl(input_path, output_path=None):
    """Fix category casing in a JSONL file."""
    if output_path is None:
        output_path = input_path  # overwrite in place

    lines = []
    fixed_count = 0
    total_items = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # Fix target field
            if "target" in record:
                try:
                    target = json.loads(record["target"])
                    if "items" in target:
                        for item in target["items"]:
                            total_items += 1
                            original = item.get("name", "")
                            normalized = normalize_name(original)
                            if original != normalized:
                                item["name"] = normalized
                                fixed_count += 1
                        record["target"] = json.dumps(target)
                except (json.JSONDecodeError, KeyError):
                    pass

            lines.append(json.dumps(record, ensure_ascii=False))

    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"  {input_path} → {output_path}")
    print(f"    Total items: {total_items}, Fixed: {fixed_count}")
    return fixed_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix category name casing in JSONL files")
    parser.add_argument("files", nargs="+", help="JSONL files to fix")
    parser.add_argument("--dry-run", action="store_true", help="Don't write, just report")
    args = parser.parse_args()

    total_fixed = 0
    for filepath in args.files:
        if not os.path.exists(filepath):
            print(f"  SKIP: {filepath} not found")
            continue
        if args.dry_run:
            print(f"\n[DRY RUN] Scanning {filepath}...")
            # Read and count without writing
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "target" in record:
                        try:
                            target = json.loads(record["target"])
                            if "items" in target:
                                for item in target["items"]:
                                    original = item.get("name", "")
                                    normalized = normalize_name(original)
                                    if original != normalized:
                                        print(f"    Would fix: '{original}' → '{normalized}'")
                                        total_fixed += 1
                        except (json.JSONDecodeError, KeyError):
                            pass
        else:
            print(f"\nFixing {filepath}...")
            total_fixed += fix_jsonl(filepath)

    print(f"\nTotal fixes: {total_fixed}")


if __name__ == "__main__":
    main()
