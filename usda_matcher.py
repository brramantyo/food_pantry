#!/usr/bin/env python3
"""
usda_matcher.py
===============
Match pantry item predictions to USDA FoodData Central entries.

Supports:
  1. Category-level matching: pantry category → top USDA items in that category
  2. Semantic search: free-text query → nearest USDA items by embedding similarity
  3. Hybrid: category filter + semantic reranking

Can run standalone for testing, or be imported as a module.

Usage:
  # Test with a query
  python usda_matcher.py --query "granola bar cinnamon" --top-k 10

  # Match all Task 1 predictions
  python usda_matcher.py --predictions eval_results_v12b.json --top-k 5

  # Interactive mode
  python usda_matcher.py --interactive
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict


class USDAMatcher:
    """Semantic + category-based matcher for USDA foods."""

    def __init__(self, usda_dir="./usda_data", model_name="all-MiniLM-L6-v2"):
        self.usda_dir = usda_dir
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.metadata = None
        self.nutrients = None  # full nutrient data from flat JSONL
        self.category_index = None  # category → list of indices

        self._load_index()
        self._build_category_index()

    def _load_index(self):
        """Load prebuilt embeddings and metadata."""
        print("Loading USDA index...")

        # Load embeddings
        emb_path = os.path.join(self.usda_dir, "usda_embeddings.npy")
        self.embeddings = np.load(emb_path).astype(np.float32)  # upcast for dot product
        print(f"  Embeddings: {self.embeddings.shape}")

        # Load metadata
        meta_path = os.path.join(self.usda_dir, "usda_metadata.jsonl")
        self.metadata = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        print(f"  Metadata: {len(self.metadata)} items")

        # Load full nutrient data (branded + sr_legacy)
        self.nutrients = {}
        for fname in ["branded_foods_flat.jsonl", "sr_legacy_flat.jsonl"]:
            fpath = os.path.join(self.usda_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            fdc_id = item.get("fdc_id")
                            if fdc_id and item.get("nutrients"):
                                self.nutrients[fdc_id] = item["nutrients"]
        print(f"  Nutrients: {len(self.nutrients)} items with data")

        # Load sentence transformer
        print(f"  Loading model: {self.model_name}...")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
        print("  Ready!")

    def _build_category_index(self):
        """Build inverted index: USDA category → list of indices."""
        self.category_index = defaultdict(list)
        for i, meta in enumerate(self.metadata):
            cat = meta.get("brand_category", "").strip().lower()
            if cat:
                self.category_index[cat].append(i)
        print(f"  Category index: {len(self.category_index)} unique categories")

    def search_semantic(self, query, top_k=10):
        """Pure semantic search — encode query, find nearest neighbors."""
        query_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores = self.embeddings @ query_emb.T  # dot product = cosine sim (normalized)
        scores = scores.flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            meta = self.metadata[idx].copy()
            meta["score"] = float(scores[idx])
            fdc_id = meta.get("fdc_id")
            if fdc_id in self.nutrients:
                meta["nutrients"] = self.nutrients[fdc_id]
            results.append(meta)
        return results

    def search_by_category(self, usda_category, top_k=20):
        """Get all items in a specific USDA category."""
        cat_lower = usda_category.strip().lower()
        indices = self.category_index.get(cat_lower, [])
        results = []
        for idx in indices[:top_k]:
            meta = self.metadata[idx].copy()
            meta["score"] = 1.0  # category match
            fdc_id = meta.get("fdc_id")
            if fdc_id in self.nutrients:
                meta["nutrients"] = self.nutrients[fdc_id]
            results.append(meta)
        return results

    def search_hybrid(self, query, pantry_category=None, top_k=10, category_boost=0.15):
        """
        Hybrid search: semantic similarity + optional category filtering/boosting.

        If pantry_category is given, items matching related USDA categories
        get a score boost.
        """
        # Load category mapping
        mapping_path = os.path.join(self.usda_dir, "category_mapping.json")
        related_cats = set()
        if pantry_category and os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
            if pantry_category in mapping:
                for uc in mapping[pantry_category].get("usda_categories", []):
                    related_cats.add(uc.lower())

        # Semantic search
        query_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores = (self.embeddings @ query_emb.T).flatten()

        # Category boost
        if related_cats:
            for i, meta in enumerate(self.metadata):
                cat = meta.get("brand_category", "").strip().lower()
                if any(rc in cat or cat in rc for rc in related_cats):
                    scores[i] += category_boost

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            meta = self.metadata[idx].copy()
            meta["score"] = float(scores[idx])
            fdc_id = meta.get("fdc_id")
            if fdc_id in self.nutrients:
                meta["nutrients"] = self.nutrients[fdc_id]
            results.append(meta)
        return results

    def match_pantry_prediction(self, pantry_category, package_type=None, top_k=5):
        """
        Match a Task 1 prediction to USDA entries.

        Args:
            pantry_category: e.g., "Granola Products"
            package_type: e.g., "box", "bag", "can" (optional extra signal)
            top_k: number of matches to return
        """
        # Build query from category + package type
        query = pantry_category
        if package_type:
            query += f" {package_type}"

        return self.search_hybrid(query, pantry_category=pantry_category, top_k=top_k)

    def get_category_nutrition_summary(self, pantry_category, top_k=50):
        """
        Get average nutritional profile for a pantry category.
        Matches top_k USDA items and averages their nutrients.
        """
        results = self.match_pantry_prediction(pantry_category, top_k=top_k)

        # Collect nutrients
        nutrient_sums = defaultdict(list)
        for r in results:
            nuts = r.get("nutrients", {})
            for key, val in nuts.items():
                if val is not None:
                    nutrient_sums[key].append(val)

        # Average
        summary = {}
        for key, values in nutrient_sums.items():
            if values:
                summary[key] = {
                    "mean": round(sum(values) / len(values), 1),
                    "min": round(min(values), 1),
                    "max": round(max(values), 1),
                    "n_samples": len(values),
                }

        return {
            "pantry_category": pantry_category,
            "matched_items": len(results),
            "nutrition_summary": summary,
            "top_matches": [
                {"description": r["description"], "brand": r.get("brand_owner", ""), "score": r["score"]}
                for r in results[:5]
            ],
        }


def format_result(result, show_nutrients=True):
    """Pretty-print a single search result."""
    desc = result.get("description", "?")
    brand = result.get("brand_owner", "")
    cat = result.get("brand_category", "")
    score = result.get("score", 0)
    fdc_id = result.get("fdc_id", "?")

    line = f"  [{score:.3f}] {desc}"
    if brand:
        line += f" — {brand}"
    if cat:
        line += f" ({cat})"
    line += f"  [FDC#{fdc_id}]"
    print(line)

    if show_nutrients and "nutrients" in result:
        nuts = result["nutrients"]
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
            print(f"    Nutrients: {' | '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description="USDA Food Matcher")
    parser.add_argument("--usda-dir", type=str, default="./usda_data")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--query", type=str, default=None, help="Search query")
    parser.add_argument("--category", type=str, default=None, help="Pantry category")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to eval results JSON from Task 1")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--nutrition-summary", action="store_true",
                        help="Show average nutrition for each pantry category")
    args = parser.parse_args()

    matcher = USDAMatcher(usda_dir=args.usda_dir, model_name=args.model)

    if args.nutrition_summary:
        # Show nutrition summary for all 21 categories
        categories = [
            "Baby Food", "Beans and Legumes - Canned or Dried",
            "Bread and Bakery Products", "Canned Tomato Products",
            "Carbohydrate Meal", "Condiments and Sauces",
            "Dairy and Dairy Alternatives", "Desserts and Sweets",
            "Drinks", "Fresh Fruit", "Fruits - Canned or Processed",
            "Granola Products", "Meat and Poultry - Canned",
            "Meat and Poultry - Fresh", "Nut Butters and Nuts",
            "Ready Meals", "Savory Snacks and Crackers",
            "Seafood - Canned", "Soup",
            "Vegetables - Canned", "Vegetables - Fresh",
        ]
        print("\n" + "=" * 60)
        print("NUTRITION SUMMARY BY PANTRY CATEGORY")
        print("=" * 60)
        for cat in categories:
            summary = matcher.get_category_nutrition_summary(cat)
            print(f"\n--- {cat} ---")
            print(f"  Matched {summary['matched_items']} USDA items")
            nuts = summary.get("nutrition_summary", {})
            if "energy_kcal" in nuts:
                e = nuts["energy_kcal"]
                print(f"  Energy: {e['mean']} kcal (range: {e['min']}–{e['max']}, n={e['n_samples']})")
            if "protein_g" in nuts:
                p = nuts["protein_g"]
                print(f"  Protein: {p['mean']}g (range: {p['min']}–{p['max']}g)")
            if "carbohydrate_g" in nuts:
                c = nuts["carbohydrate_g"]
                print(f"  Carbs: {c['mean']}g (range: {c['min']}–{c['max']}g)")
            if "total_fat_g" in nuts:
                f_ = nuts["total_fat_g"]
                print(f"  Fat: {f_['mean']}g (range: {f_['min']}–{f_['max']}g)")
            print(f"  Top matches:")
            for m in summary["top_matches"][:3]:
                print(f"    → {m['description']} ({m['brand']})")
        return

    if args.query:
        print(f"\nSearching: '{args.query}'")
        if args.category:
            print(f"Category filter: {args.category}")
            results = matcher.search_hybrid(args.query, pantry_category=args.category, top_k=args.top_k)
        else:
            results = matcher.search_semantic(args.query, top_k=args.top_k)
        print(f"Top {len(results)} results:\n")
        for r in results:
            format_result(r)
        return

    if args.predictions:
        print(f"\nMatching Task 1 predictions from: {args.predictions}")
        with open(args.predictions, "r") as f:
            eval_data = json.load(f)

        predictions = eval_data.get("predictions", eval_data.get("results", []))
        if isinstance(eval_data, list):
            predictions = eval_data

        all_matches = []
        for pred in predictions:
            # Extract predicted categories
            pred_text = pred.get("predicted", pred.get("prediction", ""))
            try:
                pred_obj = json.loads(pred_text) if isinstance(pred_text, str) else pred_text
                items = pred_obj.get("items", [])
            except (json.JSONDecodeError, AttributeError):
                continue

            image = pred.get("image", "?")
            print(f"\n{'='*50}")
            print(f"Image: {os.path.basename(image)}")

            for item in items:
                name = item.get("name", "")
                pkg = item.get("package_type", "")
                print(f"\n  Category: {name} ({pkg})")
                matches = matcher.match_pantry_prediction(name, package_type=pkg, top_k=args.top_k)
                for m in matches:
                    format_result(m)
                all_matches.append({
                    "image": image,
                    "pantry_category": name,
                    "package_type": pkg,
                    "usda_matches": [
                        {
                            "fdc_id": m.get("fdc_id"),
                            "description": m.get("description"),
                            "brand": m.get("brand_owner", ""),
                            "score": m.get("score"),
                            "nutrients": m.get("nutrients", {}),
                        }
                        for m in matches
                    ]
                })

        # Save matches
        output_path = args.predictions.replace(".json", "_usda_matches.json")
        with open(output_path, "w") as f:
            json.dump(all_matches, f, indent=2, ensure_ascii=False)
        print(f"\nMatches saved to {output_path}")
        return

    if args.interactive:
        print("\n=== USDA Food Matcher — Interactive Mode ===")
        print("Type a food description to search. 'quit' to exit.\n")
        while True:
            query = input("Query> ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            results = matcher.search_semantic(query, top_k=args.top_k)
            print(f"\nTop {len(results)} results:")
            for r in results:
                format_result(r)
            print()
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
