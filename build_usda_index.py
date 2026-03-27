#!/usr/bin/env python3
"""
build_usda_index.py
===================
Build a sentence transformer embedding index over USDA food descriptions
for semantic search matching.

Creates:
  - usda_data/usda_embeddings.npy    (float16 embeddings matrix)
  - usda_data/usda_metadata.jsonl    (parallel metadata for each embedding)
  - usda_data/usda_index_info.json   (index stats)

Usage:
  python build_usda_index.py --usda-dir ./usda_data
  python build_usda_index.py --usda-dir ./usda_data --model all-MiniLM-L6-v2
"""

import argparse
import json
import os
import sys
import time
import numpy as np

def load_flat_foods(jsonl_path, max_items=None):
    """Load preprocessed flat JSONL foods."""
    foods = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items and i >= max_items:
                break
            line = line.strip()
            if line:
                foods.append(json.loads(line))
    return foods


def build_search_text(food):
    """Build a rich text string for embedding from food metadata."""
    parts = []

    desc = food.get("description", "")
    if desc:
        parts.append(desc)

    brand = food.get("brand_owner", "")
    if brand:
        parts.append(brand)

    category = food.get("brand_category", "")
    if category:
        parts.append(category)

    # Add first 200 chars of ingredients for extra signal
    ingredients = food.get("ingredients", "")
    if ingredients:
        parts.append(ingredients[:200])

    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Build USDA semantic search index")
    parser.add_argument("--usda-dir", type=str, default="./usda_data")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=None,
                        help="Limit items for testing (None = all)")
    parser.add_argument("--include-sr", action="store_true", default=True,
                        help="Include SR Legacy foods")
    args = parser.parse_args()

    print("=" * 60)
    print("Building USDA Semantic Search Index")
    print(f"Model: {args.model}")
    print("=" * 60)

    # 1. Load preprocessed data
    branded_path = os.path.join(args.usda_dir, "branded_foods_flat.jsonl")
    sr_path = os.path.join(args.usda_dir, "sr_legacy_flat.jsonl")

    print(f"\nLoading branded foods from {branded_path}...")
    foods = load_flat_foods(branded_path, args.max_items)
    print(f"  Loaded {len(foods):,} branded items")

    if args.include_sr and os.path.exists(sr_path):
        print(f"Loading SR Legacy from {sr_path}...")
        sr_foods = load_flat_foods(sr_path)
        print(f"  Loaded {len(sr_foods):,} SR Legacy items")
        foods.extend(sr_foods)

    print(f"\nTotal foods to index: {len(foods):,}")

    # 2. Build search texts
    print("Building search texts...")
    texts = [build_search_text(f) for f in foods]
    print(f"  Average text length: {sum(len(t) for t in texts) / len(texts):.0f} chars")

    # 3. Load sentence transformer
    print(f"\nLoading sentence transformer: {args.model}...")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer(args.model)
    print(f"  Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    # 4. Encode in batches
    print(f"\nEncoding {len(texts):,} texts (batch_size={args.batch_size})...")
    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # for cosine similarity via dot product
    )

    elapsed = time.time() - start
    print(f"  Encoding done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} items/sec)")

    # 5. Save embeddings as float16 to save space
    embeddings_f16 = embeddings.astype(np.float16)
    emb_path = os.path.join(args.usda_dir, "usda_embeddings.npy")
    np.save(emb_path, embeddings_f16)
    emb_size = os.path.getsize(emb_path) / (1024 * 1024)
    print(f"  Embeddings saved: {emb_path} ({emb_size:.1f} MB)")

    # 6. Save metadata (parallel to embeddings)
    meta_path = os.path.join(args.usda_dir, "usda_metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for food in foods:
            # Save only what we need for matching (not full nutrients)
            meta = {
                "fdc_id": food.get("fdc_id"),
                "description": food.get("description", ""),
                "brand_owner": food.get("brand_owner", ""),
                "brand_category": food.get("brand_category", ""),
                "data_type": food.get("data_type", ""),
                "serving_size": food.get("serving_size"),
                "serving_size_unit": food.get("serving_size_unit", ""),
                "household_serving": food.get("household_serving", ""),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    meta_size = os.path.getsize(meta_path) / (1024 * 1024)
    print(f"  Metadata saved: {meta_path} ({meta_size:.1f} MB)")

    # 7. Save index info
    info = {
        "model": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "num_items": len(foods),
        "num_branded": sum(1 for f in foods if f.get("data_type") == "branded"),
        "num_sr_legacy": sum(1 for f in foods if f.get("data_type") == "sr_legacy"),
        "dtype": "float16",
        "normalized": True,
        "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    info_path = os.path.join(args.usda_dir, "usda_index_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Index info saved: {info_path}")

    print(f"\n{'=' * 60}")
    print("INDEX BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Items indexed: {info['num_items']:,}")
    print(f"  Branded: {info['num_branded']:,}")
    print(f"  SR Legacy: {info['num_sr_legacy']:,}")
    print(f"  Embedding dim: {info['embedding_dim']}")
    print(f"  Total size: {emb_size + meta_size:.1f} MB")


if __name__ == "__main__":
    main()
