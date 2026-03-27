# Contrastive Learning Plan for Food Pantry Classification

## Why Contrastive Learning? (Prof Zheng + Prof Sun suggestion)

Our current approach (generative classification with Florence-2) struggles with:
1. **Multi-label under-detection** — model predicts 1 class when 2-3 are present
2. **Fine-grained confusion** — Ready Meals vs Carbohydrate Meal look similar
3. **Class imbalance** — some classes have <10 training samples

Contrastive learning can help because it:
- Learns **discriminative embeddings** (push apart similar-but-different classes)
- Handles **multi-label** naturally (each item gets its own embedding)
- Is **robust to class imbalance** (works per-pair, not per-class)
- Can leverage **pre-trained CLIP/Florence-2 features** without full retraining

## Approach Options

### Option A: Supervised Contrastive Fine-tuning (SupCon)
**Paper:** Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020)

- Use Florence-2 vision encoder as backbone
- Add projection head → 128-dim embedding space
- Train with SupCon loss: same-class pairs pulled together, different-class pushed apart
- Then train linear classifier on frozen embeddings

**Pros:** Simple, proven, handles imbalance well
**Cons:** Needs detection-first pipeline (one item per crop)

### Option B: CLIP-style Contrastive (image-text)
**Paper:** Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)

- Use Florence-2 as both image and text encoder
- Contrastive loss between image embeddings and category text embeddings
- Category texts: "A photo of Baby Food in a pantry", "A photo of Granola Products", etc.
- Zero-shot capable after training

**Pros:** Leverages Florence-2's multimodal nature, zero-shot generalization
**Cons:** Need to design good text prompts per category

### Option C: Multi-label Contrastive
- Modified SupCon that handles multi-label images
- Images sharing ANY category are partial positives
- Weight contrastive pairs by label overlap (Jaccard similarity)

**Pros:** Directly addresses our multi-label problem
**Cons:** More complex, less standard

## Recommended: Option A (SupCon) + Detection Pipeline

Combining with the detection-first pipeline that both profs want:

```
Image → Florence-2 OD (detect boxes)
  → Crop each box
    → Florence-2 vision encoder → SupCon embedding
      → Nearest-class classifier → Category
        → USDA match → Nutrition
```

This solves multi-label (each crop = single item) AND gets better features.

## Implementation Plan

### Phase 1: Extract features (no training)
1. Use Florence-2 vision encoder to extract features for all training crops
2. Evaluate kNN classifier on these features → baseline embedding quality
3. ~1 hour work

### Phase 2: SupCon fine-tuning
1. Add projection head (MLP: 1024 → 512 → 128)
2. Train with SupCon loss on training set crops
3. Use augmentation (same as v11) for positive pairs
4. ~2-3 hours training

### Phase 3: Evaluate
1. Extract embeddings for test set
2. kNN + linear probe evaluation
3. Compare with current Florence-2 generative approach
4. Per-class analysis (especially weak classes)

## Key References

1. Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020) — SupCon loss
2. Radford et al., "CLIP" (2021) — image-text contrastive
3. Pettersson et al., "Multimodal Fine-Grained Grocery Product Recognition" (2024) — grocery domain
4. Klasson et al., "A Hierarchical Grocery Store Image Dataset" (WACV 2019) — grocery dataset

## Estimated Timeline

- Feature extraction baseline: 1 day
- SupCon training: 1 day  
- Evaluation + comparison: 0.5 day
- Total: ~2.5 days
