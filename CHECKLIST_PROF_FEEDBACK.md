# Professor Feedback Checklist

## From Prof Zizhan Zheng (Collaborator)

### 1. ✅ Full confusion matrix analysis
- **Status:** DONE (in `report_task1_v2.tex` — failure analysis section)
- Identified main failure mode: under-detection in multi-label images, NOT inter-class confusion
- Per-class precision/recall/F1 breakdown across all experiments

### 2. ✅ Loss reweighting / class-balanced sampling
- **Status:** DONE (built into training since v7)
- Class-balanced oversampling with `min_samples_per_class`
- Confusion-pair boosting for weakest classes (v11: Dairy, Veg Fresh, Nut Butters, Granola, Ready Meals)
- Focal loss explored in v9 (gamma=1.0 + label smoothing)

### 3. ✅ Florence-2 REGION_TO_OCR as input signal
- **Status:** DONE — explored in 2 variants
- **OCR v2 (aggressive):** Dropped F1 from 57.6% → 48.4% (too noisy)
- **OCR v3 (conservative):** +50% F1 for Granola, +9.4% for Snacks, +6.8% for Soup. Marginal overall.
- Conclusion: OCR helps for select categories but introduces too much noise for general use
- Documented in `report_task1_extended.tex`

### 4. ✅ Detection/segmentation-first pipeline
- **Status:** DONE — explored in 3 variants
- **Vanilla OD → classify:** 52.2% F1 ❌
- **Fine-tuned OD (labels only):** ~40% F1 ❌ (label hallucination)
- **Fine-tuned OD → crop → v11 classify:** 56.4% F1 ❌ (recall 87%↑ but precision 42%↓)
- Key finding: v11 over-predicts on individual crops. Direct classification still best.
- Documented in `report_task1_extended.tex`

### 5. 🔲 VLM with chain-of-thought reasoning
- **Status:** NOT YET EXPLORED
- Could use GPT-4V / Claude vision as a comparison baseline
- Or prompt Florence-2 with multi-step reasoning
- **Priority:** LOW — would be nice for report but not critical

---

## From Prof Yin Sun (Advisor)

### 1. ✅ Run through Step 2 (USDA matching) before improving Step 1
- **Status:** DONE
- Full end-to-end pipeline evaluated on ALL 166 test images
- Results: nutrition profiles for 21 categories, error propagation analysis
- Key finding: misclassification overestimates calories +48%, protein stable (+1g)
- Documented in `report_task2_full.tex`

### 2. ✅ "If step 1 classification is inaccurate, it does not mean the nutrition outcome is bad"
- **Status:** DIRECTLY ANSWERED with data
- Error propagation table in `report_task2_full.tex`
- Correct avg: 167 kcal | Wrong avg: 247 kcal | Δ: +81 kcal
- Protein almost unaffected (+1g)
- Pantry-level aggregation further reduces per-item errors

### 3. ✅ Failure examples with reasoning
- **Status:** DONE
- `report_task1_v2.tex` Section "Failure Analysis" with specific examples
- `report_task2_full.tex` qualitative examples (correct + incorrect + reasoning)
- Sample predictions in all evaluation logs

### 4. ⚠️ Reports too infrequent
- **Status:** IMPROVED — 2 new LaTeX sections written today
  - `report_task1_extended.tex` — all alternative experiments
  - `report_task2_full.tex` — full Task 2 evaluation
  - Auto-generated LaTeX tables from `eval_pipeline_full_table.tex`
- **TODO:** Compile into single document and send to prof

### 5. ✅ Segmentation before classification? Can Florence-2 do that?
- **Status:** ANSWERED through detection experiments
- Florence-2 CAN do OD (bounding boxes) — we fine-tuned it for pantry items
- Detection works (recall 87%) but classifier breaks on individual crops
- Segmentation (SAM) would have same problem — not worth separate experiment
- Documented reasoning in report

### 6. ✅ Contrastive learning?
- **Status:** DONE + IN PROGRESS
- v1: 77.8% Valid Micro F1 (frozen encoder, SupCon + BCE) — job 15430 finishing
- v2a: Better hyperparams (bs=16, alpha=0.3, auto-threshold) — queued (job 15444)
- v2b: Partial unfreeze last 2 encoder layers — queued (job 15445)
- Ensemble: v11 × contrastive — queued (job 15443)
- **Surprisingly competitive:** matches v11 (76.5%) with only 1.7M trainable params vs 204M frozen

---

## Summary

| Request | Status | Notes |
|---------|--------|-------|
| Confusion matrix analysis | ✅ Done | In report_task1_v2.tex |
| Loss reweighting / balanced sampling | ✅ Done | Since v7, with oversampling + focal loss |
| OCR input signal | ✅ Done | v2 (aggressive) ❌, v3 (conservative) marginal |
| Detection/segmentation pipeline | ✅ Done | 3 variants tested, all worse than direct |
| VLM chain-of-thought | 🔲 Not yet | Low priority |
| Run Task 2 before improving Task 1 | ✅ Done | Full 166-image eval with nutrition |
| Error propagation analysis | ✅ Done | +81 kcal overestimate, protein stable |
| Failure examples with reasoning | ✅ Done | In both report sections |
| More frequent reports | ⚠️ Improving | 2 new sections today, need to compile & send |
| Segmentation + Florence-2 | ✅ Answered | OD works, but crop classification fails |
| Contrastive learning | ✅ Running | v1 done, v2a/v2b/ensemble queued |
