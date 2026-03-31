#!/usr/bin/env python3
"""
Analyze different fusion strategies from eval_od_contrastive.json
No training needed — just recombines existing predictions.
"""
import json
from collections import Counter

with open("eval_od_contrastive.json") as f:
    data = json.load(f)

results = data["results"]

strats = {}
for name in ["full_only", "crops_only", "union", "intersection", "intersection_fallback"]:
    strats[name] = {"tp": Counter(), "fp": Counter(), "fn": Counter()}

for r in results:
    target = set(r["target"])
    full = set(r["full_image"])
    crops = set(r["crops"])

    preds_map = {
        "full_only": full,
        "crops_only": crops,
        "union": full | crops,
        "intersection": full & crops,
        "intersection_fallback": (full & crops) if crops else full,
    }

    for name, preds in preds_map.items():
        s = strats[name]
        for cls in target | preds:
            if cls in target and cls in preds:
                s["tp"][cls] += 1
            elif cls in preds:
                s["fp"][cls] += 1
            elif cls in target:
                s["fn"][cls] += 1

print(f"\n{'='*75}")
print(f"  FUSION STRATEGY ANALYSIS (OD + Contrastive)")
print(f"{'='*75}")
print(f"  {'Strategy':<25} {'Micro P':>8} {'Micro R':>8} {'Micro F1':>9} {'Exact Match':>12}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*9} {'-'*12}")

for name in ["full_only", "crops_only", "union", "intersection", "intersection_fallback"]:
    s = strats[name]
    tp = sum(s["tp"].values())
    fp = sum(s["fp"].values())
    fn = sum(s["fn"].values())
    p = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * p * rec / max(p + rec, 1e-8)

    em = 0
    for res in results:
        target = set(res["target"])
        full = set(res["full_image"])
        crops = set(res["crops"])
        pred_map = {
            "full_only": full,
            "crops_only": crops,
            "union": full | crops,
            "intersection": full & crops,
            "intersection_fallback": (full & crops) if crops else full,
        }
        if pred_map[name] == target:
            em += 1

    print(f"  {name:<25} {p:>7.1%} {rec:>7.1%} {f1:>8.1%} {em:>5}/{len(results)} ({em/len(results)*100:.1f}%)")

# Show cases where intersection helps vs hurts
print(f"\n{'='*75}")
print(f"  INTERSECTION vs FULL — Case Analysis")
print(f"{'='*75}")
helped = 0
hurt = 0
same = 0
for r in results:
    target = set(r["target"])
    full = set(r["full_image"])
    crops = set(r["crops"])
    inter = (full & crops) if crops else full

    full_correct = target & full
    inter_correct = target & inter
    full_wrong = full - target
    inter_wrong = inter - target

    full_f1 = 2*len(full_correct) / max(len(full_correct)*2 + len(full_wrong) + len(target - full), 1)
    inter_f1 = 2*len(inter_correct) / max(len(inter_correct)*2 + len(inter_wrong) + len(target - inter), 1)

    if inter_f1 > full_f1:
        helped += 1
    elif inter_f1 < full_f1:
        hurt += 1
    else:
        same += 1

print(f"  Intersection HELPED: {helped}")
print(f"  Intersection HURT:   {hurt}")
print(f"  Same:                {same}")
print(f"  Total:               {len(results)}")
