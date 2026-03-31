#!/usr/bin/env python3
import json

for split in ['train', 'valid']:
    with open(f'{split}/_annotations.coco.json') as f:
        coco = json.load(f)
    excluded = {6, 12, 18, 23}
    total = len(coco['annotations'])
    valid = sum(1 for a in coco['annotations'] if a['category_id'] not in excluded)
    too_small = 0
    for a in coco['annotations']:
        if a['category_id'] in excluded:
            continue
        bbox = a['bbox']
        w = float(bbox[2])
        h = float(bbox[3])
        pw = w * 1.3
        ph = h * 1.3
        if pw < 32 or ph < 32:
            too_small += 1
    print(f'{split}: total={total}, valid={valid}, too_small={too_small}, usable={valid - too_small}')

print(f'\nExpected GT crops = sum of usable across both splits')
