# -*- coding: utf-8 -*-
"""probe_dior_root.py — 反查每个 blob 用的 DIOR 数据根（内联复刻 load_dior，避免模块重载）
"""
import os, sys, glob, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

DIOR_ID2NAME = {0: 'airplane', 1: 'ship', 2: 'storage tank', 3: 'baseball diamond',
                4: 'tennis court', 5: 'basketball court', 6: 'ground track field',
                7: 'harbor', 8: 'bridge', 9: 'vehicle', 10: 'expressway service area',
                11: 'expressway toll station', 12: 'dam', 13: 'windmill', 14: 'overpass',
                15: 'golf course', 16: 'roundabout', 17: 'soccer ball field',
                18: 'basketball court', 19: 'stadium'}

D = r'C:/Users/张三/modeling/dior_data'
ROOTS = sorted({str(Path(p).parent) for p in glob.glob(D + '/**/vehicle_*.jpg', recursive=True)} |
               {str(Path(p).parent) for p in glob.glob(D + '/**/ship_*.jpg', recursive=True)})
PARQS = sorted(set(glob.glob(D + '/**/*.parquet', recursive=True)))
print('ROOTS:'); [print('  ', r) for r in ROOTS]
print('PARQS:'); [print('  ', p) for p in PARQS]


def build_gt_index(pq, name2id, cache={}):
    if pq in cache:
        return cache[pq]
    df = pd.read_parquet(pq)
    idx = defaultdict(list)
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        for i, cat in enumerate(row['objects']['category']):
            cid = int(cat)
            nm = DIOR_ID2NAME.get(cid)
            if nm != name2id:
                continue
            x, y, w, h = [int(v) for v in row['objects']['bbox'][i]]
            idx[img_id].append([x, y, x + w, y + h])
    cache[pq] = idx
    return idx


def load_dior_like(root, pq, cls_name, n):
    test_ids = set()
    for f in glob.glob(os.path.join(root, '*.jpg')):
        for part in reversed(Path(f).stem.split('_')):
            if part.isdigit():
                test_ids.add(int(part)); break
    name2id = {v: k for k, v in DIOR_ID2NAME.items()}
    cls_id = name2id.get(cls_name)
    gt_index = build_gt_index(pq, cls_name)  # reuse by class name
    pats = [os.path.join(root, f'{cls_name}_*.jpg'),
            os.path.join(root, f"{cls_name.replace(' ', '_')}_*.jpg")]
    paths = []
    for pat in pats:
        paths = sorted(glob.glob(pat))
        if paths:
            break
    out = []
    for p in paths:
        img_id = None
        for part in reversed(Path(p).stem.split('_')):
            if part.isdigit():
                img_id = int(part); break
        if img_id is None or img_id not in test_ids:
            continue
        g = gt_index.get(img_id, [])
        if not g:
            continue
        out.append((p, g))
        if len(out) >= n:
            break
    return out


def main():
    blobs = {
        'qual_car_n15': (r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache/qual_car_off20_n15.pkl', 'vehicle', 20, 15),
        'n100_car': (r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache/fwd_dior2coco_car_off20_n100.pkl', 'vehicle', 20, 100),
        'n100_boat': (r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache/fwd_dior2coco_boat_off20_n100.pkl', 'ship', 20, 100),
        'n8_boat': (r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache/fwd_dior2coco_boat_off20_n8.pkl', 'ship', 20, 8),
    }
    for tag, (bf, cls, off, n) in blobs.items():
        bl = pickle.load(open(bf, 'rb'))
        bc = [len(p['gt']) for p in bl['per_img']]
        print(f'\n[{tag}] cls={cls} blob n={len(bc)} sum={sum(bc)} counts[:12]={bc[:12]}')
        for root in ROOTS:
            for pq in PARQS:
                try:
                    seq = load_dior_like(root, pq, cls, off + n)
                except Exception as e:
                    print(f'   {root.split("/")[-1]:16s} {Path(pq).name:26s} ERR {type(e).__name__}')
                    continue
                seq = seq[off:off + n]
                c = [len(g) for _, g in seq]
                if c and c == bc[:len(c)] and len(c) == len(bc):
                    print(f'   *** MATCH  root={root.split("/")[-1]:16s} pq={Path(pq).name}')
                elif c:
                    print(f'   {root.split("/")[-1]:16s} {Path(pq).name:26s} n={len(c)} sum={sum(c)}')


if __name__ == '__main__':
    main()
