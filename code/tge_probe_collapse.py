# -*- coding: utf-8 -*-
"""tge_probe_collapse.py — 坍缩假说检验: 源原型类间余弦, 源域 vs 目标域

假说: DIOR 源原型坍缩 (类间余弦 0.964, 已测) → 判别失效 + 负原型同向无效;
      COCO 源原型 (自然场景类) 应不坍缩 → 解释反向 B2 有效 + 外推推偏。

检验: COCO 3 类源原型 (airplane/boat/vehicle-聚合) 类间余弦 vs DIOR 0.964。
输出: 控制台表 + 实验数据/tge_probe_collapse.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_collapse.json'


def proto_from_gts(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for box in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 > x1 + 10 and y2 > y1 + 10:
                crops.append(img.crop((x1, y1, x2, y2)))
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def main():
    # COCO 源原型
    print('[1/2] COCO 源原型 (GT 框, N=500) ...')
    coco = {}
    for name, cls in [('airplane', 'airplane'), ('boat', 'boat'), ('vehicle', None)]:
        if cls is None:  # vehicle: 4 细类合并
            imgs_all, gts_all = [], []
            for c in ['car', 'truck', 'bus', 'motorcycle']:
                im, gt = U.load_coco(c, n=500)
                imgs_all += im
                gts_all += gt
            coco[name] = proto_from_gts(imgs_all, gts_all)
        else:
            im, gt = U.load_coco(cls, n=500)
            coco[name] = proto_from_gts(im, gt)
        print(f'  {name}: 就绪')

    # DIOR 源原型 (对照, 与 entangle 脚本同协议)
    print('[2/2] DIOR 源原型 (对照) ...')
    dior = {}
    for cls in ['airplane', 'ship', 'vehicle']:
        im, gt = U.load_dior(cls)
        dior[cls] = proto_from_gts(im, gt)

    def pair_cos(protos):
        names = list(protos)
        vals = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                vals.append(float((protos[a] @ protos[b]).item()))
        return float(np.mean(vals)), vals

    c_mean, c_vals = pair_cos(coco)
    d_mean, d_vals = pair_cos(dior)
    print('\n类间余弦:')
    print(f'  COCO 源原型: 均值 {c_mean:.3f}   ({", ".join(f"{v:.3f}" for v in c_vals)})')
    print(f'  DIOR 源原型: 均值 {d_mean:.3f}   ({", ".join(f"{v:.3f}" for v in d_vals)})')
    ok = c_mean < d_mean - 0.1
    print(f'  → 坍缩假说 (COCO 不坍缩, DIOR 坍缩): {"✅ 支持" if ok else "❌ 不支持"}')

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'coco_mean': round(c_mean, 4), 'dior_mean': round(d_mean, 4),
                   'coco_pairs': c_vals, 'dior_pairs': d_vals}, f, indent=2)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
