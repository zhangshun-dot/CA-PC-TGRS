# -*- coding: utf-8 -*-
"""route_proxy_extend.py — 分离度 AUC 全量重算 (12 类统一协议)

与 route_proxy_probe.py 的 AUC 协议完全一致:
  源原型 CDVM vis (max(0, f·P_src − λ·f·P_neg)) 在 full-pool 候选框上的 rank-AUC
  (L12 / pre-filter 0.03 / COCO 前 N 图 / DIOR full20 全部)

本次改动 (2026-08-22):
  * N_COCO: 100 → 500 —— bus/motorcycle 在 N=100 下正例太少 (n_pos≈个位数),
    AUC 0.1805/0.2560 是测量噪声 (假低点), 会打脸边界图 thesis。
  * 12 类全量重算 (6 COCO 跨域 + 6 DIOR 同域), 不再复用旧 6 类 N=100 结果,
    保证横轴协议一致。旧 route_proxy_probe.json 保留作 N=100 存档。

输出: 实验数据/route_proxy_probe_12.json (全量覆盖, 12 类)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import unified_eval as U

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'route_proxy_probe_12.json'

LAYER = 11
N_COCO = 500          # 2026-08-22: 100 → 500, 压 bus/motorcycle 测量噪声

ALL_TARGETS = [
    ('COCO-airplane',   'coco', 'airplane',    'a photo of',         'airplane'),
    ('COCO-boat',       'coco', 'boat',        'a photo of',         'ship'),
    ('COCO-car',        'coco', 'car',         'a photo of',         'vehicle'),
    ('COCO-truck',      'coco', 'truck',       'a photo of',         'vehicle'),
    ('COCO-bus',        'coco', 'bus',         'a photo of',         'vehicle'),
    ('COCO-motorcycle', 'coco', 'motorcycle',  'a photo of',         'vehicle'),
    ('DIOR-airplane',   'dior', 'airplane',    'an overhead image of', 'airplane'),
    ('DIOR-ship',       'dior', 'ship',        'an overhead image of', 'ship'),
    ('DIOR-vehicle',    'dior', 'vehicle',     'an overhead image of', 'vehicle'),
    ('DIOR-harbor',     'dior', 'harbor',      'an overhead image of', 'harbor'),
    ('DIOR-storage tank','dior', 'storage tank','an overhead image of', 'storage tank'),
    ('DIOR-bridge',     'dior', 'bridge',      'an overhead image of', 'bridge'),
]


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def rank_auc(vis_all, lab_all):
    vis_all = np.asarray(vis_all); lab_all = np.asarray(lab_all)
    pos = vis_all[lab_all == 1]; neg = vis_all[lab_all == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    n_pos, n_neg = len(pos), len(neg)
    order = np.argsort(vis_all)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(vis_all))
    return (rank[lab_all == 1].mean() - (n_pos - 1) / 2) / n_neg


def main():
    bank = U.get_bank()
    # 前置检查: 所有 DIOR 类是否有源原型
    for name, ds, cls, prefix, dior_cls in ALL_TARGETS:
        if ds == 'dior' and dior_cls not in bank:
            raise SystemExit(f'!! bank 缺 {dior_cls} 原型，无法补算 AUC')

    rows = {}
    for name, dataset, cls, prefix, dior_cls in ALL_TARGETS:
        print(f'[{name}]  ← DIOR {dior_cls} 源原型')
        images, gts = U.load_coco(cls, n=N_COCO) if dataset == 'coco' else U.load_dior(cls)
        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        vis_all, labels = [], []
        for (boxes, scores), img, g in zip(dets, images, gts):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box in boxes[keep]:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    vis_all.append(0.0); labels.append(0); continue
                f = U.encode_dinov2(img.crop((x1, y1, x2, y2)), LAYER)
                v = max(0.0, float((f @ P_src).item()) - U.NEG_LAM * float((f @ neg).item()))
                lab = 1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in g) else 0
                vis_all.append(v); labels.append(lab)

        auc = rank_auc(vis_all, labels)
        n_pos = sum(labels)
        print(f'  N={len(images)}图  候选框={len(vis_all)}  正例={n_pos}  AUC={auc:.4f}')
        rows[name] = {'auc': round(auc, 4),
                      'n_images': len(images), 'n_boxes': len(vis_all), 'n_pos': n_pos,
                      'R_c': None, 'head_ratio': None, 'stability': None}

    # 全量写入（不复用旧 N=100 结果，保证横轴口径一致）
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print('\n重算结果 (12 类统一协议):')
    print(f'{"类":<20s} {"N图":>5s} {"候选框":>7s} {"正例":>6s} {"AUC":>8s}')
    print('-' * 50)
    for k, v in rows.items():
        print(f'{k:<20s} {v["n_images"]:>5d} {v["n_boxes"]:>7d} {v["n_pos"]:>6d} {v["auc"]:>8.4f}')
    print(f'\n✅ -> {OUT}  ({len(rows)} 类, N_COCO={N_COCO})')


if __name__ == '__main__':
    main()
