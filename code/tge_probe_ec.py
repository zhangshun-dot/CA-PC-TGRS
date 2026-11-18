# -*- coding: utf-8 -*-
"""tge_probe_ec.py — 外推条件负原型 (EC-NegProto) 验证

发现: 负原型在源原型上无效 (+0.0005), 在外推原型上显著 (+0.0331), 纯交互 +0.0326。
算法化: 负原型的构建空间从"源域 DIOR 类"搬到"外推空间 (COCO 外推原型)":

  P_neg_EC(y) = mean over y'≠y { P̂_y' }      ← 外推空间负原型 (条件于外推)
  P_neg_src(y) = DIOR top-3 近邻类均值         ← 源域负原型 (现状)

机制: 外推放大的是"目标域类间纠缠" (P̂_bus 与 P̂_car 更近), 源域负原型切得不精准;
外推空间负原型精确切外推后产生的纠缠 —— "外推条件负原型"的字面算法化。

对照 (N=500 口径, 与消融一致):
  λ=0 无负原型 (基准) / 源域负原型 (现状) / EC 外推空间负原型 (候选)
  λ ∈ {0.25, 0.5, 0.75} 全区间报告 (不挑单点)

输出: 实验数据/tge_probe_ec.json + 控制台表
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
OUT = DATA / 'tge_probe_ec.json'

BETA = 2.0
N_COCO = 500
LAMS = [0.25, 0.5, 0.75]
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


def main():
    # 1) DIOR 源原型 (负原型用)
    print('[1/3] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        crops = []
        for img, g in zip(images, gts):
            for box in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 > x1 + 10 and y2 > y1 + 10:
                    crops.append(img.crop((x1, y1, x2, y2)))
        if not crops:
            continue
        F = T.encode_clip_batch(crops)
        p = F.mean(dim=0)
        protos[cls] = p / p.norm()

    # 2) 外推原型 (6 类) + 源域负原型 (每类)
    print('[2/3] 外推原型 + 两种负原型 ...')
    P_src = {}
    P_hat = {}
    P_neg_src = {}
    for name, coco_cls, src_cls in TARGETS:
        P_src[name] = protos[src_cls]
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        delta = d_y - s_xs
        p = P_src[name] + BETA * delta.to(U.DEVICE)
        P_hat[name] = (p / p.norm()).cpu()
        # 源域负原型: DIOR top-3 近邻均值
        pos = P_src[name].to(U.DEVICE)
        others = torch.stack([protos[c] for c in protos if c != src_cls]).to(U.DEVICE)
        neg = others[(others @ pos).topk(3).indices].mean(dim=0)
        P_neg_src[name] = (neg / neg.norm()).cpu()
    # 外推空间负原型: 其余 5 类外推原型均值
    P_neg_ec = {name: torch.stack([P_hat[n2] for n2, *_ in TARGETS if n2 != name]).mean(dim=0)
                for name, *_ in TARGETS}
    P_neg_ec = {k: v / v.norm() for k, v in P_neg_ec.items()}

    # 3) 每类 AP50 (N=500 全测, 与 tge_probe_ap 协议一致)
    print('[3/3] AP50 对比 ...')
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)
        crops, boxes_all, gd_all, lab_all, img_ids = [], [], [], [], []
        for ii, ((boxes, scores), img, g) in enumerate(zip(dets, images, gts)):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                lab = 1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in g) else 0
                boxes_all.append([x1, y1, x2, y2])
                gd_all.append(float(sc))
                lab_all.append(lab)
                img_ids.append(ii)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    crops.append(None)
                else:
                    crops.append(img.crop((x1, y1, x2, y2)))
        real_idx = [i for i, c in enumerate(crops) if c is not None]
        F = np.zeros((len(crops), 512))
        if real_idx:
            F[real_idx] = T.encode_clip_batch([crops[i] for i in real_idx]).cpu().numpy()
        gd = np.clip(np.array(gd_all), 1e-6, 1.0)
        gt_flat = [g for gs in gts for g in gs]
        P = P_hat[name]

        def ap_with(neg, lam):
            vis = np.maximum(0.0, F @ P.numpy() - lam * (F @ neg.numpy()))
            fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
            return U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)

        # λ=0: 无负原型 (仅外推, 基准) —— 逐图 AP50 (2026-08-23 修正)
        fused_no_neg = np.sqrt(gd * np.clip(np.maximum(0.0, F @ P.numpy()), 1e-6, 1.0))
        ap_no_neg = U.compute_ap50_img(U.split_by_img(boxes_all, fused_no_neg, img_ids,
                                                      len(images)), gts)
        ap_src = {str(l): round(ap_with(P_neg_src[name], l), 4) for l in LAMS}
        ap_ec = {str(l): round(ap_with(P_neg_ec[name], l), 4) for l in LAMS}

        rows[name] = {'no_neg': round(ap_no_neg, 4), 'neg_src': ap_src, 'neg_ec': ap_ec}
        print(f'  {name:<16s} λ0={ap_no_neg:.4f}  '
              + '  '.join(f'srcλ{l}={ap_src[str(l)]:.4f}' for l in LAMS) + '  '
              + '  '.join(f'ECλ{l}={ap_ec[str(l)]:.4f}' for l in LAMS))

    print('\n' + '=' * 104)
    print(f'{"类":<16s} {"λ=0":>8s} {"src.25":>8s} {"src.50":>8s} {"src.75":>8s} '
          f'{"EC.25":>8s} {"EC.50":>8s} {"EC.75":>8s}')
    print('-' * 104)
    for name, *_ in TARGETS:
        r = rows[name]
        print(f'{name:<16s} {r["no_neg"]:>8.4f} ' +
              ' '.join(f'{r["neg_src"][str(l)]:>8.4f}' for l in LAMS) + ' ' +
              ' '.join(f'{r["neg_ec"][str(l)]:>8.4f}' for l in LAMS))
    print('-' * 104)
    print(f'{"均值":<16s} {np.mean([rows[n]["no_neg"] for n in rows]):>8.4f} ' +
          ' '.join(f'{np.mean([rows[n]["neg_src"][str(l)] for n in rows]):>8.4f}' for l in LAMS) + ' ' +
          ' '.join(f'{np.mean([rows[n]["neg_ec"][str(l)] for n in rows]):>8.4f}' for l in LAMS))

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'beta': BETA, 'n_coco': N_COCO, 'lams': LAMS,
                              'protocol': 'N=500, sqrt(gd*vis), image-safe per-image AP50 '
                                          '(2026-08-23 修正), 无/源域/EC 负原型'},
                   'classes': rows}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
