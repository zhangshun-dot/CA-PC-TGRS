# -*- coding: utf-8 -*-
"""tge_probe_beta.py — β 主表口径扫描: β ∈ {1.0, 2.0, 4.0} (N=100 AP 层)

期刊补强 (β 灵敏度补全到主表口径): AUC 层已有 β 曲线, 这里在 N=100 AP 层
验证 β=2.0 峰位 (λ=0.5, 与主表一致)。特征编码一次 (β 只改 P_hat = P_src + βΔ),
每 β 一次矩阵乘。输出: 实验数据/tge_probe_beta.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_beta.json'

N_COCO = 100
K = 10
BETAS = [1.0, 2.0, 4.0]
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


def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def eval_class_w(images, gts, dets, W):
    """逐图 AP50, W = P_hat - λ·P_neg 预计算 (β 扫描只改 W)。"""
    crops, boxes_all, gd_all, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            img_ids.append(ii)
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
    if crops:
        F = T.encode_clip_batch(crops).cpu().numpy()
        vis = np.maximum(0.0, F @ W)
        fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
        ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    else:
        ap = base
    return base, ap


def main():
    print('[1/2] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/2] COCO 6 类 × β %s (N=%d, 逐图 AP50, 特征一次编码) ...' % (BETAS, N_COCO))
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]

        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs

        base, _ = eval_class_w(test_imgs, test_gts, test_dets,
                               (P_src.cpu().numpy() - U.NEG_LAM * P_neg.cpu().numpy()))
        row = {'baseline': round(base, 4)}
        for b in BETAS:
            P_hat = P_src + b * Delta.to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()
            W = (P_hat.cpu().numpy() - U.NEG_LAM * P_neg.cpu().numpy())
            _, ap = eval_class_w(test_imgs, test_gts, test_dets, W)
            row['cspc_b%.1f' % b] = round(ap, 4)
            row['gain_b%.1f' % b] = round(ap - base, 4)
        rows[name] = row
        print('  %-16s base=%.4f  ' % (name, base) + '  '.join(
            'β=%.1f: %.4f(%+.4f)' % (b, row['cspc_b%.1f' % b], row['gain_b%.1f' % b])
            for b in BETAS))

    mean_row = {k: round(float(np.mean([rows[n][k] for n in rows])), 4)
                for k in ['baseline'] + [f'cspc_b{b}' for b in BETAS] + [f'gain_b{b}' for b in BETAS]}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_coco': N_COCO, 'betas': BETAS, 'neg_lam': U.NEG_LAM,
                              'protocol': 'β 主表口径扫描, N=100 逐图 AP50, 特征一次编码'},
                   'classes': rows, 'mean': mean_row}, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 84)
    print('%-16s %9s' % ('class', 'baseline') + ''.join('%13s' % ('β=%.1f' % b) for b in BETAS))
    print('-' * 84)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-16s %9.4f' % (name, r['baseline']) + ''.join(
            '%13.4f' % r['cspc_b%.1f' % b] for b in BETAS))
    print('-' * 84)
    print('%-16s %9.4f' % ('mean', mean_row['baseline']) + ''.join(
        '%13.4f' % mean_row['cspc_b%.1f' % b] for b in BETAS))
    print('-' * 84)
    print('增益: ' + '  '.join('β=%.1f: %+.4f' % (b, mean_row['gain_b%.1f' % b]) for b in BETAS))
    print('=' * 84)
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
