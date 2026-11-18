# -*- coding: utf-8 -*-
"""tge_probe_srcsize.py — 源域样本量消融: DIOR 原型用 5/10/20 图构建 → 外推效果

期刊补强 (对源域数据量的稳健性): 展示 A-CSPC 的增益在源域样本缩减时如何退化。
  - n_src ∈ {5, 10, 20}: U.load_dior(cls, n=n_src) 构建 CLIP 源原型
  - 评估 COCO 6 类 (N=100, 逐图 AP50, 同一框池缓存), 行: baseline / A-CSPC@5 / @10 / @20
协议与主表完全一致 (tge_probe_fs 口径)。输出: 实验数据/tge_probe_srcsize.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_srcsize.json'

BETA = 2.0
N_COCO = 100
K = 10
SRC_NS = [5, 10, 20]
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


def build_protos(n_src):
    """DIOR 20 类源原型, 每类用前 n_src 图。"""
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls, n=n_src)
        crops = []
        for img, g in zip(images, gts):
            for box in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 > x1 + 10 and y2 > y1 + 10:
                    crops.append(img.crop((x1, y1, x2, y2)))
        if not crops:
            continue
        protos[cls] = clip_proto(crops)
    return protos


def eval_class(images, gts, dets, P_cal, P_neg):
    """逐图 AP50 (同 tge_probe_fs)。"""
    boxes_all, gd_all, feats, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(img.crop((x1, y1, x2, y2)))
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            img_ids.append(ii)
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
    if feats:
        F = T.encode_clip_batch(feats).cpu().numpy()
        vis = np.maximum(0.0, F @ P_cal.cpu().numpy()
                         - U.NEG_LAM * (F @ P_neg.cpu().numpy()))
        fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
        ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    else:
        ap = base
    return base, ap


def main():
    print('[1/2] DIOR 源原型 (n_src = %s) ...' % SRC_NS)
    protos_by_n = {n: build_protos(n) for n in SRC_NS}
    for n in SRC_NS:
        print('  n=%2d: %d 类原型就绪' % (n, len(protos_by_n[n])))

    print('[2/2] COCO 6 类评估 (N=%d, 逐图 AP50) ...' % N_COCO)
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]

        base, _ = eval_class(test_imgs, test_gts, test_dets,
                             protos_by_n[SRC_NS[-1]][src_cls].to(U.DEVICE),
                             T.get_neg_proto(protos_by_n[SRC_NS[-1]], src_cls))
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]

        row = {'baseline': round(base, 4)}
        for n in SRC_NS:
            protos = protos_by_n[n]
            P_src = protos[src_cls].to(U.DEVICE)
            P_neg = T.get_neg_proto(protos, src_cls)
            P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()
            _, ap = eval_class(test_imgs, test_gts, test_dets, P_hat, P_neg)
            row['cspc_%d' % n] = round(ap, 4)
            row['gain_%d' % n] = round(ap - base, 4)
        rows[name] = row
        print('  %-16s base=%.4f  ' % (name, base) +
              '  '.join('n%2d: %.4f(%+.4f)' % (n, row['cspc_%d' % n], row['gain_%d' % n])
                        for n in SRC_NS))

    mean_row = {k: round(float(np.mean([rows[n][k] for n in rows])), 4)
                for k in ['baseline'] + [f'cspc_{n}' for n in SRC_NS] + [f'gain_{n}' for n in SRC_NS]}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_coco': N_COCO, 'beta': BETA, 'src_n': SRC_NS,
                              'protocol': 'DIOR 源原型 n_src 消融, N=100 逐图 AP50, 同主表口径'},
                   'classes': rows, 'mean': mean_row}, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 96)
    print('%-16s %9s %9s %9s %9s' % ('class', 'baseline', 'n=5', 'n=10', 'n=20'))
    print('-' * 96)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-16s %9.4f %9.4f %9.4f %9.4f' %
              (name, r['baseline'], r['cspc_5'], r['cspc_10'], r['cspc_20']))
    print('-' * 96)
    print('%-16s %9.4f %9.4f %9.4f %9.4f   <- mean' %
          ('mean', mean_row['baseline'], mean_row['cspc_5'],
           mean_row['cspc_10'], mean_row['cspc_20']))
    print('-' * 96)
    print('增益:  ' + '  '.join('n=%2d: %+.4f' % (n, mean_row['gain_%d' % n]) for n in SRC_NS))
    print('=' * 96)
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
