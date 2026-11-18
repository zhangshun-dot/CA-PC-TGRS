# -*- coding: utf-8 -*-
"""tge_probe_prefilter.py — pre-filter 阈值灵敏度: 0.01 / 0.03 / 0.05

期刊补强 (协议阈值稳健性): 主表关键行 (baseline / A-CSPC) 在三个 pre-filter
阈值下重跑, 验证方法结论 (外推增益为正且显著) 不依赖阈值选择。
协议: N=100 逐图 AP50, 同一 th=0 检测缓存, 只改过滤阈值。
输出: 实验数据/tge_probe_prefilter.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_prefilter.json'

BETA = 2.0
N_COCO = 100
K = 10
PF_LIST = [0.01, 0.03, 0.05]
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


def eval_class(images, gts, dets, P_cal, P_neg, pf):
    """逐图 AP50, pre_filter 参数化。"""
    boxes_all, gd_all, feats, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > pf
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
    print('[1/2] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/2] COCO 6 类 × pre-filter %s (N=%d, 逐图 AP50) ...' % (PF_LIST, N_COCO))
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]

        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()

        row = {}
        for pf in PF_LIST:
            base, ap = eval_class(test_imgs, test_gts, test_dets, P_hat, P_neg, pf)
            row['base_%.2f' % pf] = round(base, 4)
            row['cspc_%.2f' % pf] = round(ap, 4)
            row['gain_%.2f' % pf] = round(ap - base, 4)
        rows[name] = row
        print('  %-16s ' % name + '  '.join(
            'pf=%.2f: %.4f→%.4f(%+.4f)' % (pf, row['base_%.2f' % pf],
                                           row['cspc_%.2f' % pf], row['gain_%.2f' % pf])
            for pf in PF_LIST))

    mean_row = {k: round(float(np.mean([rows[n][k] for n in rows])), 4)
                for k in ['base_%.2f' % pf for pf in PF_LIST] +
                         ['cspc_%.2f' % pf for pf in PF_LIST] +
                         ['gain_%.2f' % pf for pf in PF_LIST]}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_coco': N_COCO, 'beta': BETA, 'pf_list': PF_LIST,
                              'protocol': 'pre-filter 灵敏度, N=100 逐图 AP50, 同主表口径'},
                   'classes': rows, 'mean': mean_row}, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 96)
    print('%-16s' % 'class' + ''.join('%9s %9s %9s' % ('b.%.2f' % pf, 'A.%.2f' % pf, 'Δ.%.2f' % pf)
                                      for pf in PF_LIST))
    print('-' * 96)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-16s' % name + ''.join(
            '%9.4f %9.4f %+9.4f' % (r['base_%.2f' % pf], r['cspc_%.2f' % pf],
                                    r['gain_%.2f' % pf]) for pf in PF_LIST))
    print('-' * 96)
    print('%-16s' % 'mean' + ''.join(
        '%9.4f %9.4f %+9.4f' % (mean_row['base_%.2f' % pf], mean_row['cspc_%.2f' % pf],
                                mean_row['gain_%.2f' % pf]) for pf in PF_LIST))
    print('=' * 96)
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
