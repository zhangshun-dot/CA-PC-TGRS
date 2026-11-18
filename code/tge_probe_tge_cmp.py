# -*- coding: utf-8 -*-
"""tge_probe_tge_cmp.py — ExDet TGE 迁移基线对照: 语言外推的增量拆解 (N=100, 逐图 AP50)

对照行 (同一框池, 同一协议, 只改打分规则):
  baseline — GDINO 原始分数
  TGE      — 纯语言外推 P̂ = norm(P_src + β·Δ), 无负原型 (λ=0)  ← ExDet TGE 的忠实迁移 (单模板)
  A-CSPC   — 外推 + 源域负原型相对化 (λ=0.5)                    ← 本文方法

增量分解 (路径式, 每类 + 均值):
  TGE 增量          = TGE − baseline     (语言外推主效应)
  负原型条件贡献     = A-CSPC − TGE       (非对称原则: 负原型在外推上的条件增益)
  总增量            = A-CSPC − baseline
协议与 tge_probe_fs 完全一致 (N=100, pre-filter 0.03, 后 90 图测试, 逐图 AP50)。
输出: 实验数据/tge_probe_tge_cmp.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_tge_cmp.json'

BETA = 2.0
N_COCO = 100
K = 10                      # 前 K 图校准段 (此处仅切后 90 图测试, 与主表一致)
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
    """所有 GT 框 crop (同 tge_probe_fs.gt_crops)。"""
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def eval_class(images, gts, dets, P_cal, P_neg=None, neg_lam=U.NEG_LAM):
    """同 tge_probe_fs.eval_class (逐图 AP50), P_neg=None → 无负原型项 (TGE 行)。

    P_neg 为 None: vis = max(0, F@P_cal)           ← 纯语言外推
    P_neg 给定:    vis = max(0, F@P_cal − λ·F@P_neg) ← A-CSPC
    """
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
        if P_neg is not None:
            vis = np.maximum(0.0, F @ P_cal.cpu().numpy()
                             - neg_lam * (F @ P_neg.cpu().numpy()))
        else:
            vis = np.maximum(0.0, F @ P_cal.cpu().numpy())
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

    print('[2/2] COCO 6 类三行对照 (N=%d, 逐图 AP50) ...' % N_COCO)
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)

        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()

        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]
        base, _ = eval_class(test_imgs, test_gts, test_dets, P_src, P_neg)
        _, ap_tge = eval_class(test_imgs, test_gts, test_dets, P_hat, None)
        _, ap_cspc = eval_class(test_imgs, test_gts, test_dets, P_hat, P_neg)

        rows[name] = {'baseline': round(base, 4),
                      'tge_lam0': round(ap_tge, 4),
                      'acspc_lam05': round(ap_cspc, 4),
                      'tge_gain': round(ap_tge - base, 4),
                      'neg_cond_gain': round(ap_cspc - ap_tge, 4),
                      'total_gain': round(ap_cspc - base, 4)}
        print('  %-16s base=%.4f  TGE(λ0)=%.4f  A-CSPC(λ.5)=%.4f  '
              '[TGE %+.4f | 负 %+.4f | 总 %+.4f]'
              % (name, base, ap_tge, ap_cspc,
                 ap_tge - base, ap_cspc - ap_tge, ap_cspc - base))

    # 均值 + 增量分解
    mean_row = {}
    for key in ['baseline', 'tge_lam0', 'acspc_lam05',
                'tge_gain', 'neg_cond_gain', 'total_gain']:
        mean_row[key] = round(float(np.mean([rows[n][key] for n in rows])), 4)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_coco': N_COCO, 'beta': BETA, 'neg_lam': U.NEG_LAM,
                              'protocol': 'gdino th=0 -> pre-filter 0.03 -> image-safe per-image '
                                          'AP50, 后90图测试, 同一框池只改打分规则'},
                   'classes': rows, 'mean': mean_row},
                  f, indent=2, ensure_ascii=False)

    print()
    print('=' * 100)
    print('%-16s %9s %9s %9s %9s %9s %9s' %
          ('class', 'baseline', 'TGE(λ0)', 'A-CSPC', 'TGE增益', '负贡献', '总增益'))
    print('-' * 100)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-16s %9.4f %9.4f %9.4f %+9.4f %+9.4f %+9.4f' %
              (name, r['baseline'], r['tge_lam0'], r['acspc_lam05'],
               r['tge_gain'], r['neg_cond_gain'], r['total_gain']))
    print('-' * 100)
    print('%-16s %9.4f %9.4f %9.4f %+9.4f %+9.4f %+9.4f   <- 6-class mean' %
          ('mean6', mean_row['baseline'], mean_row['tge_lam0'], mean_row['acspc_lam05'],
           mean_row['tge_gain'], mean_row['neg_cond_gain'], mean_row['total_gain']))
    print('=' * 100)
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
