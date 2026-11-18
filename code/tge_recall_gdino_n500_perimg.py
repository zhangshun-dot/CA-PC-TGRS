# -*- coding: utf-8 -*-
"""GDINO 框池 N=500 逐图 per-GT recall —— 论文 P1 "N=500 统计" 的正式来源。

定义 (同论文 3.2): 每图 GT 仅被本图候选框池中任一 IoU>=0.5 的框覆盖。
pre-filter 0.03, 含小框。注意: 不用全局合并池 (tge_recall_probe.py 旧口径会
跨图误匹配而偏高 0.05~0.12), 逐图为正确口径。
输出: tge_recall_gdino_n500_perimg.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import json
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_recall_gdino_n500_perimg.json'
N_COCO = 500
TGT_TMPL = 'a photo of {}'
TARGETS = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']


def iou_pairwise(boxes, gts):
    b = np.asarray(boxes, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    xx1 = np.maximum(b[:, None, 0], g[None, :, 0])
    yy1 = np.maximum(b[:, None, 1], g[None, :, 1])
    xx2 = np.minimum(b[:, None, 2], g[None, :, 2])
    yy2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (ab[:, None] + ag[None, :] - inter + 1e-8)


def main():
    rows = {}
    for cls in TARGETS:
        images, gts = U.load_coco(cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(cls), f'COCO-{cls}')
        n = min(len(images), len(dets))
        images, dets, gts = images[:n], dets[:n], gts[:n]
        n_gt, hit = 0, 0
        for (boxes, scores), img, gs in zip(dets, images, gts):
            bl = []
            for box, sc in zip(boxes, scores):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if sc > U.PRE_FILTER:
                    bl.append([x1, y1, x2, y2])
            n_gt += len(gs)
            if bl and gs:
                m = iou_pairwise(np.asarray(bl, dtype=np.float64), np.asarray(gs, dtype=np.float64))
                hit += int((m.max(axis=0) >= 0.5).sum())
        r = hit / n_gt if n_gt else float('nan')
        rows[cls] = {'n_images': n, 'n_gt': n_gt, 'recall_pf_perimg': round(r, 4)}
        print('%-10s N=%3d GT=%4d  逐图 per-GT recall=%.4f' % (cls, n, n_gt, r))
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'detector': 'GDINO-tiny', 'n_coco': N_COCO,
                              'definition': 'per-image per-GT recall, IoU>=0.5, '
                                            'pre-filter 0.03, 含小框, 逐图口径'},
                   'classes': rows}, f, indent=2, ensure_ascii=False)
    print('-> %s' % OUT)


if __name__ == '__main__':
    main()
