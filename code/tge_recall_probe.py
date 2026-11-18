# -*- coding: utf-8 -*-
"""tge_recall_probe.py — 框池召回审计: GDINO 候选框池对 GT 的覆盖

评估协议"框池固定, 不新增召回"需要数字支撑 (论文 4.7 局限节/评审防御):
  recall@proposal = GT 框被任一框池框 IoU>=0.5 覆盖的比例 (GT 互斥匹配)

三个口径:
  full   — GDINO th=0 全部输出框 (理论上限)
  pf     — pre-filter 0.03 之后的框池
  pf_ns  — pre-filter 0.03 + 小框(<=10px)过滤 (实际评估协议用的框池)

同一框池缓存 (tge_probe_dets_COCO-*.pkl), 与主表 N=100 协议完全一致。
输出: 实验数据/tge_recall_probe.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_recall_probe.json'

N_COCO = 100
TGT_TMPL = 'a photo of {}'

TARGETS = [
    ('COCO-airplane',   'airplane'),
    ('COCO-boat',       'boat'),
    ('COCO-car',        'car'),
    ('COCO-truck',      'truck'),
    ('COCO-bus',        'bus'),
    ('COCO-motorcycle', 'motorcycle'),
]


def iou_pairwise(boxes, gts):
    """(N,4) x (M,4) -> (N,M) IoU 矩阵 (同 U.iou 公式)。"""
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
    print('[1/2] COCO 6 类: 框池召回 (N=%d, 复用检测缓存) ...' % N_COCO)
    rows = {}
    for name, coco_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)

        gt_flat = [g for gs in gts for g in gs]
        n_gt = len(gt_flat)

        # 三口径框池 (每框: (box, keep_flag, is_small))
        boxes_full, boxes_pf, boxes_pfns = [], [], []
        for (boxes, scores), img in zip(dets, images):
            keep = scores > U.PRE_FILTER
            for box, sc in zip(boxes, scores):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                small = x2 <= x1 + 10 or y2 <= y1 + 10
                boxes_full.append([x1, y1, x2, y2])
                if sc > U.PRE_FILTER:
                    boxes_pf.append([x1, y1, x2, y2])
                    if not small:
                        boxes_pfns.append([x1, y1, x2, y2])

        def recall(box_pool):
            if not box_pool or n_gt == 0:
                return 0.0
            m = iou_pairwise(box_pool, gt_flat)
            return float((m.max(axis=0) >= 0.5).mean())

        r_full = recall(boxes_full)
        r_pf = recall(boxes_pf)
        r_pfns = recall(boxes_pfns)
        rows[name] = {
            'n_images': len(images), 'n_gt': n_gt,
            'n_boxes_full': len(boxes_full), 'n_boxes_pf': len(boxes_pf),
            'n_boxes_pf_nosmall': len(boxes_pfns),
            'avg_boxes_per_img_full': round(len(boxes_full) / max(len(images), 1), 1),
            'avg_boxes_per_img_pf': round(len(boxes_pf) / max(len(images), 1), 1),
            'recall_full': round(r_full, 4),
            'recall_pf': round(r_pf, 4),
            'recall_pf_nosmall': round(r_pfns, 4),
        }
        print('  %-16s GT=%4d  boxes=%5d/%5d/%5d  recall=%.3f/%.3f/%.3f'
              % (name, n_gt, len(boxes_full), len(boxes_pf), len(boxes_pfns),
                 r_full, r_pf, r_pfns))

    # 均值行
    def mean_of(key):
        return round(float(np.mean([rows[n][key] for n in rows])), 4)

    mean_row = {'n_images': N_COCO,
                'n_gt': int(np.mean([rows[n]['n_gt'] for n in rows])),
                'n_boxes_full': int(np.mean([rows[n]['n_boxes_full'] for n in rows])),
                'n_boxes_pf': int(np.mean([rows[n]['n_boxes_pf'] for n in rows])),
                'n_boxes_pf_nosmall': int(np.mean([rows[n]['n_boxes_pf_nosmall'] for n in rows])),
                'avg_boxes_per_img_full': mean_of('avg_boxes_per_img_full'),
                'avg_boxes_per_img_pf': mean_of('avg_boxes_per_img_pf'),
                'recall_full': mean_of('recall_full'),
                'recall_pf': mean_of('recall_pf'),
                'recall_pf_nosmall': mean_of('recall_pf_nosmall')}

    out = {
        'config': {'n_coco': N_COCO, 'pre_filter': U.PRE_FILTER,
                   'protocol': 'recall@proposal: GT covered by any pool box IoU>=0.5, '
                               'same dets cache as main table (tge_probe_dets_*.pkl)',
                   'recall_definition': 'per-GT hit, no mutual exclusion'},
        'classes': rows, 'mean': mean_row,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 92)
    print('%-16s %6s %8s %8s %8s' % ('class', 'n_gt', 'rec_full', 'rec_pf', 'rec_pfns'))
    print('-' * 92)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-16s %6d %8.4f %8.4f %8.4f' % (name, r['n_gt'], r['recall_full'],
                                                r['recall_pf'], r['recall_pf_nosmall']))
    print('-' * 92)
    print('%-16s %6d %8.4f %8.4f %8.4f' % ('mean', mean_row['n_gt'], mean_row['recall_full'],
                                            mean_row['recall_pf'], mean_row['recall_pf_nosmall']))
    print('=' * 92)
    print('-> %s' % OUT)


if __name__ == '__main__':
    main()
