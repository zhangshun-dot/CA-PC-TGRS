# -*- coding: utf-8 -*-
"""tge_diag_owlv2.py — 诊断 OWLv2 logits 分布与 GT 匹配的关系

问题: sigmoid 协议 (A-CSPC -0.48) 比旧版直接 clip (A-CSPC -0.03) 崩得多。
只读 pkl 缓存, 不跑模型。验证:
  1. logits 分布 (sigmoid 前后的尺度)
  2. GT 匹配框 vs 未匹配框的 logits 区分度
  3. 每 GT 的重复框数 (IoU>=0.5) —— 判断重复框是否是 AP 崩的源头
  4. pre-filter 0.03 (sigmoid 概率尺度) 实际滤掉的框比例
"""
import pickle
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import unified_eval as U

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
N = 100
TARGETS = [
    ('COCO-airplane', 'airplane'), ('COCO-boat', 'boat'), ('COCO-car', 'car'),
    ('COCO-truck', 'truck'), ('COCO-bus', 'bus'), ('COCO-motorcycle', 'motorcycle'),
]


def main():
    print('%-16s %8s %8s %8s | %8s %8s | %8s %8s | %8s %8s | %8s' % (
        'class', 'n_img', 'n_box', 'n_keep', 'lo_med', 'lo_p90',
        'gt_med', 'ng_med', 'dup/gt', 'logit>=1', '滤掉%'))
    print('-' * 118)
    for name, coco_cls in TARGETS:
        dets = pickle.load(open(DATA / f'tge_probe_dets_OWLV2-{name}.pkl', 'rb'))
        images, gts = U.load_coco(coco_cls, n=N)
        all_log = np.concatenate([s for _, s in dets])
        keep = all_log > np.log(0.03 / (1 - 0.03))  # sigmoid > 0.03 -> logit > -3.476

        gt_logs, ng_logs, n_dup, n_gt = [], [], 0, 0
        for (boxes, scores), g in zip(dets, gts):
            for gb in g:
                n_gt += 1
                ious = np.array([U.iou(b, gb) for b in boxes])
                hits = ious >= 0.5
                n_dup += int(hits.sum())
                if hits.any():
                    gt_logs.append(float(scores[hits].max()))
            for b, s in zip(boxes, scores):
                if not any(U.iou(b, gb) >= 0.5 for gb in g):
                    ng_logs.append(float(s))

        print('%-16s %8d %8d %8d | %8.2f %8.2f | %8.2f %8.2f | %8.2f %8.1f%% %8.1f%%' % (
            name, len(images), len(all_log), int(keep.sum()),
            np.median(all_log), np.percentile(all_log, 90),
            np.median(gt_logs) if gt_logs else float('nan'),
            np.median(ng_logs) if ng_logs else float('nan'),
            n_dup / n_gt if n_gt else float('nan'),
            100 * float((all_log >= 1).mean()),  # 旧版 clip 后并列 1.0 的框占比
            100 * float((~keep).mean())))
    print()
    print('说明:')
    print('  n_keep  = sigmoid > 0.03 (logit > -3.48) 过滤后保留框数')
    print('  lo_med  = 全部框 logits 中位数; lo_p90 = 90 分位')
    print('  gt_med  = 与 GT IoU>=0.5 的框 (每 GT 取最高) logit 中位数')
    print('  ng_med  = 未匹配 GT 的框 logit 中位数')
    print('  dup/gt  = 平均每个 GT 有多少个 IoU>=0.5 的重复框 (逐图贪心下都是 FP 候选)')
    print('  logit>=1 = 旧版协议下 gd 并列 1.0 的框占比')
    print('  滤掉%    = pre-filter 0.03 滤掉的框占比')


if __name__ == '__main__':
    main()
