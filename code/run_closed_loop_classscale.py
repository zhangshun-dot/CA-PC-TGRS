# -*- coding: utf-8 -*-
"""run_closed_loop_classscale.py — #11 类尺度/长尾敏感性（CA-PC 增益是否依赖目标尺度/类频）

复用三检测器 COCO 评测缓存，逐单元(27)算:
  - avg_gt_area: 该单元 GT 框平均面积(px^2, log) —— 目标尺度代理
  - coco_freq:   COCO val 该 eval 类实例数 —— 类频/长尾代理
然后 Spearman 相关 gain(dc×gate vs base) ~ avg_gt_area / coco_freq（分检测器 + 汇总）。
回答: 闭环增益是否偏向某尺度/某频次类（若是，则长尾/小目标场景需特别说明）。

用法:
  "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_classscale.py
"""
import os, sys, json, glob
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
CACHES = {'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'),
          'owlv2': Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache'),
          'yolo':  Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache')}
CSV_IN = r'C:\Users\张三\Desktop\实验数据\closed_loop_cells.csv'
FIGDIR = r'C:\Users\张三\Desktop\实验数据\figs'
COCO_ANN = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
os.makedirs(FIGDIR, exist_ok=True)


def box_area(b):
    if len(b) == 0:
        return 0.0
    w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]
    return float(np.mean(w * h))


def main():
    # COCO val 类频
    ann = json.load(open(COCO_ANN))
    cat2id = {c['name']: c['id'] for c in ann['categories']}
    freq = {}
    for c in ann['categories']:
        freq[c['name']] = sum(1 for o in ann['annotations'] if o['category_id'] == c['id'])
    print(f'[#11 类尺度] COCO val 类频已载入 ({len(freq)} 类)')

    rows = []
    for det, cdir in CACHES.items():
        for f in sorted(cdir.glob('*_n*.pkl')):
            name = f.stem
            if name.startswith('fwd_dior2coco'):
                direction, eval_cls = 'fwd', name.split('_')[2]
            elif name.startswith('rev_coco2dior'):
                direction, eval_cls = 'rev', name.split('_')[2]
            else:
                continue
            blob = pickle.load(open(f, 'rb')) if False else __import__('pickle').load(open(f, 'rb'))
            per_img = blob['per_img']
            areas = [box_area(p['gt']) for p in per_img if len(p['gt']) > 0]
            avg_area = float(np.mean(areas)) if areas else 0.0
            rows.append({'det': det, 'direction': direction, 'eval': eval_cls,
                         'avg_gt_area': avg_area, 'log_area': np.log10(avg_area + 1),
                         'coco_freq': freq.get(eval_cls, 0),
                         'log_freq': np.log10(freq.get(eval_cls, 1))})
    df_scale = pd.DataFrame(rows)
    df_cell = pd.read_csv(CSV_IN)
    df = df_cell.merge(df_scale, on=['det', 'direction', 'eval'], how='left')
    df.to_csv(r'C:\Users\张三\Desktop\实验数据\closed_loop_classscale.csv', index=False)

    print('\n[#11 类尺度]  gain(dc×gate vs base) ~ 目标尺度 / 类频  (Spearman)')
    for det in ['gdino', 'owlv2', 'yolo', 'ALL']:
        sub = df if det == 'ALL' else df[df['det'] == det]
        sub = sub.dropna(subset=['gain_dc_x_conf_vs_base', 'log_area', 'log_freq'])
        if len(sub) < 3:
            continue
        r_a, p_a = spearmanr(sub['log_area'], sub['gain_dc_x_conf_vs_base'])
        r_f, p_f = spearmanr(sub['log_freq'], sub['gain_dc_x_conf_vs_base'])
        print(f'  {det:<6} n={len(sub):<3} gain~log_area ρ={r_a:+.3f}(p={p_a:.3f})   '
              f'gain~log_freq ρ={r_f:+.3f}(p={p_f:.3f})')

    # 图: gain vs log_area（按检测器）
    fig, ax = plt.subplots(figsize=(8, 5))
    for det, c in (('gdino', '#1f77b4'), ('owlv2', '#d62728'), ('yolo', '#2ca02c')):
        s = df[df['det'] == det].dropna(subset=['gain_dc_x_conf_vs_base', 'log_area'])
        ax.scatter(s['log_area'], s['gain_dc_x_conf_vs_base'], c=c, label=det, s=45, alpha=0.8, edgecolor='k', lw=0.4)
    ax.axhline(0, ls='--', c='gray')
    ax.set_xlabel('log10(mean GT box area, px^2)'); ax.set_ylabel('CA-PC gain (dc×gate − base)')
    ax.set_title('#11 class-scale sensitivity: gain vs object scale'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig7_classscale.png', dpi=130); plt.close(fig)
    print(f'[fig7] -> {FIGDIR}/fig7_classscale.png')
    print(f'[csv] -> 实验数据/closed_loop_classscale.csv ({len(df)} cells)')


if __name__ == '__main__':
    main()
