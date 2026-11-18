# -*- coding: utf-8 -*-
"""run_closed_loop_bidir_matrix.py — #2 双向跨检测器矩阵（3 检测器 x fwd/rev 定向增益）

输入: 实验数据/closed_loop_cells.csv (run_closed_loop_balanced.py 产物)
输出:
  - 实验数据/figs/fig6_bidir_matrix.png   检测器 x 方向 平均定向增益 heatmap + 边框图
  - stdout: 每检测器 fwd/rev 定向增益(=dc×gate(conf) AP50 − 基线) 与 oracle%

定向增益定义: gain_dir(det, direction) = mean over units( ap_dc_x_conf − base )
  (即闭环在"该方向"上相对零外推基线的真实提升；rev 即反向 CA-PC)

用法:
  "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_bidir_matrix.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CSV_IN = r'C:\Users\张三\Desktop\实验数据\closed_loop_cells.csv'
FIGDIR = r'C:\Users\张三\Desktop\实验数据\figs'

df = pd.read_csv(CSV_IN)
DET_ORDER = ['gdino', 'owlv2', 'yolo']
DET_LABEL = {'gdino': 'GDINO-tiny', 'owlv2': 'OWLv2', 'yolo': 'YOLO-World'}

# 每单元定向增益（dc×gate(conf) − 基线）
df['gain_fwd'] = np.where(df['direction'] == 'fwd', df['ap_dc_x_conf'] - df['base'], np.nan)
df['gain_rev'] = np.where(df['direction'] == 'rev', df['ap_dc_x_conf'] - df['base'], np.nan)

print('[#2 双向跨检测器矩阵]  定向增益 = dc×gate(conf) AP50 − 零外推基线(β=0)\n')
print(f'  {"detector":<12}{"fwd gain":>10}{"rev gain":>10}{"fwd oracle%":>13}{"rev oracle%":>13}')
M_gain = np.zeros((len(DET_ORDER), 2))   # [det, dir]  mean directed gain
M_orac = np.zeros((len(DET_ORDER), 2))
for i, det in enumerate(DET_ORDER):
    sub = df[df['det'] == det]
    gf = sub['gain_fwd'].mean(); gr = sub['gain_rev'].mean()
    of = (sub[sub['direction'] == 'fwd']['ap_dc_x_conf'] / sub[sub['direction'] == 'fwd']['ap_star']).mean() * 100
    orc = (sub[sub['direction'] == 'rev']['ap_dc_x_conf'] / sub[sub['direction'] == 'rev']['ap_star']).mean() * 100
    M_gain[i, 0], M_gain[i, 1] = gf, gr
    M_orac[i, 0], M_orac[i, 1] = of, orc
    print(f'  {DET_LABEL[det]:<12}{gf:>+10.4f}{gr:>+10.4f}{of:>12.1f}%{orc:>12.1f}%')

# 也报告"双向都正"的单元占比（闭环在两个方向都成立的比例）
both_pos = df.groupby(['det', 'direction'])['ap_dc_x_conf'].apply(lambda s: (s - df.loc[s.index, 'base']).gt(0).mean())
print('\n  各检测器各方向"定向增益>0"单元占比:')
for det in DET_ORDER:
    for d in ('fwd', 'rev'):
        v = both_pos.get((det, d), np.nan)
        print(f'    {DET_LABEL[det]:<12} {d:<4} {v*100:.1f}%')

# 图: 检测器 x 方向 定向增益 heatmap（含 oracle% 标注）
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, (M, title, unit) in zip(axes, [(M_gain, 'mean directed gain (dc×gate − base)', 'AP50'),
                                        (M_orac, 'oracle coverage % (dc×gate / β*)', '%')]):
    sns.heatmap(M, annot=True, fmt='.3f' if unit == 'AP50' else '.1f', cmap='YlGnBu',
                xticklabels=['fwd (DIOR→COCO)', 'rev (COCO→DIOR)'],
                yticklabels=[DET_LABEL[d] for d in DET_ORDER], cbar_kws={'label': unit}, ax=ax)
    ax.set_title(title)
fig.suptitle('#2 bidirectional cross-detector matrix (3 detectors x fwd/rev)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{FIGDIR}/fig6_bidir_matrix.png', dpi=130); plt.close(fig)
print(f'\n[fig6] -> {FIGDIR}/fig6_bidir_matrix.png')

# 存 json 供文档引用
json.dump({'M_gain': M_gain.tolist(), 'M_oracle': M_orac.tolist(),
           'dets': DET_ORDER, 'dirs': ['fwd', 'rev']},
          open(r'C:\Users\张三\Desktop\实验数据\bidir_matrix.json', 'w'), indent=2)
