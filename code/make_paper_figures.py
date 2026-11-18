# -*- coding: utf-8 -*-
"""make_paper_figures.py — 论文图表素材（英文标签，保存 PNG 300dpi）

数据全部读自 strict_ablation.json（严格消融 + 配对 bootstrap 口径修正版），
保证与报告/汇总一致，避免手工抄错。

输出到 实验数据/figures/：
  fig1_shot_curve.png    — K={1,2,3,5,10} few-shot AP（α=0.5, 6类宏平均），
                            附 baseline 与 source-only 水平参考线
  fig2_alpha_sensitivity.png — K=10 时 α∈[0,1] AP（6类宏平均），标注平台区间
  fig3_class_delta.png   — 逐类配对 Δ_few-shot−source-only 条形图（strict vs vehicle 着色）
  fig4_group_means.png   — 分组均值 baseline / source-only / few-shot 并列条形图
"""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
IN = SAVE_DIR / 'strict_ablation.json'
FIG_DIR = SAVE_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

ORDER = ['COCO-airplane', 'COCO-boat', 'COCO-car', 'COCO-truck', 'COCO-bus', 'COCO-motorcycle']
SHORT = {'COCO-airplane': 'airplane', 'COCO-boat': 'boat', 'COCO-car': 'car',
         'COCO-truck': 'truck', 'COCO-bus': 'bus', 'COCO-motorcycle': 'motorcycle'}
GROUP_COLOR = {'strict': '#1f77b4', 'vehicle_pressure': '#ff7f0e'}


def main():
    d = json.loads(IN.read_text(encoding='utf-8'))
    per = d['per_class']
    gmean = d['group_mean']
    pb = d['paired_bootstrap']
    curve = d['shot_curve']
    alpha = d['alpha_sensitivity']

    # ── Fig1: shot curve ──
    ks = sorted(int(k) for k in curve)
    ys = [curve[str(k)] for k in ks]
    base6 = np.mean([per[c]['baseline'] for c in ORDER])
    so6 = np.mean([per[c]['source_only'] for c in ORDER])
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
    ax.plot(ks, ys, 'o-', color='#2ca02c', lw=2, ms=5, label='few-shot (α=0.5)')
    ax.axhline(base6, ls='--', color='gray', lw=1.2, label=f'baseline {base6:.3f}')
    ax.axhline(so6, ls=':', color='#1f77b4', lw=1.2, label=f'source-only {so6:.3f}')
    ax.set_xlabel('K (calibration images / class)')
    ax.set_ylabel('AP50 (6-class macro mean)')
    ax.set_xticks(ks)
    ax.set_ylim(0.35, 0.55)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_shot_curve.png')
    plt.close(fig)

    # ── Fig2: alpha sensitivity ──
    as_ = sorted(float(k) for k in alpha)
    ya = [alpha[str(a)] for a in as_]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)
    ax.plot(as_, ya, 's-', color='#9467bd', lw=2, ms=5)
    ax.axvspan(0.0, 0.5, color='#2ca02c', alpha=0.10, label='α∈[0,0.5] plateau')
    ax.axhline(base6, ls='--', color='gray', lw=1.2, label=f'baseline {base6:.3f}')
    ax.set_xlabel('α (source-prototype weight)')
    ax.set_ylabel('AP50 (6-class macro mean, K=10)')
    ax.set_xticks(as_)
    ax.set_ylim(0.35, 0.55)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_alpha_sensitivity.png')
    plt.close(fig)

    # ── Fig3: per-class paired delta ──
    names = [SHORT[c] for c in ORDER]
    ds = [per[c]['d_fs_minus_so'] for c in ORDER]
    grp = [per[c]['group'] for c in ORDER]
    colors = [GROUP_COLOR[g] for g in grp]
    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=300)
    bars = ax.bar(names, ds, color=colors, alpha=0.85, edgecolor='black', lw=0.4)
    ax.axhline(0, color='black', lw=0.8)
    for b, v in zip(bars, ds):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.005 if v >= 0 else -0.015),
                f'{v:+.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=8)
    ax.set_ylabel(r'$\Delta$ AP50 (few-shot − source-only, paired)')
    ax.set_ylim(min(ds) - 0.03, max(ds) + 0.03)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GROUP_COLOR['strict'], label='strict'),
                       Patch(color=GROUP_COLOR['vehicle_pressure'], label='vehicle pressure')],
              fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_class_delta.png')
    plt.close(fig)

    # ── Fig4: group means ──
    labels = ['strict\n(airplane/boat/car)', 'vehicle\n(car/truck/bus/moto)', '6-class\nmacro mean']
    keys = [k for k in gmean]  # 保持 JSON 顺序
    base_v = [gmean[k]['baseline'] for k in keys]
    so_v = [gmean[k]['source_only'] for k in keys]
    fs_v = [gmean[k]['fewshot'] for k in keys]
    x = np.arange(len(keys))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=300)
    ax.bar(x - w, base_v, w, label='baseline', color='gray', edgecolor='black', lw=0.3)
    ax.bar(x, so_v, w, label='source-only CDVM', color='#1f77b4', edgecolor='black', lw=0.3)
    ax.bar(x + w, fs_v, w, label='few-shot K10 α=0.5', color='#2ca02c', edgecolor='black', lw=0.3)
    for xi, b, s, f in zip(x, base_v, so_v, fs_v):
        ax.text(xi, s + 0.006, f'Δ{s-b:+.3f}'.replace('Δ-', 'Δ'), ha='center', fontsize=6, color='#1f77b4')
        ax.text(xi, f + 0.012, f'Δ{f-b:+.3f}'.replace('Δ-', 'Δ'), ha='center', fontsize=7, color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('AP50')
    ax.set_ylim(0, 0.62)
    ax.legend(fontsize=8, loc='upper left', ncol=3, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_group_means.png')
    plt.close(fig)

    print('✅ 图表已生成:')
    for p in sorted(FIG_DIR.glob('*.png')):
        print(f'   {p}')


if __name__ == '__main__':
    main()
