# -*- coding: utf-8 -*-
"""run_closed_loop_analysis.py — 证据包：per-class CSV + 五类可视化 + 门控信号相关性

输入: 实验数据/run_closed_loop_balanced.json (含 cells_detail + summary + diag)
输出:
  - 实验数据/closed_loop_cells.csv        (per-class 明细, 论文补充材料)
  - 实验数据/figs/fig1_scatter_beta.png   signal(dc_mag/n_pred/max_conf) vs oracle β*
  - 实验数据/figs/fig2_gamma_curve.png    γ(conf) 多门控函数/尺度比较
  - 实验数据/figs/fig3_oracle_bar.png      per-class oracle% (dc×gate(conf)) 按检测器分组
  - 实验数据/figs/fig4_heatmap.png        detector × method 平均 AP50 / 增益 heatmap
  - 实验数据/figs/fig5_reliability.png     max_conf 分布 + gate(conf) 叠加 (conf gate 行为)
  - stdout: 门控信号 vs β*/gain 的 Spearman/Pearson 表

用法:
  "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_analysis.py
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

JSON_IN = r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'
CSV_OUT = r'C:\Users\张三\Desktop\实验数据\closed_loop_cells.csv'
FIGDIR = r'C:\Users\张三\Desktop\实验数据\figs'
os.makedirs(FIGDIR, exist_ok=True)

DET_ORDER = ['gdino', 'owlv2', 'yolo']
DET_LABEL = {'gdino': 'GDINO-tiny', 'owlv2': 'OWLv2', 'yolo': 'YOLO-World'}
PAL = {'gdino': '#1f77b4', 'owlv2': '#d62728', 'yolo': '#2ca02c'}

d = json.load(open(JSON_IN, encoding='utf-8'))
cells = d['cells_detail']
summary = d['summary']
cfg = d['config']
c0 = cfg['gate']['c0']
PRIOR_SCALE = cfg['gate']['prior_scale']

df = pd.DataFrame(cells)
df['det_label'] = df['det'].map(DET_LABEL)


# ── 1. per-class CSV ──
cols = ['det', 'det_label', 'direction', 'src', 'eval', 'dc_mag', 'n_pred', 'max_conf', 'score_var',
        'beta_star', 'ap_star', 'base', 'ap_fixed2', 'ap_dc', 'ap_dc_x_count',
        'ap_dc_x_conf', 'ap_dc_x_conf_perdet', 'ap_dc_x_sigmoid', 'ap_dc_x_piece', 'ap_gate_only',
        'oracle_ratio_dc_x_conf', 'oracle_ratio_dc_x_conf_perdet',
        'gain_dc_x_conf_vs_base', 'gain_dc_x_conf_vs_fix']
df[cols].round(6).to_csv(CSV_OUT, index=False)
print(f'[CSV] -> {CSV_OUT}  ({len(df)} cells)')


# ── 2. fig1: scatter signal vs oracle β* ──
signals = [('dc_mag', 'dc_mag (\u2016E_photo\u2212E_overhead\u2016)'),
           ('max_conf', 'max_conf (target-domain mean max conf)'),
           ('n_pred', 'n_pred (# predicted boxes)'),
           ('score_var', 'score_var (box-conf std)')]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, (skey, stitle) in zip(axes, signals):
    for det in DET_ORDER:
        sub = df[df['det'] == det]
        x, y = sub[skey].values, sub['beta_star'].values
        ax.scatter(x, y, c=PAL[det], label=DET_LABEL[det], s=45, alpha=0.8, edgecolor='k', lw=0.4)
        if len(x) >= 3:
            z = np.polyfit(x, y, 1); xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, np.polyval(z, xs), c=PAL[det], ls='--', lw=1)
            rho, p = spearmanr(x, y)
            ax.text(0.05, 0.92 - 0.08 * DET_ORDER.index(det), f'{DET_LABEL[det]}: ρ={rho:+.2f} (p={p:.3f})',
                    transform=ax.transAxes, color=PAL[det], fontsize=9, va='top')
    ax.set_xlabel(stitle); ax.set_ylabel('oracle β*')
    ax.set_title(f'signal vs β*  ({stitle})'); ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
fig.suptitle('Zero-GT signal vs oracle beta* (Spearman per detector)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f'{FIGDIR}/fig1_scatter_beta.png', dpi=130); plt.close(fig)
print('[fig1] scatter vs β* saved')


# ── 3. fig2: γ(conf) 门控函数比较 ──
def g_tanh(conf, scale, c): return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0, 1))
def g_sigm(conf, scale, c): return float(1.0 / (1.0 + np.exp(scale * (conf - c))))
def g_piece(conf, scale, c): return float(np.clip(1.0 - scale * (conf - c), 0, 1))
def g_iso(conf, scale, c):  return float(1.0 / (1.0 + np.exp((conf - c) / (1.0 / scale))))

conf_grid = np.linspace(0, 1, 200)
funcs = [('tanh(3(c−c0))', g_tanh, 3.0), ('sigmoid', g_sigm, 3.0),
         ('piecewise', g_piece, 3.0), ('1/(1+e^((c−c0)/k))', g_iso, 3.0)]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, scale in zip(axes, [3.0, 4.5]):
    for name, fn, _ in funcs:
        ax.plot(conf_grid, [fn(c, scale, c0) for c in conf_grid], label=name, lw=1.6)
    ax.axvline(c0, ls=':', c='gray'); ax.text(c0, 0.02, f'c0={c0:.2f}', fontsize=8, color='gray')
    ax.set_xlabel('conf'); ax.set_ylabel('gate(conf)')
    ax.set_title(f'gate function (scale={scale})'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('Gate function / scale sensitivity (pooled c0, gamma = dc x gate)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f'{FIGDIR}/fig2_gamma_curve.png', dpi=130); plt.close(fig)
print('[fig2] γ(conf) curves saved')


# ── 4. fig3: per-class oracle% (dc×gate(conf)) 分组柱状 ──
fig, ax = plt.subplots(figsize=(15, 5))
xpos = np.arange(len(df))
colors = [PAL[det] for det in df['det']]
ax.bar(xpos, df['oracle_ratio_dc_x_conf'].values * 100, color=colors, edgecolor='k', lw=0.3)
ticks = [f"{DET_LABEL[det]}\n{evalcl}\n({dir_})" for det, dir_, evalcl in zip(df['det'], df['direction'], df['eval'])]
ax.set_xticks(xpos); ax.set_xticklabels(ticks, fontsize=6, rotation=90)
ax.axhline(95, ls='--', c='k', lw=1); ax.text(0.5, 95.5, '95% oracle coverage', fontsize=8)
ax.set_ylabel('oracle coverage % (dc×gate(conf) AP50 / oracle)')
ax.set_title('per-class oracle coverage (3 detectors x 9 units)')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig3_oracle_bar.png', dpi=130); plt.close(fig)
print('[fig3] per-class oracle bar saved')


# ── 5. fig4: detector × method heatmap ──
methods = ['base', 'ap_fixed2', 'ap_dc', 'ap_dc_x_count', 'ap_dc_x_conf',
           'ap_dc_x_conf_perdet', 'ap_dc_x_sigmoid', 'ap_dc_x_piece', 'ap_gate_only', 'ap_star']
mlabel = {'base': 'beta=0 base', 'ap_fixed2': 'fixed beta=2.0', 'ap_dc': 'pure dc',
          'ap_dc_x_count': 'dc x n_pred', 'ap_dc_x_conf': 'dc x tanh gate',
          'ap_dc_x_conf_perdet': 'dc x per-det gate', 'ap_dc_x_sigmoid': 'dc x sigmoid',
          'ap_dc_x_piece': 'dc x piece', 'ap_gate_only': 'gate only (no dc)', 'ap_star': 'oracle'}
M = np.zeros((len(DET_ORDER), len(methods)))
for i, det in enumerate(DET_ORDER):
    sub = df[df['det'] == det]
    for j, m in enumerate(methods):
        M[i, j] = sub[m].mean()
fig, ax = plt.subplots(figsize=(13, 3.2))
sns.heatmap(M * 100, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'AP50'},
            xticklabels=[mlabel[m] for m in methods], yticklabels=[DET_LABEL[d] for d in DET_ORDER], ax=ax)
ax.set_title('detector x method mean AP50 (%)  (#13 two-factor / #14 c0 ablations)')
fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig4_heatmap.png', dpi=130); plt.close(fig)
print('[fig4] heatmap saved')


# ── 6. fig5: reliability-ish (max_conf 分布 + gate) ──
fig, ax = plt.subplots(figsize=(10, 4.5))
for det in DET_ORDER:
    sub = df[df['det'] == det]
    ax.hist(sub['max_conf'].values, bins=12, alpha=0.4, label=f'{DET_LABEL[det]} (n={len(sub)})',
            color=PAL[det], density=True)
ax2 = ax.twinx()
ax2.plot(conf_grid, [g_tanh(c, 3.0, c0) for c in conf_grid], 'k--', lw=2, label='gate(conf)=1−tanh(3(c−c0))')
ax.set_xlabel('max_conf (target-domain mean max conf)'); ax.set_ylabel('density')
ax2.set_ylabel('gate(conf)', color='k'); ax2.legend(loc='upper right', fontsize=8)
ax.set_title('conf-gate behavior: max_conf distribution (left) and gate(conf) curve (right)')
ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f'{FIGDIR}/fig5_reliability.png', dpi=130); plt.close(fig)
print('[fig5] reliability-ish saved')


# ── 7. 门控信号相关性表（Spearman/Pearson vs β* 与 gain） ──
print('\n[门控信号相关性]  signal vs oracle β* / gain (三检测器)')
for det in DET_ORDER:
    sub = df[df['det'] == det]
    print(f'  [{DET_LABEL[det]}]')
    for skey in ['dc_mag', 'n_pred', 'max_conf', 'score_var']:
        x = sub[skey].values; yb = sub['beta_star'].values; yg = sub['gain_dc_x_conf_vs_base'].values
        rs, ps = spearmanr(x, yb); rp, pp = pearsonr(x, yb)
        rg, pg = spearmanr(x, yg)
        print(f'    {skey:<10} vs β*:  ρS={rs:+.3f}(p={ps:.3f}) ρP={rp:+.3f}(p={pp:.3f})   vs gain: ρS={rg:+.3f}(p={pg:.3f})')

print(f'\n全部产物在: {FIGDIR} 与 {CSV_OUT}')
