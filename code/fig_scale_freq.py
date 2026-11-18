# -*- coding: utf-8 -*-
"""fig_scale_freq.py — 尺度/类频敏感性双面板图 (A1, 保守口径)

关键口径(见 run_sizebin_pairedci.py 的说明):
  同一 (direction,src,eval) cell 的 4 个检测器**共享同一组 GT**(同一 avg_gt_area)与同一 coco_freq,
  所以 36 行并非独立观测; 行级 p 值反保守。本图据 cell 均值(n=9; 类频面板只取前向 6 cell)
  报告 Spearman, 并把 4 个个体点画成淡色, 使重复结构可见。
  另: 反向 cell 的目标类名(ship/vehicle)不是 COCO 类, 类频代理无定义 -> 类频面板只用前向 cell。
数据: C:\\Users\\张三\\Desktop\\实验数据\\closed_loop_scalefreq_clean.csv
输出: figs/fig4_scalefreq.pdf/.png
"""
import csv, math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = r'C:\Users\张三\Desktop\实验数据\closed_loop_scalefreq_clean.csv'
OUT = r'C:\Users\张三\WorkBuddy\2026-09-23-16-06-04\figs\fig4_scalefreq'


def pear(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    return pear(rank(x), rank(y))


def betacf(a, b, xx):
    EPS, FPMIN = 3e-12, 1e-300
    qab = a + b; qap = a + 1; qam = a - 1; c = 1; d = 1 - qab * xx / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1 / d; h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * xx / ((qam + m2) * (a + m2)); d = 1 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * xx / ((a + m2) * (qap + m2)); d = 1 + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1 + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1 / d; de = d * c; h *= de
        if abs(de - 1) < EPS: break
    return h


def betai(a, b, xx):
    if xx <= 0: return 0.0
    if xx >= 1: return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(xx) + b * math.log(1 - xx))
    if xx < (a + 1) / (a + b + 2): return bt * betacf(a, b, xx) / a
    return 1 - bt * betacf(b, a, 1 - xx) / b


def pval(r, n):
    if abs(r) >= 1 or n <= 2: return 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r)); d = n - 2
    return betai(d / 2, 0.5, d / (d + t * t))


rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
grp = defaultdict(list)
for r in rows:
    grp[(r['direction'], r['src'], r['eval'])].append(r)

cells = {k: {'la': float(v[0]['log_area']), 'lf': float(v[0]['log_freq']),
             'gain': sum(float(x['gain']) for x in v) / len(v), 'n': len(v)} for k, v in grp.items()}
fwd = [k for k in cells if k[0] == 'fwd']

la9 = [cells[k]['la'] for k in cells]; g9 = [cells[k]['gain'] for k in cells]
lf6 = [cells[k]['lf'] for k in fwd]; g6 = [cells[k]['gain'] for k in fwd]
ra = spearman(la9, g9); pa = pval(ra, len(la9))
rf = spearman(lf6, g6); pf = pval(rf, len(lf6))
print(f'[cell-level] n_area={len(la9)} rho_area={ra:+.3f} p={pa:.3f} | n_freq={len(lf6)} rho_freq={rf:+.3f} p={pf:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=200)


def scatter_cellmean(ax, xkey, keys, rho, p, xlabel, title, ckey):
    for k in keys:                                   # 淡色个体点(4 检测器)
        for r in grp[k]:
            ax.scatter(float(r[ckey]), float(r['gain']), s=16, c='#9aa5b1',
                       alpha=0.45, edgecolors='none', zorder=1)
    xs = [cells[k][xkey] for k in keys]; ys = [cells[k]['gain'] for k in keys]
    ax.scatter(xs, ys, s=64, c='#c53030', marker='o', alpha=0.95,
               edgecolors='white', linewidths=0.7, zorder=3, label='cell mean')
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    slope = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx
    xr = [min(xs), max(xs)]
    ax.plot(xr, [my + slope * (v - mx) for v in xr], color='#c53030', ls='--', lw=1.0, alpha=0.8, zorder=2)
    ax.axhline(0, color='#999', lw=0.7, zorder=0)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(r'$\Delta$AP50 (CA-PC $-$ base)', fontsize=8.5)
    ax.set_title(f'{title}  ($n={n}$ cells)\nSpearman $\\rho={rho:+.2f}$ ($p={p:.2f}$)', fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    for s in ax.spines.values():
        s.set_color('#bbb')
    ax.grid(alpha=0.18, lw=0.5)


scatter_cellmean(axes[0], 'la', list(cells), ra, pa, r'$\log$ mean GT object area', 'Scale', 'log_area')
scatter_cellmean(axes[1], 'lf', fwd, rf, pf, r'$\log$ COCO class frequency', 'Class frequency (forward)', 'log_freq')
axes[0].legend(fontsize=7, loc='lower right', frameon=False)
fig.tight_layout(pad=0.4)
fig.savefig(OUT + '.pdf', bbox_inches='tight')
fig.savefig(OUT + '.png', bbox_inches='tight', dpi=200)
print('saved', OUT)
