# -*- coding: utf-8 -*-
r"""tge_soft_routing_analysis.py — 软路由实验·结果分析 (探索阶段, 非确认)

读 tge_soft_routing.json, 输出:
  1) 逐类混合曲线 AP50(w), 每域内部峰判定 (软路由是否有收益空间)
  2) 机制信号与 gain(ext−B2) 的 Spearman (22 前向类)
  3) 单信号二分路由分离度 (⚠️ 阈值事后计算, 仅探索, 需重新预注册才可作证据)
  4) 汇总 JSON

运行: python tge_soft_routing_analysis.py   (~秒级)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
sr = json.load(open(DATA / 'tge_soft_routing.json', encoding='utf-8'))
cls = sr['classes']
WS = sr['config']['ws']
FORWARD = [n for n in cls if not n.startswith('REV-')]
GAIN = {n: cls[n]['gain_extB2'] for n in cls}
GROUPS = {g: [n for n in FORWARD if n.startswith(p)] for g, p in
          [('COCO', 'COCO-'), ('DOTA', 'DOTA-'), ('VOC', 'VOC-'), ('VisDrone', 'VD-')]}

# ── 1) 混合曲线 + 内部峰 ──
print('=' * 100)
print('逐类混合曲线 AP50(w)  [w=0→B2, 1→ext]   ●=内部峰>两端')
print('=' * 100)
interior = {}
for n in sorted(FORWARD, key=lambda x: -GAIN[x]):
    c = cls[n]['curve']
    vals = [c[str(w)] for w in WS]
    ends = max(vals[0], vals[-1])
    ip = max(vals[1:4]) if len(vals) > 4 else -1
    has_peak = ip > ends + 1e-6
    interior[n] = has_peak
    mark = '  ●' if has_peak else ''
    print('%-18s %+8.4f | ' % (n, GAIN[n]) +
          ' '.join('%5.3f' % v for v in vals) + ('  peak=%.3f' % ip if has_peak else '') + mark)
print('-' * 100)
per_dom = {g: sum(1 for n in ns if interior[n]) for g, ns in GROUPS.items()}
print('内部峰类数/域: %s' % per_dom)
n_peak = sum(interior.values())
print('22 类中 %d 类存在内部峰' % n_peak)

# 域级 oracle 峰 (均值曲线)
print('\n域级混合曲线 (类均值 AP50):')
print('%-10s %8s %8s %8s %8s %8s   %s' % ('domain', *[str(w) for w in WS], 'peak'))
for g, ns in GROUPS.items():
    if not ns:
        continue
    mean_curve = [float(np.mean([cls[n]['curve'][str(w)] for n in ns])) for w in WS]
    ends = max(mean_curve[0], mean_curve[-1])
    ip = max(mean_curve[1:4]) if len(mean_curve) > 4 else -1
    has = ip > ends + 1e-6
    print('%-10s %8.4f %8.4f %8.4f %8.4f %8.4f   %s' %
          (g, *mean_curve, ('内部峰 %.3f' % ip) if has else '单调'))

# ── 2) 机制信号 × gain ──
print('\n' + '=' * 100)
print('机制信号 vs gain(ext−B2), 前向 22 类 Spearman')
print('=' * 100)
SIG_KEYS = ['agree', 'sep_src', 'gd_hi_frac', 'small_frac', 'low_band_recovery',
            'low_band_lift', 'g1_roundtrip', 'high_band_ext_agree']
corr = {}
for key in SIG_KEYS:
    pairs = [(cls[n]['feats'].get(key), GAIN[n]) for n in FORWARD]
    pairs = [(x, y) for x, y in pairs if x is not None]
    if len(pairs) >= 6:
        rho, p = spearmanr([x for x, _ in pairs], [y for _, y in pairs])
        corr[key] = {'n': len(pairs), 'rho': round(float(rho), 4), 'p': round(float(p), 4)}
        print('%-20s n=%2d  ρ=%+.3f  p=%.4f  %s' % (key, len(pairs), rho, p,
                                                    '*' if p < 0.05 else ''))
    else:
        print('%-20s n=%2d (不足)' % (key, len(pairs)))

# ── 3) 单信号二分路由 (探索, ⚠️ 阈值事后) ──
print('\n' + '=' * 100)
print('单信号二分路由 (前向 22 类; ⚠️ 阈值事后扫描, 仅探索, 需重新预注册)')
print('=' * 100)
neg = [n for n in FORWARD if GAIN[n] < 0]
pos = [n for n in FORWARD if GAIN[n] >= 0]
print('失效 %d 类 / 有益 %d 类; 全外推基准 = %.3f (%d/%d)' %
      (len(neg), len(pos), len(pos) / len(FORWARD), len(pos), len(FORWARD)))
routing = {}
for key in SIG_KEYS:
    valid = [n for n in FORWARD if cls[n]['feats'].get(key) is not None]
    if len(valid) < 8:
        continue
    xs = np.array([cls[n]['feats'][key] for n in valid])
    ys = np.array([GAIN[n] >= 0 for n in valid])
    best = (0, None, None)
    for th in np.percentile(xs, np.linspace(5, 95, 19)):
        for d in [-1, 1]:
            pred = (d * xs) < (d * th)
            acc = np.mean(pred == ys)
            if acc > best[0]:
                best = (acc, th, d)
    acc, th, d = best
    routing[key] = {'acc': round(float(acc), 4), 'n': len(valid), 'th': round(float(th), 4),
                    'dir': d}
    print('%-20s 事后最优 acc=%.3f (%d/%d)' % (key, acc, int(acc * len(valid)), len(valid)))

# ── 4) 域级最近邻 (φ 用新信号, 复验 S2) ──
print('\n' + '=' * 100)
print('域特征 φ(新信号) 最近邻 (复验 S2)')
print('=' * 100)
dom_phi = {g: {k: float(np.nanmean([cls[n]['feats'].get(k, np.nan) for n in ns]))
               for k in ['agree', 'sep_src', 'gd_hi_frac', 'low_band_recovery', 'low_band_lift']}
           for g, ns in GROUPS.items() if ns}
FEAT2 = ['agree', 'sep_src', 'low_band_recovery']
D = list(dom_phi)
Z = np.array([[dom_phi[n][k] for k in FEAT2] for n in D])
Zm = Z.mean(0); Zs = Z.std(0) + 1e-9
Zz = (Z - Zm) / Zs
for i, held in enumerate(D):
    ds = {dev: float(np.linalg.norm(Zz[i] - Zz[j])) for j, dev in enumerate(D) if dev != held}
    nn = min(ds, key=ds.get)
    print('%-10s → NN %-10s  (dist %.3f)' % (held, nn, ds[nn]))

nn_out = {}
for i, held in enumerate(D):
    ds = {dev: float(np.linalg.norm(Zz[i] - Zz[j])) for j, dev in enumerate(D) if dev != held}
    nn_out[held] = {'nn': min(ds, key=ds.get), 'dist': round(ds[min(ds, key=ds.get)], 3)}

out = {'interior_peaks': interior, 'per_domain_peaks': per_dom,
       'corr': corr, 'routing_posthoc': routing, 'domains': D, 'nn': nn_out}
with open(DATA / 'tge_soft_routing_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('\n→ tge_soft_routing_analysis.json')
