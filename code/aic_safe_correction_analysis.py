# -*- coding: utf-8 -*-
"""aic_safe_correction_analysis.py — Safe-Correction 组合门控分析

输入: aic_comp_{gdino_tiny,gdino_base,owlv2}.json + aic_visdrone_results.json
做法:
  1) 汇总所有「类 × 域」点的 g2(域偏移对齐度) 与实际增益 (A-CSPC − baseline)
  2) 验证 g2 是否预测增益方向 (Spearman)
  3) τ 选择只用 gdino-tiny 域(修正全有效的域), 然后冻结 τ 应用到全部 22 个点
     → Safe-Correction: g2≥τ 才修正, 否则回退 baseline
  4) 对比 always-baseline / always-A-CSPC / Safe-Correction 的均值与最差情况
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

points = []  # (domain, cls, g2, base_ap50, acspc_ap50, base_map, acspc_map)

for dom, fn in [('COCO-gdino-tiny', 'aic_comp_gdino_tiny.json'),
                ('COCO-gdino-base', 'aic_comp_gdino_base.json'),
                ('COCO-owlv2', 'aic_comp_owlv2.json'),
                ('VisDrone', 'aic_visdrone_results.json')]:
    p = DATA / fn
    if not p.exists():
        print(f'缺 {fn}，跳过 {dom}')
        continue
    r = json.load(open(p))
    dom_g2 = r.get('config', {}).get('g2', {})
    for c, cr in r['per_class'].items():
        g2 = cr.get('g2', dom_g2.get(c))
        points.append((dom, c, g2,
                       cr['baseline']['ap50'], cr['A-CSPC']['ap50'],
                       cr['baseline']['map'], cr['A-CSPC']['map']))

print(f'共 {len(points)} 个类-域点\n')

# 1) g2 与增益的关系
gains = [b[4] - b[3] for b in points]
g2s = [b[2] for b in points]
rho, pv = spearmanr(g2s, gains)
print(f'Spearman(g2, gain) = {rho:.3f} (p={pv:.4f})')
print(f'修正全有效的点: {sum(1 for g in gains if g > 0)}/{len(gains)}')

# 2) τ 只用 gdino-tiny 域选（那里修正全部有效）
tiny_g2 = [b[2] for b in points if b[0] == 'COCO-gdino-tiny']
tau_star = round(min(tiny_g2) - 0.01, 3)
print(f'τ* = {tau_star} (gdino-tiny 域最小 g2 − 0.01, 不偷看其他域)\n')

# 3) 三种策略对比
def strat(name, pick):
    a50s, maps_ = [], []
    for pt in points:
        use = pick(pt)
        m = 'acspc' if use else 'base'
        a50s.append(pt[4] if use else pt[3])
        maps_.append(pt[6] if use else pt[5])
    a50s, maps_ = np.array(a50s), np.array(maps_)
    d_vs_base = a50s - np.array([p[3] for p in points])
    print(f'{name:22s} mean AP50={a50s.mean():.4f}  mean mAP={maps_.mean():.4f}  '
          f'最差单点变化={d_vs_base.min():+.4f}  低于baseline的点数={int((d_vs_base < -1e-9).sum())}')
    return a50s.mean(), maps_.mean()


def tau_from(dev_points):
    """与原选择规则一致：开发集里增益为正的点的 g2 最小值 − 0.01。"""
    g2_pos = [b[2] for b in dev_points if (b[4] - b[3]) > 0]
    if not g2_pos:
        return 1.0
    return round(min(g2_pos) - 0.01, 3)


domains = sorted(set(p[0] for p in points))

print('\n=== Safe-Correction 留出验证 (leave-one-domain-out) ===')
print(f'{"held-out":16s} {"τ_dev":>7s} {"base":>8s} {"alwaysA-CSPC":>13s} {"Safe(LOO)":>10s} {"最差Δ":>9s}')
loo_rows = {}
for held in domains:
    dev = [p for p in points if p[0] != held]
    tau_dev = tau_from(dev)
    a50_base, a50_c, a50_s = [], [], []
    worst = 0.0
    for pt in points:
        if pt[0] != held:
            continue
        base, acspc = pt[3], pt[4]
        use = pt[2] >= tau_dev
        a50_base.append(base); a50_c.append(acspc); a50_s.append(acspc if use else base)
        worst = min(worst, (acspc if use else base) - base)
    a50_base, a50_c, a50_s = map(np.array, (a50_base, a50_c, a50_s))
    loo_rows[held] = {'tau_dev': tau_dev,
                      'base': round(a50_base.mean(), 4),
                      'always_acspc': round(a50_c.mean(), 4),
                      'safe_loo': round(a50_s.mean(), 4),
                      'worst_delta': round(float(worst), 4)}
    print(f'{held:16s} {tau_dev:7.3f} {a50_base.mean():8.4f} {a50_c.mean():13.4f} '
          f'{a50_s.mean():10.4f} {worst:+9.4f}')

always_base = strat('always-baseline', lambda p: False)
always_corr = strat('always-A-CSPC', lambda p: True)
safe = strat(f'Safe-Correction(τ*={tau_star})', lambda p: p[2] >= tau_star)

# 4) 逐点明细
print('\n逐点明细 (g2 → 决策 / 增益):')
for dom, c, g2, b50, a50, bm, am in sorted(points, key=lambda x: x[2]):
    use = g2 >= tau_star
    gain = a50 - b50
    ok = '✓' if (gain > 0) == use or abs(gain) < 1e-9 else '✗'
    print(f'  {dom:16s} {c:11s} g2={g2:.3f}  gain={gain:+.4f}  '
          f'{"修正" if use else "回退":2s} {ok}')

out = {
    'n_points': len(points),
    'spearman_g2_gain': {'rho': round(float(rho), 4), 'p': round(float(pv), 4)},
    'tau_star': tau_star,
    'loo_validation': loo_rows,
    'always_baseline': {'mean_ap50': round(float(always_base[0]), 4),
                        'mean_map': round(float(always_base[1]), 4)},
    'always_acspc': {'mean_ap50': round(float(always_corr[0]), 4),
                     'mean_map': round(float(always_corr[1]), 4)},
    'safe_correction': {'mean_ap50': round(float(safe[0]), 4),
                        'mean_map': round(float(safe[1]), 4)},
    'points': [{'domain': d, 'cls': c, 'g2': g, 'gain_ap50': round(a - b, 4)}
               for d, c, g, b, a, bm, am in points],
}
json.dump(out, open(DATA / 'aic_safe_correction_analysis.json', 'w'),
          ensure_ascii=False, indent=2)
print(f'\n已写入 {DATA / "aic_safe_correction_analysis.json"}')
