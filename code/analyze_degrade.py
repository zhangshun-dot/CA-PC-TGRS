# -*- coding: utf-8 -*-
"""analyze_degrade.py — 退化轴(非遥感、独立 holdout)结果分析

核心问题:
  Q1 独立 headroom 律: gain_CA-PC 是否随【外部】headroom 变化?
     H_ext = AP_clean_base - AP_deg_base   (只用 base 打分器测, 与 CA-PC 无关 → 非循环)
  Q2 方向特异性: gain_CA-PC 是否显著大于 gain_ca_wrong(错误条件轴对照)?
     若两者无差异 → 该轴上的增益是"任意条件文本扰动"效应, 不是标定锚方向。
"""
import json, math, itertools

J = r'C:\Users\张三\Desktop\实验数据\run_degradation.json'
d = json.load(open(J, encoding='utf-8'))
rows = d['rows']


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan'), float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float('nan')
    if abs(r) < 1 and n > 2:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        # 双侧 p 的近似 (t 分布, 用正态近似给量级即可)
        p = math.erfc(abs(t) / math.sqrt(2))
    else:
        p = float('nan')
    return r, p


clean = {}
for r in rows:
    if r['corr'] == 'clean':
        clean[(r['det'], r['cls'])] = r

deg = [r for r in rows if r['corr'] != 'clean']
print(f'clean cells={len(clean)}  degraded cells={len(deg)}')

recs = []
for r in deg:
    c = clean[(r['det'], r['cls'])]
    H = c['base'] - r['base']                     # 外部 headroom (base 打分器)
    recs.append(dict(det=r['det'], cls=r['cls'], corr=r['corr'], sev=r['sev'],
                     base_clean=c['base'], base=r['base'], H=H,
                     g_ca=r['ca_pc'] - r['base'], g_wrong=r['ca_wrong'] - r['base'],
                     g_tge=r['tge'] - r['base'], g_naive=r['naive'] - r['base'],
                     g_star=r['ca_star'] - r['base'], dc=r['dc_mag']))

print('\n%-10s %7s %7s %7s | %8s %8s %8s %8s %8s' % (
    'cls', 'bClean', 'bDeg', 'H_ext', 'g_CA', 'g_wrong', 'g_TGE', 'g_naive', 'g_star'))
for cls in dict.fromkeys(x['cls'] for x in recs):
    g = [x for x in recs if x['cls'] == cls]
    f = lambda k: sum(x[k] for x in g) / len(g)
    print('%-10s %7.4f %7.4f %7.4f | %+8.4f %+8.4f %+8.4f %+8.4f %+8.4f' % (
        cls, f('base_clean'), f('base'), f('H'), f('g_ca'), f('g_wrong'),
        f('g_tge'), f('g_naive'), f('g_star')))

print('\n--- Q1: 增益 vs 外部 headroom ---')
for key in ('g_ca', 'g_wrong', 'g_tge'):
    xs = [x['H'] for x in recs]
    ys = [x[key] for x in recs]
    r, p = pearson(xs, ys)
    print('  r(H_ext, %-8s) = %+0.3f  p≈%.3g  n=%d' % (key, r, p, len(xs)))
xs = [x['base'] for x in recs]      # 绝对退化后 base 水平
for key in ('g_ca', 'g_wrong'):
    ys = [x[key] for x in recs]
    r, p = pearson(xs, ys)
    print('  r(base_deg, %-8s) = %+0.3f  p≈%.3g' % (key, r, p))

print('\n--- Q2: 方向特异性 (配对) ---')
dif = [x['g_ca'] - x['g_wrong'] for x in recs]
n = len(dif)
mu = sum(dif) / n
sd = math.sqrt(sum((v - mu) ** 2 for v in dif) / (n - 1)) if n > 1 else 0
se = sd / math.sqrt(n) if n else 0
print('  mean(g_CA - g_wrong) = %+.4f  sd=%.4f  se=%.4f  t≈%.2f  n=%d' % (
    mu, sd, se, (mu / se if se else 0), n))
win = sum(1 for v in dif if v > 1e-9)
lose = sum(1 for v in dif if v < -1e-9)
print('  CA-PC > wrong: %d cells, < : %d cells, ==: %d' % (win, lose, n - win - lose))

print('\n--- 每 cell 明细 (按 H 降序) ---')
print('%-8s %-14s %3s %7s %8s %8s %8s %8s' % ('cls', 'corr', 'S', 'bDeg', 'H', 'g_CA', 'g_wrong', 'g_star'))
for x in sorted(recs, key=lambda z: -z['H']):
    print('%-8s %-14s %3d %7.4f %8.4f %+8.4f %+8.4f %+8.4f' % (
        x['cls'], x['corr'], x['sev'], x['base'], x['H'], x['g_ca'], x['g_wrong'], x['g_star']))
