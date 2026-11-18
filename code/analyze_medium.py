# -*- coding: utf-8 -*-
"""analyze_medium.py — 媒介迁移轴(非遥感)结果分析

回答三问：
  Q0 轴是否成立：base 在媒介图上是否大幅掉(H_ext = AP_clean − AP_medium)？
     媒介文本锚是否比照片文本锚更贴近媒介 crop 特征(align_md > align_ph)？
  Q1 独立 headroom：r(H_ext, gain_CA)（H_ext 只用 base 打分器测 → 非循环）
  Q2 合法零假设：real 方向 gain 在 {real}∪{49 随机方向} 中的 rank → p_c；
     以及 real vs 错媒介方向。
"""
import json, math

J = r'C:\Users\张三\Desktop\实验数据\run_medium_shift.json'
d = json.load(open(J, encoding='utf-8'))
rows = d['rows']


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan'), float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs); syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return float('nan'), float('nan')
    r = sxy / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r)) if abs(r) < 1 else 9.9
    return r, math.erfc(abs(t) / math.sqrt(2))


clean = {(r['det'], r['cls']): r for r in rows if r['medium'] == 'clean'}
med = [r for r in rows if r['medium'] != 'clean']
print(f'clean={len(clean)}  medium cells={len(med)}  media={sorted(set(r["medium"] for r in med))}')

print('\n=== Q0 轴成立性 ===')
print('%-9s %-8s %8s %8s %8s | %9s %9s | %s' % (
    'cls', 'medium', 'bClean', 'bMed', 'H_ext', 'align_md', 'align_ph', '锚对有意义?'))
ok_axis = []
for r in med:
    c = clean.get((r['det'], r['cls']))
    H = (c['base'] - r['base']) if c else float('nan')
    good = r['align_medium'] > r['align_photo']
    ok_axis.append(good)
    print('%-9s %-8s %8.4f %8.4f %8.4f | %9.4f %9.4f | %s' % (
        r['cls'], r['medium'], (c['base'] if c else float('nan')), r['base'], H,
        r['align_medium'], r['align_photo'], 'YES' if good else 'no'))
print('锚对有意义(align_md>align_ph)的格: %d/%d' % (sum(ok_axis), len(ok_axis)))

print('\n=== Q1 独立 headroom 律 ===')
recs = []
for r in med:
    c = clean.get((r['det'], r['cls']))
    if not c:
        continue
    recs.append(dict(cls=r['cls'], medium=r['medium'], H=c['base'] - r['base'], base=r['base'],
                     g_ca=r['ca_pc'] - r['base'], g_tge=r['tge'] - r['base'],
                     g_naive=r['naive'] - r['base'], g_wrong=(None if r['ca_wrong_medium'] is None
                                                              else r['ca_wrong_medium'] - r['base']),
                     p_c=r['p_c'], rank=r['rank'], rand_mean=r['rand_mean'],
                     rand_gain=r['rand_mean'] - r['base'],
                     align=r['align_medium'] - r['align_photo'], dc=r['dc_mag']))
if recs:
    for key in ('g_ca', 'g_tge'):
        r_, p_ = pearson([x['H'] for x in recs], [x[key] for x in recs])
        print('  r(H_ext, %-7s) = %+0.3f  p≈%.3g  n=%d' % (key, r_, p_, len(recs)))
    r_, p_ = pearson([x['base'] for x in recs], [x['g_ca'] for x in recs])
    print('  r(base_med, g_ca ) = %+0.3f  p≈%.3g' % (r_, p_))
    r_, p_ = pearson([x['align'] for x in recs], [x['g_ca'] for x in recs])
    print('  r(align_md-ph, g_ca) = %+0.3f  p≈%.3g' % (r_, p_))

print('\n=== Q2 合法零假设(随机方向 rank) ===')
if recs:
    pcs = [x['p_c'] for x in recs]
    print('  mean p_c = %.3f   格数 p_c<0.05: %d/%d   p_c<0.10: %d/%d' % (
        sum(pcs) / len(pcs), sum(1 for p in pcs if p < 0.05), len(pcs),
        sum(1 for p in pcs if p < 0.10), len(pcs)))
    dg = [x['g_ca'] - x['rand_gain'] for x in recs]
    print('  mean(g_ca - rand_gain) = %+.4f   (real 优于随机均值)' % (sum(dg) / len(dg)))
    gw = [x for x in recs if x['g_wrong'] is not None]
    if gw:
        dw = [x['g_ca'] - x['g_wrong'] for x in gw]
        print('  mean(g_ca - g_wrongMed) = %+.4f  (real 优于错媒介)  n=%d' % (sum(dw) / len(dw), len(dw)))

print('\n=== 明细 ===')
print('%-9s %-8s %7s %8s %8s %8s %9s %9s %6s' % (
    'cls', 'medium', 'bMed', 'H_ext', 'g_CA', 'g_TGE', 'g_wrongMed', 'rand_mean', 'p_c'))
for x in sorted(recs, key=lambda z: -z['H']):
    print('%-9s %-8s %7.4f %8.4f %+8.4f %+8.4f %9s %9.4f %6.2f' % (
        x['cls'], x['medium'], x['base'], x['H'], x['g_ca'], x['g_tge'],
        ('%+.4f' % x['g_wrong']) if x['g_wrong'] is not None else '  n/a  ',
        x['rand_mean'], x['p_c']))
