# -*- coding: utf-8 -*-
"""analyze_restoration.py — 复原判决实验分析（两层理论终判）

判读（对应 run_restoration.py 里的 P1/P2）:
  P1  R_rest = (AP_res_base − AP_deg_base) / H_ext      高 → 图像侧复原能吃掉观测损失
                                                         → level-1 主导 + 图像侧补偿是活方向
  P2  g_CA(res) vs g_CA(deg)                             复原后增益缩水 → 增益是观测补偿
                                                         不缩水 → 增益与观测条件无关
数据源: run_restoration.json（复原图）+ run_degradation.json（退化图同口径 base/ca）
"""
import json, math

REST = r'C:\Users\张三\Desktop\实验数据\run_restoration.json'
DEG = r'C:\Users\张三\Desktop\实验数据\run_degradation.json'

rs = json.load(open(REST, encoding='utf-8'))['rows']
dg = json.load(open(DEG, encoding='utf-8'))['rows']
deg = {(r['det'], r['cls'], r['corr'], r['sev']): r for r in dg if r['corr'] != 'clean'}

print('%-10s %-14s S%d' % ('cls', 'corr', 0), end='')
print('  %7s %7s %7s %7s | %7s %7s | %7s %7s' % (
    'bClean', 'bDeg', 'bRes', 'H_ext', 'R_rest', 'R_ca*', 'gCA_deg', 'gCA_res'))
recs = []
for r in rs:
    d = deg.get((r['det'], r['cls'], r['corr'], r['sev']))
    if d is None or r['ap_clean_base'] is None:
        continue
    H = r['ap_clean_base'] - d['base']
    R_rest = (r['base'] - d['base']) / H if H > 1e-6 else float('nan')
    g_ca_deg = d['ca_pc'] - d['base']
    g_ca_res = r['ca_pc'] - r['base']
    g_star_deg = d['ca_star'] - d['base']
    R_star = ((r['base'] + max(g_star_deg, 0)) - d['base']) / H if H > 1e-6 else float('nan')
    recs.append(dict(cls=r['cls'], corr=r['corr'], sev=r['sev'], H=H,
                     R_rest=R_rest, g_ca_deg=g_ca_deg, g_ca_res=g_ca_res))
    print('%-10s %-14s S%-d  %7.4f %7.4f %7.4f %7.4f | %7.2f %7.2f | %+7.4f %+7.4f' % (
        r['cls'], r['corr'], r['sev'], r['ap_clean_base'], d['base'], r['base'], H,
        R_rest, R_star, g_ca_deg, g_ca_res))


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float('nan')
    p = float('nan')
    if abs(r) < 1 and n > 2:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        p = math.erfc(abs(t) / math.sqrt(2))
    return r, p


print('\n===== 汇总 =====')
n = len(recs)
print(f'cells={n}')
for key in ('R_rest',):
    vals = [x[key] for x in recs if x[key] == x[key]]
    if vals:
        print(f'  P1  R_rest: mean={sum(vals)/len(vals):+.3f}  median={sorted(vals)[len(vals)//2]:+.3f}  '
              f'>0.5占比={sum(v > 0.5 for v in vals)}/{len(vals)}')
# P2: 复原后 CA-PC 增益是否缩水（配对）
dif = [x['g_ca_res'] - x['g_ca_deg'] for x in recs]
mu = sum(dif) / len(dif)
sd = math.sqrt(sum((v - mu) ** 2 for v in dif) / (len(dif) - 1)) if len(dif) > 1 else 0
se = sd / math.sqrt(len(dif)) if dif else 0
print(f'  P2  mean(gCA_res − gCA_deg) = {mu:+.4f}  sd={sd:.4f}  se={se:.4f}  '
      f't≈{mu / se if se else float("nan"):.2f}  n={len(dif)}')
print(f'      缩水(更接近0)格数: {sum(abs(x["g_ca_res"]) < abs(x["g_ca_deg"]) for x in recs)}/{len(recs)}')
r, p = pearson([x['H'] for x in recs], [x['g_ca_res'] for x in recs])
print(f'  附  r(H_ext, gCA_res) = {r:+.3f}  p≈{p:.3g}')
r, p = pearson([x['H'] for x in recs], [x['R_rest'] for x in recs if True])
print('  附  r(H_ext, R_rest) = %+0.3f' % r)
