# -*- coding: utf-8 -*-
r"""tge_three_tier_oracle.py — 三档 oracle 体检 (选择器探索协议·实验 3)

三档 a(x) ∈ {baseline(不干预), B2, ext}。核心问题不是"选哪个系数", 而是
"是否应该重打分"。本脚本按冻结判据检查 oracle 上限是否足够:

判据 (用户 2026-08-27 冻结):
  1) 逐类三档 oracle = max(baseline, B2, ext)
  2) 若三档 oracle 相对最佳固定方案的平均提升 <0.01, 或主要收益只来自一两个类 → 停止
  3) VisDrone baseline−ext=0.0029 是否有配对 bootstrap 支持 (区间跨零 → 仅风险控制候选)
  4) 只有 oracle 上限充足才开发带回退的三档选择器 (max_a LCB(Δ̂_a) ≤ 0 → baseline)

数据: tge_soft_routing.json (逐类 base/B2/ext, 同协议)。
运行: python tge_three_tier_oracle.py   (~秒级)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
sr = json.load(open(DATA / 'tge_soft_routing.json', encoding='utf-8'))
lodo = json.load(open(DATA / 'tge_dccs_lodo.json', encoding='utf-8'))
cls = sr['classes']
FORWARD = [n for n in cls if not n.startswith('REV-')]
DOM_PREFIX = [('COCO', 'COCO-'), ('DOTA', 'DOTA-'), ('VOC', 'VOC-'), ('VisDrone', 'VD-')]

rows = []
for n in FORWARD:
    c = cls[n]
    rows.append({'name': n,
                 'base': c['curve']['base'], 'b2': c['curve']['0.0'], 'ext': c['curve']['1.0']})
for r in rows:
    acts = {'baseline': r['base'], 'B2': r['b2'], 'ext': r['ext']}
    best = max(acts, key=acts.get)
    r['oracle'] = acts[best]
    r['oracle_action'] = best

print('=' * 100)
print('逐类三档 oracle (base/B2/ext → 最优动作)')
print('=' * 100)
print('%-20s %8s %8s %8s %9s  %-10s' % ('class', 'base', 'B2', 'ext', 'oracle', 'action'))
for r in sorted(rows, key=lambda x: -x['oracle']):
    print('%-20s %8.4f %8.4f %8.4f %9.4f  %-10s' %
          (r['name'], r['base'], r['b2'], r['ext'], r['oracle'], r['oracle_action']))

# ── 汇总 (按域 + 总体) ──
print('\n' + '=' * 100)
print('域级 & 总体: oracle vs 最佳固定方案')
print('=' * 100)
print('%-10s %10s %10s %10s %10s %10s %10s' %
      ('domain', 'mean_base', 'mean_B2', 'mean_ext', 'mean_orac', 'best_fix', 'headroom'))
groups = {'ALL': FORWARD}
for g, p in DOM_PREFIX:
    groups[g] = [n for n in FORWARD if n.startswith(p)]

agg = {}
for g, names in groups.items():
    mb = np.mean([r['base'] for r in rows if r['name'] in names])
    mb2 = np.mean([r['b2'] for r in rows if r['name'] in names])
    me = np.mean([r['ext'] for r in rows if r['name'] in names])
    mo = np.mean([r['oracle'] for r in rows if r['name'] in names])
    best_fix = max(mb, mb2, me)
    head = mo - best_fix
    agg[g] = {'n': len(names), 'mean_base': mb, 'mean_B2': mb2, 'mean_ext': me,
              'mean_oracle': mo, 'best_fix': best_fix, 'headroom': head}
    print('%-10s %10.4f %10.4f %10.4f %10.4f %10.4f %+10.4f' %
          (g, mb, mb2, me, mo, best_fix, head))

# oracle 动作分布
from collections import Counter
dist = Counter(r['oracle_action'] for r in rows)
print('\noracle 动作分布 (22 前向类): %s' % dict(dist))

# ── 判据 2: 收益集中度 (相对 ext, 即最佳固定方案) ──
print('\n' + '=' * 100)
print('判据 2: 三档 oracle 相对 ext 的逐类收益 + 集中度')
print('=' * 100)
per_class_head = sorted([(r['name'], r['oracle'] - r['ext']) for r in rows],
                        key=lambda x: -x[1])
tot = sum(h for _, h in per_class_head)
top1, top2 = per_class_head[0], per_class_head[1]
top3 = per_class_head[2]
print('总收益(相对 ext, 22 类): %+.4f  → 均值 %+.4f' % (tot, tot / len(rows)))
print('Top1 %s %+.4f (%.1f%%) | Top2 %s %+.4f (%.1f%%) | Top3 %s %+.4f (%.1f%%)' %
      (top1[0], top1[1], 100 * top1[1] / tot,
       top2[0], top2[1], 100 * (top1[1] + top2[1]) / tot,
       top3[0], top3[1], 100 * (top1[1] + top2[1] + top3[1]) / tot))
wo_top2 = tot - top1[1] - top2[1]
print('去掉 Top2 后剩余收益均值: %+.4f' % (wo_top2 / len(rows)))

# ── 判据 3: VisDrone baseline−ext 配对 bootstrap ──
print('\n' + '=' * 100)
print('判据 3: VisDrone ext−base 配对 bootstrap (tge_dccs_lodo.json)')
print('=' * 100)
ci = lodo['boot_ci']['Fixed(2.0,0.5)']['visdrone']
pt = lodo['summary']['Fixed(2.0,0.5)']['per_fold']['visdrone']
cross = ci[0] <= 0 <= ci[1]
print('ext−base 点估计 %+.4f, 95%% CI [%+.4f, %+.4f] → %s'
      % (pt, ci[0], ci[1], '区间跨零, 无配对 bootstrap 支持' if cross else '区间不跨零'))
print('baseline−ext = %+.4f  → %s' % (-pt, '仅风险控制候选' if cross else '可称明确第三机制'))

# ── 综合判定 ──
print('\n' + '=' * 100)
print('综合判定')
print('=' * 100)
a_ok = agg['ALL']['headroom'] >= 0.01
conc = (top1[1] + top2[1]) / tot >= 0.5
print('判据1 平均提升: %+.4f %s 0.01' % (agg['ALL']['headroom'], '≥' if a_ok else '<'))
print('判据2 集中度:   Top2 占 %.0f%% %s 50%%' % (100 * (top1[1] + top2[1]) / tot,
                                                 '≥' if conc else '<'))
print('判据3 VisDrone: %s' % ('区间跨零 → 风险控制候选' if cross else '有 bootstrap 支持'))
verdict = 'STOP (D)' if (conc or (not a_ok) or cross) else 'PROCEED to B'
print('综合判定: %s' % verdict)

out = {'rows': [{k: r[k] for k in ['name', 'base', 'b2', 'ext', 'oracle', 'oracle_action']}
                for r in rows],
       'agg': {g: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
               for g, d in agg.items()},
       'action_dist': dict(dist),
       'per_class_head': {n: round(h, 4) for n, h in per_class_head},
       'top2_share': round((top1[1] + top2[1]) / tot, 4),
       'headroom_wo_top2': round(wo_top2 / len(rows), 4),
       'visdrone_bs': {'point': round(pt, 4), 'ci': ci, 'crosses_zero': cross},
       'verdict': verdict}
with open(DATA / 'tge_three_tier_oracle.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('\n→ tge_three_tier_oracle.json')
