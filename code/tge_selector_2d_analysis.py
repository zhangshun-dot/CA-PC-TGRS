# -*- coding: utf-8 -*-
r"""tge_selector_2d_analysis.py — 决策选择器 2D 分离度分析 (24 类 + VisDrone 4 = 28)

问题: 是否存在目标无关的二维规则, 能把"外推有害"类 (gain<0) 从"外推有益"类
(gain>0) 分开? 对比基准: 全外推 (无选择) 与 g2 gate。

输入:
  tge_proxy_selector.json       (13 类, agree/sep/gd_hi)
  tge_proxy_selector_24.json    (24 类, 同上)
  tge_proxy_selector_visdrone.json (VisDrone 4 类, 新增)
  tge_dccs_grid.json            (diff_stats: small_frac/gd_entropy/boxes_per_img/gd_mean_conf)

核心输出:
  1) 28 类合并表 (class, gain, agree, sep, small_frac, gd_hi, group)
  2) 逐信号 × 逐组 Spearman (增益可预测性)
  3) 前向域 22 类: 单一信号的 2D 线性分离评估 (sep×agree, sep×small_frac)
  4) 前向域路由准确率: 规则(外推 iff s>θ) vs 全外推 vs oracle, 最坏类损失

⚠️ 红线: 任何阈值 θ 都是事后观察, 只用于"机制发现"; 不宣称设计出已验证选择器。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
FORWARD_PREFIX = ('COCO-', 'DOTA-', 'VOC-', 'VD-')

# ── 读入 ──
p13 = json.load(open(DATA / 'tge_proxy_selector.json', encoding='utf-8'))
p24 = json.load(open(DATA / 'tge_proxy_selector_24.json', encoding='utf-8'))
pvd = json.load(open(DATA / 'tge_proxy_selector_visdrone.json', encoding='utf-8'))
dccs = json.load(open(DATA / 'tge_dccs_grid.json', encoding='utf-8'))

proxy = dict(p13['proxy'])
proxy.update(p24['proxy'])          # 24 覆盖 13
proxy.update(pvd['proxy'])          # VisDrone
gain = dict(p13['gain_map'])
gain.update(p24['gain_map'])
gain.update(pvd['gain_map'])

# 补 small_frac (来自 dccs_grid diff_stats, 覆盖 COCO/DOTA/VisDrone)
for dom_key in ['coco', 'dota', 'visdrone']:
    for name, st in dccs[dom_key]['diff_stats'].items():
        if name in proxy and 'small_frac' not in proxy[name]:
            proxy[name]['small_frac'] = st['small_frac']

rows = []
for name, g in gain.items():
    pr = proxy.get(name)
    if not pr:
        continue
    group = ('REV' if name.startswith('REV-') else
             'COCO' if name.startswith('COCO-') else
             'DOTA' if name.startswith('DOTA-') else
             'VOC' if name.startswith('VOC-') else 'VisDrone')
    rows.append({'name': name, 'gain': g, 'group': group,
                 'agree': pr.get('agree_spearman', np.nan),
                 'kendall': pr.get('agree_kendall', np.nan),
                 'sep': pr.get('sep_src', np.nan),
                 'small': pr.get('small_frac', np.nan),
                 'gd_hi': pr.get('gd_hi_frac', np.nan)})

# ── 1) 合并表 ──
print('=' * 108)
print('%-20s %9s %8s %8s %8s %8s %-9s' % ('class', 'gain', 'agree', 'sep', 'small', 'gd_hi', 'group'))
print('-' * 108)
for r in sorted(rows, key=lambda x: (x['group'], -x['gain'])):
    mark = '  ← 失效' if r['gain'] < 0 else ''
    print('%-20s %+9.4f %8.3f %8.3f %8.3f %8.3f %-9s%s' %
          (r['name'], r['gain'], r['agree'], r['sep'], r['small'], r['gd_hi'], r['group'], mark))
print('=' * 108)

# ── 2) 逐信号 × 逐组 Spearman ──
groups = ['全部28', '前向22', 'REV6', 'COCO6', 'DOTA7', 'VOC5', 'VisDrone4']
def sel(gname):
    if gname == '全部28':
        return rows
    if gname == '前向22':
        return [r for r in rows if any(r['name'].startswith(p) for p in FORWARD_PREFIX)]
    return [r for r in rows if r['group'] == gname]

print('\n逐信号 × 逐组 Spearman(gain):')
print('%-10s %10s %10s %10s %10s' % ('group', 'agree', 'sep', 'small', 'gd_hi'))
for gn in groups:
    ss = sel(gn)
    cells = []
    for key in ['agree', 'sep', 'small', 'gd_hi']:
        xs = np.array([r[key] for r in ss if not np.isnan(r[key])])
        ys = np.array([r['gain'] for r in ss if not np.isnan(r[key])])
        if len(xs) >= 4:
            rho, p = spearmanr(xs, ys)
            cells.append('%+.3f%s' % (rho, '*' if p < 0.05 else ''))
        else:
            cells.append('—')
    print('%-10s %10s %10s %10s %10s' % (gn, *cells))

# ── 3) 前向域 22 类: 单一信号二分路由 ──
print('\n' + '=' * 108)
print('前向域 22 类 (COCO6+DOTA7+VOC5+VisDrone4) — 单信号二分路由 (阈值事后扫描, 机制探索)')
print('=' * 108)
fwd = [r for r in rows if any(r['name'].startswith(p) for p in FORWARD_PREFIX)]
neg = [r for r in fwd if r['gain'] < 0]
pos = [r for r in fwd if r['gain'] >= 0]
print('失效类 %d: %s' % (len(neg), ', '.join(r['name'] for r in neg)))
print('有益类 %d: %s' % (len(pos), ', '.join(r['name'] for r in pos)))

for key in ['agree', 'sep', 'small', 'gd_hi']:
    valid = [r for r in fwd if not np.isnan(r[key])]
    if len(valid) < 6:
        continue
    xs = np.array([r[key] for r in valid])
    best = None
    for th in np.percentile(xs, np.linspace(5, 95, 19)):
        # 规则: 外推 iff x < th (对 agree/sep 与增益负相关/正相关均覆盖)
        for direction in [-1, 1]:
            pred_ext = (direction * xs) < (direction * th)
            y_ext = np.array([r['gain'] >= 0 for r in valid])
            acc = np.mean(pred_ext == y_ext)
            if best is None or acc > best[0]:
                best = (acc, th, direction)
    acc, th, direction = best
    # 用最优方向单独报告 (约定: 规则符号)
    xs = np.array([r[key] for r in fwd])
    sign = '+' if direction == 1 else '−'
    print('%-8s 最佳二分: 外推 iff %s*%s %s %.3f → 准确率 %.3f (%d/%d)'
          % (key, sign, key, '<' if direction == 1 else '>', abs(th), acc, int(acc * len(fwd)), len(fwd)))

# ── 4) 2D 线性分离 (sep×agree, sep×small) — 用所有数据点找最好的一条斜线 ──
print('\n2D 线性分离 (前向 22 类):')
def best_line(pair_a, pair_b):
    va = [r for r in fwd if not np.isnan(r[pair_a]) and not np.isnan(r[pair_b])]
    if len(va) < 8:
        return None
    A = np.array([[r[pair_a], r[pair_b], 1.0] for r in va])
    y = np.array([r['gain'] >= 0 for r in va]) * 2 - 1
    # 用带符号增益当权重: 目标是找到一个方向 w, s.t. 增益随投影单调
    # 简化: 在 (a,b) 平面上对增益做平面拟合, 找最大梯度方向投影的秩相关
    from numpy.linalg import lstsq
    sol, *_ = lstsq(np.column_stack([A[:, 0], A[:, 1], np.ones(len(A))]),
                    np.array([r['gain'] for r in va]), rcond=None)
    proj = A @ np.array([sol[0], sol[1], 0])   # 仅用 a,b 的线性组合
    rho, p = spearmanr(proj, np.array([r['gain'] for r in va]))
    return rho, p, sol, va

for pair in [('sep', 'agree'), ('sep', 'small'), ('agree', 'small')]:
    res = best_line(*pair)
    if res is None:
        print('  %-14s n 不足' % (pair[0] + '×' + pair[1]))
        continue
    rho, p, sol, va = res
    print('  %-14s 增益线性投影 ρ=%+.3f p=%.4f  (w=%s)  n=%d'
          % (pair[0] + '×' + pair[1], rho, p,
             ('[%.2f %.2f]' % (sol[0], sol[1])), len(va)))

# ── 5) 汇总 JSON ──
out = {'rows': rows,
       'groups': {gn: {'rho': None} for gn in groups},
       'fwd_neg': [r['name'] for r in neg], 'fwd_pos': [r['name'] for r in pos]}
out['groups'] = {}
for gn in groups:
    ss = sel(gn)
    out['groups'][gn] = {key: (spearmanr([r[key] for r in ss if not np.isnan(r[key])],
                                         [r['gain'] for r in ss if not np.isnan(r[key])])
                               if len([r for r in ss if not np.isnan(r[key])]) >= 4
                               else (None, None))[0]
                         for key in ['agree', 'sep', 'small', 'gd_hi']}
with open(DATA / 'tge_selector_2d_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('\n→ tge_selector_2d_analysis.json')
