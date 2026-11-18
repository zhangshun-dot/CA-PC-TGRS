# -*- coding: utf-8 -*-
r"""tge_dccs_lodo.py — DCCS Phase B: 留一目标域 (LODO) 系数选择裁决

对 3 个目标域 {COCO, VOC, DOTA} 做留一: 内层在 2 个训练域上选 (β,λ),
在留出的目标域上一次性评估。比较:
  Fixed(1.0, 0.25) / Fixed(2.0, 0.5)      —— 固定配置
  TrainBest (naive)                        —— 训练域归一化均值最大 (无稳健性)
  DCCS-U                                    —— 稳健效用 U = mean − α·std − γ·fail_rate
  Oracle                                    —— 留出域真实最优 (上界)
报告每折选中配置 + 留出增益, 以及 oracle gap。

域间尺度不同 (COCO ±0.06 / VOC ±0.02 / DOTA ±0.05), 选参前先做域内 z-score 归一。

运行: python tge_dccs_lodo.py   (秒级)
输出: 实验数据/tge_dccs_lodo.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, re
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
BETAS = [0.5, 1.0, 2.0, 4.0]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA, GAMMA = 0.5, 0.1   # 稳健效用超参 (可调)

KEY_RE = re.compile(r'^b([0-9.]+)_l([0-9.]+)$')


def parse_grid(grid):
    """把任意 key 格式的 grid 归一成 {(round(b,2), round(l,2)): gain}。"""
    out = {}
    for key, val in grid.items():
        m = KEY_RE.match(key)
        if m:
            b, l = round(float(m.group(1)), 2), round(float(m.group(2)), 2)
            out[(b, l)] = val
    return out


def main():
    j = json.load(open(DATA / 'tge_dccs_grid.json', encoding='utf-8'))
    domains = {}
    for name in ['coco', 'dota', 'voc', 'visdrone']:
        g = parse_grid(j[name]['grid'])
        # 补全缺失键 (若某域某配置缺失, 标记 NaN)
        full = {}
        for b in BETAS:
            for l in LAMBDAS:
                full[(b, l)] = g.get((b, l), np.nan)
        domains[name] = full

    # 逐类增益 (用于类级失败率/bootstrap): {domain: {class: {(b,l): gain}}}
    per_class = {}
    for name in ['coco', 'dota', 'visdrone']:
        per_class[name] = {cls: parse_grid(j[name]['per_class'][cls]['grid'])
                           for cls in j[name]['per_class']}
    # VOC 逐类来自开发域网格
    dev = json.load(open(DATA / 'tge_dev_scan_voc.json', encoding='utf-8'))
    per_class['voc'] = {}
    for key, aps in dev['grid'].items():
        m = KEY_RE.match(key)
        if not m:
            continue
        c = (round(float(m.group(1)), 2), round(float(m.group(2)), 2))
        for cls, ap in aps.items():
            per_class['voc'].setdefault(cls, {})[c] = round(ap - dev['baseline'][cls], 4)

    def fail_rate(c, train_domains):
        """配置 c 在训练域上的类级负增益比例。"""
        cnt, neg = 0, 0
        for d in train_domains:
            for cls, gmap in per_class[d].items():
                g = gmap.get(c, np.nan)
                if np.isfinite(g):
                    cnt += 1
                    neg += int(g < 0)
        return neg / max(1, cnt)

    configs = [(b, l) for b in BETAS for l in LAMBDAS]
    names = ['coco', 'voc', 'dota', 'visdrone']

    def mean_gain(domain, c):
        return domains[domain].get(c, np.nan)

    # 域内 z-score (对 20 配置)
    z = {}
    for d in names:
        vals = np.array([domains[d][c] for c in configs], dtype=float)
        valid = vals[np.isfinite(vals)]
        mu, sd = valid.mean(), valid.std()
        z[d] = {c: (domains[d][c] - mu) / sd if np.isfinite(domains[d][c]) else np.nan
                for c in configs}

    results = {}
    methods = ['Fixed(1.0,0.25)', 'Fixed(2.0,0.5)', 'TrainBest', 'DCCS-U', 'Oracle']
    pick = {m: {} for m in methods}
    gains = {m: {} for m in methods}

    for held in names:
        train = [d for d in names if d != held]
        oracle_c = max(configs, key=lambda c: domains[held][c])
        pick['Oracle'][held] = oracle_c
        gains['Oracle'][held] = domains[held][oracle_c]
        pick['Fixed(1.0,0.25)'][held] = (1.0, 0.25)
        gains['Fixed(1.0,0.25)'][held] = domains[held][(1.0, 0.25)]
        pick['Fixed(2.0,0.5)'][held] = (2.0, 0.5)
        gains['Fixed(2.0,0.5)'][held] = domains[held][(2.0, 0.5)]

        # TrainBest: 训练域 z 均值最大
        tb = max(configs, key=lambda c: np.nanmean([z[d][c] for d in train]))
        pick['TrainBest'][held] = tb
        gains['TrainBest'][held] = domains[held][tb]

        # DCCS-U: mean − α·std − γ·fail_rate (fail = 类级负增益比例)
        best_u, best_c = None, None
        for c in configs:
            zs = np.array([z[d][c] for d in train])
            m, s = np.nanmean(zs), np.nanstd(zs)
            u = m - ALPHA * s - GAMMA * fail_rate(c, train)
            if best_u is None or u > best_u:
                best_u, best_c = u, c
        pick['DCCS-U'][held] = best_c
        gains['DCCS-U'][held] = domains[held][best_c]

    # 汇总
    print('=' * 116)
    print('%-14s %8s %8s %8s %8s %8s %8s' %
          ('方法', 'COCO', 'VOC', 'DOTA', 'VDron', '均值', 'gap'))
    print('-' * 116)
    summary = {}
    for m in methods:
        row = [gains[m][d] for d in names]
        gap = float(np.mean([gains['Oracle'][d] - gains[m][d] for d in names]))
        summary[m] = {'per_fold': {d: round(gains[m][d], 4) for d in names},
                      'mean': round(float(np.mean(row)), 4),
                      'oracle_gap': round(gap, 4),
                      'picked': {d: pick[m][d] for d in names}}
        print('%-14s %8.4f %8.4f %8.4f %8.4f %8.4f  %8.4f' %
              (m, *row, float(np.mean(row)), gap))
    print('=' * 116)
    print('\n各方法每折选中的配置:')
    for m in methods:
        print('  %-14s %s' % (m, {d: pick[m][d] for d in names}))

    # ── 不确定性分析 ──
    rng = np.random.RandomState(42)
    R = 1000
    boot_ci, neg_frac = {}, {}
    for m in methods:
        boot_ci[m], neg_frac[m] = {}, {}
        for held in names:
            c = pick[m][held]
            vals = [per_class[held][cls].get(c, np.nan) for cls in per_class[held]]
            vals = np.array([v for v in vals if np.isfinite(v)])
            if len(vals) < 2:
                boot_ci[m][held], neg_frac[m][held] = [np.nan, np.nan], np.nan
                continue
            bs = np.array([np.mean(rng.choice(vals, size=len(vals), replace=True))
                           for _ in range(R)])
            boot_ci[m][held] = [round(float(np.percentile(bs, 2.5)), 4),
                                round(float(np.percentile(bs, 97.5)), 4)]
            neg_frac[m][held] = round(float(np.mean(vals < 0)), 4)

    print('\n每折留出增益 95% bootstrap CI (类重采样, R=1000):')
    print('%-14s %s' % ('方法', '  '.join('%s [%+.3f, %+.3f]' % (d.upper(), *boot_ci[m][d])
                                          for d in names)))
    print('\n每折留出域的负增益类比例:')
    print('%-14s %s' % ('方法', '  '.join('%s=%.2f' % (d.upper(), neg_frac[m][d]) for d in names)))
    print('\n选中配置频率 (跨 %d 折):' % len(names))
    from collections import Counter
    freq = {m: {str(tuple(v)): n for v, n in Counter(tuple(v) for v in pick[m].values()).items()}
            for m in methods}
    for m in methods:
        print('  %-14s %s' % (m, freq[m]))

    out = {'config': {'protocol': 'DCCS Phase B LODO: 4 目标域留一 (COCO/VOC/DOTA/VisDrone), '
                                  '域内 z-score 归一, α=%.2f γ=%.2f, bootstrap R=%d' % (ALPHA, GAMMA, R)},
           'summary': summary, 'boot_ci': boot_ci, 'neg_frac': neg_frac,
           'selected_freq': freq}
    with open(DATA / 'tge_dccs_lodo.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ tge_dccs_lodo.json')


if __name__ == '__main__':
    main()
