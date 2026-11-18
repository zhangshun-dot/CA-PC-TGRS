# -*- coding: utf-8 -*-
r"""tge_dccs_sep_route.py — 路线 B 验证: sep 难度路由能否改善 LODO

对每个留出域, 用 DCCS-U 选出的配置, 按类判"外推 or 退回 baseline":
  sep_src(cls) < τ  → 保守 (gain=0)
  sep_src(cls) ≥ τ  → 外推 (用所选配置的网格增益)
扫 τ, 对比 Fixed(2.0,0.5) / DCCS-U(全外推)。

注: VOC 无 sep 数据 (grid 未存 diff_stats), 按全外推处理。
运行: python tge_dccs_sep_route.py
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, re
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
KEY_RE = re.compile(r'^b([0-9.]+)_l([0-9.]+)$')
TAUS = [0.08, 0.10, 0.12, 0.14]


def parse_grid(grid):
    out = {}
    for key, val in grid.items():
        m = KEY_RE.match(key)
        if m:
            out[(round(float(m.group(1)), 2), round(float(m.group(2)), 2))] = val
    return out


def main():
    j = json.load(open(DATA / 'tge_dccs_grid.json', encoding='utf-8'))
    names = ['coco', 'voc', 'dota', 'visdrone']
    grid = {d: parse_grid(j[d]['grid']) for d in names}
    per_class = {}
    for d in names:
        if d == 'voc':
            dev = json.load(open(DATA / 'tge_dev_scan_voc.json', encoding='utf-8'))
            pc = {}
            for key, aps in dev['grid'].items():
                m = KEY_RE.match(key)
                if not m:
                    continue
                c = (round(float(m.group(1)), 2), round(float(m.group(2)), 2))
                for cls, ap in aps.items():
                    pc.setdefault(cls, {})[c] = round(ap - dev['baseline'][cls], 4)
            per_class['voc'] = pc
        else:
            per_class[d] = {cls: parse_grid(j[d]['per_class'][cls]['grid'])
                            for cls in j[d]['per_class']}
    sep = {d: {cls: j[d]['diff_stats'][cls]['sep_src'] for cls in j[d]['diff_stats']}
           for d in names if 'diff_stats' in j[d]}
    # DCCS-U 每折选中的配置 (与 LODO 一致: 在训练域 z-score + 稳健效用)
    # 简化: 用 LODO 输出里 DCCS-U 的 picks
    lodo = json.load(open(DATA / 'tge_dccs_lodo.json', encoding='utf-8'))
    picks = {d: tuple(lodo['summary']['DCCS-U']['picked'][d]) for d in names}

    fixed = (2.0, 0.5)
    print('=' * 100)
    print('%-10s %8s | %s' % ('留出域', 'Fixed(2,0.5)', '  '.join('τ=%.2f' % t for t in TAUS)))
    print('-' * 100)
    for held in names:
        row = [grid[held][fixed]]
        for tau in TAUS:
            tot, cnt = 0.0, 0
            for cls in per_class[held]:
                g = per_class[held][cls].get(picks[held], 0.0)
                s = sep.get(held, {}).get(cls, None)
                routed = 0.0 if (s is not None and s < tau) else g  # 保守=0 增益
                tot += routed
                cnt += 1
            row.append(tot / max(1, cnt))
        print('%-10s %+8.4f | %s' % (held.upper(), row[0],
                                     '  '.join('%+8.4f' % r for r in row[1:])))
    print('=' * 100)
    print('注: 保守 = 退回 baseline (增益 0); VOC 无 sep 数据, 全外推。')
    print('    DCCS-U 每折配置:', picks)
    # 均值对比
    print('\n各方案 4 域均值:')
    for tau in TAUS:
        means = []
        for held in names:
            tot, cnt = 0.0, 0
            for cls in per_class[held]:
                g = per_class[held][cls].get(picks[held], 0.0)
                s = sep.get(held, {}).get(cls, None)
                tot += (0.0 if (s is not None and s < tau) else g)
                cnt += 1
            means.append(tot / max(1, cnt))
        print('  τ=%.2f: %.4f' % (tau, np.mean(means)))
    print('  Fixed(2.0,0.5): %.4f' % np.mean([grid[held][fixed] for held in names]))
    print('  DCCS-U 全外推: %.4f' % np.mean(
        [np.mean([per_class[held][c].get(picks[held], 0.0) for c in per_class[held]])
         for held in names]))


if __name__ == '__main__':
    main()
