# -*- coding: utf-8 -*-
r"""autoprior_conflict_test.py — (iii)「自动标注」的结构性可行性检验

## 为什么做这个

元问：物理先验需要专家标注吗？能不能改成自动标注？

已测：`get_gate`（OBJECT_SIZE + VIEWPOINT_DIFF，专家手写）在论文自己的 22 格里
**11/22 = 0.500，低于平凡基线「全开」15/22 = 0.6818**（`gate_vs_22cells_audit.py`）。

所以问题变成：**能不能找一个不用专家的自动先验？**

## 关键结构（本脚本要验证的）

`tge_soft_routing.py` 的 22 个 (目标域, 类) 单元格，**源域全部是 DIOR**，变的是目标域
（COCO 6 / DOTA 7 / VisDrone 4 / VOC 5）。所以：

    「源域原型 + 类文本」这类**与目标域无关**的量，在同一 (源类, 目标类) 对的所有格上
    取值完全相同 —— 如果这些格的标签相反，则**任何**与目标域无关的先验（无论专家手写
    还是自动导出）都不可能正确处理它们。这不是功效不足，是**结构性不可能**。

本脚本把这个结构显式列出来，并给出「目标域无关先验」的准确率上界。

★ 标签口径（与 gate_vs_22cells_audit.py 一致）：
    gain_extB2 = curve[w=1] − curve[w=0]  (w=0 保守映射 B2，w=1 满外推)
    >0 = 外推有益（该校正）; <0 = 有害（不该校正）

输出: 实验数据/autoprior_conflict_test.json
运行: C:\python-study\python39\python.exe autoprior_conflict_test.py
"""
import json
from pathlib import Path
from collections import defaultdict

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'autoprior_conflict_test.json'

# (cell, src_cls, tgt_cls) —— 逐字取自 tge_soft_routing.py 第 41–48 / 162–191 行
CELLS = [
    ('COCO-airplane',     'airplane',     'airplane'),
    ('COCO-boat',         'ship',         'boat'),
    ('COCO-car',          'vehicle',      'car'),
    ('COCO-truck',        'vehicle',      'truck'),
    ('COCO-bus',          'vehicle',      'bus'),
    ('COCO-motorcycle',   'vehicle',      'motorcycle'),
    ('DOTA-plane',        'airplane',     'plane'),
    ('DOTA-ship',         'ship',         'ship'),
    ('DOTA-storage-tank', 'storage tank', 'storage-tank'),
    ('DOTA-harbor',       'harbor',       'harbor'),
    ('DOTA-bridge',       'bridge',       'bridge'),
    ('DOTA-large-vehicle', 'vehicle',     'large-vehicle'),
    ('DOTA-small-vehicle', 'vehicle',     'small-vehicle'),
    ('VD-car',            'vehicle',      'car'),
    ('VD-truck',          'vehicle',      'truck'),
    ('VD-bus',            'vehicle',      'bus'),
    ('VD-motor',          'vehicle',      'motor'),
    ('VOC-aeroplane',     'airplane',     'aeroplane'),
    ('VOC-boat',          'ship',         'boat'),
    ('VOC-car',           'vehicle',      'car'),
    ('VOC-bus',           'vehicle',      'bus'),
    ('VOC-motorbike',     'vehicle',      'motorbike'),
]


def main():
    sr = json.load(open(DATA / 'tge_soft_routing.json', encoding='utf-8'))['classes']
    rows = []
    for cell, s, t in CELLS:
        g = sr[cell]['gain_extB2']
        rows.append({'cell': cell, 'src_cls': s, 'tgt_cls': t,
                     'gain_extB2': round(g, 4), 'label': int(g > 0)})

    print('=' * 100)
    print('(iii) 自动标注的结构性检验：源域全部 = DIOR，只有目标域在变')
    print('=' * 100)
    print('%-22s %-14s %-14s %11s %6s' % ('cell', 'src_cls', 'tgt_cls', 'gain_extB2', '标签'))
    for r in rows:
        print('%-22s %-14s %-14s %+11.4f %6d' % (r['cell'], r['src_cls'], r['tgt_cls'],
                                                 r['gain_extB2'], r['label']))

    # ── 按 (src_cls, tgt_cls) 分组 ──
    groups = defaultdict(list)
    for r in rows:
        groups[(r['src_cls'], r['tgt_cls'])].append(r)

    print('\n' + '=' * 100)
    print('① 严格重复键（同一个 (源类, 目标类) 字符串对）—— 目标域无关量的取值逐位相同')
    print('=' * 100)
    strict = {k: v for k, v in groups.items() if len(v) > 1}
    n_strict_conflict = 0
    for k, v in sorted(strict.items()):
        labs = [x['label'] for x in v]
        ok = len(set(labs)) == 1
        print('  (%-8s, %-13s) → %s   %s' %
              (k[0], k[1], ' '.join('%s:%+0.4f(标签%d)' % (x['cell'], x['gain_extB2'], x['label'])
                                    for x in v),
               '一致' if ok else '★★ 冲突'))
        if not ok:
            n_strict_conflict += 1
    # 若某键只出现一次，加入单例
    singles = [k for k, v in groups.items() if len(v) == 1]
    print('  (严格重复键共 %d 组，其中 %d 组标签冲突；单例键 %d 个)'
          % (len(strict), n_strict_conflict, len(singles)))

    # ── 弱同义键：把目标类文本归一化后看冲突（文本近义但字符串不同）──
    SYN = {'aeroplane': 'airplane', 'plane': 'airplane',
           'motorbike': 'motorcycle', 'motor': 'motorcycle'}
    groups2 = defaultdict(list)
    for r in rows:
        groups2[(r['src_cls'], SYN.get(r['tgt_cls'], r['tgt_cls']))].append(r)
    print('\n' + '=' * 100)
    print('② 弱同义键（把 aeroplane/plane→airplane、motorbike/motor→motorcycle 归一）')
    print('=' * 100)
    n_weak_conflict = 0
    weak_cells = set()
    for k, v in sorted(groups2.items()):
        if len(v) == 1:
            continue
        labs = [x['label'] for x in v]
        ok = len(set(labs)) == 1
        print('  (%-14s, %-13s) → %s   %s' %
              (k[0], k[1], ' '.join('%s:%+0.4f(%d)' % (x['cell'], x['gain_extB2'], x['label'])
                                    for x in v), '一致' if ok else '★★ 冲突'))
        if not ok:
            n_weak_conflict += 1
            weak_cells.update(x['cell'] for x in v)

    # ── 严格上界：目标域无关先验只能是 (src_cls, tgt_cls 字符串) 的函数 ──
    # 取值逐位相同的两格 ⇒ 任何目标域无关先验给同一答案 ⇒ 标签相反时必错其一
    best = sum(max([x['label'] for x in v].count(0), [x['label'] for x in v].count(1))
               for v in groups.values())
    n = len(rows)
    n_pos = sum(r['label'] for r in rows)
    print('\n' + '=' * 100)
    print('③ 「目标域无关先验」的严格上界（同键取值逐位相同，只能给同一答案 ⇒ 取多数）')
    print('=' * 100)
    print('  22 格中正例 %d / 负例 %d' % (n_pos, n - n_pos))
    print('  严格上界 : %d/%d = %.4f' % (best, n, best / n))
    print('  平凡基线·全开 : %d/%d = %.4f' % (n_pos, n, n_pos / n))
    print('  专家手写 get_gate : 11/%d = 0.5000  ← 实测' % n)
    print('  ★ 更关键的判定：严格冲突 %d 组 ⇒ 目标域无关先验**不可能零错**（至少错 %d 格）'
          % (n_strict_conflict, n_strict_conflict))
    print('    ⇒ 这不是"专家没调好"或"换根轴就行"，而是**结构性做不到完美**。')

    json.dump({'rows': rows, 'strict_conflicts': n_strict_conflict,
               'weak_conflicts': n_weak_conflict,
               'conflict_cells': sorted(weak_cells),
               'target_invariant_upper_bound': best / n,
               'n': n, 'n_pos': int(sum(r['label'] for r in rows))},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s' % OUT)


if __name__ == '__main__':
    main()
