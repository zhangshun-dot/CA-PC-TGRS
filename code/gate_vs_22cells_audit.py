# -*- coding: utf-8 -*-
r"""gate_vs_22cells_audit.py — 把冻结的类级物理门放进论文自己的 22 类框架里对账

## 背景

`A-CSPC_四条局限盘点与两张清单_2026-09-18.md` §3 指出：`get_gate()`（类级物理常识门，
`OBJECT_SIZE` + `VIEWPOINT_DIFF`，零目标域信息、预注册冻结）**从未在 22 类框架里被检验过**。
论文 §7 (iii) 自记「22 类中无法分离 7 个外推有害类，事后最优单信号 17/22」。

本脚本用 `tge_soft_routing.json` 里现成的 22 个 (目标域, 类) 单元格标签做对账，**无 GPU、纯重算**。

## 标签怎么来（可核对）

`tge_soft_routing.json` 每个类有软路由曲线 `curve[w]`，w=0 为保守映射(B2)、w=1 为满外推：
    gain_extB2 = ext(w=1) − b2(w=0)      ⇒ >0 = 外推对该格有益，应校正；<0 = 有害，不该校正
22 个格中 gain_extB2<0 的恰好 **7 个**，与论文自记的「7 个外推有害类」吻合 ⇒ 标签口径正确。

## 要回答

Q1 `get_gate` 作为**路由器**准不准？（对照两条平凡基线：全开 15/22、全关 7/22）
Q2 它能不能**分离那 7 个有害类**（recall_bad）？代价是什么（扔掉了多少有益类）？
Q3 与 `routing_posthoc` 的事后最优单信号 17/22 差多少？（注意后者阈值是在同一 22 点上拟合的）

输出: 实验数据/gate_vs_22cells_audit.json
运行: C:\python-study\python39\python.exe gate_vs_22cells_audit.py
"""
import json, sys
from pathlib import Path
from collections import OrderedDict

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'gate_vs_22cells_audit.json'

# ── 与 cross_detector_main_experiment.py 第 59 行逐字一致（预注册冻结版本）──
OBJECT_SIZE = {'airplane': 30., 'boat': 15., 'car': 4.5, 'bus': 12., 'truck': 12.,
               'motorcycle': 2., 'bicycle': 1.8, 'person': 1.7}
VIEWPOINT_DIFF = {'airplane': .3, 'boat': .4, 'car': .7, 'bus': .8, 'truck': .75,
                  'motorcycle': .6, 'bicycle': .5, 'person': .9}


def get_gate(cls):
    return 1 if (OBJECT_SIZE.get(cls, 999) < 10.0
                 and VIEWPOINT_DIFF.get(cls, 0) > 0.5) else 0


# 22 个单元格名 → get_gate 的类键（不可映射的显式写 None，走 .get 默认 → gate=0）
CELL2GATEKEY = OrderedDict([
    ('COCO-airplane', 'airplane'), ('COCO-boat', 'boat'), ('COCO-car', 'car'),
    ('COCO-truck', 'truck'), ('COCO-bus', 'bus'), ('COCO-motorcycle', 'motorcycle'),
    ('DOTA-plane', 'airplane'), ('DOTA-ship', 'boat'), ('DOTA-storage-tank', None),
    ('DOTA-harbor', None), ('DOTA-bridge', None),
    ('DOTA-large-vehicle', None), ('DOTA-small-vehicle', None),
    ('VD-car', 'car'), ('VD-truck', 'truck'), ('VD-bus', 'bus'), ('VD-motor', 'motorcycle'),
    ('VOC-aeroplane', 'airplane'), ('VOC-boat', 'boat'), ('VOC-car', 'car'),
    ('VOC-bus', 'bus'), ('VOC-motorbike', 'motorcycle'),
])


def main():
    sr = json.load(open(DATA / 'tge_soft_routing.json', encoding='utf-8'))
    sra = json.load(open(DATA / 'tge_soft_routing_analysis.json', encoding='utf-8'))
    cl = sr['classes']
    assert set(cl) == set(CELL2GATEKEY), \
        '单元格集合不一致: %s' % (set(cl) ^ set(CELL2GATEKEY))

    rows = []
    for name in CELL2GATEKEY:
        v = cl[name]
        g = v['gain_extB2']
        key = CELL2GATEKEY[name]
        rows.append({'cell': name, 'gate_key': key, 'gate': get_gate(key) if key else 0,
                     'gain_extB2': round(g, 4), 'label_helpful': int(g > 0),
                     'base': v['curve']['base'], 'b2': v['b2'], 'ext': v['ext']})

    n = len(rows)
    n_help = sum(r['label_helpful'] for r in rows)
    n_harm = n - n_help
    correct = sum(1 for r in rows if r['gate'] == r['label_helpful'])
    opens = [r for r in rows if r['gate'] == 1]
    tp = sum(1 for r in opens if r['label_helpful'] == 1)
    fp = sum(1 for r in opens if r['label_helpful'] == 0)

    print('=' * 96)
    print('类级物理门 get_gate  vs  论文自己的 22 个 (目标域, 类) 单元格')
    print('=' * 96)
    print('标签口径: gain_extB2 = curve[w=1] − curve[w=0]  (w=0 保守映射 B2，w=1 满外推)')
    print('有益 %d 格 / 有害 %d 格  (n=%d)' % (n_help, n_harm, n))
    print()
    print('%-22s %-11s %5s %11s %7s' % ('cell', 'gate_key', 'gate', 'gain_extB2', 'label'))
    for r in rows:
        print('%-22s %-11s %5d %+11.4f %7d%s' % (
            r['cell'], r['gate_key'] or '—', r['gate'], r['gain_extB2'], r['label_helpful'],
            '   <<< 误开(有害却开门)' if (r['gate'] == 1 and r['label_helpful'] == 0) else ''))

    print()
    print('── Q1 作为「路由器」的准确率 ──')
    print('  get_gate                : %2d/%d = %.4f' % (correct, n, correct / n))
    print('  平凡基线·全开(全外推)   : %2d/%d = %.4f' % (n_help, n, n_help / n))
    print('  平凡基线·全关(全保守)   : %2d/%d = %.4f' % (n_harm, n, n_harm / n))
    print('  ⇒ get_gate 相对全开基线 %+d 格 (%.4f)' % (correct - n_help, correct / n - n_help / n))

    print()
    print('── Q2 对「有害类」的分离能力与代价 ──')
    rej_harm = sum(1 for r in rows if r['gate'] == 0 and r['label_helpful'] == 0)
    rej_help = sum(1 for r in rows if r['gate'] == 0 and r['label_helpful'] == 1)
    print('  recall_bad  = 有害类中被关门: %d/%d = %.4f' % (rej_harm, n_harm, rej_harm / n_harm))
    print('  recall_good = 有益类中被开门: %d/%d = %.4f' % (tp, n_help, tp / n_help))
    print('  开门格数 %d, 其中有益 %d、有害 %d ⇒ 开门精度 %.4f' % (len(opens), tp, fp, tp / max(len(opens), 1)))
    print('  ★ 代价：为了拒掉 %d 个有害类，被误关的有益类是 %d 个（比拒掉的有害类还多）'
          % (rej_harm, rej_help))
    missed = [r['cell'] for r in rows if r['gate'] == 0 and r['label_helpful'] == 1]
    print('  被误关的有益格: %s' % ', '.join(missed))

    print()
    print('── Q3 与事后最优单信号对照 ──')
    rh = sra['routing_posthoc']
    print('  %-16s %8s %8s %8s' % ('signal', 'acc', 'th', 'dir'))
    for k, v in rh.items():
        print('  %-16s %8.4f %8s %8s' % (k, v['acc'], v.get('th', '-'), v.get('dir', '-')))
    best = max(rh.items(), key=lambda kv: kv[1]['acc'])
    print('  事后最优 = %s (acc=%.4f, 阈值在同一 22 点上拟合 ⇒ 乐观)'
          % (best[0], best[1]['acc']))
    print('  get_gate = %.4f（预注册、零拟合）' % (correct / n))

    print()
    print('── 覆盖缺陷 ──')
    no_prior = [r['cell'] for r in rows if r['gate_key'] is None]
    print('  get_gate 的物理先验只覆盖 8 个类键: %s' % sorted(OBJECT_SIZE))
    print('  在 22 格中有 %d 格无先验（gate 直接被 .get(...,999) 默认关掉）: %s'
          % (len(no_prior), ', '.join(no_prior)))

    json.dump({'n': n, 'n_helpful': n_help, 'n_harmful': n_harm,
               'gate_correct': correct, 'gate_acc': correct / n,
               'baseline_all_open': n_help / n, 'baseline_all_closed': n_harm / n,
               'recall_bad': rej_harm / n_harm, 'recall_good': tp / n_help,
               'n_open': len(opens), 'open_precision': tp / max(len(opens), 1),
               'wrongly_closed_helpful': missed, 'no_prior_cells': no_prior,
               'routing_posthoc': rh, 'rows': rows},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s' % OUT)


if __name__ == '__main__':
    main()
