# -*- coding: utf-8 -*-
"""analyze_coarse_medium.py — 把 run_coarse_medium.json (18 格) 汇总成 M1 落稿表。

读 run_coarse_medium.json, 输出:
  (A) 每个 (coarse, fine) 在 sketch/cartoon 上的 full/class_only/tmpl_only/neg_full 增益与 rank/p_c
  (B) 按 coarse 类聚合的均值 (sketch / cartoon / 合并)
  (C) airplane 对照组是否≈0
  (D) 整体: full 正占比 / class_only 正占比 / tmpl 近零 / neg 负占比 / 显著(p<0.05)占比
  (E) 直接给出可贴进论文的 LaTeX 行 (均值± / 正占比)
"""
import json, os, statistics as st

OUT = r'C:\Users\张三\Desktop\实验数据\run_coarse_medium.json'

def main():
    d = json.load(open(OUT, encoding='utf-8'))
    rows = d['rows']
    assert len(rows) == 18, f"期望 18 格, 实际 {len(rows)}"

    # 分区
    by_cf = {}
    for r in rows:
        if r['medium'] == 'clean':
            continue
        by_cf.setdefault((r['coarse'], r['fine']), {})[r['medium']] = r

    print("="*100)
    print("逐 (coarse->fine) × medium: gain_full / gain_class / gain_tmpl / gain_neg | rank/N p_c")
    print("="*100)
    for (c, f), mm in sorted(by_cf.items()):
        print(f"\n[{c} -> {f}]")
        for med in ['sketch', 'cartoon']:
            r = mm.get(med)
            if not r:
                continue
            print(f"  {med:8} full={r['gain_full']:+.4f} class={r['gain_class']:+.4f} "
                  f"tmpl={r['gain_tmpl']:+.4f} neg={r['gain_neg']:+.4f} "
                  f"| rank={r['rank']}/{r['n_rand']+1} p_c={r['p_c']} "
                  f"(dc_full_mag={r['dc_full_mag']})")

    # 聚合
    def agg(pred):
        gf, gc, gt, gn, pc, rnd = [], [], [], [], [], []
        for (c, f), mm in by_cf.items():
            for med, r in mm.items():
                if not pred(c, f, med):
                    continue
                gf.append(r['gain_full']); gc.append(r['gain_class'])
                gt.append(r['gain_tmpl']); gn.append(r['gain_neg'])
                pc.append(r['p_c']); rnd.append(r['rand_gain_mean'])
        return gf, gc, gt, gn, pc, rnd

    def stat(x):
        return (st.mean(x), st.pstdev(x)) if x else (float('nan'), float('nan'))

    def line(label, gf, gc, gt, gn, rnd):
        m_full, _ = stat(gf); m_cls, _ = stat(gc); m_tmpl, _ = stat(gt)
        m_neg, _ = stat(gn); m_rnd, _ = stat(rnd)
        pf = sum(1 for v in gf if v > 0) / len(gf)
        pc_ = sum(1 for v in gc if v > 0) / len(gc)
        nt = sum(1 for v in gt if abs(v) < 0.01) / len(gt)
        nn = sum(1 for v in gn if v < 0) / len(gn)
        pr = sum(1 for v in rnd if v > 0) / len(rnd)
        print(f"  {label:34} full={m_full:+.4f}({pf:.0%}) cls={m_cls:+.4f}({pc_:.0%}) "
              f"tmpl={m_tmpl:+.4f}(~0 {nt:.0%}) neg={m_neg:+.4f}(neg {nn:.0%}) rnd={m_rnd:+.4f}({pr:.0%})")

    print("\n" + "="*100)
    print("聚合 (有类迁移分量的 5 格 = vehicle×4 + ship×1, 按媒介拆分)")
    print("="*100)
    for med in ['sketch', 'cartoon']:
        gf, gc, gt, gn, pc, rnd = agg(lambda c, f, m: m == med and c in ('vehicle', 'ship'))
        line(f"[{med}] has-class-term (n={len(gf)})", gf, gc, gt, gn, rnd)

    print("\n" + "="*100)
    print("airplane 对照 (无词表差: class_only≡0, full=tmpl_only 仅媒介分量)")
    print("="*100)
    for med in ['sketch', 'cartoon']:
        gf, gc, gt, gn, pc, rnd = agg(lambda c, f, m: m == med and c == 'airplane')
        line(f"[{med}] airplane-control (n={len(gf)})", gf, gc, gt, gn, rnd)

    # LaTeX 表 tab:decomp-rs
    print("\n" + "="*100)
    print("LaTeX: tab:decomp-rs (有类迁移分量 5 格, 按媒介)")
    print("="*100)
    rows_latex = []
    for med in ['sketch', 'cartoon']:
        gf, gc, gt, gn, pc, rnd = agg(lambda c, f, m: m == med and c in ('vehicle', 'ship'))
        m_full, _ = stat(gf); m_cls, _ = stat(gc); m_tmpl, _ = stat(gt)
        m_neg, _ = stat(gn); m_rnd, _ = stat(rnd)
        pf = sum(1 for v in gf if v > 0); pcnt = sum(1 for v in gc if v > 0)
        nt = sum(1 for v in gt if abs(v) < 0.01); nn = sum(1 for v in gn if v < 0)
        n = len(gf)
        rows_latex.append((med, m_full, pf, m_cls, pcnt, m_tmpl, nt, m_neg, nn, m_rnd, n))
    print(r"\begin{tabular}{@{}lrrrrr@{}}")
    print(r"\toprule")
    print(r"Medium & Joint & Vocab & Cond & Negated & Random \\")
    print(r"& $\Delta$AP50 (pos) & $\Delta$AP50 (pos) & $\Delta$AP50 ($\approx$0) & $\Delta$AP50 (neg) & $\Delta$AP50 (pos) \\")
    print(r"\midrule")
    for (med, m_full, pf, m_cls, pcnt, m_tmpl, nt, m_neg, nn, m_rnd, n) in rows_latex:
        print(f"{med} & {m_full:+.4f} ({pf}/{n}) & {m_cls:+.4f} ({pcnt}/{n}) & "
              f"{m_tmpl:+.4f} ({nt}/{n}) & {m_neg:+.4f} ({nn}/{n}) & {m_rnd:+.4f} ({n}/{n}) \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == '__main__':
    main()
