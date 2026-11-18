# -*- coding: utf-8 -*-
"""run_closed_loop_bh.py — #10 统计稳健性 / 多重比较校正 (Benjamini-Hochberg FDR)

从 closed_loop_cells.csv 的 27 单元逐格增益，对"各先验模式 AP > 基线(β=0)"做
逐检测器符号检验(binomial sign test)，再跨 7 模式 × 3 检测器 = 21 次比较做 BH-FDR 校正，
回答: 在控制伪发现率下，闭环增益是否仍显著（而非 21 次检验的偶然）。

用法:
  "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_bh.py
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest

CSV_IN = r'C:\Users\张三\Desktop\实验数据\closed_loop_cells.csv'
OUT = r'C:\Users\张三\Desktop\实验数据\run_closed_loop_bh.json'
DETS = ['gdino', 'owlv2', 'yolo']
MODES = ['ap_dc', 'ap_dc_x_count', 'ap_dc_x_conf', 'ap_dc_x_conf_perdet',
         'ap_dc_x_sigmoid', 'ap_dc_x_piece', 'ap_gate_only']


def bh(pvals, q=0.05):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = m - rank  # 1-based from largest
        val = min(prev, p[idx] * m / i)
        adj[idx] = val
        prev = val
    return adj


def main():
    df = pd.read_csv(CSV_IN)
    tests = []
    for det in DETS:
        sub = df[df['det'] == det]
        for mode in MODES:
            gains = (sub[mode] - sub['base']).values
            k = int((gains > 0).sum()); n = len(gains)
            bt = binomtest(k, n, 0.5, alternative='greater')
            tests.append({'det': det, 'mode': mode, 'n': n, 'k_pos': k,
                          'raw_p': float(bt.pvalue)})
    raw = np.array([t['raw_p'] for t in tests])
    adj = bh(raw, 0.05)
    for t, a in zip(tests, adj):
        t['bh_p'] = float(a)
        t['sig_fdr005'] = bool(a < 0.05)
    # 汇总：每模式跨检测器
    print('[#10 BH-FDR]  21 次比较(7 模式 × 3 检测器), q=0.05')
    print(f'  {"det":<7}{"mode":<18}{"k+/n":>6}{"raw_p":>10}{"BH_p":>10}{"sig":>6}')
    for t in tests:
        print(f'  {t["det"]:<7}{t["mode"]:<18}{t["k_pos"]}/{t["n"]:<3}{t["raw_p"]:>10.4f}'
              f'{t["bh_p"]:>10.4f}{"✓" if t["sig_fdr005"] else "":>6}')
    n_sig = sum(t['sig_fdr005'] for t in tests)
    print(f'\n  BH 校正后显著(q<0.05): {n_sig}/{len(tests)} 次')
    for mode in MODES:
        sub = [t for t in tests if t['mode'] == mode]
        print(f'    {mode:<18} 3 检测器全显著 = {all(t["sig_fdr005"] for t in sub)} '
              f'(raw_p={[round(t["raw_p"],4) for t in sub]})')
    json.dump({'q': 0.05, 'n_tests': len(tests), 'n_sig_bh': n_sig, 'tests': tests},
              open(OUT, 'w'), indent=2)
    print(f'\n→ {OUT}')


if __name__ == '__main__':
    main()
