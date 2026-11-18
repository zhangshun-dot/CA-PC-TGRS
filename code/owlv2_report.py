# -*- coding: utf-8 -*-
"""owlv2_report.py — 从已完成的 checkpoint JSON 出主干对照报告（不重跑实验）

读取:
  oowlv2_cdvm_12class.json  (OWLv2×CDVM, 12 类, 每类 5 分数)
  final_results_12class.json (GDINO 主表, 同协议)
打印:
  1) OWLv2 逐类 5 分数表 (baseline→CDVM_fs) + 方法增益
  2) GDINO vs OWLv2 对照表 (baseline / CDVM_fs / 主干Δ / 方法Δ×2)
  3) cross/same/total 均值 + 判定
修复了 owlv2_cdvm_eval.py 里 fmt 字符串 + ':>+12s' 的 ValueError。
"""
import json
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OWL = DATA / 'owlv2_cdvm_12class.json'
GD = DATA / 'final_results_12class.json'

EVALS_12 = [
    ('COCO-airplane', 'coco'), ('COCO-boat', 'coco'), ('COCO-car', 'coco'),
    ('COCO-truck', 'coco'), ('COCO-bus', 'coco'), ('COCO-motorcycle', 'coco'),
    ('DIOR-airplane', 'dior'), ('DIOR-ship', 'dior'), ('DIOR-vehicle', 'dior'),
    ('DIOR-harbor', 'dior'), ('DIOR-storage tank', 'dior'), ('DIOR-bridge', 'dior'),
]


def gk(r, key):
    """递归取 key ('zero.vep'); 缺失/None → None。"""
    cur = r
    for k in key.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def isnum(x):
    return x is not None and isinstance(x, (int, float, np.floating)) and x == x


def f(x, nd=4):
    """普通数字格式 (无符号)。"""
    return f'{x:.{nd}f}' if isnum(x) else '  nan '


def fs(x, nd=4):
    """带符号数字格式。"""
    return f'{x:+.{nd}f}' if isnum(x) else '   nan '


def main():
    owl = json.load(open(OWL, encoding='utf-8'))['per_class']
    gd = json.load(open(GD, encoding='utf-8'))['per_class']

    # ── 表1: OWLv2 逐类 5 分数 ──
    print('=' * 96)
    print('OWLv2 × CDVM — 12 类逐行 (协议: OWLv2 th=0 → pre-filter 0.03 → 数据集级 AP50)')
    print(f'{"Test":<16s} {"baseline":>9s} {"VEP_zs":>8s} {"CDVM_zs":>8s} {"VEP_fs":>8s} '
          f'{"CDVM_fs":>9s} {"Δ方法":>8s}')
    print('-' * 96)
    for name, dom in EVALS_12:
        r = owl.get(name)
        if not r:
            continue
        base = gk(r, 'baseline')
        fs5 = gk(r, 'few.cdvm')
        d = (fs5 - base) if isnum(fs5) and isnum(base) else None
        print(f'{name:<16s} {f(base):>9s} {f(gk(r,"zero.vep")):>8s} {f(gk(r,"zero.cdvm")):>8s} '
              f'{f(gk(r,"few.vep")):>8s} {f(fs5):>9s} {fs(d):>8s}')
    print('-' * 96)

    def mean_of(sel, key):
        vals = [gk(owl[n], key) for n, _ in sel if n in owl]
        vals = [v for v in vals if isnum(v)]
        return float(np.mean(vals)) if vals else float('nan')

    means = {}
    for label, sel in [('cross', [x for x in EVALS_12 if x[1] == 'coco']),
                       ('same', [x for x in EVALS_12 if x[1] == 'dior']),
                       ('total', EVALS_12)]:
        b, z, zc, vf, cf = (mean_of(sel, k) for k in
                            ['baseline', 'zero.vep', 'zero.cdvm', 'few.vep', 'few.cdvm'])
        means[label] = dict(baseline=b, zero_vep=z, zero_cdvm=zc, few_vep=vf, few_cdvm=cf)
        print(f'{label:<16s} {f(b):>9s} {f(z):>8s} {f(zc):>8s} {f(vf):>8s} {f(cf):>9s} {fs(cf - b):>8s}')

    # ── 表2: GDINO vs OWLv2 对照 ──
    print('\n' + '=' * 96)
    print('主干对照: GDINO-tiny vs OWLv2  (CDVM_fs = 最终方法, 主表协议)')
    print(f'{"Test":<16s} {"GD-base":>8s} {"OWL-base":>8s} {"GD-CDVM":>8s} {"OWL-CDVM":>8s} '
          f'{"主干Δ":>9s} {"Δ方法GD":>9s} {"Δ方法OWL":>9s}')
    print('-' * 96)
    for name, dom in EVALS_12:
        g, r = gd.get(name, {}), owl.get(name)
        if not r:
            continue
        gb = gk(g, 'baseline'); gf = gk(g, 'few.cdvm')
        ob = gk(r, 'baseline'); of = gk(r, 'few.cdvm')
        print(f'{name:<16s} {f(gb):>8s} {f(ob):>8s} {f(gf):>8s} {f(of):>8s} '
              f'{fs(ob - gb):>9s} {fs(gf - gb):>9s} {fs(of - ob):>9s}')
    print('-' * 96)

    def mean_g(sel, key):
        vals = [gk(gd.get(n, {}), key) for n, _ in sel if n in gd]
        vals = [v for v in vals if isnum(v)]
        return float(np.mean(vals)) if vals else float('nan')

    print(f'{"":<16s} {"GD-base":>8s} {"OWL-base":>8s} {"GD-CDVM":>8s} {"OWL-CDVM":>8s} '
          f'{"主干Δ":>9s} {"Δ方法GD":>9s} {"Δ方法OWL":>9s}')
    for label, sel in [('cross', [x for x in EVALS_12 if x[1] == 'coco']),
                       ('same', [x for x in EVALS_12 if x[1] == 'dior']),
                       ('total', EVALS_12)]:
        gb, gf = mean_g(sel, 'baseline'), mean_g(sel, 'few.cdvm')
        ob, of = means[label]['baseline'], means[label]['few_cdvm']
        print(f'{label:<16s} {f(gb):>8s} {f(ob):>8s} {f(gf):>8s} {f(of):>8s} '
              f'{fs(ob - gb):>9s} {fs(gf - gb):>9s} {fs(of - ob):>9s}')

    # ── 判定 ──
    print('\n' + '=' * 96)
    c = means['cross']; t = means['total']
    print('判定:')
    print(f'  cross-domain:  方法增益 {fs(c["few_cdvm"] - c["baseline"])} '
          f'(baseline {f(c["baseline"])} → CDVM_fs {f(c["few_cdvm"])})')
    print(f'  same-domain:   方法增益 {fs(means["same"]["few_cdvm"] - means["same"]["baseline"])} '
          f'(baseline {f(means["same"]["baseline"])} → CDVM_fs {f(means["same"]["few_cdvm"])})')
    print(f'  total:         方法增益 {fs(t["few_cdvm"] - t["baseline"])}')
    print('''
  解读 (论文 asset):
    * OWLv2 强主干下 cross-domain 为负 → 方法增益是"修复弱 zero-shot 排序",
      不是通用视觉信号注入 (排除性证据)。
    * same-domain (DIOR 大图, OWLv2 960px 漏检严重, base 仅 0.37) 仍为正 →
      方法在"弱检测器 + 域匹配"时依然有效。
    * 与 GDINO 主表 (+0.0745) 并排: 主干越弱/域越偏, 方法增益越大。''')


if __name__ == '__main__':
    main()
