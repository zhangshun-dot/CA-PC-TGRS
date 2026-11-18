# -*- coding: utf-8 -*-
"""owlv2_cdvm_eval.py — OWLv2 × CDVM 完整对照表 (全 12 类)

背景: owlv2_probe.py 显示 OWLv2 baseline 在 COCO 跨域 3 类上平均 +0.3996
      (airplane 0.9730 / boat 0.5930 / car 0.7482 vs GDINO-tiny)。
本脚本把 CDVM 方法族整套搬到 OWLv2 主干, 跑 expand_eval 的 12 类:
  候选框来源: OWLv2 (960x960, ensemble) th=0.0
  其余协议与 gain_decompose.py 完全一致:
    L12 / K=10 / α=0.5 / NEG_LAM=0.5 / pre-filter 0.03 / 数据集级 11-point AP50
    COCO 前10校准(测试时 few-shot) 后90测试; DIOR 同域 few=zero
每类 5 分数 (同一候选框集合): baseline / VEP_zs / CDVM_zs / VEP_fs / CDVM_fs

输出: 实验数据/owlv2_cdvm_12class.json + 与 GDINO 主表(final_results_12class)对照表
断点续跑: 每类完成即写盘; 已存在的类自动跳过 (删输出文件可强制全跑)
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import time

import numpy as np
import torch

import unified_eval as U
import gain_decompose as G
import owlv2_probe as OP   # 复用 owl_detect / resolve_snapshot (探针已修好)

# ── 12 类 (与 expand_eval 一致) ──
EVALS_12 = [
    ('COCO-airplane',    'coco', 'airplane',    'natural_scene',  'a photo of',         'airplane'),
    ('COCO-boat',        'coco', 'boat',        'natural_scene',  'a photo of',         'ship'),
    ('COCO-car',         'coco', 'car',         'natural_scene',  'a photo of',         'vehicle'),
    ('COCO-truck',       'coco', 'truck',       'natural_scene',  'a photo of',         'vehicle'),
    ('COCO-bus',         'coco', 'bus',         'natural_scene',  'a photo of',         'vehicle'),
    ('COCO-motorcycle',  'coco', 'motorcycle',  'natural_scene',  'a photo of',         'vehicle'),
    ('DIOR-airplane',    'dior', 'airplane',    'remote_sensing', 'an overhead image of', 'airplane'),
    ('DIOR-ship',        'dior', 'ship',        'remote_sensing', 'an overhead image of', 'ship'),
    ('DIOR-vehicle',     'dior', 'vehicle',     'remote_sensing', 'an overhead image of', 'vehicle'),
    ('DIOR-harbor',      'dior', 'harbor',      'remote_sensing', 'an overhead image of', 'harbor'),
    ('DIOR-storage tank', 'dior', 'storage tank', 'remote_sensing', 'an overhead image of', 'storage tank'),
    ('DIOR-bridge',      'dior', 'bridge',      'remote_sensing', 'an overhead image of', 'bridge'),
]

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'owlv2_cdvm_12class.json'
GDINO_RESULTS = DATA / 'final_results_12class.json'


def fmt(x):
    return 'nan' if x != x else f'{x:.4f}'


def fmts(x):
    """带符号字符串格式（修复 fmt 字符串 + ':>+12s' 的 ValueError）。"""
    return '  nan ' if x != x else f'{x:+.4f}'


def main():
    U.EVALS = EVALS_12
    bank = U.get_bank()
    missing = [d for _, ds, _, _, _, d in EVALS_12 if ds == 'dior' and d not in bank]
    if missing:
        raise SystemExit(f'!! DIOR bank 缺类原型: {missing}')

    # 断点续跑
    done = {}
    if OUT.exists():
        try:
            prev = json.load(open(OUT, encoding='utf-8'))
            done = prev.get('per_class', {})
            if done:
                print(f'[resume] 已存在 {len(done)} 类, 跳过: {", ".join(done)}')
        except Exception as e:
            print(f'[resume] 读旧输出失败 ({e}), 重新全跑')

    summary = dict(done)
    print(f'OWLv2 × CDVM 全类评估: {len(EVALS_12)} 类 (COCO 6 跨域 + DIOR 6 同域)')
    print(f'协议: OWLv2 th=0.0 → pre-filter {U.PRE_FILTER} → 数据集级 AP50, '
          f'L12 / K={G.K} / α={G.ALPHA} / λ={G.NEG_LAM}')
    print(f'注意: OWLv2 是 960px ensemble(4 backbone), 每类比 GDINO-tiny 慢 ~10x。'
          f'建议放后台跑, 中断后重跑会从断点继续。')
    print('=' * 90)

    t0 = time.time()
    for i, (name, dataset, cls, domain, prefix, dior_cls) in enumerate(EVALS_12, 1):
        if name in summary:
            continue
        print(f'\n[{i}/{len(EVALS_12)}] {name}  ({dataset})  [累计 {time.time()-t0:.0f}s]')
        if dataset == 'coco':
            images, gts = U.load_coco(cls, n=G.N_COCO)
        else:
            images, gts = U.load_dior(cls)
        print(f'  {len(images)} 图, {sum(len(g) for g in gts)} GT')
        if len(images) < 5:
            summary[name] = {'baseline': float('nan'),
                             'zero': {'vep': float('nan'), 'cdvm': float('nan')},
                             'few': {'vep': float('nan'), 'cdvm': float('nan')}}
            continue

        t1 = time.time()
        dets = OP.owl_detect(images, f'{prefix} {cls}', 0.0)
        print(f'  OWLv2 检测 {time.time()-t1:.1f}s')

        P_src = bank[dior_cls][G.LAYER]['prototype'].to(U.DEVICE)
        neg = G.get_neg_proto(bank, dior_cls, G.LAYER)

        if dataset == 'coco':
            test_imgs, test_gts, test_dets = images[G.K:], gts[G.K:], dets[G.K:]
            fs_imgs, fs_gts = images[:G.K], gts[:G.K]
            P_tgt = G.extract_fewshot_proto(fs_imgs, fs_gts)
            P_cal = None
            if P_tgt is not None:
                P_cal = G.ALPHA * P_src + (1 - G.ALPHA) * P_tgt
                P_cal = P_cal / P_cal.norm()
            bx, gd, fe, gt = G.collect_feats(test_imgs, test_gts, test_dets)
            row = {
                'baseline': G.ap_of(bx, gd, fe, gt, None),
                'zero': {'vep': G.ap_of(bx, gd, fe, gt, P_src, neg=None),
                         'cdvm': G.ap_of(bx, gd, fe, gt, P_src, neg=neg)},
                'few': {'vep': G.ap_of(bx, gd, fe, gt, P_cal, neg=None),
                        'cdvm': G.ap_of(bx, gd, fe, gt, P_cal, neg=neg)},
            }
        else:
            bx, gd, fe, gt = G.collect_feats(images, gts, dets)
            base = G.ap_of(bx, gd, fe, gt, None)
            vep = G.ap_of(bx, gd, fe, gt, P_src, neg=None)
            cdvm = G.ap_of(bx, gd, fe, gt, P_src, neg=neg)
            row = {'baseline': base,
                   'zero': {'vep': vep, 'cdvm': cdvm},
                   'few': {'vep': vep, 'cdvm': cdvm}}

        summary[name] = row
        print(f'  baseline={fmt(row["baseline"])} | VEP_zs={fmt(row["zero"]["vep"])} '
              f'CDVM_zs={fmt(row["zero"]["cdvm"])} | VEP_fs={fmt(row["few"]["vep"])} '
              f'CDVM_fs={fmt(row["few"]["cdvm"])}')

        # 每类完成即写盘 (断点)
        def clean(v):
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if v != v:  # nan
                return None
            return v

        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump({'config': {
                'backbone': 'OWLv2-base-patch16-ensemble (960x960)',
                'layer': G.LAYER, 'k': G.K, 'alpha': G.ALPHA, 'neg_lam': G.NEG_LAM,
                'n_coco': G.N_COCO,
                'protocol': 'OWLv2 th=0.0 -> pre-filter 0.03 -> dataset-level AP50',
                'note': 'candidate boxes from OWLv2; rerank identical to gain_decompose'},
                'per_class': clean(summary)}, f, indent=2, ensure_ascii=False)
        print(f'  已写盘 ({len(summary)}/{len(EVALS_12)})')

    # ══════════ 汇总 + 与 GDINO 主表对照 ══════════
    cols = [('baseline', 'baseline'),
            ('zero_vep', 'zero.vep'), ('zero_cdvm', 'zero.cdvm'),
            ('few_vep', 'few.vep'), ('few_cdvm', 'few.cdvm')]
    print('\n\n' + '=' * 100)
    print('OWLv2 × CDVM — 12 类逐行')
    print(f'{"Test":<16s}' + ''.join(f'{c[0]:>12s}' for c in cols))
    print('-' * 100)
    for name in EVALS_12:
        r = summary.get(name)
        if r is None:
            continue
        print(f'{name:<16s}' + ''.join(f'{fmt(G.get_key(r, c[1])):>12s}' for c in cols))
    print('-' * 100)

    def mean_of(sel, key):
        vals = [G.get_key(summary[n], key) for n in sel if n in summary]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float('nan')

    means = {}
    for label, sel in [('cross', [n for n, d, *_ in EVALS_12 if d == 'coco']),
                       ('same', [n for n, d, *_ in EVALS_12 if d == 'dior']),
                       ('total', [n for n, *_ in EVALS_12])]:
        means[label] = {c[0]: mean_of(sel, c[1]) for c in cols}
        print(f'{label:<16s}' + ''.join(f'{fmt(means[label][c[0]]):>12s}' for c in cols))

    # ── 对照 GDINO 主表 (final_results_12class.json) ──
    print('\n' + '=' * 100)
    print('主干对照: GDINO-tiny vs OWLv2  (CDVM_fs = 最终方法)')
    print(f'{"Test":<16s} {"GD-base":>9s} {"OWL-base":>9s} {"GD-CDVM":>9s} {"OWL-CDVM":>9s}'
          f' {"主干Δ(base)":>12s} {"方法Δ(GD)":>11s} {"方法Δ(OWL)":>11s}')
    print('-' * 100)
    gd_res = {}
    if GDINO_RESULTS.exists():
        gd_res = json.load(open(GDINO_RESULTS, encoding='utf-8')).get('per_class', {})
    else:
        print('  (未找到 GDINO 主表, 只打印 OWLv2 自身)')
    for name, *_ in EVALS_12:
        r = summary.get(name)
        if r is None:
            continue
        g = gd_res.get(name, {})
        gd_base = g.get('baseline', float('nan'))
        gd_fs = G.get_key(g, 'few.cdvm') if g else float('nan')
        o_base = r['baseline']
        o_fs = G.get_key(r, 'few.cdvm')
        d_base = o_base - gd_base
        d_gd = gd_fs - gd_base
        d_owl = o_fs - o_base
        print(f'{name:<16s} {fmt(gd_base):>9s} {fmt(o_base):>9s} {fmt(gd_fs):>9s} {fmt(o_fs):>9s}'
              f' {fmts(d_base):>12s} {fmts(d_gd):>11s} {fmts(d_owl):>11s}')

    # 判定
    def avg(lst):
        lst = [v for v in lst if v == v]
        return float(np.mean(lst)) if lst else float('nan')

    m_base_gd = avg([gd_res.get(n, {}).get('baseline', float('nan')) for n, *_ in EVALS_12])
    m_base_owl = avg([summary[n]['baseline'] for n, *_ in EVALS_12 if n in summary])
    m_gd_fs = avg([G.get_key(gd_res[n], 'few.cdvm') for n, *_ in EVALS_12 if n in gd_res])
    m_owl_fs = avg([G.get_key(summary[n], 'few.cdvm') for n, *_ in EVALS_12 if n in summary])

    print('\n' + '=' * 100)
    print('判定 (主表协议, 12 类均值):')
    print(f'  baseline:        GDINO {fmt(m_base_gd)}  vs  OWLv2 {fmt(m_base_owl)}   Δ={fmt(m_base_owl - m_base_gd)}')
    print(f'  CDVM_fs:         GDINO {fmt(m_gd_fs)}  vs  OWLv2 {fmt(m_owl_fs)}   Δ={fmt(m_owl_fs - m_gd_fs)}')
    print(f'  方法增益(GD):    {fmt(m_gd_fs - m_base_gd)}')
    print(f'  方法增益(OWL):   {fmt(m_owl_fs - m_base_owl)}')
    d_owl = m_owl_fs - m_base_owl
    if d_owl == d_owl:
        if d_owl >= 0.01:
            print(f'  → 方法在 OWLv2 上也有增益 (+{d_owl:.4f}) → 方法主干无关性成立, 最强叙事')
        elif d_owl >= 0:
            print(f'  → 方法在 OWLv2 上持平 ({d_owl:+.4f}) → 增益被强主干吸收')
        else:
            print(f'  → 方法在 OWLv2 上为负 ({d_owl:+.4f}) → OWL 主干下方法失效, 需谨慎解读')

    # 最终落盘 (含对照)
    final = {
        'config': {
            'backbone': 'OWLv2-base-patch16-ensemble (960x960)',
            'layer': G.LAYER, 'k': G.K, 'alpha': G.ALPHA, 'neg_lam': G.NEG_LAM,
            'n_coco': G.N_COCO,
            'protocol': 'OWLv2 th=0.0 -> pre-filter 0.03 -> dataset-level AP50',
            'note': 'candidate boxes from OWLv2; rerank identical to gain_decompose'},
        'per_class': clean(summary),
        'means': means,
        'comparison_gdino': {
            'gd_baseline_total': m_base_gd, 'owl_baseline_total': m_base_owl,
            'backbone_gain': m_base_owl - m_base_gd,
            'gd_cdvm_fs_total': m_gd_fs, 'owl_cdvm_fs_total': m_owl_fs,
            'method_gain_on_gdino': m_gd_fs - m_base_gd,
            'method_gain_on_owlv2': m_owl_fs - m_base_owl},
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f'\n✅ 完成 -> {OUT}')


if __name__ == '__main__':
    main()
