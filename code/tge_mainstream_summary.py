# -*- coding: utf-8 -*-
r"""tge_mainstream_summary.py — 主流方法全量对比汇总 (COCO val2017 全量统一协议)

读三个方法的全量结果 + A-CSPC 主表, 输出统一对比表:
  baseline / CoOp_zs / CoOp_fs / CLIP-Adapter_fs / NegPrompt_fs / A-CSPC(1.0,0.25) / A-CSPC(2.0,0.5)

前提: 先跑完
  python tge_probe_coop.py        → tge_probe_coop_full.json
  python tge_probe_adapter.py     → tge_probe_adapter_full.json
  python tge_probe_negprompt.py   → tge_probe_negprompt_full.json
A-CSPC 主表读 tge_v2_main_b2_l0.5.json (全量, β=2.0 λ=0.5)。
判定: A-CSPC(2.0,0.5) 是否在全量口径上超过各训练型方法。
"""
import json
from pathlib import Path

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
NAMES = ['COCO-airplane', 'COCO-boat', 'COCO-car', 'COCO-truck', 'COCO-bus', 'COCO-motorcycle']


def load(p):
    return json.load(open(DATA / p, encoding='utf-8')) if (DATA / p).exists() else None


def get(js, key):
    return js.get(key) if js else None


def main():
    coop = load('tge_probe_coop_full.json')
    adap = load('tge_probe_adapter_full.json')
    negp = load('tge_probe_negprompt_full.json')
    main = load('tge_v2_main_b2_l0.5.json')
    miss = [p for p, j in [('coop_full', coop), ('adapter_full', adap),
                           ('negprompt_full', negp), ('v2_main', main)] if j is None]
    if miss:
        print('❌ 缺文件: %s   (先跑对应方法)' % ', '.join(miss))
        return

    rows = {}
    for n in NAMES:
        r = {'base': None}
        c = (main['classes'] or {}).get(n, {})
        r['base'] = c.get('base50')
        r['ACSPC_20_05'] = c.get('ours50')
        if coop:
            cc = coop['classes'][n]
            r['CoOp_zs'] = cc['coop_zs']
            r['CoOp_fs'] = cc['coop_fs']
        if adap:
            ac = adap['classes'][n]
            r['Adapter_fs'] = ac['adapter_fs']
        if negp:
            nc = negp['classes'][n]
            r['NegPrompt_fs'] = nc.get('negprompt_fs', nc.get('adapter_fs'))
        rows[n] = r

    # 统一基线 (全量应为 0.3707 均值, 各方法应逐位一致)
    bases = [r['base'] for r in rows.values() if r['base'] is not None]
    print('基线一致性检查: 均值=%.4f, 各方法逐位一致=%s' % (sum(bases)/len(bases),
                                                  all(abs(b-bases[0])<1e-4 for b in bases)))

    cols = ['base', 'CoOp_zs', 'CoOp_fs', 'Adapter_fs', 'NegPrompt_fs', 'ACSPC_20_05']
    print('=' * (12 + 15 * len(cols)))
    print('%-10s' % 'class' + ''.join('%15s' % c for c in cols))
    print('-' * (12 + 15 * len(cols)))
    for n in NAMES:
        r = rows[n]
        print('%-10s' % n + ''.join('%15.4f' % (r.get(c, float('nan')) or float('nan'))
                                    for c in cols))
    print('-' * (12 + 15 * len(cols)))
    means = {c: sum(r.get(c) or 0 for r in rows.values()) / len(rows) for c in cols}
    print('%-10s' % 'mean' + ''.join('%15.4f' % means[c] for c in cols))
    print('=' * (12 + 15 * len(cols)))

    # 判定
    a = means['ACSPC_20_05']
    print('\n判定 (全量统一口径, A-CSPC(2.0,0.5) mean=%.4f):' % a)
    for m in ['CoOp_zs', 'CoOp_fs', 'Adapter_fs', 'NegPrompt_fs']:
        if m in means and means[m] is not None:
            d = a - means[m]
            print('  vs %-12s %.4f   A-CSPC %s %.4f' %
                  (m, means[m], '胜出' if d > 0 else '落后', abs(d)))


if __name__ == '__main__':
    main()
