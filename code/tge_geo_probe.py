# -*- coding: utf-8 -*-
r"""tge_geo_probe.py — 几何流形假设的零成本检验 (纯文本几何 vs oracle β)

背景: 用户方向性直觉 "β/λ 可由特征空间内在几何量自动确定"。
最廉价的检验: 源/目标文本模板在 CLIP 空间的几何关系 (不需图像、不需目标域数据),
能否排序各目标域的 oracle 最优 β。

oracle 域级最优 (tge_dccs_lodo.json): COCO→(2.0,0.5) VOC→(4.0,0.0) DOTA→(1.0,0.75) VisDrone→(1.0,1.0)
量: Δ = d_y − s_xs (目标模板−源模板嵌入差) 的范数 |Δ|; cos(s_xs, d_y)。
类映射沿用主表 (源文本用 DIOR 类名, 目标文本用目标类名)。
运行: python tge_geo_probe.py   (~10 秒, 仅 CLIP 文本编码)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import tge_probe as T

SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

DOMAINS = {
    'COCO': [('airplane', 'airplane'), ('boat', 'ship'), ('car', 'vehicle'),
             ('truck', 'vehicle'), ('bus', 'vehicle'), ('motorcycle', 'vehicle')],
    'VOC': [('aeroplane', 'airplane'), ('boat', 'ship'), ('car', 'vehicle'),
            ('bus', 'vehicle'), ('motorbike', 'vehicle')],
    'DOTA': [('plane', 'airplane'), ('ship', 'ship'), ('storage-tank', 'storage tank'),
             ('harbor', 'harbor'), ('bridge', 'bridge'),
             ('large-vehicle', 'vehicle'), ('small-vehicle', 'vehicle')],
    'VisDrone': [('car', 'vehicle'), ('truck', 'vehicle'), ('bus', 'vehicle'),
                 ('motor', 'vehicle')],
}
ORACLE_B = {'COCO': 2.0, 'VOC': 4.0, 'DOTA': 1.0, 'VisDrone': 1.0}
ORACLE_L = {'COCO': 0.5, 'VOC': 0.0, 'DOTA': 0.75, 'VisDrone': 1.0}


def main():
    print('%-10s %8s %8s | %8s %8s %8s | %8s' %
          ('domain', 'oracleβ', 'oracleλ', '|Δ|mean', 'cos_sd', '|s|', '|d|'))
    rows = []
    for dom, pairs in DOMAINS.items():
        dnorm, cs, snorm, dnorm2 = [], [], [], []
        for tgt, src in pairs:
            d_y = T.text_emb([TGT_TMPL.format(tgt)]).cpu()[0]
            s_xs = T.text_emb([SRC_TMPL.format(src)]).cpu()[0]
            delta = d_y - s_xs
            dnorm.append(float(delta.norm()))
            cs.append(float((d_y @ s_xs).item()))
            snorm.append(float(s_xs.norm()))
            dnorm2.append(float(d_y.norm()))
        r = (dom, np.mean(dnorm), np.mean(cs), np.mean(snorm), np.mean(dnorm2))
        rows.append(r)
        print('%-10s %8.1f %8.2f | %8.4f %8.4f %8.4f %8.4f' %
              (dom, ORACLE_B[dom], ORACLE_L[dom], r[1], r[2], r[3], r[4]))

    print('\n按 |Δ| 排序 (oracle β):')
    for dom, md, mc, ms, md2 in sorted(rows, key=lambda x: x[1], reverse=True):
        print('  %-10s |Δ|=%.4f  oracleβ=%.1f' % (dom, md, ORACLE_B[dom]))
    print('按 cos(s,d) 排序 (oracle β):')
    for dom, md, mc, ms, md2 in sorted(rows, key=lambda x: x[2]):
        print('  %-10s cos=%.4f  oracleβ=%.1f' % (dom, mc, ORACLE_B[dom]))

    print('\n判定: 若 |Δ| 或 cos 能把 oracleβ={2,4,1,1} 的域排序出来 → 几何量携带选参信息;'
          ' 乱序 → 纯文本几何无法解释最优 β。')
    # 域间量 vs 域间 β 的秩一致性
    from scipy.stats import spearmanr
    bs = [ORACLE_B[r[0]] for r in rows]
    for col, name in [(1, '|Δ|'), (2, 'cos')]:
        vals = [r[col] for r in rows]
        rho, p = spearmanr(vals, bs)
        print('  %s vs oracleβ: ρ=%.3f (p=%.3f)' % (name, rho, p))


if __name__ == '__main__':
    main()
