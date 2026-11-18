# -*- coding: utf-8 -*-
"""tge_probe_gate.py — 诊断驱动选择性外推: gate 可行性分析 (19 场景类, 零重跑)

主线 (DOTA 数据修正): B2 = 稳健默认; 外推 = 判别失效 + 文本方向可靠类的选择性救援。
本脚本把"经验规律"升级为"算法"检验:
  1. 19 场景类的 baseline/B2/extrap 从已有 JSON 读取 (tge_probe_ap_st / dota / rev)
  2. 每个场景类补算两个零标签 gate 候选:
     g1 = 往返残差(β=2)   (源域可得: 源原型+文本, 不碰目标域)
     g2 = cos(Δ, D_dom)   (纯文本: 外推方向与全局域方向的一致性)
  3. 检验 g1/g2 与外推收益 (extrap−B2) 的相关性 (harbor 类应被识别)
  4. 固定规则对比: 全外推 / 全 B2 / gate 选择, 报告均值 + 最坏类损失

红线: 规则提前固定 (阈值写死), 不用目标域标签选择。
输出: 实验数据/tge_probe_gate.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_gate.json'
BETA = 2.0
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

# (场景类, 源域原型类, 源前缀, 目标文本类, 目标前缀, json 来源)
SCENES = [
    # COCO 正向 6 (tge_probe_ap_st.json)
    ('COCO-airplane', 'dior', 'airplane', SRC_TMPL, 'airplane', TGT_TMPL, 'ap_st'),
    ('COCO-boat',     'dior', 'ship',     SRC_TMPL, 'boat',     TGT_TMPL, 'ap_st'),
    ('COCO-car',      'dior', 'vehicle',  SRC_TMPL, 'car',      TGT_TMPL, 'ap_st'),
    ('COCO-truck',    'dior', 'vehicle',  SRC_TMPL, 'truck',    TGT_TMPL, 'ap_st'),
    ('COCO-bus',      'dior', 'vehicle',  SRC_TMPL, 'bus',      TGT_TMPL, 'ap_st'),
    ('COCO-motorcycle','dior','vehicle',  SRC_TMPL, 'motorcycle',TGT_TMPL,'ap_st'),
    # DOTA 线1 正向 7 (tge_probe_dota.json dior_to_dota)
    ('DOTA-plane',        'dior', 'airplane',     SRC_TMPL, 'airplane',     TGT_TMPL, 'dota1'),
    ('DOTA-ship',         'dior', 'ship',         SRC_TMPL, 'ship',         TGT_TMPL, 'dota1'),
    ('DOTA-storage-tank', 'dior', 'storage tank', SRC_TMPL, 'storage tank', TGT_TMPL, 'dota1'),
    ('DOTA-harbor',       'dior', 'harbor',       SRC_TMPL, 'harbor',       TGT_TMPL, 'dota1'),
    ('DOTA-bridge',       'dior', 'bridge',       SRC_TMPL, 'bridge',       TGT_TMPL, 'dota1'),
    ('DOTA-large-vehicle','dior', 'vehicle',      SRC_TMPL, 'vehicle',      TGT_TMPL, 'dota1'),
    ('DOTA-small-vehicle','dior', 'vehicle',      SRC_TMPL, 'vehicle',      TGT_TMPL, 'dota1'),
    # VOC 正向 5 (tge_probe_voc.json, 第二正向目标域, 2026-08-23 加)
    ('VOC-aeroplane',  'dior', 'airplane', SRC_TMPL, 'aeroplane', TGT_TMPL, 'voc'),
    ('VOC-boat',       'dior', 'ship',     SRC_TMPL, 'boat',      TGT_TMPL, 'voc'),
    ('VOC-car',        'dior', 'vehicle',  SRC_TMPL, 'car',       TGT_TMPL, 'voc'),
    ('VOC-bus',        'dior', 'vehicle',  SRC_TMPL, 'bus',       TGT_TMPL, 'voc'),
    ('VOC-motorbike',  'dior', 'vehicle',  SRC_TMPL, 'motorbike', TGT_TMPL, 'voc'),
    # 反向 DIOR 3 (tge_probe_rev.json)
    ('REV-DIOR-airplane', 'coco', 'airplane', TGT_TMPL, 'airplane', SRC_TMPL, 'rev'),
    ('REV-DIOR-ship',     'coco', 'boat',     TGT_TMPL, 'ship',     SRC_TMPL, 'rev'),
    ('REV-DIOR-vehicle',  'coco', 'vehicle',  TGT_TMPL, 'vehicle',  SRC_TMPL, 'rev'),
    # 反向 DOTA 3 (tge_probe_dota.json coco_to_dota)
    ('REV-DOTA-plane',        'coco', 'airplane', TGT_TMPL, 'plane',         SRC_TMPL, 'dota2'),
    ('REV-DOTA-ship',         'coco', 'boat',     TGT_TMPL, 'ship',          SRC_TMPL, 'dota2'),
    ('REV-DOTA-large-vehicle','coco', 'vehicle',  TGT_TMPL, 'large-vehicle', SRC_TMPL, 'dota2'),
]


def build_proto(src_domain, cls):
    if src_domain == 'dior':
        images, gts = U.load_dior(cls)
    else:
        if cls == 'vehicle':
            images, gts = [], []
            for c in ['car', 'truck', 'bus', 'motorcycle']:
                im, gt = U.load_coco(c, n=500)
                images += im
                gts += gt
        else:
            images, gts = U.load_coco(cls, n=500)
    crops = []
    for img, g in zip(images, gts):
        for box in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 > x1 + 10 and y2 > y1 + 10:
                crops.append(img.crop((x1, y1, x2, y2)))
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def main():
    # 1) 源原型 + D_dom
    print('[1/3] 源原型 + 全局域方向 ...')
    protos = {}
    for dom in ['dior', 'coco']:
        for cls in (['airplane', 'ship', 'vehicle', 'storage tank', 'harbor', 'bridge']
                    if dom == 'dior' else ['airplane', 'boat', 'vehicle']):
            protos[(dom, cls)] = build_proto(dom, cls)
            print(f'  ({dom},{cls}) ok')

    dom_src = torch.stack([T.text_avg([SRC_TMPL], c) for c in U.DIOR_ID2NAME.values()]).cpu()
    dom_tgt = torch.stack([T.text_avg([TGT_TMPL], c) for c in U.DIOR_ID2NAME.values()]).cpu()
    dom_dirs = dom_tgt - dom_src
    dom_dirs = dom_dirs / dom_dirs.norm(dim=-1, keepdim=True)
    D_dom = dom_dirs.mean(dim=0)
    D_dom = D_dom / D_dom.norm()

    # 2) 读 JSON
    print('[2/3] 读取三份结果 ...')
    ap_st = json.load(open(DATA / 'tge_probe_ap_st.json', encoding='utf-8'))['classes']
    dota = json.load(open(DATA / 'tge_probe_dota.json', encoding='utf-8'))
    rev = json.load(open(DATA / 'tge_probe_rev.json', encoding='utf-8'))['classes']
    voc = json.load(open(DATA / 'tge_probe_voc.json', encoding='utf-8'))['classes']

    # 3) 每场景: gate 变量 + 收益
    rows = []
    for name, src_dom, src_cls, src_tmpl, tgt_cls, tgt_tmpl, src_json in SCENES:
        if src_json == 'ap_st':
            r = ap_st[name]
            base, b2, ext = r['baseline'], r['B2_clip'], r['B3_joint2']
        elif src_json == 'dota1':
            r = dota['dior_to_dota'][name[len('DOTA-'):]]   # 完整类名 (含连字符)
            base, b2, ext = r['baseline'], r['B2_src_neg'], r['extrap']
        elif src_json == 'dota2':
            r = dota['coco_to_dota'][name[len('REV-DOTA-'):]]
            base, b2, ext = r['baseline'], r['B2_src_neg'], r['extrap']
        elif src_json == 'voc':
            r = voc[name]
            base, b2, ext = r['baseline'], r['b2'], r['acspc']
        else:
            r = rev[name[len('REV-'):]]   # 'REV-DIOR-x' → 'DIOR-x'
            base, b2, ext = r['baseline'], r['B2_src'], r['extrap']

        P_src = protos[(src_dom, src_cls)]
        s_xs = T.text_emb([src_tmpl.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([tgt_tmpl.format(tgt_cls)]).cpu()[0]
        Delta = d_y - s_xs
        g1 = T.roundtrip_residual(P_src, Delta, BETA)           # 往返残差
        g2 = float((Delta / (Delta.norm() + 1e-8)) @ D_dom)     # 方向一致性
        gain = ext - b2                                          # 外推相对 B2 的收益
        rows.append({'name': name, 'src': src_json, 'base': base, 'b2': b2, 'ext': ext,
                     'gain': gain, 'g1': g1, 'g2': g2})

    print(f'\n{"场景":<24s} {"base":>7s} {"B2":>7s} {"ext":>7s} {"ext−B2":>8s} {"g1残差":>7s} {"g2方向":>7s}')
    print('-' * 80)
    for r in rows:
        print(f'{r["name"]:<24s} {r["base"]:>7.4f} {r["b2"]:>7.4f} {r["ext"]:>7.4f} '
              f'{r["gain"]:>+8.4f} {r["g1"]:>7.3f} {r["g2"]:>+7.3f}')

    # 4) 相关性
    from scipy.stats import spearmanr
    g = np.array([r['gain'] for r in rows])
    for key, lab in [('g1', '往返残差'), ('g2', '方向一致性')]:
        v = np.array([r[key] for r in rows])
        rho, p = spearmanr(v, g)
        print(f'\nSpearman({lab}, ext−B2): ρ={rho:+.3f} p={p:.4f} ({"显著" if p < 0.05 else "不显著"})')

    # 5) 正式算法版: 符号规则 g2 > 0 → 外推, else B2
    #    (测试前固定, 零阈值调参; 数据检查: 正向 g2 ≥ +0.70, 反向 ≤ −0.76, 符号规则完全分离)
    sel = ['extrap' if r['g2'] > 0.0 else 'B2' for r in rows]
    mean_all_ext = float(np.mean([r['ext'] for r in rows]))
    mean_all_b2 = float(np.mean([r['b2'] for r in rows]))
    mean_gate = float(np.mean([r['ext'] if s == 'extrap' else r['b2'] for r, s in zip(rows, sel)]))
    worst_ext = min(r['ext'] - r['base'] for r in rows)
    worst_b2 = min(r['b2'] - r['base'] for r in rows)
    worst_gate = min((r['ext'] if s == 'extrap' else r['b2']) - r['base']
                     for r, s in zip(rows, sel))
    # 触发准确率: gate 决策 vs 每类实际最优 (max(ext,b2)) 的匹配
    oracle = ['extrap' if r['ext'] >= r['b2'] else 'B2' for r in rows]
    hits = sum(1 for s, o in zip(sel, oracle) if s == o)
    acc = hits / len(rows)
    print(f'\n正式规则 (测试前固定): g2 > 0 → 外推, else B2   [零阈值, 零调参]')
    print(f'{"策略":<12s} {"均值AP":>8s} {"最坏类ΔAP":>10s} {"外推类数":>8s}')
    print(f'{"全外推":<12s} {mean_all_ext:>8.4f} {worst_ext:>+10.4f} {len(rows):>8d}')
    print(f'{"全B2":<12s} {mean_all_b2:>8.4f} {worst_b2:>+10.4f} {0:>8d}')
    print(f'{"gate选择":<12s} {mean_gate:>8.4f} {worst_gate:>+10.4f} {sum(1 for s in sel if s == "extrap"):>8d}')
    print(f'触发准确率 (vs 类级最优): {acc:.2f} ({hits}/{len(rows)})')
    print(f'推理开销: gate = 1 次文本嵌入 + 1 次点积, <1ms/类, 零训练')

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'rows': rows, 'rule': 'g2 > 0 -> extrap, else B2 (pre-fixed, zero-threshold)',
                   'trigger_acc': acc, 'hits': hits, 'n': len(rows),
                   'strategy': {'all_ext': mean_all_ext, 'all_b2': mean_all_b2,
                                'gate': mean_gate, 'worst_ext': worst_ext,
                                'worst_b2': worst_b2, 'worst_gate': worst_gate}},
                  f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
