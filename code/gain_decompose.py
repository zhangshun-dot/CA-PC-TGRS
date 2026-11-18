# -*- coding: utf-8 -*-
"""gain_decompose.py — 把 CDVM few-shot 总增益拆成可加的四段（telescoping）

协议与 final_method_table.py（final_results.json 主表）完全一致:
  L12 / K=10 / α=0.5 / NEG_LAM=0.5 / N_COCO=100（前10校准,后90测试）
  候选框 = GDINO th=0.0 → pre-filter 0.03；数据集级 11-point AP50

每类测 5 个分数（同一候选框集合, 只改打分）:
  baseline   — GDINO 原始分数
  VEP_zs     — 纯源域正原型, 无负原型            max(0, f·P_src)
  CDVM_zs    — 源域正原型 + 负原型抑制            max(0, f·P_src − λ·f·P_neg)
  VEP_fs     — α=0.5 目标域锚定正原型, 无负原型   max(0, f·P_cal)
  CDVM_fs    — 锚定正原型 + 负原型抑制            max(0, f·P_cal − λ·f·P_neg)

分解（精确可加）:
  总增益(CDVM_fs − baseline)
    = [VEP_zs − baseline]     正原型的零样本增益
    + [CDVM_zs − VEP_zs]     负原型相对化的零样本增益
    + [VEP_fs − VEP_zs]      目标域锚定（few-shot）增益, 无负原型
    + [CDVM_fs − VEP_fs]     锚定后负原型的边际增益
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
import unified_eval as U

LAYER = 11          # L12
K = 10
NEG_LAM = 0.5
ALPHA = 0.5
N_COCO = 100
OUT = r'C:\Users\张三\Desktop\实验数据\gain_decomposition.json'


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def encode_l12(crop):
    dino, pre = U.get_dinov2()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = dino(t)
    f = o.hidden_states[LAYER + 1][0, 0, :]
    return f / f.norm()


def extract_fewshot_proto(images, gts):
    feats = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(encode_l12(img.crop((x1, y1, x2, y2))))
    if not feats:
        return None
    p = torch.stack(feats).mean(dim=0)
    return p / p.norm()


def collect_feats(images, gts, dets):
    """候选框 + GDINO 分数 + L12 特征（一次编码，供所有打分变体复用）。"""
    boxes_all, gd_all, feats = [], [], []
    for (boxes, scores), img in zip(dets, images):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(encode_l12(img.crop((x1, y1, x2, y2))))
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
    gt_flat = [g for gs in gts for g in gs]
    return boxes_all, gd_all, feats, gt_flat


def ap_of(boxes_all, gd_all, feats, gt_flat, P_pos, neg=None):
    """给定正原型（可选负原型），算 AP50。neg=None → VEP 变体。"""
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    if P_pos is None:
        return U.compute_ap50(boxes_all, gd, gt_flat)  # baseline
    vis = []
    for f in feats:
        v = float((f @ P_pos).item())
        if neg is not None:
            v = v - NEG_LAM * float((f @ neg).item())
        vis.append(max(0.0, v))
    return U.compute_ap50(boxes_all, np.sqrt(gd * np.clip(vis, 1e-6, 1.0)), gt_flat)


def get_key(r, key):
    """支持 'zero.vep' 这样的点号路径递归取值。"""
    cur = r
    for k in key.split('.'):
        cur = cur[k]
    return cur


def main():
    bank = U.get_bank()
    summary = {}

    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        print('\n' + '=' * 70)
        print(f'[{name}]  {"跨域" if dataset == "coco" else "同域"}')
        if dataset == 'coco':
            images, gts = U.load_coco(cls, n=N_COCO)
        else:
            images, gts = U.load_dior(cls)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)

        if dataset == 'coco':
            test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]
            fs_imgs, fs_gts = images[:K], gts[:K]
            P_tgt = extract_fewshot_proto(fs_imgs, fs_gts)
            P_cal = None
            if P_tgt is not None:
                P_cal = ALPHA * P_src + (1 - ALPHA) * P_tgt
                P_cal = P_cal / P_cal.norm()

            bx, gd, fe, gt = collect_feats(test_imgs, test_gts, test_dets)
            base = ap_of(bx, gd, fe, gt, None)
            vep_zs = ap_of(bx, gd, fe, gt, P_src, neg=None)
            cdvm_zs = ap_of(bx, gd, fe, gt, P_src, neg=neg)
            vep_fs = ap_of(bx, gd, fe, gt, P_cal, neg=None)
            cdvm_fs = ap_of(bx, gd, fe, gt, P_cal, neg=neg)
            row = {'baseline': base,
                   'zero': {'vep': vep_zs, 'cdvm': cdvm_zs},
                   'few': {'vep': vep_fs, 'cdvm': cdvm_fs}}
            print(f'  baseline={base:.4f} | VEP_zs={vep_zs:.4f} CDVM_zs={cdvm_zs:.4f}'
                  f' | VEP_fs={vep_fs:.4f} CDVM_fs={cdvm_fs:.4f}')
        else:
            bx, gd, fe, gt = collect_feats(images, gts, dets)
            base = ap_of(bx, gd, fe, gt, None)
            vep = ap_of(bx, gd, fe, gt, P_src, neg=None)
            cdvm = ap_of(bx, gd, fe, gt, P_src, neg=neg)
            row = {'baseline': base,
                   'zero': {'vep': vep, 'cdvm': cdvm},
                   'few': {'vep': vep, 'cdvm': cdvm}}
            print(f'  baseline={base:.4f} | VEP_zs={vep:.4f} CDVM_zs={cdvm:.4f} (同域, few=zero)')

        summary[name] = row

    # ── 汇总表 ──
    cols = [('baseline', 'baseline'),
            ('zero_vep', 'zero.vep'), ('zero_cdvm', 'zero.cdvm'),
            ('few_vep', 'few.vep'), ('few_cdvm', 'few.cdvm')]
    print('\n\n' + '=' * 95)
    print(f'{"Test":<16s}' + ''.join(f'{c[0]:>12s}' for c in cols))
    print('-' * 95)
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{get_key(r, c[1]):>12.4f}' for c in cols))
    print('-' * 95)

    def mean_of(sel, key):
        return float(np.mean([get_key(summary[n], key) for n in sel]))

    means = {}
    for label, sel in [('cross', [n for n in summary if n.startswith('COCO')]),
                       ('same', [n for n in summary if n.startswith('DIOR')]),
                       ('total', list(summary))]:
        m = {'baseline': mean_of(sel, 'baseline'),
             'zero_vep': mean_of(sel, 'zero.vep'), 'zero_cdvm': mean_of(sel, 'zero.cdvm'),
             'few_vep': mean_of(sel, 'few.vep'), 'few_cdvm': mean_of(sel, 'few.cdvm')}
        means[label] = m
        print(f'{label:<16s}' + ''.join(f'{m[c[0]]:>12.4f}' for c in cols))
    # ── 精确分解（total, 可与 final_results 的 0.3503→0.4280 对账）──
    # 正确 telescoping 路径: base → VEP_zs → CDVM_zs → CDVM_fs
    #   [1] 正原型零样本  = VEP_zs − base
    #   [2] 负原型相对化  = CDVM_zs − VEP_zs
    #   [3] 测试时校准    = CDVM_fs − CDVM_zs   (负原型保持, 只换原型)
    # 附加: 纯锚定增益(VEP_fs−VEP_zs) 与 锚定后负原型边际(CDVM_fs−VEP_fs)
    #       用于检查两个机制是否正交可加 (差值≈0 → 可加)
    t = means['total']
    pos0 = t['zero_vep'] - t['baseline']
    neg0 = t['zero_cdvm'] - t['zero_vep']
    fs_stage = t['few_cdvm'] - t['zero_cdvm']
    anchor_pure = t['few_vep'] - t['zero_vep']
    neg_fs = t['few_cdvm'] - t['few_vep']
    dec = {
        'total_gain_fewshot': round(t['few_cdvm'] - t['baseline'], 4),
        'pos_proto_zero': round(pos0, 4),
        'neg_proto_zero': round(neg0, 4),
        'calib_fewshot': round(fs_stage, 4),
        'anchor_pure': round(anchor_pure, 4),
        'neg_after_anchor': round(neg_fs, 4),
        'interaction': round(anchor_pure - fs_stage, 4),
        'check_sum': round(pos0 + neg0 + fs_stage, 4),
    }
    print('\n' + '=' * 95)
    print('总增益精确分解 (total mean, 路径 base→VEP_zs→CDVM_zs→CDVM_fs):')
    print(f'  CDVM_few − baseline             = {dec["total_gain_fewshot"]:+.4f}')
    print(f'    [1] 正原型零样本 VEP_zs−base   = {dec["pos_proto_zero"]:+.4f}')
    print(f'    [2] 负原型相对化 (zs)          = {dec["neg_proto_zero"]:+.4f}')
    print(f'    [3] 测试时校准 CDVM_fs−CDVM_zs  = {dec["calib_fewshot"]:+.4f}')
    print(f'    校验和 [1]+[2]+[3]             = {dec["check_sum"]:+.4f}')
    print(f'  机制正交性检查:')
    print(f'    纯锚定增益 VEP_fs−VEP_zs       = {dec["anchor_pure"]:+.4f}')
    print(f'    锚定后负原型边际 CDVM_fs−VEP_fs = {dec["neg_after_anchor"]:+.4f}')
    print(f'    交叉项(≈0 ⇒ 两机制正交可加)     = {dec["interaction"]:+.4f}')

    out = {
        'config': {'layer': LAYER, 'k': K, 'alpha': ALPHA, 'neg_lam': NEG_LAM,
                   'n_coco': N_COCO, 'protocol': 'GDINO th=0.0 -> pre-filter 0.03 -> dataset-level AP50',
                   'note': 'same protocol as final_method_table.py (final_results.json)'},
        'per_class': summary,
        'means': means,
        'decomposition_total': dec,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n✅ 分解完成 -> {OUT}')


if __name__ == '__main__':
    main()
