# -*- coding: utf-8 -*-
r"""tge_v2_ablations.py — v2 冻结协议 §6 新增消融 (全量 COCO, β=1.0, λ=0.25)

1) 负原型池大小: top-k ∈ {1,3,5,10} 近邻类均值作 P_neg, 测 A-CSPC 增益
2) 分数分桶恢复分析: 候选框按 baseline gd 分数分 5 桶, 看 A-CSPC 在每个桶
   里多恢复多少真实目标 (TP), 定位"校正发生在低分框"的机制
3) gate 判定: 对 6 个 COCO 类算 g2 = cos(Δ, D_dom), 报告外推/保守选择
   (注: COCO 6 类全为正向 g2>0 → gate ≡ 全外推, 与 19 类混合方向集不同)

复用 tge_v2_main 的检测缓存与特征收集 (无需重新检测)。
运行: python tge_v2_ablations.py   (~20 分钟, 主要是一次 CLIP 特征编码)
输出: 实验数据/tge_v2_ablations.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, pickle, argparse
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         fused_arrays, dataset_ap, v2_cached_dets,
                         TARGETS, FROZEN_BETA, FROZEN_LAM, K_CALIB,
                         SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
TOPK_LIST = [1, 3, 5, 10]


def get_neg_proto_k(protos, pos_cls, k):
    """k 参数化的负原型: 与正原型最接近的 top-k 其他类原型均值。"""
    pos = protos[pos_cls]
    others = torch.stack([protos[c] for c in protos if c != pos_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(k).indices].mean(dim=0)
    return neg / neg.norm()


def score_bin_recovery(per_img, n_bins=5):
    """候选框按 baseline gd 分数分桶, 统计每桶真目标数及其 gd/fused 平均分。

    TP 匹配: 逐图按分数降序贪心匹配 GT (IoU≥0.5, 逐图互斥), 与 AP50 口径一致。
    目标: 展示 A-CSPC 把低分真目标提升 (低桶 fused_tp_mean > gd_tp_mean)。
    调用前须已对 per_img 写入 fused 列。
    """
    rows = []
    for p in per_img[K_CALIB:]:
        boxes, scores, gt = p['box'], p['gd'], p['gt']
        if len(boxes) == 0 or len(gt) == 0:
            continue
        fused = p['fused']
        matched = [False] * len(gt)
        order = np.argsort(-scores)
        for bi in order:
            ious = [U.iou(boxes[bi].tolist(), gb.tolist()) for gb in gt]
            j = int(np.argmax(ious))
            is_tp = 1 if ious[j] >= 0.5 and not matched[j] else 0
            if is_tp:
                matched[j] = True
            rows.append((float(scores[bi]), float(fused[bi]), is_tp))
    if not rows:
        return []
    arr = np.array(rows)
    gd_s, fu_s, tp = arr[:, 0], arr[:, 1], arr[:, 2]
    bounds = np.percentile(gd_s, np.linspace(0, 100, n_bins + 1))
    out = []
    for i in range(n_bins):
        lo, hi = bounds[i], bounds[i + 1]
        m = (gd_s >= lo) & ((gd_s <= hi) if i == n_bins - 1 else (gd_s < hi))
        if m.sum() == 0:
            continue
        tpc = tp[m]
        gd_tp = gd_s[m][tpc == 1]
        fu_tp = fu_s[m][tpc == 1]
        out.append({'bin': i, 'gd_range': [round(lo, 4), round(hi, 4)], 'n': int(m.sum()),
                    'n_tp': int(tpc.sum()), 'tp_frac': round(float(tpc.sum()) / m.sum(), 4),
                    'gd_tp_mean': round(float(gd_tp.mean()), 4) if len(gd_tp) else 0.0,
                    'fused_tp_mean': round(float(fu_tp.mean()), 4) if len(fu_tp) else 0.0,
                    'tp_score_lift': round(float(fu_tp.mean() - gd_tp.mean()), 4)
                                     if len(gd_tp) else 0.0})
    return out


def main():
    global FROZEN_BETA, FROZEN_LAM
    import time
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC v2 消融 (可指定 β/λ)')
    ap.add_argument('--beta', type=float, default=FROZEN_BETA)
    ap.add_argument('--lam', type=float, default=FROZEN_LAM)
    args = ap.parse_args()
    FROZEN_BETA, FROZEN_LAM = args.beta, args.lam
    tag = 'b%g_l%g' % (FROZEN_BETA, FROZEN_LAM)
    out_path = DATA / ('tge_v2_ablations_%s.json' % tag)
    print('配置: β=%.2f λ=%.2f  →  输出 %s' % (FROZEN_BETA, FROZEN_LAM, out_path.name))
    print('[DIOR 20 类 CLIP 源原型]')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[COCO 6 类: 特征一次编码 (复用检测缓存)]')
    class_data = []
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            continue
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        P_hat = P_src + FROZEN_BETA * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        class_data.append({'name': name, 'per_img': per_img, 'F': F, 'gts': gts,
                           'P_src': P_src, 'P_hat': P_hat, 'src_cls': src_cls,
                           'n_test': len(images) - K_CALIB})
        print('  %-16s N_test=%3d' % (name, len(images) - K_CALIB))

    # ── 1) 负原型池大小 ──
    print('\n[1/3] 负原型池大小 top-k 消融 ...')
    topk_res = {}
    for k in TOPK_LIST:
        gains = {}
        for d in class_data:
            P_neg = get_neg_proto_k(protos, d['src_cls'], k)
            W = (d['P_hat'].cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
            fused_arrays(d['per_img'], d['F'], W)
            a_ours = dataset_ap(d['per_img'], 'fused')
            a_base = dataset_ap(d['per_img'], 'gd')
            gains[d['name']] = round(a_ours - a_base, 4)
        topk_res[str(k)] = {'per_class_gain': gains,
                            'mean_gain': round(float(np.mean(list(gains.values()))), 4)}
        print('  top-%d: mean=%+.4f' % (k, topk_res[str(k)]['mean_gain']))

    # ── 2) 分数分桶恢复分析 (top-3 负原型, 冻结 λ) ──
    print('\n[2/3] 分数分桶恢复分析 ...')
    bin_res = {}
    for d in class_data:
        P_neg = get_neg_proto_k(protos, d['src_cls'], 3)
        W = (d['P_hat'].cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
        fused_arrays(d['per_img'], d['F'], W)
        bin_res[d['name']] = score_bin_recovery(d['per_img'])
    # 打印每类低分桶 TP 提升示例
    for name, bins in bin_res.items():
        if bins:
            lift0 = bins[0].get('tp_score_lift', 0)
            print('  %-16s 最低分桶 TP 分数提升=%+.4f' % (name, lift0))

    # ── 3) gate 判定 ──
    print('\n[3/3] g2 gate 判定 (6 COCO 类) ...')
    # D_dom 与论文口径一致 (tge_probe_gate.py): DIOR 20 类 (d(x)−s(x)) 单位化平均
    # ⚠️ text_emb 返回 (1,512), 必须 [0] 取行, 否则 stack 成 (20,1,512)
    dom_src = torch.stack([T.text_emb([SRC_TMPL.format(c)]).cpu()[0]
                           for c in U.DIOR_ID2NAME.values()])
    dom_tgt = torch.stack([T.text_emb([TGT_TMPL.format(c)]).cpu()[0]
                           for c in U.DIOR_ID2NAME.values()])
    dom_dirs = dom_tgt - dom_src
    dom_dirs = dom_dirs / dom_dirs.norm(dim=-1, keepdim=True)
    D_dom = dom_dirs.mean(dim=0)
    D_dom = D_dom / D_dom.norm()
    gate_res = {}
    try:
        for d in class_data:
            src_cls = d['src_cls']
            coco_cls = [t[1] for t in TARGETS if t[0] == d['name']][0]
            s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
            d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
            Delta = (d_y - s_xs)
            g2 = float((Delta / (Delta.norm() + 1e-8)) @ D_dom)
            gate_res[d['name']] = {'g2': round(g2, 4), 'decision': 'extrap' if g2 > 0 else 'B2'}
            print('  %-16s g2=%+.4f -> %s' % (d['name'], g2, gate_res[d['name']]['decision']))
        n_extrap = sum(1 for v in gate_res.values() if v['decision'] == 'extrap')
        print('  → COCO 6 类中外推 %d/6 (全正向 → gate ≡ 全外推, 与 19 类混合方向集不同)' % n_extrap)
    except Exception as e:
        print('  ⚠️ gate 判定失败 (topk/分桶结果仍会写出): %s' % e)
        gate_res = {'error': str(e)}

    out = {'config': {'protocol': 'v2 frozen: β=1.0 λ=0.25, full val2017; '
                                  'neg-proto topk / score-bin recovery / g2 gate',
                      'beta': FROZEN_BETA, 'neg_lam': FROZEN_LAM},
           'topk_ablation': topk_res,
           'score_bin_recovery': bin_res,
           'gate': gate_res}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ %s   (耗时 %.1f 分钟)' % (out_path.name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
