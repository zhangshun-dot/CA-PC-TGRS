# -*- coding: utf-8 -*-
r"""tge_v2_perclass_ci.py — v2 表 2 逐类显著性 (COCO 全量, 主配置 β=2.0, λ=0.5)

逐类 bootstrap: 类别内测试图像有放回重采样, 每轮重算该类 baseline/A-CSPC 的
AP50 → Δ 分布, 输出逐类 mean / 95% percentile CI / 单边 p (count+1)/(R+1),
并对 6 类 p 做 Benjamini-Hochberg FDR 校正 (q 值)。
另附 6 类均值增益 bootstrap (R=1000 × 3 种子, 与 tge_v2_main 一致)。

复用 tge_v2 检测缓存, 重编码一次 CLIP 特征。
运行: python tge_v2_perclass_ci.py [--beta 2.0 --lam 0.5]   (~15 分钟)
输出: 实验数据/tge_v2_perclass_ci_b2_l0.5.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, argparse
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         fused_arrays, v2_cached_dets,
                         TARGETS, FROZEN_BETA, FROZEN_LAM, K_CALIB,
                         SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

R_BOOT = 1000
SEEDS = [42, 20260823, 7]


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(n)
    out[order] = q
    return out


def per_class_ap(sel, key):
    return U.compute_ap50_img([(p['box'], p[key]) for p in sel], [p['gt'] for p in sel])


def main():
    global FROZEN_BETA, FROZEN_LAM
    import time
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC v2 逐类显著性 bootstrap')
    ap.add_argument('--beta', type=float, default=FROZEN_BETA)
    ap.add_argument('--lam', type=float, default=FROZEN_LAM)
    args = ap.parse_args()
    FROZEN_BETA, FROZEN_LAM = args.beta, args.lam
    tag = 'b%g_l%g' % (FROZEN_BETA, FROZEN_LAM)
    out_path = DATA / ('tge_v2_perclass_ci_%s.json' % tag)
    print('配置: β=%.2f λ=%.2f  →  输出 %s' % (FROZEN_BETA, FROZEN_LAM, out_path.name))

    print('[1/3] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/3] COCO 6 类: 特征一次编码 + A-CSPC fused ...')
    class_data = []
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            continue
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        P_hat = P_src + FROZEN_BETA * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
        fused_arrays(per_img, F, W)
        class_data.append({'name': name, 'per_img': per_img})
        print('  %-16s N_test=%3d' % (name, len(images) - K_CALIB))

    print('[3/3] 逐类 bootstrap + 均值 bootstrap + BH-FDR ...')
    per_class = {}
    for d in class_data:
        n_t = len(d['per_img']) - K_CALIB
        idx_base = list(range(K_CALIB, len(d['per_img'])))
        # 逐类重采样
        gains = []
        rng = np.random.RandomState(42)
        for _ in range(R_BOOT):
            idx = rng.randint(K_CALIB, len(d['per_img']), size=n_t)
            sel = [d['per_img'][i] for i in idx]
            a_base = per_class_ap(sel, 'gd')
            a_ours = per_class_ap(sel, 'fused')
            gains.append(a_ours - a_base)
        gains = np.array(gains)
        p_single = float((np.sum(gains <= 0) + 1) / (R_BOOT + 1))
        per_class[d['name']] = {'delta_mean': float(gains.mean()),
                                'ci95': [float(np.percentile(gains, 2.5)),
                                         float(np.percentile(gains, 97.5))],
                                'p_one_sided': round(p_single, 4),
                                'frac_nonpos': float(np.mean(gains <= 0))}
        print('  %-16s Δ=%.4f CI=[%+.4f, %+.4f] p=%.4f' %
              (d['name'], gains.mean(), np.percentile(gains, 2.5),
               np.percentile(gains, 97.5), p_single))

    # BH-FDR
    names = list(per_class.keys())
    pvals = [per_class[n]['p_one_sided'] for n in names]
    qs = bh_fdr(pvals)
    for i, n in enumerate(names):
        per_class[n]['q_fdr'] = round(float(qs[i]), 4)

    # 均值增益 bootstrap (3 种子, 与主表同口径)
    mean_boot = {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        gmeans = []
        for _ in range(R_BOOT):
            dsum = 0.0
            for d in class_data:
                n_t = len(d['per_img']) - K_CALIB
                idx = rng.randint(K_CALIB, len(d['per_img']), size=n_t)
                sel = [d['per_img'][i] for i in idx]
                dsum += per_class_ap(sel, 'fused') - per_class_ap(sel, 'gd')
            gmeans.append(dsum / len(class_data))
        gmeans = np.array(gmeans)
        mean_boot[str(seed)] = {'mean': float(gmeans.mean()),
                                'ci95': [float(np.percentile(gmeans, 2.5)),
                                         float(np.percentile(gmeans, 97.5))],
                                'frac_nonpos': float(np.mean(gmeans <= 0))}

    out = {'config': {'protocol': 'v2 表2 逐类显著性: COCO 全量, β=%.2f λ=%.2f, R=%d, 按图重采样, '
                                  '单边 p=(count+1)/(R+1), BH-FDR' % (FROZEN_BETA, FROZEN_LAM, R_BOOT)},
           'per_class': per_class, 'mean_boot': mean_boot}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n逐类显著性:')
    for n in names:
        r = per_class[n]
        print('  %-16s p=%.4f q_FDR=%.4f  CI=[%+.4f, %+.4f]' %
              (n, r['p_one_sided'], r['q_fdr'], r['ci95'][0], r['ci95'][1]))
    print('均值 bootstrap (3 种子):')
    for seed in SEEDS:
        b = mean_boot[str(seed)]
        print('  seed=%d mean=%+.4f CI=[%+.4f, %+.4f] 非正比例=%.4f' %
              (seed, b['mean'], b['ci95'][0], b['ci95'][1], b['frac_nonpos']))
    print('→ %s   (耗时 %.1f 分钟)' % (out_path.name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
