# -*- coding: utf-8 -*-
r"""tge_alpha_scan.py — 检测器感知融合 α 扫描 (方案②可行性诊断)

背景: 诊断 (tge_owlv2_diag.json) 判定 OWLv2 负增益源于 vis 信号远弱于强 detector 排序
(owl_vis 0.18 ≪ owl_base 0.68), 而非尺度。修复方向 = 融合强度 α 随检测器感知:
  fused_α = gd^α · vis^(1-α)   (α=0.5 即现状 sqrt 几何平均)

本脚本回答三个问题 (决定方案②是否可行):
  1. 每个检测器的增益曲线 g(α) 形状: OWLv2 是否有 α<1 的正峰 (可转正)?
  2. 最优 α 是否随检测器可分: GDINO-tiny/base ≈ 小α (多用 vis), OWLv2 ≈ 大α (信 detector)?
  3. 最优 α 跨类是否稳定 (同检测器内 6 类最优 α 是否集中)?

协议: 与主表一致 (β=2.0 λ=0.5, K_CALIB=10, COCO 全量, 逐图 AP50)。
α ∈ {0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}。
检测缓存: tge_v2_dets_{COCO|GDINOB|OWLV2}-{name}.pkl (均已存在)。
运行: python tge_alpha_scan.py   (~15-30 分钟, CLIP 框特征编码)
输出: 实验数据/tge_alpha_scan.json + 控制台表
"""
import sys, json, pickle
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         TARGETS, SRC_TMPL, TGT_TMPL, DATA)

BETA, LAM = 2.0, 0.5
K_CALIB = 10
ALPHAS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DETECTORS = [('GDINO-tiny', 'COCO'), ('GDINO-base', 'GDINOB'), ('OWLv2', 'OWLV2')]


def ap50(per_img, key):
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    return float(U.compute_ap50_img([(p['box'], p[key]) for p in sel],
                                    [p['gt'] for p in sel]))


def fused_power(per_img, F, W, alpha):
    for p in per_img:
        n = len(p['gd'])
        if n == 0:
            p['fused'] = np.zeros(0)
            continue
        gd = np.clip(p['gd'], 1e-6, 1.0)
        vis = np.clip(np.maximum(0.0, F[p['crop_idx']] @ W), 1e-6, 1.0)
        p['fused'] = gd ** alpha * vis ** (1.0 - alpha)
    return per_img


def main():
    import time
    t0 = time.time()
    print('[1/3] DIOR 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    print('  原型池 %d 类' % len(protos))

    # 每检测器 × 每类: 加载 + CLIP 特征
    print('[2/3] 加载缓存 + CLIP 特征 (3 检测器) ...')
    data = {}   # (det_tag, name) -> (per_img, F, W)
    for dtag, pfx in DETECTORS:
        for name, coco_cls, src_cls in TARGETS:
            p = DATA / f'tge_v2_dets_{pfx}-{name}.pkl'
            if not p.exists():
                print('  !! 缺缓存 %s' % p.name)
                continue
            images, gts = load_coco_all(coco_cls)
            if len(images) < K_CALIB + 10:
                continue
            dets = pickle.load(open(p, 'rb'))
            per_img, F = collect_features(images, gts, dets)
            P_src = protos[src_cls].to(U.DEVICE)
            P_neg = T.get_neg_proto(protos, src_cls)
            s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
            d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
            P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()
            W = (P_hat.cpu().numpy() - LAM * P_neg.cpu().numpy())
            data[(dtag, name)] = (per_img, F, W)
    print('  就绪 %d 组合' % len(data))

    print('[3/3] α 扫描 ...')
    out = {}
    print('=' * 104)
    print('%-16s |' % 'class', ' '.join('%+8s' % ('a%.2f' % a) for a in ALPHAS))
    print('-' * 104)
    for dtag, _ in DETECTORS:
        rows = {}
        best_by_cls = []
        for name, coco_cls, _ in TARGETS:
            if (dtag, name) not in data:
                continue
            per_img, F, W = data[(dtag, name)]
            base = ap50(per_img, 'gd')
            gains = {}
            for a in ALPHAS:
                fused_power(per_img, F, W, a)
                gains[str(a)] = round(ap50(per_img, 'fused') - base, 4)
            best_a = max(ALPHAS, key=lambda a: gains[str(a)])
            best_by_cls.append((name, best_a, gains[str(best_a)]))
            rows[name] = {'base': round(base, 4), 'gains': gains}
            print('%-16s |' % name,
                  ' '.join('%+8.4f' % gains[str(a)] for a in ALPHAS))
        # 域级 mean 曲线
        mean_g = {str(a): round(float(np.mean([rows[n]['gains'][str(a)] for n in rows])), 4)
                  for a in ALPHAS}
        out[dtag] = {'per_class': rows, 'mean_gain_curve': mean_g,
                     'best_alpha': float(max(ALPHAS, key=lambda a: mean_g[str(a)])),
                     'best_by_class': best_by_cls,
                     'alpha_argmax_counts': {
                         str(a): sum(1 for _, ba, _ in best_by_cls if ba == a)
                         for a in ALPHAS}}
        print('-' * 104)
        print('%-16s |' % 'MEAN',
              ' '.join('%+8.4f' % mean_g[str(a)] for a in ALPHAS), '  <- %s' % dtag)
        print('-' * 104)

    # 关键汇总: 每个 α 下三检测器 mean gain 是否全 ≥0
    print('\n关键汇总 (α → 三检测器 mean gain):')
    print('%-8s %10s %10s %10s %8s' % ('alpha', 'GDINO-tiny', 'GDINO-base', 'OWLv2', 'all>=0'))
    all_pos = []
    for a in ALPHAS:
        row = [out[d]['mean_gain_curve'][str(a)] for d, _ in DETECTORS]
        ok = all(v >= 0 for v in row)
        all_pos.append((a, ok))
        print('%-8.2f %+10.4f %+10.4f %+10.4f %8s' % (a, row[0], row[1], row[2], 'Y' if ok else ''))
    best_all = [a for a, ok in all_pos if ok]
    print('\n使三检测器均 ≥0 的 α: %s' % (best_all if best_all else '无单一 α'))
    print('每检测器最优 α: %s' % {d: out[d]['best_alpha'] for d, _ in DETECTORS})

    with open(DATA / 'tge_alpha_scan.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ 实验数据/tge_alpha_scan.json   (耗时 %.1f 分钟)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
