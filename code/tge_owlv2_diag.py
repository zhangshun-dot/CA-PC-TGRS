# -*- coding: utf-8 -*-
r"""tge_owlv2_diag.py — OWLv2 负增益根源诊断 (双向尺度假说检验)

背景: 主表检测器谱系 GDINO-tiny +0.0646 / GDINO-base +0.0790 / OWLv2 -0.0316。
OWLv2 baseline 极高 (0.68) 但 A-CSPC 全类打负。两个假说:
  A. gd 尺度失配: OWLv2 scores 是 raw logits (0~0.7), GDINO 是概率 (0~1);
     fused = sqrt(gd*vis) 对 gd 标度敏感 → OWLv2 的 detector 强信号被 logit 低标度稀释。
  B. baseline 饱和: OWLv2 排序已近最优, 任何 vis 重打分都是净破坏 (结构问题, 修尺度也没用)。

双向诊断 (每类, β=2.0 λ=0.5, K_CALIB=10, COCO 全量, 与主表同口径):
  1. OWLv2: fused(gd_raw)        → 现状 (应复现 -0.0316)
  2. OWLv2: fused(gd_quantile→GDINO 分布) → 若显著转正 → 假说A成立 (尺度是主因)
  3. OWLv2: vis-only 排序 AP     → 若远低于 gd-only 排序 AP → vis 信号弱于 detector
  4. GDINO-tiny: fused(gd*0.3)   → 模拟低尺度, 若从 +0.0646 变负 → 反向证实尺度影响

注意: quantile-map 是诊断工具, 不是论文方法。修复方案会选干净的单调校准。
运行: python tge_owlv2_diag.py   (~15-25 分钟, CLIP 框特征编码)
输出: 实验数据/tge_owlv2_diag.json + 控制台表
"""
import sys, json, pickle
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         TARGETS, SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

BETA, LAM = 2.0, 0.5
K_CALIB = 10


def quantile_map(x_src, x_tgt):
    """x_tgt 每个值 → x_src 分布中同分位位置的值 (单调, 保序)。"""
    xs = np.sort(np.asarray(x_src, dtype=float))
    ranks = np.searchsorted(np.sort(np.asarray(x_tgt, dtype=float)),
                            np.asarray(x_tgt, dtype=float), side='right') / max(1, len(x_tgt))
    ranks = np.clip(ranks, 0.0, 1.0)
    return np.quantile(xs, ranks)


def ap50(per_img, scores_key):
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    preds = [(p['box'], p[scores_key]) for p in sel]
    gts = [p['gt'] for p in sel]
    return float(U.compute_ap50_img(preds, gts))


def build(per_img, F, W, gd_scale, use_vis=True):
    """给 per_img 每个 p 加 fused 列: fused = sqrt(gd' * vis)。gd'=gd*gd_scale (或替换)。"""
    for p in per_img:
        n = len(p['gd'])
        if n == 0:
            p['fused'] = np.zeros(0)
            continue
        gd2 = np.clip(p['gd'] * gd_scale, 1e-6, 1.0)
        vis = np.maximum(0.0, F[p['crop_idx']] @ W)
        if use_vis:
            p['fused'] = np.sqrt(gd2 * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = gd2
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

    print('[2/3] 加载缓存 (GDINO-tiny + OWLv2) ...')
    gdino_cached, owl_cached = {}, {}
    for name, coco_cls, src_cls in TARGETS:
        gp = DATA / f'tge_v2_dets_COCO-{name}.pkl'
        op = DATA / f'tge_v2_dets_OWLV2-{name}.pkl'
        if gp.exists():
            gdino_cached[name] = pickle.load(open(gp, 'rb'))
        if op.exists():
            owl_cached[name] = pickle.load(open(op, 'rb'))
    print('  GDINO-tiny 缓存 %d/6, OWLv2 缓存 %d/6' % (len(gdino_cached), len(owl_cached)))

    print('[3/3] 逐类诊断 ...')
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            continue
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - LAM * P_neg.cpu().numpy())

        r = {'name': name}
        # ── OWLv2 ──
        if name in owl_cached:
            per_img, F = collect_features(images, gts, owl_cached[name])
            r['owl_base'] = round(ap50(per_img, 'gd'), 4)
            # vis-only (gd 不参与, 仅看 vis 排序)
            build(per_img, F, W, 0.0, use_vis=True)     # fused = sqrt(eps * vis) ≈ vis 排序
            # 上面 0.0 会 clip 到 1e-6, sqrt(1e-6*vis) 仍保 vis 单调 → 等价 vis-only 排序
            r['owl_vis_only'] = round(ap50(per_img, 'fused'), 4)
            # 现状 fused (gd raw logit)
            build(per_img, F, W, 1.0, use_vis=True)
            r['owl_fused'] = round(ap50(per_img, 'fused'), 4)
            r['owl_gain'] = round(r['owl_fused'] - r['owl_base'], 4)
            # gd 分布
            all_gd = np.concatenate([p['gd'] for p in per_img])
            r['owl_gd_q'] = [round(float(np.percentile(all_gd, q)), 4) for q in (10, 50, 90)]
        # ── GDINO-tiny (供 quantile map + 低尺度对照) ──
        if name in gdino_cached:
            per_img, F = collect_features(images, gts, gdino_cached[name])
            r['gd_base'] = round(ap50(per_img, 'gd'), 4)
            all_gd = np.concatenate([p['gd'] for p in per_img])
            r['gd_gd_q'] = [round(float(np.percentile(all_gd, q)), 4) for q in (10, 50, 90)]
            r['gd_vis_only'] = round(ap50(build(per_img, F, W, 0.0, True), 'fused'), 4)
            r['gd_fused'] = round(ap50(build(per_img, F, W, 1.0, True), 'fused'), 4)
            r['gd_gain'] = round(r['gd_fused'] - r['gd_base'], 4)
            r['gd_fused_lowscale'] = round(ap50(build(per_img, F, W, 0.3, True), 'fused'), 4)
            r['gd_gain_lowscale'] = round(r['gd_fused_lowscale'] - r['gd_base'], 4)

        # ── 假说A: OWLv2 gd quantile-map 到 GDINO 分布 ──
        if name in owl_cached and name in gdino_cached:
            per_img, F = collect_features(images, gts, owl_cached[name])
            all_gd_owl = np.concatenate([p['gd'] for p in per_img])
            # 用 GDINO-tiny 同类框池 gd 分布做目标分布
            gpi, _ = collect_features(images, gts, gdino_cached[name])
            all_gd_gdi = np.concatenate([p['gd'] for p in gpi])
            gd_qm = quantile_map(all_gd_gdi, all_gd_owl)
            # 按 per_img 切回
            idx = 0
            for p in per_img:
                n = len(p['gd'])
                p['gd_qm'] = gd_qm[idx:idx + n]
                idx += n
            for p in per_img:
                n = len(p['gd'])
                vis = np.maximum(0.0, F[p['crop_idx']] @ W)
                p['fused'] = np.sqrt(np.clip(p['gd_qm'], 1e-6, 1.0) * np.clip(vis, 1e-6, 1.0))
            r['owl_fused_qm'] = round(ap50(per_img, 'fused'), 4)
            r['owl_gain_qm'] = round(r['owl_fused_qm'] - r['owl_base'], 4)

        rows[name] = r
        o = r.get('owl_gain', 0)
        g = r.get('gd_gain', 0)
        qm = r.get('owl_gain_qm', 0)
        print('  %-16s | OWL base=%.3f gain=%+.4f vis=%.3f qm=%+.4f | '
              'GD base=%.3f gain=%+.4f lowsc=%+.4f' %
              (name, r.get('owl_base', 0), o, r.get('owl_vis_only', 0), qm,
               r.get('gd_base', 0), g, r.get('gd_gain_lowscale', 0)))

    def mean(k):
        vals = [r[k] for r in rows.values() if k in r]
        return round(float(np.mean(vals)), 4) if vals else None

    print('=' * 100)
    print('%-18s %8s %8s %8s %8s | %8s %8s %8s %8s' %
          ('class', 'owl_base', 'owl_gain', 'owl_vis', 'owl_qm', 'gd_base', 'gd_gain', 'gd_low', 'gd_vis'))
    for r in rows.values():
        print('%-18s %8.4f %+8.4f %8.4f %+8.4f | %8.4f %+8.4f %+8.4f %8.4f' %
              (r['name'], r.get('owl_base', 0), r.get('owl_gain', 0),
               r.get('owl_vis_only', 0), r.get('owl_gain_qm', 0),
               r.get('gd_base', 0), r.get('gd_gain', 0),
               r.get('gd_gain_lowscale', 0), r.get('gd_vis_only', 0)))
    print('-' * 100)
    print('%-18s %8.4f %+8.4f %8.4f %+8.4f | %8.4f %+8.4f %+8.4f %8.4f   <- mean' %
          ('mean', mean('owl_base'), mean('owl_gain'), mean('owl_vis_only'),
           mean('owl_gain_qm'), mean('gd_base'), mean('gd_gain'),
           mean('gd_gain_lowscale'), mean('gd_vis_only')))
    print('=' * 100)
    print('解读: owl_qm 大幅转正 → 假说A(尺度失配)成立; owl_vis 远低于 owl_base → vis 信号弱于 detector')
    out = {'config': {'beta': BETA, 'lam': LAM, 'protocol': 'bidirectional gd-scale diagnosis',
                      'k_calib': K_CALIB}, 'rows': rows}
    with open(DATA / 'tge_owlv2_diag.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('→ 实验数据/tge_owlv2_diag.json   (耗时 %.1f 分钟)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
