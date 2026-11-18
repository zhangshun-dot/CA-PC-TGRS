# -*- coding: utf-8 -*-
r"""tge_soft_routing.py — 选择器探索协议·实验 1: 软路由收益空间 + 机制定向信号

目标（预注册，见 selector_exploration_protocol.md）:
  1) oracle 混合曲线 AP50(w): w 在 B2(β=0,λ=.5) 与 ext(β=2,λ=.5) 原型空间线性混合。
     若某域存在内部峰 w*∈(0,1) 且优于两端 → 软路由有收益空间, 值得为它设计信号。
     若全部单调 → 软路由退化为硬选。
  2) 机制定向信号 (冻结): high_band_ext_agree / low_band_recovery / roundtrip_residual(g1),
     附已有 agree/sep_src/gd_hi_frac/small_frac/conf/entropy/boxes_per_img。
     判定能否把 7 个失效类从 15 个有益类中分离。

协议 (与 tge_dccs_grid 一致): COCO 全量 K=10 / DOTA N=10 k0 / VisDrone N=20 k0 / VOC N=100 k0。
零目标 GT/AP; 信号全部从目标框池 (detector 分数 + CLIP 框特征 + 原型 + 文本) 计算。

运行: python tge_soft_routing.py   (~30-40 分钟, 主要是 CLIP 框特征编码)
输出: 实验数据/tge_soft_routing.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau, entropy
import unified_eval as U
import tge_probe as T
import tge_probe_voc as VOC
import tge_v2_visdrone as VD
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         fused_arrays, v2_cached_dets, SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

BETA = 2.0
LAM = U.NEG_LAM  # 0.5
WS = [0.0, 0.25, 0.5, 0.75, 1.0]
HIGH_THR = 0.3      # 高段外推一致性: s_det > 0.3 (冻结)
LOW_PCT = 30        # 低分桶恢复: s_det < 30th percentile (冻结)
DOTA_IMG = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
DOTA_LBL = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'
DOTA7 = ['plane', 'ship', 'storage-tank', 'harbor', 'bridge', 'large-vehicle', 'small-vehicle']
DOTA_TO_DIOR = {'plane': 'airplane', 'ship': 'ship', 'storage-tank': 'storage tank',
                'harbor': 'harbor', 'bridge': 'bridge',
                'large-vehicle': 'vehicle', 'small-vehicle': 'vehicle'}
VOC_TARGETS = [('VOC-aeroplane', 'aeroplane', 'airplane'), ('VOC-boat', 'boat', 'ship'),
               ('VOC-car', 'car', 'vehicle'), ('VOC-bus', 'bus', 'vehicle'),
               ('VOC-motorbike', 'motorbike', 'vehicle')]
VIS_TARGETS = [('VD-car', 'car', 'vehicle'), ('VD-truck', 'truck', 'vehicle'),
               ('VD-bus', 'bus', 'vehicle'), ('VD-motor', 'motor', 'vehicle')]


def load_dota(dota_cls, n=10):
    quads = defaultdict(list)
    for fn in Path(DOTA_LBL).glob('*.txt'):
        img_name = fn.stem + '.png'
        if not (Path(DOTA_IMG) / img_name).exists():
            continue
        for line in open(fn):
            parts = line.strip().split()
            if len(parts) < 9 or parts[8] != dota_cls:
                continue
            try:
                q = list(map(float, parts[:8]))
                xs, ys = q[0::2], q[1::2]
                quads[img_name].append([min(xs), min(ys), max(xs), max(ys)])
            except Exception:
                continue
    from PIL import Image
    images, gts = [], []
    for img_name, boxes in quads.items():
        images.append(Image.open(Path(DOTA_IMG) / img_name).convert('RGB'))
        gts.append(boxes)
        if len(images) >= n:
            break
    return images, gts


def probe(per_img, F, P_src, P_hat):
    """冻结特征集 + 混合曲线数据。返回 (features, curve)。"""
    gds, srcs, hats, n_boxes, n_small = [], [], [], 0, 0
    for p in per_img:
        n = len(p['gd'])
        if n:
            gds.append(p['gd'])
            srcs.append(F[p['crop_idx']] @ P_src.cpu().numpy())
            hats.append(F[p['crop_idx']] @ P_hat.cpu().numpy())
            if p['box'].size:
                areas = (p['box'][:, 2] - p['box'][:, 0]) * (p['box'][:, 3] - p['box'][:, 1])
                n_small += int(np.sum(areas < 32 * 32))
        n_boxes += n
    gd = np.concatenate(gds) if gds else np.zeros(0)
    src = np.concatenate(srcs) if srcs else np.zeros(0)
    hat = np.concatenate(hats) if hats else np.zeros(0)
    n_img = max(1, len(per_img))

    feats = {}
    if len(gd) >= 10:
        feats['agree'] = round(float(spearmanr(gd, src)[0]), 4)
        feats['sep_src'] = round(float(src.std()), 4)
        feats['gd_hi_frac'] = round(float(np.mean(gd > HIGH_THR)), 4)
        feats['small_frac'] = round(n_small / max(1, n_boxes), 4)
        feats['gd_mean_conf'] = round(float(gd.mean()), 4)
        feats['gd_entropy'] = round(float(entropy(np.histogram(gd, bins=20, range=(0, 1))[0] + 1e-9)), 4)
        feats['boxes_per_img'] = round(n_boxes / n_img, 2)
        # 机制定向
        hi = gd > HIGH_THR
        feats['high_band_ext_agree'] = (round(float(spearmanr(gd[hi], hat[hi])[0]), 4)
                                        if hi.sum() >= 10 else None)
        lo = gd < np.percentile(gd, LOW_PCT)
        feats['low_band_recovery'] = (round(float(hat[lo].mean()), 4) if lo.sum() else None)
        feats['low_band_lift'] = (round(float(hat[lo].mean() - src[lo].mean()), 4)
                                  if lo.sum() else None)
        feats['n_boxes'] = int(len(gd))
    return feats


def main():
    import time
    t0 = time.time()
    print('[1/5] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    results = {}

    def eval_class(name, images, gts, dets, src_cls, tgt_cls, k_test):
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(tgt_cls)]).cpu()[0]
        Delta = d_y - s_xs
        P_hat = P_src + BETA * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        feats = probe(per_img, F, P_src, P_hat)
        feats['g1_roundtrip'] = round(float(T.roundtrip_residual(P_src.cpu(), Delta, BETA)), 4)

        sel = [per_img[i] for i in range(k_test, len(per_img))]
        a_base = U.compute_ap50_img([(p['box'], p['gd']) for p in sel], [p['gt'] for p in sel])
        W0 = (P_src.cpu().numpy() - LAM * P_neg.cpu().numpy())
        W1 = (P_hat.cpu().numpy() - LAM * P_neg.cpu().numpy())
        curve = {'base': round(float(a_base), 4)}
        for w in WS:
            W = (1 - w) * W0 + w * W1
            fused_arrays(per_img, F, W)
            a_w = U.compute_ap50_img([(p['box'], p['fused']) for p in sel], [p['gt'] for p in sel])
            curve[str(w)] = round(float(a_w), 4)
        results[name] = {'curve': curve, 'b2': curve['0.0'], 'ext': curve['1.0'],
                         'gain_extB2': round(curve['1.0'] - curve['0.0'], 4), 'feats': feats,
                         'n_img': len(images)}
        print('  %-16s base=%.4f  B2=%.4f  ext=%.4f  (ext−B2)%+.4f  '
              'hi_agree=%s lo_rec=%s' % (name, a_base, curve['0.0'], curve['1.0'],
                                         curve['1.0'] - curve['0.0'],
                                         feats.get('high_band_ext_agree'),
                                         feats.get('low_band_recovery')))

    # COCO 全量 (K=10)
    print('[2/5] COCO 全量 (K=10) ...')
    for name, coco_cls, src_cls in [('COCO-airplane', 'airplane', 'airplane'),
                                    ('COCO-boat', 'boat', 'ship'),
                                    ('COCO-car', 'car', 'vehicle'),
                                    ('COCO-truck', 'truck', 'vehicle'),
                                    ('COCO-bus', 'bus', 'vehicle'),
                                    ('COCO-motorcycle', 'motorcycle', 'vehicle')]:
        images, gts = load_coco_all(coco_cls)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        eval_class(name, images, gts, dets, src_cls, coco_cls, k_test=10)

    # DOTA N=10 (k0)
    print('[3/5] DOTA 7 (N=10, k0) ...')
    for dc in DOTA7:
        name = 'DOTA-' + dc
        images, gts = load_dota(dc)
        if not images:
            continue
        dets = U.gdino_detect(images, SRC_TMPL.format(DOTA_TO_DIOR[dc]), 0.0)
        eval_class(name, images, gts, dets, DOTA_TO_DIOR[dc], dc, k_test=0)

    # VisDrone N=20 (k0)
    print('[4/5] VisDrone 4 (N=20, k0) ...')
    for name, vis_cls, src_cls in VIS_TARGETS:
        images, gts = VD.load_visdrone(vis_cls)
        if len(images) < 10:
            continue
        dets = U.gdino_detect(images, TGT_TMPL.format(vis_cls), 0.0)
        eval_class(name, images, gts, dets, src_cls, vis_cls, k_test=0)

    # VOC N=100 (k0)
    print('[5/5] VOC 5 (N=100, k0) ...')
    for name, voc_cls, src_cls in VOC_TARGETS:
        images, gts = VOC.load_voc(voc_cls)
        if len(images) < 10:
            continue
        dets = T.cached_dets(images, TGT_TMPL.format(voc_cls), name)
        eval_class(name, images, gts, dets, src_cls, voc_cls, k_test=0)

    out = {'config': {'protocol': 'soft-routing oracle curve + mechanism signals (frozen)',
                      'beta': BETA, 'lam': LAM, 'ws': WS, 'high_thr': HIGH_THR, 'low_pct': LOW_PCT},
           'classes': results}
    with open(DATA / 'tge_soft_routing.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ tge_soft_routing.json   (耗时 %.1f 分钟)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
