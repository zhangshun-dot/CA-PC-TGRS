# -*- coding: utf-8 -*-
r"""tge_dccs_grid.py — DCCS Phase A: 多目标域 β×λ 增益网格 + 无标签难度统计

目标: 为 Phase B (LODO 严格 zero-shot 选择器) 提供数据。
对每个目标侧域 (COCO 全量 / VOC N=100 / DOTA N=10, 源=DIOR):
  1) 增益矩阵 gain(β, λ) = mean_类 [ AP(A-CSPC(β,λ)) − AP(baseline) ]
  2) 无标签难度统计: sep_src (CLIP 源原型 std), gd 均置信度, gd 熵,
     每图框数, 小目标占比

网格统一: β∈{0.5, 1, 2, 4} × λ∈{0, 0.25, 0.5, 0.75, 1} (与 tge_dev_scan_voc 一致,
VOC 直接复用其网格)。COCO 复用 v2 检测缓存; DOTA 重新检测。

运行: python tge_dccs_grid.py   (~35 分钟)
输出: 实验数据/tge_dccs_grid.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from scipy.stats import entropy
import unified_eval as U
import tge_probe as T
import tge_probe_voc as VOC
import tge_v2_visdrone as VD
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         v2_cached_dets, TARGETS, SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

DOTA_IMG = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
DOTA_LBL = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'
BETAS = [0.5, 1.0, 2.0, 4.0]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
K_CALIB = 10


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


def diff_stats(per_img, F, P_src):
    """无标签难度统计 (只用候选框 + 源原型, 零目标 GT)。"""
    gds, srcs, n_boxes = [], [], 0
    n_small = 0
    for p in per_img:
        n = len(p['gd'])
        n_boxes += n
        if n:
            gds.append(p['gd'])
            srcs.append(F[p['crop_idx']] @ P_src.cpu().numpy())
            if p['box'].size:
                areas = (p['box'][:, 2] - p['box'][:, 0]) * (p['box'][:, 3] - p['box'][:, 1])
                n_small += int(np.sum(areas < 32 * 32))
    gd = np.concatenate(gds) if gds else np.zeros(0)
    src = np.concatenate(srcs) if srcs else np.zeros(0)
    n_img = max(1, len(per_img))
    return {'sep_src': round(float(src.std()), 4) if len(src) else 0.0,
            'gd_mean_conf': round(float(gd.mean()), 4) if len(gd) else 0.0,
            'gd_entropy': round(float(entropy(np.histogram(gd, bins=20, range=(0, 1))[0] + 1e-9)), 4)
                          if len(gd) else 0.0,
            'boxes_per_img': round(n_boxes / n_img, 2),
            'small_frac': round(n_small / max(1, n_boxes), 4)}


def main():
    import time
    t0 = time.time()
    print('[1/4] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    def eval_grid(images, gts, dets, name, coco_cls, src_cls, k_test=K_CALIB):
        """单个目标类的 β×λ 增益。k_test: 评估起始图 (COCO 剔除 K 校准图, DOTA N=10 用 0)。"""
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        from tge_v2_main import fused_arrays

        def ap50(key):
            sel = [per_img[i] for i in range(k_test, len(per_img))]
            return U.compute_ap50_img([(p['box'], p[key]) for p in sel], [p['gt'] for p in sel])

        a_base = ap50('gd')
        gains = {}
        for b in BETAS:
            P_hat = P_src + b * Delta.to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()
            for lam in LAMBDAS:
                W = (P_hat.cpu().numpy() - lam * P_neg.cpu().numpy())
                fused_arrays(per_img, F, W)
                gains[f'b%g_l%g' % (b, lam)] = round(ap50('fused') - a_base, 4)
        return round(a_base, 4), gains, per_img, F, P_src

    # ── COCO 全量 (DIOR 源) ──
    print('[2/4] COCO 全量 (DIOR 源) β×λ 网格 ...')
    coco_grid, coco_stats, coco_class_gain = {}, {}, {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        a_base, gains, per_img, F, P_src = eval_grid(images, gts, dets, name, coco_cls, src_cls)
        coco_class_gain[name] = {'base': a_base, 'grid': gains}
        coco_stats[name] = diff_stats(per_img, F, P_src)
        print('  %-16s base=%.4f' % (name, a_base))
    coco_grid = {}
    for b in BETAS:
        for lam in LAMBDAS:
            key = 'b%g_l%g' % (b, lam)
            coco_grid[key] = round(float(np.mean(
                [coco_class_gain[n]['grid'][key] for n in coco_class_gain])), 4)

    # ── DOTA (DIOR 源, N=10/类) ──
    print('[3/4] DOTA (DIOR 源) β×λ 网格 ...')
    dota_targets = [('DOTA-plane', 'plane', 'airplane'), ('DOTA-ship', 'ship', 'ship'),
                    ('DOTA-storage-tank', 'storage-tank', 'storage tank'),
                    ('DOTA-harbor', 'harbor', 'harbor'), ('DOTA-bridge', 'bridge', 'bridge'),
                    ('DOTA-large-vehicle', 'large-vehicle', 'vehicle'),
                    ('DOTA-small-vehicle', 'small-vehicle', 'vehicle')]
    dota_class_gain, dota_stats = {}, {}
    for name, dota_cls, dior_cls in dota_targets:
        images, gts = load_dota(dota_cls)
        dets = U.gdino_detect(images, SRC_TMPL.format(dior_cls), 0.0)
        a_base, gains, per_img, F, P_src = eval_grid(images, gts, dets, name, dota_cls, dior_cls,
                                                     k_test=0)  # DOTA N=10, 无 K 校准剔除
        dota_class_gain[name] = {'base': a_base, 'grid': gains}
        dota_stats[name] = diff_stats(per_img, F, P_src)
        print('  %-16s base=%.4f' % (name, a_base))
    dota_grid = {}
    for b in BETAS:
        for lam in LAMBDAS:
            key = 'b%g_l%g' % (b, lam)
            dota_grid[key] = round(float(np.mean(
                [dota_class_gain[n]['grid'][key] for n in dota_class_gain])), 4)

    # ── VisDrone (DIOR 源, 4 类同名映射, 作第 4 目标域) ──
    print('[4/4a] VisDrone (DIOR 源) β×λ 网格 ...')
    vis_targets = [('VD-car', 'car', 'vehicle'), ('VD-truck', 'truck', 'vehicle'),
                   ('VD-bus', 'bus', 'vehicle'), ('VD-motor', 'motor', 'vehicle')]
    vis_class_gain, vis_stats = {}, {}
    for name, vis_cls, dior_cls in vis_targets:
        images, gts = VD.load_visdrone(vis_cls)
        if len(images) < 10:
            print('  !! %s 图太少, 跳过' % vis_cls)
            continue
        dets = U.gdino_detect(images, TGT_TMPL.format(vis_cls), 0.0)
        a_base, gains, per_img, F, P_src = eval_grid(images, gts, dets, name, vis_cls,
                                                     dior_cls, k_test=0)
        vis_class_gain[name] = {'base': a_base, 'grid': gains}
        vis_stats[name] = diff_stats(per_img, F, P_src)
        print('  %-16s base=%.4f' % (name, a_base))
    vis_grid = {}
    for b in BETAS:
        for lam in LAMBDAS:
            key = 'b%g_l%g' % (b, lam)
            vis_grid[key] = round(float(np.mean(
                [vis_class_gain[n]['grid'][key] for n in vis_class_gain])), 4)

    # ── VOC (复用 tge_dev_scan_voc 网格) ──
    print('[4/4] VOC (复用开发域网格) ...')
    voc_grid = {}
    try:
        dev = json.load(open(DATA / 'tge_dev_scan_voc.json', encoding='utf-8'))
        voc_base = {c: dev['baseline'][c] for c in dev['baseline']}
        voc_grid = {key: round(float(np.mean(
            [dev['grid'][key][c] - voc_base[c] for c in voc_base])), 4)
            for key in dev['grid']}
    except Exception as e:
        print('  ⚠️ VOC 网格复用失败 (COCO/DOTA 结果仍会写出): %s' % e)

    out = {'config': {'protocol': 'DCCS Phase A: β×λ 增益网格 (DIOR 源), 目标域 {COCO全量, VOC N100, '
                                  'DOTA N10, VisDrone N20}, 网格 β∈{0.5,1,2,4}×λ∈{0,0.25,0.5,0.75,1}',
                      'betas': BETAS, 'lambdas': LAMBDAS},
           'coco': {'grid': coco_grid, 'per_class': coco_class_gain, 'diff_stats': coco_stats},
           'dota': {'grid': dota_grid, 'per_class': dota_class_gain, 'diff_stats': dota_stats},
           'visdrone': {'grid': vis_grid, 'per_class': vis_class_gain, 'diff_stats': vis_stats},
           'voc': {'grid': voc_grid, 'source': 'tge_dev_scan_voc.json'}}
    with open(DATA / 'tge_dccs_grid.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print('\n增益矩阵 (均值, 跨类):')
    print('%-14s' % '', end='')
    for b in BETAS:
        print('β=%g     ' % b, end='')
    print()
    for lam in LAMBDAS:
        print('λ=%-5s' % lam, end='')
        for b in BETAS:
            key = 'b%g_l%g' % (b, lam)
            print('  %+8.4f ' % coco_grid[key], end='')
        print()
    print('\n→ tge_dccs_grid.json   (耗时 %.1f 分钟)' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
