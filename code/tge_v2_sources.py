# -*- coding: utf-8 -*-
r"""tge_v2_sources.py — v2 多源域全量: NWPU / DOTA → COCO (全量 val2017, 主配置可指定)

刷新 §4.6 表 3: 在 COCO 全量 + (β, λ) 下重跑 NWPU VHR-10 与 DOTA v1.5 两个
独立遥感源域 (与 DIOR→COCO 主表同协议, 复用 tge_v2 的 COCO 检测缓存)。

类映射 (与主表同构):
  NWPU: airplane→airplane, ship→boat, vehicle→car/truck/bus/motorcycle
  DOTA: plane→airplane, ship→boat, vehicle(large+small 合并)→car/truck/bus/motorcycle

复用 tge_probe_nwpu.load_nwpu 与 tge_probe_dota_src.load_dota。
运行: python tge_v2_sources.py [--beta 2.0 --lam 0.5]   (~15 分钟)
输出: 实验数据/tge_v2_sources_b2_l0.5.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, pickle, argparse
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
import tge_probe_nwpu as NW
import tge_probe_dota_src as DS
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         fused_arrays, dataset_ap, v2_cached_dets,
                         FROZEN_BETA, FROZEN_LAM, K_CALIB,
                         SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

# (源域名, 目标类, 源原型类) 映射结构与主表相同
NWPU_TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]
DOTA_TARGETS = [
    ('COCO-airplane',   'airplane',   'plane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


def build_nwpu_protos():
    protos = {}
    for cls in NW.NWPU_CLASSES:
        images, gts = NW.load_nwpu(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    return protos


def build_dota_protos():
    protos = {}
    for cls in DS.DOTA_CLASSES:
        images, gts = DS.load_dota(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    # vehicle 合并: large + small
    imgs_v, gts_v = [], []
    for vc in ('large-vehicle', 'small-vehicle'):
        im2, gt2 = DS.load_dota(vc)
        imgs_v += im2
        gts_v += gt2
    protos['vehicle'] = clip_proto(gt_crops(imgs_v, gts_v))
    return protos


def run_source(name, protos, targets):
    rows = {}
    for eval_name, coco_cls, src_cls in targets:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            continue
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), eval_name)
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
        a_base = dataset_ap(per_img, 'gd')
        a_ours = dataset_ap(per_img, 'fused')
        rec = recall(per_img)
        rows[eval_name] = {'base': round(a_base, 4), 'ours': round(a_ours, 4),
                           'gain': round(a_ours - a_base, 4), 'recall': round(rec, 4),
                           'n_test': len(images) - K_CALIB}
        print('  %-16s base=%.4f ours=%.4f (%+.4f) rec=%.4f' %
              (eval_name, a_base, a_ours, a_ours - a_base, rec))
    return rows


def recall(per_img):
    n_gt, hit = 0, 0
    for p in per_img:
        gts, boxes = p['gt'], p['box']
        n_gt += len(gts)
        if len(boxes) == 0:
            continue
        matched = np.zeros(len(gts), dtype=bool)
        for b in boxes:
            if matched.all():
                break
            for j, gb in enumerate(gts):
                if matched[j]:
                    continue
                if U.iou(b.tolist(), gb.tolist()) >= 0.5:
                    matched[j] = True
        hit += int(matched.sum())
    return hit / n_gt if n_gt else float('nan')


def main():
    global FROZEN_BETA, FROZEN_LAM
    import time
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC v2 多源域全量 (NWPU/DOTA→COCO)')
    ap.add_argument('--beta', type=float, default=FROZEN_BETA)
    ap.add_argument('--lam', type=float, default=FROZEN_LAM)
    args = ap.parse_args()
    FROZEN_BETA, FROZEN_LAM = args.beta, args.lam
    tag = 'b%g_l%g' % (FROZEN_BETA, FROZEN_LAM)
    out_path = DATA / ('tge_v2_sources_%s.json' % tag)
    print('配置: β=%.2f λ=%.2f  →  输出 %s' % (FROZEN_BETA, FROZEN_LAM, out_path.name))

    print('[1/2] NWPU VHR-10 → COCO (10 类池) ...')
    rows_n = run_source('NWPU', build_nwpu_protos(), NWPU_TARGETS)
    print('[2/2] DOTA v1.5 → COCO (16 类池 + vehicle 合并) ...')
    rows_d = run_source('DOTA', build_dota_protos(), DOTA_TARGETS)

    def mean_of(rows):
        return {'base': round(float(np.mean([rows[c]['base'] for c in rows])), 4),
                'ours': round(float(np.mean([rows[c]['ours'] for c in rows])), 4),
                'gain': round(float(np.mean([rows[c]['gain'] for c in rows])), 4)}

    out = {'config': {'protocol': 'v2: NWPU/DOTA→COCO, COCO val2017 全量, β=%.2f λ=%.2f, '
                                  'K=10 calib, per-image AP50' % (FROZEN_BETA, FROZEN_LAM)},
           'nwpu': {'classes': rows_n, 'mean': mean_of(rows_n)},
           'dota': {'classes': rows_d, 'mean': mean_of(rows_d)}}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 90)
    print('源域      %-14s %9s %9s %9s' % ('', 'base', 'ours', '增益'))
    print('-' * 90)
    for label, rows in [('NWPU', rows_n), ('DOTA', rows_d)]:
        m = mean_of(rows)
        print('%-12s %-14s %9.4f %9.4f %+9.4f' % (label, 'mean', m['base'], m['ours'], m['gain']))
    print('=' * 90)
    print('→ %s   (耗时 %.1f 分钟)' % (out_path.name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
