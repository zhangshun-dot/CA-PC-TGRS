# -*- coding: utf-8 -*-
r"""tge_v2_multidetector_cfixed.py — §4.8多检测器对比重跑（C_fixed融合）

对比三个检测器：GDINO-tiny, GDINO-base, OWLv2
每个检测器对比：几何平均融合 vs C_fixed校准加权融合

运行: python tge_v2_multidetector_cfixed.py
输出: 实验数据/tge_v2_multidetector_cfixed_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
K_CALIB = 10
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
PRE_FILTER = 0.03

DEV_DIOR = [
    ('DIOR-COCO-airplane',   'airplane',   'airplane'),
    ('DIOR-COCO-boat',       'boat',       'ship'),
    ('DIOR-COCO-car',        'car',        'vehicle'),
    ('DIOR-COCO-truck',      'truck',      'vehicle'),
    ('DIOR-COCO-bus',        'bus',        'vehicle'),
    ('DIOR-COCO-motorcycle', 'motorcycle', 'vehicle'),
]

DETECTORS = {
    'GDINO-tiny': {'prefix': 'tge_v2_dets_COCO-COCO', 'score_key': 'gd'},
    'GDINO-base': {'prefix': 'tge_v2_dets_GDINOB-COCO', 'score_key': 'gd'},
    'OWLv2':      {'prefix': 'tge_v2_dets_OWLV2-COCO', 'score_key': 'gd'},
}

CF_W = 0.35
CF_TAU_DET = 2.0
CF_TAU_VIS = 2.0


def clip_proto(crops):
    if not crops: return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()

def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10: continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops

def collect_features(images, gts, dets, score_key='gd'):
    per_img, crops_all = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10: continue
            if float(sc) < PRE_FILTER: continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        score_key: np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                        'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64)})
    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return per_img, F

def dataset_ap(per_img, key):
    if len(per_img) > K_CALIB:
        sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    else:
        sel = per_img
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    return U.compute_ap50_img(preds, gts)

def evaluate_geometric(per_img, F, score_key, P_pos_ext, P_neg, lam):
    W = P_pos_ext - lam * P_neg
    for p in per_img:
        n = len(p[score_key])
        if n:
            vis = np.maximum(0.0, F[p['crop_idx']] @ W)
            p['fused'] = np.sqrt(p[score_key] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)
    return dataset_ap(per_img, 'fused')

def evaluate_cfixed(per_img, F, score_key, P_pos_ext, P_neg, lam):
    for p in per_img:
        n = len(p[score_key])
        if n:
            f_k = F[p['crop_idx']]
            sim_pos = f_k @ P_pos_ext
            sim_neg = f_k @ P_neg
            vis = np.maximum(0.0, sim_pos - lam * sim_neg)
            gd_calib = np.clip(p[score_key] ** (1.0 / CF_TAU_DET), 1e-6, 1.0)
            vis_calib = np.clip(vis ** (1.0 / CF_TAU_VIS), 1e-6, 1.0)
            p['fused'] = CF_W * gd_calib + (1 - CF_W) * vis_calib
        else:
            p['fused'] = np.zeros(0)
    return dataset_ap(per_img, 'fused')


def main():
    t0 = time.time()
    print('=' * 100)
    print('§4.8多检测器对比重跑（几何平均 vs C_fixed校准加权融合）')
    print('=' * 100)

    # 构建原型
    print('\n[1/3] 构建DIOR原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None: protos[cls] = p

    # 预计算原型和外推方向
    proto_data = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        P_src = protos[src_cls].cpu().numpy()
        P_neg = T.get_neg_proto(protos, src_cls).cpu().numpy()
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0].numpy()
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0].numpy()
        Delta = d_y - s_xs
        P_pos_ext = P_src + 2.0 * Delta
        P_pos_ext = P_pos_ext / (np.linalg.norm(P_pos_ext) + 1e-8)
        proto_data[name] = {'P_pos_ext': P_pos_ext, 'P_neg': P_neg}

    # 逐个检测器评估
    print('\n[2/3] 逐个检测器评估 ...')
    all_results = {}

    for det_name, det_cfg in DETECTORS.items():
        print(f'\n  检测器: {det_name}')
        prefix = det_cfg['prefix']
        score_key = det_cfg['score_key']

        det_results = {'geometric': {}, 'cfixed': {}}
        base_aps = {}

        for name, coco_cls, src_cls in DEV_DIOR:
            # 加载检测缓存
            cache_path = DATA / f'{prefix}-{coco_cls}.pkl'
            if not cache_path.exists():
                print(f'    {name}: 缓存不存在，跳过')
                continue

            images, gts = U.load_coco(coco_cls, n=100000)
            dets = pickle.load(open(cache_path, 'rb'))
            # 确保dets长度与images一致
            if len(dets) > len(images):
                dets = dets[:len(images)]

            per_img, F = collect_features(images, gts, dets, score_key)
            base_ap = dataset_ap(per_img, score_key)
            base_aps[name] = base_ap

            pd = proto_data[name]
            # 几何平均
            ap_geo = evaluate_geometric(per_img, F, score_key, pd['P_pos_ext'], pd['P_neg'], 0.5)
            gain_geo = ap_geo - base_ap
            det_results['geometric'][name] = float(gain_geo)

            # C_fixed
            ap_cf = evaluate_cfixed(per_img, F, score_key, pd['P_pos_ext'], pd['P_neg'], 0.5)
            gain_cf = ap_cf - base_ap
            det_results['cfixed'][name] = float(gain_cf)

            print(f'    {name}: base={base_ap:.4f}, geo={gain_geo:+.4f}, cfixed={gain_cf:+.4f}')

        # 均值
        geo_gains = list(det_results['geometric'].values())
        cf_gains = list(det_results['cfixed'].values())
        det_results['geometric_mean'] = float(np.mean(geo_gains)) if geo_gains else 0
        det_results['cfixed_mean'] = float(np.mean(cf_gains)) if cf_gains else 0
        det_results['geometric_pos'] = sum(1 for g in geo_gains if g > 0)
        det_results['cfixed_pos'] = sum(1 for g in cf_gains if g > 0)
        det_results['base_mean'] = float(np.mean(list(base_aps.values()))) if base_aps else 0

        all_results[det_name] = det_results
        print(f'    均值: base={det_results["base_mean"]:.4f}, '
              f'geo={det_results["geometric_mean"]:+.4f} ({det_results["geometric_pos"]}/6), '
              f'cfixed={det_results["cfixed_mean"]:+.4f} ({det_results["cfixed_pos"]}/6)')

    # 汇总对比
    print('\n' + '=' * 100)
    print('汇总对比（DIOR→COCO六类）')
    print('=' * 100)
    print(f'\n{"检测器":<15} {"基线AP":>10} {"几何平均":>12} {"C_fixed":>12} {"提升":>10} {"正增益":>10}')
    print('-' * 70)
    for det_name in DETECTORS.keys():
        r = all_results[det_name]
        improvement = r['cfixed_mean'] - r['geometric_mean']
        print(f'{det_name:<15} {r["base_mean"]:>10.4f} '
              f'{r["geometric_mean"]:>+12.4f} {r["cfixed_mean"]:>+12.4f} '
              f'{improvement:>+10.4f} {r["cfixed_pos"]:>4d}/6')

    # 保存
    output = {
        'config': {'fusion_cfixed': {'w': CF_W, 'tau_det': CF_TAU_DET, 'tau_vis': CF_TAU_VIS},
                   'beta': 2.0, 'lambda': 0.5},
        'results': all_results,
    }
    out_path = DATA / 'tge_v2_multidetector_cfixed_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
