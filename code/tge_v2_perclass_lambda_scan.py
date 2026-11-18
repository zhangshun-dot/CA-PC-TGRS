# -*- coding: utf-8 -*-
r"""tge_v2_perclass_lambda_scan.py — 逐类 λ 扫描

固定 β=2.0, 对每个 COCO 类别扫描 λ ∈ {0, 0.25, 0.5, 0.75, 1.0}
找到每类最优 λ, 对比固定 λ=0.5
验证物理先验 C(混淆度) 能否预测最优 λ
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import time
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

BETA = 2.0
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
K_CALIB = 10
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
DETECTOR = 'gdino-tiny'
PRE_FILTER = 0.03

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


def load_coco_all(cls_name):
    return U.load_coco(cls_name, n=100000)


def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def v2_cached_dets(images, prompt, name):
    p = DATA / f'tge_v2_dets_COCO-{name}.pkl'
    if p.exists():
        print(f'    [缓存] {p.name}')
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def collect_features(images, gts, dets):
    per_img = []
    crops_all = []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                        'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64)})
    if crops_all:
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    return per_img, F


def dataset_ap(per_img, key):
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    return U.compute_ap50_img(preds, gts)


def main():
    t0 = time.time()
    print('=' * 90)
    print('逐类 λ 扫描 (β=2.0 固定, λ ∈ {0, 0.25, 0.5, 0.75, 1.0})')
    print('=' * 90)

    print('[1/3] DIOR 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/3] COCO 6 类检测 + 特征 ...')
    class_data = []
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        # A-CSPC: P̂ = norm(P_src + β·Δ), β=2.0 固定
        P_hat = P_src + BETA * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        class_data.append({'name': name, 'per_img': per_img, 'F': F,
                           'P_hat': P_hat.cpu().numpy(), 'P_neg': P_neg.cpu().numpy(),
                           'n_test': len(images) - K_CALIB})
        print('  %-16s N_test=%3d 框=%6d' % (name, len(images)-K_CALIB,
                                                  sum(len(p['gd']) for p in per_img)))

    print('[3/3] 逐类 λ 扫描 ...')
    results = {}
    for d in class_data:
        name = d['name']
        base50 = dataset_ap(d['per_img'], 'gd')
        lambda_results = {}
        for lam in LAMBDAS:
            W = d['P_hat'] - lam * d['P_neg']
            # 计算 fused
            for p in d['per_img']:
                n = len(p['gd'])
                if n:
                    vis = np.maximum(0.0, d['F'][p['crop_idx']] @ W)
                    p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
                else:
                    p['fused'] = np.zeros(0)
            ap50 = dataset_ap(d['per_img'], 'fused')
            lambda_results[lam] = {'ap50': round(ap50, 4), 'gain': round(ap50 - base50, 4)}
        # 找最优 λ
        best_lam = max(LAMBDAS, key=lambda l: lambda_results[l]['gain'])
        results[name] = {'base50': round(base50, 4),
                         'lambda_scan': lambda_results,
                         'best_lambda': best_lam,
                         'best_gain': lambda_results[best_lam]['gain'],
                         'fixed_lam05_gain': lambda_results[0.5]['gain']}
        gains_str = '  '.join(f'λ={l}:{lambda_results[l]["gain"]:+.3f}' for l in LAMBDAS)
        print('  %-16s base=%.4f  %s  | 最优λ=%.2f (%+.4f)' %
              (name, base50, gains_str, best_lam, lambda_results[best_lam]['gain']))

    # 汇总
    print('\n' + '=' * 90)
    print('汇总表')
    print('=' * 90)
    print('%-16s %8s %8s %8s %8s %8s %8s %10s %10s' %
          ('类别', 'λ=0', 'λ=0.25', 'λ=0.5', 'λ=0.75', 'λ=1.0', 'base', '最优λ', '最优增益'))
    print('-' * 100)
    total_fixed = 0
    total_best = 0
    for name, r in results.items():
        gains = [r['lambda_scan'][l]['gain'] for l in LAMBDAS]
        print('%-16s %+8.4f %+8.4f %+8.4f %+8.4f %+8.4f %8.4f %10.2f %+10.4f' %
              (name, gains[0], gains[1], gains[2], gains[3], gains[4], r['base50'],
               r['best_lambda'], r['best_gain']))
        total_fixed += r['fixed_lam05_gain']
        total_best += r['best_gain']
    n = len(results)
    print('-' * 100)
    print('%-16s %8s %8s %+8.4f %8s %8s %8s %10s %+10.4f' %
          ('平均', '', '', total_fixed/n, '', '', '', '', total_best/n))
    print('\n固定 λ=0.5 平均增益: %+.4f' % (total_fixed/n))
    print('逐类最优 λ 平均增益: %+.4f' % (total_best/n))
    print('提升: %+.4f' % (total_best/n - total_fixed/n))

    # 物理先验 C(混淆度) vs 最优 λ
    print('\n' + '=' * 90)
    print('物理先验 C(混淆度) vs 最优 λ')
    print('=' * 90)
    # COCO 6类的混淆度 (从 v6 计算逻辑: 域内最近邻距离)
    # airplane 最近邻 boat, boat 最近邻 airplane, car 最近邻 truck, truck 最近邻 car, bus 最近邻 truck, motorcycle 最近邻 bus
    # 用 V,M,S,I 四维欧氏距离
    prior_lib_local = {
        "airplane":   {"V": 0.65, "M": 0.15, "S": 0.20, "I": 0.35},
        "boat":       {"V": 0.90, "M": 0.70, "S": 0.30, "I": 0.55},
        "car":        {"V": 0.50, "M": 0.10, "S": 0.70, "I": 0.50},
        "truck":      {"V": 0.60, "M": 0.30, "S": 0.40, "I": 0.55},
        "bus":        {"V": 0.65, "M": 0.50, "S": 0.40, "I": 0.40},
        "motorcycle": {"V": 0.95, "M": 0.55, "S": 0.90, "I": 0.45},
    }
    coco_cats = ["airplane", "boat", "car", "truck", "bus", "motorcycle"]
    confusion = {}
    for cat in coco_cats:
        p = prior_lib_local[cat]
        vec = [p["V"], p["M"], p["S"], p["I"]]
        min_dist = float("inf")
        for other in coco_cats:
            if other == cat: continue
            po = prior_lib_local[other]
            vec_o = [po["V"], po["M"], po["S"], po["I"]]
            dist = np.sqrt(sum((a-b)**2 for a,b in zip(vec, vec_o)))
            if dist < min_dist: min_dist = dist
        confusion[cat] = round(np.exp(-min_dist * 5.0), 3)

    print('%-16s %8s %10s %12s' % ('类别', 'C(混淆)', '最优λ', '固定λ=0.5增益'))
    print('-' * 50)
    for name, r in results.items():
        cat = name.replace('COCO-', '')
        print('%-16s %8.3f %10.2f %+12.4f' % (name, confusion[cat], r['best_lambda'], r['fixed_lam05_gain']))

    # 保存
    out = {'config': {'beta': BETA, 'lambdas': LAMBDAS, 'protocol': 'per-class lambda scan'},
           'results': results, 'confusion': confusion,
           'summary': {'fixed_lam05_mean_gain': round(total_fixed/n, 4),
                        'perclass_best_mean_gain': round(total_best/n, 4),
                        'improvement': round(total_best/n - total_fixed/n, 4)}}
    out_path = DATA / 'tge_v2_perclass_lambda_scan.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ %s  (耗时 %.1f 分钟)' % (out_path.name, (time.time()-t0)/60))


if __name__ == '__main__':
    main()
