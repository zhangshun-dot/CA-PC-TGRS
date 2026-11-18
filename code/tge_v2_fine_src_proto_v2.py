# -*- coding: utf-8 -*-
r"""tge_v2_fine_src_proto_v2.py — 方案A: 细类别源原型 + 粗 s_xs

P_src_fine(c) = norm(P_vehicle + gamma * (text_emb("a photo of c") - text_emb("a photo of vehicle")))
s_xs 保持粗 "an overhead image of vehicle" (跟原方法一样)
Delta = d_y(c) - s_xs(vehicle)  (跟原方法一样)

gamma=0 等于原方法; gamma>0 是在原方法基础上的增量

airplane/boat 保持原细到细原型 (G=0)
扫描 gamma in {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}
beta=2.0, lambda=0.5 固定
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

BETA = 2.0
LAMBDA = 0.5
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
K_CALIB = 10
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
DETECTOR = 'gdino-tiny'
PRE_FILTER = 0.03

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane', False),
    ('COCO-boat',       'boat',       'ship',     False),
    ('COCO-car',        'car',        'vehicle',  True),
    ('COCO-truck',      'truck',      'vehicle',  True),
    ('COCO-bus',        'bus',        'vehicle',  True),
    ('COCO-motorcycle', 'motorcycle', 'vehicle',  True),
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
    print('方案A: 细类别源原型 + 粗 s_xs (β=2.0, λ=0.5 固定)')
    print('=' * 90)

    # [1] DIOR 源原型
    print('[1/4] DIOR 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    P_vehicle = protos['vehicle'].to(U.DEVICE)

    # [2] 文本嵌入
    print('[2/4] 文本嵌入 ...')
    d_y = {}
    for name, coco_cls, src_cls, needs_fine in TARGETS:
        d_y[coco_cls] = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]

    # s_xs: 全部用粗版本 (跟原方法一样)
    s_xs_coarse = {}
    for name, coco_cls, src_cls, needs_fine in TARGETS:
        s_xs_coarse[coco_cls] = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]

    # 用于构造偏移量的文本嵌入 (目标模板)
    text_c = {}
    for name, coco_cls, src_cls, needs_fine in TARGETS:
        if needs_fine:
            text_c[coco_cls] = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
    text_vehicle = T.text_emb([TGT_TMPL.format('vehicle')]).cpu()[0]

    # [3] COCO 检测 + 特征
    print('[3/4] COCO 6 类检测 + 特征 ...')
    class_data = []
    for name, coco_cls, src_cls, needs_fine in TARGETS:
        images, gts = load_coco_all(coco_cls)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_neg = T.get_neg_proto(protos, src_cls)
        class_data.append({'name': name, 'coco_cls': coco_cls, 'src_cls': src_cls,
                           'needs_fine': needs_fine, 'per_img': per_img, 'F': F,
                           'P_neg': P_neg.cpu().numpy(),
                           'n_test': len(images) - K_CALIB})
        print('  %-16s N_test=%3d 框=%6d  fine_proto=%s' %
              (name, len(images)-K_CALIB, sum(len(p['gd']) for p in per_img), needs_fine))

    # [4] gamma 扫描
    print('[4/4] gamma 扫描 ...')
    results = {}
    for d in class_data:
        name = d['name']
        coco_cls = d['coco_cls']
        base50 = dataset_ap(d['per_img'], 'gd')
        gamma_results = {}

        # Delta = d_y - s_xs (粗版本, 跟原方法一样)
        Delta = d_y[coco_cls] - s_xs_coarse[coco_cls]

        for gamma in GAMMAS:
            if d['needs_fine']:
                # P_src_fine = norm(P_vehicle + gamma * (text(c) - text(vehicle)))
                offset = (text_c[coco_cls] - text_vehicle).to(U.DEVICE)
                P_src = P_vehicle + gamma * offset
                P_src = P_src / P_src.norm()
            else:
                P_src = protos[d['src_cls']].to(U.DEVICE)

            # A-CSPC: P_hat = norm(P_src + beta * Delta)
            P_hat = P_src + BETA * Delta.to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()

            W = P_hat.cpu().numpy() - LAMBDA * d['P_neg']

            for p in d['per_img']:
                n = len(p['gd'])
                if n:
                    vis = np.maximum(0.0, d['F'][p['crop_idx']] @ W)
                    p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
                else:
                    p['fused'] = np.zeros(0)
            ap50 = dataset_ap(d['per_img'], 'fused')
            gamma_results[gamma] = {'ap50': round(ap50, 4), 'gain': round(ap50 - base50, 4)}

        best_gamma = max(GAMMAS, key=lambda g: gamma_results[g]['gain'])
        results[name] = {'base50': round(base50, 4),
                         'gamma_scan': gamma_results,
                         'best_gamma': best_gamma,
                         'best_gain': gamma_results[best_gamma]['gain'],
                         'gamma0_gain': gamma_results[0.0]['gain'],
                         'needs_fine': d['needs_fine']}
        gains_str = '  '.join(f'γ={g}:{gamma_results[g]["gain"]:+.3f}' for g in GAMMAS)
        print('  %-16s base=%.4f  %s  | 最优γ=%.2f (%+.4f)' %
              (name, base50, gains_str, best_gamma, gamma_results[best_gamma]['gain']))

    # 汇总
    print('\n' + '=' * 90)
    print('汇总表')
    print('=' * 90)
    header = '%-16s %8s' % ('类别', 'base')
    for g in GAMMAS:
        header += ' %8s' % ('γ=%.2f' % g)
    header += ' %10s %10s' % ('最优γ', '最优增益')
    print(header)
    print('-' * 140)

    total_best = 0
    n = len(results)
    for name, r in results.items():
        gains = [r['gamma_scan'][g]['gain'] for g in GAMMAS]
        row = '%-16s %8.4f' % (name, r['base50'])
        for g in gains:
            row += ' %+8.4f' % g
        row += ' %10.2f %+10.4f' % (r['best_gamma'], r['best_gain'])
        print(row)
        total_best += r['best_gain']

    print('-' * 140)
    print('%-16s %8s' % ('平均(全部6类)', ''), end='')
    for g in GAMMAS:
        avg = sum(results[name]['gamma_scan'][g]['gain'] for name in results) / n
        print(' %+8.4f' % avg, end='')
    print(' %10s %+10.4f' % ('', total_best/n))

    # 只看需要细原型的 4 类
    print('\n--- 只看需要细原型的 4 类 (car/truck/bus/motorcycle) ---')
    print('%-16s %8s' % ('类别', 'base'), end='')
    for g in GAMMAS:
        print(' %8s' % ('γ=%.2f' % g), end='')
    print(' %10s %10s' % ('最优γ', '最优增益'))
    print('-' * 120)
    fine_total_best = 0
    for name, r in results.items():
        if not r['needs_fine']:
            continue
        gains = [r['gamma_scan'][g]['gain'] for g in GAMMAS]
        row = '%-16s %8.4f' % (name, r['base50'])
        for g in gains:
            row += ' %+8.4f' % g
        row += ' %10.2f %+10.4f' % (r['best_gamma'], r['best_gain'])
        print(row)
        fine_total_best += r['best_gain']
    print('-' * 120)
    print('%-16s %8s' % ('平均(4类)', ''), end='')
    for g in GAMMAS:
        avg = sum(results[name]['gamma_scan'][g]['gain'] for name in results if results[name]['needs_fine']) / 4
        print(' %+8.4f' % avg, end='')
    print(' %10s %+10.4f' % ('', fine_total_best/4))

    print('\n=== 关键对比 ===')
    print('原方法 (γ=0, 粗原型+粗s_xs): 6类均值 +0.0646')
    print('逐类最优 γ:                   6类均值 %+.4f' % (total_best/n))
    print('提升 (vs 原方法):              %+.4f' % (total_best/n - 0.0646))
    print('细原型4类原方法均值: %+.4f' % ((0.0809+0.0774+0.0735+0.1189)/4))
    print('细原型4类最优γ均值:  %+.4f' % (fine_total_best/4))

    # 保存
    out = {'config': {'beta': BETA, 'lambda': LAMBDA, 'gammas': GAMMAS,
                       'protocol': 'fine-grained source prototype + coarse s_xs (scheme A)'},
           'results': results,
           'summary': {'original_6class_mean': 0.0646,
                        'best_gamma_6class_mean': round(total_best/n, 4),
                        'improvement_vs_original': round(total_best/n - 0.0646, 4),
                        'original_fine4_mean': round((0.0809+0.0774+0.0735+0.1189)/4, 4),
                        'best_gamma_fine4_mean': round(fine_total_best/4, 4)}}
    out_path = DATA / 'tge_v2_fine_src_proto_v2.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ %s  (耗时 %.1f 分钟)' % (out_path.name, (time.time()-t0)/60))


if __name__ == '__main__':
    main()
