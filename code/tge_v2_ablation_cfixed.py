# -*- coding: utf-8 -*-
r"""tge_v2_ablation_cfixed.py — §4.4消融实验重跑（C_fixed融合）

对比四种配置，都用C_fixed校准加权融合：
  1. 完整A-CSPC：正原型外推 + 负原型保守（β=2.0, λ=0.5）
  2. 去外推：β=0，仅源域视觉原型+负原型
  3. 去负原型：λ=0，仅正原型外推
  4. EC：负原型也外推（对称外推）

运行: python tge_v2_ablation_cfixed.py
输出: 实验数据/tge_v2_ablation_cfixed_results.json
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

# C_fixed最优参数（从主实验重算得到）
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

def v2_cached_dets(images, prompt, name):
    p = DATA / f'tge_v2_dets_COCO-COCO-{name}.pkl'
    if p.exists(): return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets

def collect_features(images, gts, dets):
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
                        'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
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


def evaluate_config(per_img, F, gd_key, P_pos, P_neg, beta, lam, use_ec=False):
    """通用评估函数：C_fixed校准加权融合"""
    for p in per_img:
        n = len(p['gd'])
        if n == 0:
            p['fused'] = np.zeros(0)
            continue
        f_k = F[p['crop_idx']]
        # 正原型
        if beta > 0:
            # 外推方向需要在外部计算，这里用预计算的P_pos（已外推或未外推）
            sim_pos = f_k @ P_pos
        else:
            sim_pos = f_k @ P_pos
        # 负原型
        if lam > 0:
            if use_ec:
                # EC：负原型也外推（用P_neg_ext，在外部计算）
                sim_neg = f_k @ P_neg
            else:
                sim_neg = f_k @ P_neg
            vis = np.maximum(0.0, sim_pos - lam * sim_neg)
        else:
            vis = np.maximum(0.0, sim_pos)
        # C_fixed校准加权融合
        gd_calib = np.clip(p['gd'] ** (1.0 / CF_TAU_DET), 1e-6, 1.0)
        vis_calib = np.clip(vis ** (1.0 / CF_TAU_VIS), 1e-6, 1.0)
        p['fused'] = CF_W * gd_calib + (1 - CF_W) * vis_calib
    return dataset_ap(per_img, 'fused')


def main():
    t0 = time.time()
    print('=' * 100)
    print('§4.4消融实验重跑（C_fixed校准加权融合，w=0.35, τ=2.0）')
    print('=' * 100)

    # 构建原型
    print('\n[1/3] 构建原型和加载数据 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None: protos[cls] = p

    class_data = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        images, gts = U.load_coco(coco_cls, n=100000)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), coco_cls)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].cpu().numpy()
        P_neg = T.get_neg_proto(protos, src_cls).cpu().numpy()
        # 外推方向
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0].numpy()
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0].numpy()
        Delta = d_y - s_xs
        # 外推后的正原型
        P_pos_ext = P_src + 2.0 * Delta
        P_pos_ext = P_pos_ext / (np.linalg.norm(P_pos_ext) + 1e-8)
        # EC：外推后的负原型
        P_neg_ext = P_neg + 2.0 * Delta
        P_neg_ext = P_neg_ext / (np.linalg.norm(P_neg_ext) + 1e-8)
        base50 = dataset_ap(per_img, 'gd')
        class_data[name] = {
            'per_img': per_img, 'F': F, 'base50': base50,
            'P_src': P_src, 'P_neg': P_neg, 'P_pos_ext': P_pos_ext, 'P_neg_ext': P_neg_ext,
        }
        print(f'  {name}: {len(per_img)}张图, baseAP={base50:.4f}')

    # 四种配置
    print('\n[2/3] 评估四种配置 ...')
    configs = {
        '完整A-CSPC': {'beta': 2.0, 'lam': 0.5, 'use_ext_pos': True, 'use_ec': False},
        '去外推(β=0)': {'beta': 0, 'lam': 0.5, 'use_ext_pos': False, 'use_ec': False},
        '去负原型(λ=0)': {'beta': 2.0, 'lam': 0, 'use_ext_pos': True, 'use_ec': False},
        'EC(负原型也外推)': {'beta': 2.0, 'lam': 0.5, 'use_ext_pos': True, 'use_ec': True},
    }

    results = {}
    for config_name, cfg in configs.items():
        print(f'\n  配置: {config_name}')
        gains = {}
        for name, cd in class_data.items():
            P_pos = cd['P_pos_ext'] if cfg['use_ext_pos'] else cd['P_src']
            P_neg = cd['P_neg_ext'] if cfg['use_ec'] else cd['P_neg']
            ap = evaluate_config(cd['per_img'], cd['F'], 'gd', P_pos, P_neg,
                                 cfg['beta'], cfg['lam'], cfg['use_ec'])
            gain = ap - cd['base50']
            gains[name] = gain
            print(f'    {name}: {gain:+.4f}')
        mean_gain = np.mean(list(gains.values()))
        n_pos = sum(1 for g in gains.values() if g > 0)
        results[config_name] = {
            'mean_gain': float(mean_gain),
            'n_positive': n_pos,
            'n_total': len(gains),
            'gains': {n: float(g) for n, g in gains.items()},
        }
        print(f'    均值: {mean_gain:+.4f}, 正增益: {n_pos}/{len(gains)}')

    # 汇总对比
    print('\n' + '=' * 100)
    print('汇总对比（C_fixed融合，DIOR→COCO六类）')
    print('=' * 100)
    print(f'\n{"配置":<25} {"均值增益":>12} {"正增益":>10} {"vs完整":>12}')
    print('-' * 60)
    full_gain = results['完整A-CSPC']['mean_gain']
    for config_name in configs.keys():
        r = results[config_name]
        vs_full = r['mean_gain'] - full_gain
        print(f'{config_name:<25} {r["mean_gain"]:>+12.4f} {r["n_positive"]:>4d}/{r["n_total"]:<3d} {vs_full:>+12.4f}')

    # 与原论文（几何平均）对比
    print(f'\n原论文（几何平均）完整A-CSPC: +0.0646')
    print(f'C_fixed完整A-CSPC: {full_gain:+.4f}')
    print(f'相对提升: {(full_gain-0.0646)/abs(0.0646)*100:+.1f}%')

    # 保存
    output = {
        'config': {'fusion': 'C_fixed', 'w': CF_W, 'tau_det': CF_TAU_DET, 'tau_vis': CF_TAU_VIS},
        'results': results,
        'original_geometric_full': 0.0646,
    }
    out_path = DATA / 'tge_v2_ablation_cfixed_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
