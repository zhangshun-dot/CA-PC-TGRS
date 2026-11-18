# -*- coding: utf-8 -*-
"""boat_ablation.py — 确认(1.0,0.25)配置下boat的逐类增益

复用 tge_v2_main 的检测缓存与特征收集。
运行: python boat_ablation.py
输出: 实验数据/boat_ablation_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, time
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         fused_arrays, dataset_ap, v2_cached_dets,
                         TARGETS, K_CALIB, SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

# 目标域无关配置 (DIOR→VOC开发域选参)
BETA = 1.0
LAM = 0.25

def get_neg_proto(protos, pos_cls):
    """获取负原型: 与正原型最接近的top-3其他类原型均值。"""
    pos = protos[pos_cls]
    others = torch.stack([protos[c] for c in protos if c != pos_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()

def main():
    t0 = time.time()
    out_path = DATA / 'boat_ablation_results.json'

    print('=== Boat转负实验确认 ===')
    print(f'配置: β={BETA}, λ={LAM} (目标域无关配置)')
    print()

    # 1. 加载DIOR源原型
    print('[1/3] 加载DIOR 20类CLIP源原型...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    print(f'  加载 {len(protos)} 个源原型')

    # 2. 加载COCO 6类数据
    print('\n[2/3] 加载COCO 6类检测缓存与特征...')
    class_data = []
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print(f'  !! {name} 图太少, 跳过')
            continue
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        class_data.append({'name': name, 'per_img': per_img, 'F': F,
                           'P_src': P_src, 'P_neg': P_neg, 'Delta': Delta,
                           'n_test': len(images) - K_CALIB})
        print(f'  {name}: N_test={len(images)-K_CALIB}')

    # 3. 计算(1.0,0.25)配置下的逐类增益
    print('\n[3/3] 计算(1.0,0.25)配置下的逐类增益...')
    results = {}
    for d in class_data:
        P_src = d['P_src']
        P_neg = d['P_neg']
        Delta = d['Delta']

        # 正原型外推
        P_hat = P_src + BETA * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()

        # 融合权重
        W = (P_hat.cpu().numpy() - LAM * P_neg.cpu().numpy())

        # 计算AP
        fused_arrays(d['per_img'], d['F'], W)
        a_ours = dataset_ap(d['per_img'], 'fused')
        a_base = dataset_ap(d['per_img'], 'gd')
        gain = round(a_ours - a_base, 4)
        results[d['name']] = {'baseline': round(a_base, 4), 'ours': round(a_ours, 4), 'gain': gain}
        print(f'  {d["name"]}: baseline={a_base:.4f}, ours={a_ours:.4f}, gain={gain:+.4f}')

    # 计算均值
    mean_gain = round(float(np.mean([r['gain'] for r in results.values()])), 4)
    print(f'\n  6类均值增益: {mean_gain:+.4f}')

    # 特别关注boat
    boat_gain = results.get('COCO-boat', {}).get('gain', 'N/A')
    print(f'  Boat增益: {boat_gain}')

    # 保存结果
    out = {
        'config': {
            'protocol': '目标域无关配置 (DIOR→VOC开发域选参), 全量COCO val2017',
            'beta': BETA, 'lam': LAM,
            'note': '用于确认boat转负的真实数值'
        },
        'per_class': results,
        'mean_gain': mean_gain,
        'boat_gain': boat_gain
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f'\n→ {out_path}   (耗时 {((time.time()-t0)/60):.1f} 分钟)')

if __name__ == '__main__':
    main()
