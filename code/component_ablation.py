# -*- coding: utf-8 -*-
"""component_ablation.py — A-CSPC 组件消融实验

计算四个配置的6类均值AP50增益：
1. A-CSPC主配置: β=2.0, λ=0.5 (正原型外推 + 源域负原型)
2. 去除正原型外推: β=0, λ=0.5 (仅源域视觉原型 + 负原型)
3. 去除负原型: β=2.0, λ=0 (正原型外推, 无负原型)
4. 对称外推: β=2.0, λ=0.5 (正负原型都沿文本方向外推)

复用 tge_v2_main 的检测缓存与特征收集。
运行: python component_ablation.py
输出: 实验数据/component_ablation_results.json
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

# 主配置参数 (目标域调参配置, 与论文主表一致)
BETA = 2.0
LAM = 0.5

def get_neg_proto(protos, pos_cls):
    """获取负原型: 与正原型最接近的top-3其他类原型均值。"""
    pos = protos[pos_cls]
    others = torch.stack([protos[c] for c in protos if c != pos_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()

def compute_config_gain(class_data, protos, config_name, beta, lam, symmetric=False):
    """计算指定配置的6类均值增益。"""
    gains = {}
    for d in class_data:
        P_src = d['P_src']
        P_neg = d['P_neg']
        Delta = d['Delta']

        # 正原型
        if beta > 0:
            P_hat = P_src + beta * Delta.to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()
        else:
            P_hat = P_src  # 不外推

        # 负原型
        if symmetric and beta > 0:
            # 对称外推: 负原型也沿文本方向外推
            P_neg_hat = P_neg + beta * Delta.to(U.DEVICE)
            P_neg_hat = P_neg_hat / P_neg_hat.norm()
        else:
            P_neg_hat = P_neg  # 负原型保留在源域空间

        # 融合权重
        W = (P_hat.cpu().numpy() - lam * P_neg_hat.cpu().numpy())

        # 计算AP
        fused_arrays(d['per_img'], d['F'], W)
        a_ours = dataset_ap(d['per_img'], 'fused')
        a_base = dataset_ap(d['per_img'], 'gd')
        gains[d['name']] = round(a_ours - a_base, 4)

    mean_gain = round(float(np.mean(list(gains.values()))), 4)
    print(f'  {config_name}: 均值增益={mean_gain:+.4f}')
    for name, gain in gains.items():
        print(f'    {name}: {gain:+.4f}')
    return {'per_class_gain': gains, 'mean_gain': mean_gain}

def main():
    t0 = time.time()
    out_path = DATA / 'component_ablation_results.json'

    print('=== A-CSPC 组件消融实验 ===')
    print(f'配置: β={BETA}, λ={LAM} (主配置)')
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

    # 2. 加载COCO数据和特征
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

    # 3. 计算四个配置的增益
    print('\n[3/3] 计算四个配置的增益...')
    results = {}

    # 配置1: A-CSPC主配置
    print('\n配置1: A-CSPC主配置 (正原型外推 + 源域负原型)')
    results['acspc_main'] = compute_config_gain(class_data, protos,
        'A-CSPC (β=2.0, λ=0.5)', beta=BETA, lam=LAM, symmetric=False)

    # 配置2: 去除正原型外推 (β=0)
    print('\n配置2: 去除正原型外推 (仅源域视觉原型 + 负原型)')
    results['no_pos_extrap'] = compute_config_gain(class_data, protos,
        '仅源域视觉原型 (β=0, λ=0.5)', beta=0, lam=LAM, symmetric=False)

    # 配置3: 去除负原型 (λ=0)
    print('\n配置3: 去除负原型 (正原型外推, 无负原型)')
    results['no_neg_proto'] = compute_config_gain(class_data, protos,
        '仅正原型外推 (β=2.0, λ=0)', beta=BETA, lam=0, symmetric=False)

    # 配置4: 对称外推 (正负原型都外推)
    print('\n配置4: 对称外推 (正负原型都沿文本方向外推)')
    results['symmetric_extrap'] = compute_config_gain(class_data, protos,
        '对称外推 (β=2.0, λ=0.5, 正负都外推)', beta=BETA, lam=LAM, symmetric=True)

    # 保存结果
    out = {
        'config': {
            'protocol': 'A-CSPC component ablation, full val2017, GDINO-tiny, CLIP ViT-B/32',
            'beta': BETA, 'lam': LAM,
            'note': '主配置β=2.0,λ=0.5为目标域调参配置'
        },
        'results': results
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 汇总表
    print('\n' + '=' * 80)
    print('组件消融结果汇总:')
    print('-' * 80)
    print(f'{"配置":<35} {"均值增益":>10}')
    print('-' * 80)
    print(f'{"A-CSPC主配置 (β=2.0,λ=0.5)":<35} {results["acspc_main"]["mean_gain"]:>+10.4f}')
    print(f'{"去除正原型外推 (β=0)":<35} {results["no_pos_extrap"]["mean_gain"]:>+10.4f}')
    print(f'{"去除负原型 (λ=0)":<35} {results["no_neg_proto"]["mean_gain"]:>+10.4f}')
    print(f'{"对称外推 (正负都外推)":<35} {results["symmetric_extrap"]["mean_gain"]:>+10.4f}')
    print('=' * 80)
    print(f'\n→ {out_path}   (耗时 {((time.time()-t0)/60):.1f} 分钟)')

if __name__ == '__main__':
    main()
