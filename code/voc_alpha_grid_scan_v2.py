# -*- coding: utf-8 -*-
"""voc_alpha_grid_scan_v2.py — VOC目标域完整α网格搜索 + 初始间隔m^0计算
使用sota_comparison中的现成函数，保证和之前实验一致
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
from scipy import stats

# 从sota_comparison导入所有需要的函数和常量
from sota_comparison import (
    SOURCE_LOADERS,
    DIOR_TARGETS, NWPU_TARGETS, DOTA_TARGETS, VISDRONE_TARGETS,
    build_proto_tge, evaluate_with_features, evaluate_baseline,
    compute_box_size_variance, compute_text_confusability,
    BETA, K_PCA, DATA, T
)

# 从voc_sota_comparison导入load_target_voc
from voc_sota_comparison import load_target_voc

# α扫描范围
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# 源域类别映射（COCO目标域）
REMOTE_TARGETS = {
    'DIOR': DIOR_TARGETS,
    'NWPU': NWPU_TARGETS,
    'DOTA': DOTA_TARGETS,
    'VisDrone': VISDRONE_TARGETS,
}

# COCO类别到VOC类别的映射
COCO2VOC = {
    'airplane': 'aeroplane',
    'boat': 'boat',
    'car': 'car',
    'bus': 'bus',
    'motorcycle': 'motorbike',
    'bicycle': 'bicycle',
    # VOC没有truck类，跳过
}

# 源域类别映射（转换为VOC目标域）
REMOTE_TARGETS_VOC = {}
for src_name, class_list in REMOTE_TARGETS.items():
    voc_list = []
    for coco_cls, src_cls, src_display in class_list:
        if coco_cls in COCO2VOC:
            voc_cls = COCO2VOC[coco_cls]
            voc_list.append((voc_cls, src_cls, src_display))
    REMOTE_TARGETS_VOC[src_name] = voc_list


def compute_initial_margin(features, proto_c, proto_neg):
    """计算初始间隔m_c^(0)：top1 - top2的置信度中位数
    features: (N, D) 候选框特征
    proto_c: (D,) 目标类别原型
    proto_neg: (D,) 负类别原型
    """
    # 计算目标类和负类的相似度
    sim_c = features @ proto_c  # (N,)
    sim_neg = features @ proto_neg  # (N,)

    # top1 - top2
    top1 = np.maximum(sim_c, sim_neg)
    top2 = np.minimum(sim_c, sim_neg)
    margins = top1 - top2

    # 取中位数作为m_c^(0)
    m_0 = np.median(margins)
    return float(m_0), margins


def run_alpha_scan_for_class(src_name, src_cls, target_cls, n_images=200):
    """对单个类别跑α网格搜索"""
    print(f'\n  处理 {src_name}/{src_cls} -> VOC/{target_cls}...')

    # 加载源域数据
    src_loader = SOURCE_LOADERS[src_name][0]  # 元组第一个元素是加载函数
    src_data = src_loader(src_cls, n=150)  # 传入字符串，不是列表
    if src_data is None or len(src_data.get('features', [])) == 0:
        print(f'    警告: 源域数据为空')
        return None

    # 加载VOC目标域数据
    target_data = load_target_voc(target_cls, n=n_images)
    if target_data is None or len(target_data.get('features', [])) == 0:
        print(f'    警告: VOC目标域数据为空')
        return None

    features = target_data['features']
    per_img = target_data['per_img']

    # 构建原型（和sota_comparison一致）
    proto_c, proto_neg, text_src, text_tgt = build_proto_tge(src_data, src_cls, target_cls)

    # 计算初始间隔m^0
    m_0, margins = compute_initial_margin(features, proto_c, proto_neg)

    # 计算外推方向
    extrapolation_dir = text_tgt - text_src
    extrapolation_dir = extrapolation_dir / (np.linalg.norm(extrapolation_dir) + 1e-8)

    # 计算mu_c：校正方向在特征上的平均投影
    # 只取预测为目标类别的样本
    sim_c = features @ proto_c
    sim_neg = features @ proto_neg
    pred_c_mask = sim_c > sim_neg
    if pred_c_mask.sum() > 0:
        mu_c = np.mean(features[pred_c_mask] @ extrapolation_dir)
    else:
        mu_c = 0.0

    # α网格搜索
    ap_results = {}
    for alpha in ALPHAS:
        # 外推原型
        extrapolated_proto = proto_c + alpha * extrapolation_dir

        # 评估（使用sota_comparison的evaluate_with_features）
        ap = evaluate_with_features(features, per_img, extrapolated_proto, proto_neg)
        ap_results[alpha] = float(ap)

    # 找最优α
    best_alpha = max(ap_results, key=ap_results.get)
    best_ap = ap_results[best_alpha]
    baseline_ap = ap_results[0.0]

    return {
        'src': src_name,
        'src_cls': src_cls,
        'target_cls': target_cls,
        'n_images': n_images,
        'm_0': m_0,
        'mu_c': float(mu_c),
        'baseline_ap': baseline_ap,
        'best_alpha': best_alpha,
        'best_ap': best_ap,
        'ap_curve': ap_results,
        'alpha_gt_0_better': best_ap > baseline_ap + 0.001,
    }


def main():
    print('=' * 80)
    print('VOC目标域完整α网格搜索 + 初始间隔m^0计算')
    print('验证Corollary 2的理论预测')
    print('=' * 80)

    all_results = []

    for src_name, class_list in REMOTE_TARGETS_VOC.items():
        print(f'\n{"=" * 80}')
        print(f'源域: {src_name}')
        print(f'{"=" * 80}')

        for voc_cls, src_cls, src_display in class_list:
            result = run_alpha_scan_for_class(src_name, src_cls, voc_cls)
            if result:
                all_results.append(result)
                print(f'    {src_cls} -> VOC/{voc_cls}: m^0={result["m_0"]:.4f}, '
                      f'baseline AP={result["baseline_ap"]:.4f}, '
                      f'best α={result["best_alpha"]:.2f}, best AP={result["best_ap"]:.4f}, '
                      f'α>0更好={result["alpha_gt_0_better"]}')

    # 保存结果
    output_file = DATA / 'voc_alpha_grid_scan_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {output_file}')

    # 分析结果
    print(f'\n{"=" * 80}')
    print('分析：m^0 vs 最优α')
    print(f'{"=" * 80}')

    m_0_values = [r['m_0'] for r in all_results]
    best_alphas = [r['best_alpha'] for r in all_results]
    baseline_aps = [r['baseline_ap'] for r in all_results]
    alpha_gt_0 = [r['alpha_gt_0_better'] for r in all_results]

    print(f'\n总类别数: {len(all_results)}')
    print(f'α>0更好的类别数: {sum(alpha_gt_0)} ({sum(alpha_gt_0)/len(all_results)*100:.1f}%)')
    print(f'最优α=0的类别数: {sum(1 for a in best_alphas if a == 0.0)}')
    print(f'最优α>0的类别数: {sum(1 for a in best_alphas if a > 0.0)}')

    # 相关性分析
    if len(all_results) >= 3:
        rho_m0_alpha, p_m0_alpha = stats.spearmanr(m_0_values, best_alphas)
        rho_m0_baseline, p_m0_baseline = stats.spearmanr(m_0_values, baseline_aps)
        rho_baseline_alpha, p_baseline_alpha = stats.spearmanr(baseline_aps, best_alphas)

        print(f'\n相关性分析:')
        print(f'  m^0 vs 最优α: ρ={rho_m0_alpha:.4f}, p={p_m0_alpha:.4f}')
        print(f'  m^0 vs baseline AP: ρ={rho_m0_baseline:.4f}, p={p_m0_baseline:.4f}')
        print(f'  baseline AP vs 最优α: ρ={rho_baseline_alpha:.4f}, p={p_baseline_alpha:.4f}')

    # 分组：m^0高 vs m^0低
    if len(all_results) >= 4:
        median_m0 = np.median(m_0_values)
        high_m0 = [r for r in all_results if r['m_0'] >= median_m0]
        low_m0 = [r for r in all_results if r['m_0'] < median_m0]

        print(f'\n分组分析 (m^0中位数={median_m0:.4f}):')
        print(f'  高m^0组 (N={len(high_m0)}): 平均最优α={np.mean([r["best_alpha"] for r in high_m0]):.4f}, '
              f'α>0比例={sum(r["alpha_gt_0_better"] for r in high_m0)/len(high_m0)*100:.1f}%')
        print(f'  低m^0组 (N={len(low_m0)}): 平均最优α={np.mean([r["best_alpha"] for r in low_m0]):.4f}, '
              f'α>0比例={sum(r["alpha_gt_0_better"] for r in low_m0)/len(low_m0)*100:.1f}%')

    # 打印详细结果表
    print(f'\n{"=" * 80}')
    print('详细结果表')
    print(f'{"=" * 80}')
    print(f'{"源域":<10s} {"源类别":<15s} {"VOC类别":<15s} {"m^0":>8s} {"baselineAP":>11s} {"最优α":>7s} {"最优AP":>8s} {"α>0更好":>9s}')
    print('-' * 95)
    for r in all_results:
        print(f'{r["src"]:<10s} {r["src_cls"]:<15s} {r["target_cls"]:<15s} '
              f'{r["m_0"]:>8.4f} {r["baseline_ap"]:>11.4f} {r["best_alpha"]:>7.2f} '
              f'{r["best_ap"]:>8.4f} {"是" if r["alpha_gt_0_better"] else "否":>9s}')


if __name__ == '__main__':
    main()
