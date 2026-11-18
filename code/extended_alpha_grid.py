# -*- coding: utf-8 -*-
"""extended_alpha_grid.py — 扩展α网格到6.0，重搜所有样本
1. 把α网格从[0,0.1,...,2.0]扩展到[0,0.1,...,2.0,2.5,3.0,4.0,5.0,6.0]
2. 在COCO和VOC目标域上重跑所有样本
3. 检查α*是否还顶在上界
4. 检查可辨识性是否改善
5. 计算ΔAP(α)曲线，用于后续换靶子分析
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np

from sota_comparison import (
    SOURCE_LOADERS,
    DIOR_TARGETS, NWPU_TARGETS, DOTA_TARGETS, VISDRONE_TARGETS,
    build_proto_tge, evaluate_with_features,
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

REMOTE_TARGETS = {
    'DIOR': DIOR_TARGETS,
    'NWPU': NWPU_TARGETS,
    'DOTA': DOTA_TARGETS,
    'VisDrone': VISDRONE_TARGETS,
}

# 扩展的α网格
EXTENDED_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]


def compute_sim_factors(target_data, src_proto):
    """计算sim_pos_mean和sim_pos_std"""
    features = target_data['features']
    sim_pos = features @ src_proto
    return {
        'sim_pos_mean': float(np.mean(sim_pos)),
        'sim_pos_std': float(np.std(sim_pos)),
    }


def process_sample(src_name, src_cls, target_cls, target_domain, target_loader_func):
    """处理单个样本"""
    # 加载源域
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']

    # 负原型
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

    # 加载目标域
    target_data = target_loader_func(target_cls, n=200)
    if target_data is None:
        return None

    # 计算sim因子
    sim_factors = compute_sim_factors(target_data, src_proto)

    # 扩展α网格搜索
    ap_curve = {}
    for alpha in EXTENDED_ALPHAS:
        if alpha == 0.0:
            proto = src_proto
        else:
            proto = build_proto_tge(src_proto, src_cls, target_cls, beta=alpha)
        ap = evaluate_with_features(target_data, proto, neg_proto=proto_neg)
        ap_curve[str(alpha)] = float(ap)

    # 找最优α
    best_alpha = EXTENDED_ALPHAS[0]
    best_ap = ap_curve[str(EXTENDED_ALPHAS[0])]
    for alpha in EXTENDED_ALPHAS:
        if ap_curve[str(alpha)] > best_ap:
            best_ap = ap_curve[str(alpha)]
            best_alpha = alpha

    # 计算可辨识性：平台宽度（AP在峰值±0.005内的α范围）
    threshold = 0.005
    plateau_alphas = [a for a in EXTENDED_ALPHAS if abs(ap_curve[str(a)] - best_ap) <= threshold]
    plateau_width = max(plateau_alphas) - min(plateau_alphas) if plateau_alphas else 0
    identifiable = plateau_width < 1.0  # 平台宽度小于1.0才算可辨识

    # 计算ΔAP曲线
    baseline_ap = ap_curve['0.0']
    delta_ap_curve = {str(a): ap_curve[str(a)] - baseline_ap for a in EXTENDED_ALPHAS}

    result = {
        'src': src_name,
        'src_cls': src_cls,
        'target_cls': target_cls,
        'target_domain': target_domain,
        'sim_pos_mean': sim_factors['sim_pos_mean'],
        'sim_pos_std': sim_factors['sim_pos_std'],
        'best_alpha': float(best_alpha),
        'best_ap': float(best_ap),
        'baseline_ap': float(baseline_ap),
        'plateau_width': float(plateau_width),
        'identifiable': identifiable,
        'hit_upper_bound': best_alpha == EXTENDED_ALPHAS[-1],
        'hit_lower_bound': best_alpha == EXTENDED_ALPHAS[0],
        'ap_curve': ap_curve,
        'delta_ap_curve': delta_ap_curve,
    }

    return result


def main():
    print('=' * 80)
    print('扩展α网格到6.0，重搜所有样本')
    print('=' * 80)
    print(f'\n扩展α网格: {EXTENDED_ALPHAS}')

    all_results = []

    # COCO目标域
    from sota_comparison import load_target_coco
    print(f'\n--- COCO目标域 ---')
    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            print(f'  处理 {src_name}/{src_cls} -> COCO/{coco_cls}...')
            result = process_sample(src_name, src_cls, coco_cls, 'COCO', load_target_coco)
            if result:
                all_results.append(result)
                print(f'    best α={result["best_alpha"]:.1f}, baseline={result["baseline_ap"]:.4f}, '
                      f'best={result["best_ap"]:.4f}, plateau={result["plateau_width"]:.1f}, '
                      f'identifiable={result["identifiable"]}, hit_upper={result["hit_upper_bound"]}')

    # VOC目标域
    from voc_sota_comparison import load_target_voc
    COCO2VOC = {
        'airplane': 'aeroplane', 'boat': 'boat', 'car': 'car',
        'bus': 'bus', 'motorcycle': 'motorbike', 'bicycle': 'bicycle',
    }
    print(f'\n--- VOC目标域 ---')
    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            if coco_cls not in COCO2VOC:
                continue
            voc_cls = COCO2VOC[coco_cls]
            print(f'  处理 {src_name}/{src_cls} -> VOC/{voc_cls}...')
            result = process_sample(src_name, src_cls, voc_cls, 'VOC', load_target_voc)
            if result:
                all_results.append(result)
                print(f'    best α={result["best_alpha"]:.1f}, baseline={result["baseline_ap"]:.4f}, '
                      f'best={result["best_ap"]:.4f}, plateau={result["plateau_width"]:.1f}, '
                      f'identifiable={result["identifiable"]}, hit_upper={result["hit_upper_bound"]}')

    # 汇总统计
    print(f'\n{"=" * 80}')
    print('汇总统计')
    print(f'{"=" * 80}')

    n_total = len(all_results)
    n_hit_upper = sum(1 for r in all_results if r['hit_upper_bound'])
    n_hit_lower = sum(1 for r in all_results if r['hit_lower_bound'])
    n_identifiable = sum(1 for r in all_results if r['identifiable'])
    n_interior = n_total - n_hit_upper - n_hit_lower

    print(f'\n总样本数: {n_total}')
    print(f'α*=6.0（触上界）: {n_hit_upper} ({n_hit_upper/n_total*100:.1f}%)')
    print(f'α*=0（触下界）: {n_hit_lower} ({n_hit_lower/n_total*100:.1f}%)')
    print(f'内部点: {n_interior} ({n_interior/n_total*100:.1f}%)')
    print(f'可辨识（平台宽度<1.0）: {n_identifiable} ({n_identifiable/n_total*100:.1f}%)')

    # 分数据集
    for domain in ['COCO', 'VOC']:
        subset = [r for r in all_results if r['target_domain'] == domain]
        n = len(subset)
        n_upper = sum(1 for r in subset if r['hit_upper_bound'])
        n_id = sum(1 for r in subset if r['identifiable'])
        print(f'\n  {domain} (N={n}):')
        print(f'    触上界: {n_upper} ({n_upper/n*100:.1f}%)')
        print(f'    可辨识: {n_id} ({n_id/n*100:.1f}%)')

    # 保存结果
    with open(DATA / 'extended_alpha_grid_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'extended_alphas': EXTENDED_ALPHAS,
            'results': all_results,
            'summary': {
                'n_total': n_total,
                'n_hit_upper': n_hit_upper,
                'n_hit_lower': n_hit_lower,
                'n_interior': n_interior,
                'n_identifiable': n_identifiable,
            }
        }, f, indent=2, ensure_ascii=False)

    print(f'\n结果已保存到: {DATA / "extended_alpha_grid_results.json"}')


if __name__ == '__main__':
    main()
