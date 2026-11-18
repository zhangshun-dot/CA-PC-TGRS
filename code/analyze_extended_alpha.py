# -*- coding: utf-8 -*-
"""analyze_extended_alpha.py — 分析扩α网格后的结果
闭式解clip回[0,2.0]，用size_viewpoint_and门控
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# 物理先验知识库
OBJECT_SIZE = {
    'airplane': 30.0, 'boat': 15.0, 'car': 4.5, 'bus': 12.0,
    'motorcycle': 2.0, 'bicycle': 1.8, 'truck': 12.0,
    'aeroplane': 30.0, 'motorbike': 2.0,
}
VIEWPOINT_DIFF = {
    'airplane': 0.3, 'boat': 0.4, 'car': 0.7, 'bus': 0.8,
    'motorcycle': 0.6, 'bicycle': 0.5, 'truck': 0.7,
    'aeroplane': 0.3, 'motorbike': 0.6,
}


def normalize_class(cls):
    mapping = {'aeroplane': 'airplane', 'motorbike': 'motorcycle'}
    return mapping.get(cls, cls)


def compute_alpha_closed_form(m0, mu, rho=0.13, mu_scale=0.01, alpha_max=2.0):
    mu_eff = mu * mu_scale
    alpha = max(0.0, (rho - m0) / max(mu_eff, 1e-8))
    return float(np.clip(alpha, 0.0, alpha_max))


def get_gate(target_cls, gate_type='size_viewpoint_and'):
    cls_norm = normalize_class(target_cls)
    size = OBJECT_SIZE.get(cls_norm, 10.0)
    vp = VIEWPOINT_DIFF.get(cls_norm, 0.5)

    if gate_type == 'size':
        return 1.0 if size < 10.0 else 0.0
    elif gate_type == 'viewpoint':
        return 1.0 if vp > 0.5 else 0.0
    elif gate_type == 'size_viewpoint_and':
        return 1.0 if (size < 10.0 and vp > 0.5) else 0.0
    elif gate_type == 'size_viewpoint_or':
        return 1.0 if (size < 10.0 or vp > 0.5) else 0.0
    else:
        return 1.0


def interpolate_ap(ap_curve, alpha):
    """在AP曲线上插值得到指定α的AP"""
    alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
    aps = [ap_curve[str(a)] for a in alphas_sorted]
    return float(np.interp(alpha, alphas_sorted, aps))


def main():
    print('=' * 80)
    print('分析扩α网格后的结果（闭式解clip回[0,2.0]）')
    print('=' * 80)

    # 加载扩网格后的数据
    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'\n总样本数: {len(all_samples)}')

    # α=2.0 vs 最优α
    print(f'\n{"=" * 80}')
    print('α=2.0 vs 最优α（扩网格到6.0）')
    print(f'{"=" * 80}')

    gains_at_2 = []
    gains_at_best = []
    best_alphas = []
    for s in all_samples:
        ap_2 = s['ap_curve'].get('2.0', s['ap_curve'].get('2'))
        baseline = s['ap_curve']['0.0']
        gains_at_2.append(ap_2 - baseline)
        gains_at_best.append(s['oracle_gain'])
        best_alphas.append(s['best_alpha'])

    print(f'\n  α=2.0时平均增益: {np.mean(gains_at_2):+.4f}')
    print(f'  最优α时平均增益: {np.mean(gains_at_best):+.4f}')
    print(f'  提升: {np.mean(gains_at_best) - np.mean(gains_at_2):+.4f}')
    print(f'  最优α分布:')
    from collections import Counter
    alpha_counts = Counter(best_alphas)
    for alpha in sorted(alpha_counts.keys()):
        count = alpha_counts[alpha]
        print(f'    α={alpha:.1f}: {count}个样本 ({count/len(best_alphas)*100:.1f}%)')

    # 闭式解+各种门控的效果
    print(f'\n{"=" * 80}')
    print('闭式解+各种门控的效果（ρ=0.13, mu_scale=0.01, α_max=2.0）')
    print(f'{"=" * 80}')

    gate_types = ['none', 'size', 'viewpoint', 'size_viewpoint_and', 'size_viewpoint_or']

    for gate_type in gate_types:
        gains = []
        alpha_hats = []
        for s in all_samples:
            alpha_closed = compute_alpha_closed_form(s['m0'], s['mu'], rho=0.13, mu_scale=0.01, alpha_max=2.0)
            if gate_type == 'none':
                gate = 1.0
            else:
                gate = get_gate(s['target_cls'], gate_type)
            alpha_hat = gate * alpha_closed
            alpha_hats.append(alpha_hat)
            ap = interpolate_ap(s['ap_curve'], alpha_hat)
            baseline = s['ap_curve']['0.0']
            gains.append(ap - baseline)

        mean_gain = np.mean(gains)
        pos_count = sum(g > 0 for g in gains)
        alpha_std = np.std(alpha_hats)
        alpha_zero_ratio = sum(a == 0 for a in alpha_hats) / len(alpha_hats)

        print(f'\n  {gate_type}:')
        print(f'    平均增益: {mean_gain:+.4f}')
        print(f'    正例: {pos_count}/{len(gains)}')
        print(f'    α̂ std: {alpha_std:.4f}')
        print(f'    α̂=0比例: {alpha_zero_ratio*100:.1f}%')

        # 分域结果
        for domain in ['COCO', 'VOC']:
            domain_gains = [g for g, s in zip(gains, all_samples) if s['target_domain'] == domain]
            print(f'    {domain}: {np.mean(domain_gains):+.4f} ({sum(g > 0 for g in domain_gains)}/{len(domain_gains)})')

    # Oracle的headroom分析
    print(f'\n{"=" * 80}')
    print('Headroom分析')
    print(f'{"=" * 80}')

    # 最好的固定α
    best_constant_alpha = 0.0
    best_constant_gain = -float('inf')
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
        gains = []
        for s in all_samples:
            ap = interpolate_ap(s['ap_curve'], alpha)
            baseline = s['ap_curve']['0.0']
            gains.append(ap - baseline)
        mean_gain = np.mean(gains)
        if mean_gain > best_constant_gain:
            best_constant_gain = mean_gain
            best_constant_alpha = alpha

    oracle_gain = np.mean(gains_at_best)
    headroom = oracle_gain - best_constant_gain

    print(f'\n  最好的固定α: {best_constant_alpha:.1f}, 增益: {best_constant_gain:+.4f}')
    print(f'  Oracle增益: {oracle_gain:+.4f}')
    print(f'  Headroom: {headroom:+.4f}')

    # 最好的方法（size_viewpoint_and）的headroom捕获率
    best_method_gain = None
    for gate_type in gate_types:
        gains = []
        for s in all_samples:
            alpha_closed = compute_alpha_closed_form(s['m0'], s['mu'], rho=0.13, mu_scale=0.01, alpha_max=2.0)
            if gate_type == 'none':
                gate = 1.0
            else:
                gate = get_gate(s['target_cls'], gate_type)
            alpha_hat = gate * alpha_closed
            ap = interpolate_ap(s['ap_curve'], alpha_hat)
            baseline = s['ap_curve']['0.0']
            gains.append(ap - baseline)
        mean_gain = np.mean(gains)
        if best_method_gain is None or mean_gain > best_method_gain:
            best_method_gain = mean_gain
            best_method_name = gate_type

    capture_rate = (best_method_gain - best_constant_gain) / headroom * 100 if headroom > 0 else 0
    print(f'\n  最好的方法: {best_method_name}, 增益: {best_method_gain:+.4f}')
    print(f'  Headroom捕获率: {capture_rate:.1f}%')

    # 保存结果
    results = {
        'gains_at_2_mean': float(np.mean(gains_at_2)),
        'gains_at_best_mean': float(oracle_gain),
        'best_constant_alpha': float(best_constant_alpha),
        'best_constant_gain': float(best_constant_gain),
        'headroom': float(headroom),
        'best_method': best_method_name,
        'best_method_gain': float(best_method_gain),
        'capture_rate': float(capture_rate),
    }
    results_file = DATA / 'extended_alpha_analysis.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  结果已保存: {results_file}')


if __name__ == '__main__':
    main()
