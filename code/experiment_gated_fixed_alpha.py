# -*- coding: utf-8 -*-
"""experiment_gated_fixed_alpha.py — 门控开启的类用更大的固定α
看看能否突破clip到2.0的限制
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

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
    alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
    aps = [ap_curve[str(a)] for a in alphas_sorted]
    return float(np.interp(alpha, alphas_sorted, aps))


def main():
    print('=' * 80)
    print('实验：门控开启的类用更大的固定α')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'\n总样本数: {len(all_samples)}')

    # Oracle
    oracle_gains = [s['oracle_gain'] for s in all_samples]
    print(f'\nOracle平均增益: {np.mean(oracle_gains):+.4f}')

    # 最好的固定α
    print(f'\n{"=" * 80}')
    print('最好的固定α（所有类用同一个α）')
    print(f'{"=" * 80}')
    best_constant = None
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
        gains = [interpolate_ap(s['ap_curve'], alpha) - s['ap_curve']['0.0'] for s in all_samples]
        mean_gain = np.mean(gains)
        pos = sum(g > 0 for g in gains)
        print(f'  α={alpha:.1f}: {mean_gain:+.4f} ({pos}/{len(gains)})')
        if best_constant is None or mean_gain > best_constant[1]:
            best_constant = (alpha, mean_gain)
    print(f'\n  最好的固定α: {best_constant[0]:.1f}, 增益: {best_constant[1]:+.4f}')

    # 各种门控 + 开启类用不同固定α
    print(f'\n{"=" * 80}')
    print('门控 + 开启类用固定α（关闭类用α=0）')
    print(f'{"=" * 80}')

    gate_types = ['size', 'viewpoint', 'size_viewpoint_and', 'size_viewpoint_or']
    alpha_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]

    best_overall = None
    for gate_type in gate_types:
        print(f'\n  --- {gate_type} ---')
        for alpha_open in alpha_values:
            gains = []
            opened_count = 0
            for s in all_samples:
                gate = get_gate(s['target_cls'], gate_type)
                if gate > 0:
                    opened_count += 1
                    alpha_hat = alpha_open
                else:
                    alpha_hat = 0.0
                ap = interpolate_ap(s['ap_curve'], alpha_hat)
                baseline = s['ap_curve']['0.0']
                gains.append(ap - baseline)
            mean_gain = np.mean(gains)
            pos = sum(g > 0 for g in gains)
            print(f'    α_open={alpha_open:.1f}: {mean_gain:+.4f} ({pos}/{len(gains)}), 开启{opened_count}个')
            if best_overall is None or mean_gain > best_overall[2]:
                best_overall = (gate_type, alpha_open, mean_gain, pos)

    print(f'\n{"=" * 80}')
    print(f'最好的组合: {best_overall[0]} + α_open={best_overall[1]:.1f}')
    print(f'  平均增益: {best_overall[2]:+.4f}')
    print(f'  正例: {best_overall[3]}/{len(all_samples)}')
    print(f'  Oracle: {np.mean(oracle_gains):+.4f}')
    print(f'  Headroom捕获率: {(best_overall[2] - best_constant[1]) / (np.mean(oracle_gains) - best_constant[1]) * 100:.1f}%')

    # 保存结果
    results = {
        'best_constant_alpha': float(best_constant[0]),
        'best_constant_gain': float(best_constant[1]),
        'oracle_gain': float(np.mean(oracle_gains)),
        'best_gate': best_overall[0],
        'best_alpha_open': float(best_overall[1]),
        'best_gain': float(best_overall[2]),
        'best_pos': int(best_overall[3]),
    }
    results_file = DATA / 'gated_fixed_alpha_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  结果已保存: {results_file}')


if __name__ == '__main__':
    main()
