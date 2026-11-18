# -*- coding: utf-8 -*-
"""detailed_gated_alpha_analysis.py — 详细分析门控+固定α的结果
分域、逐类、和闭式解对比
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

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
    else:
        return 1.0


def compute_alpha_closed_form(m0, mu, rho=0.13, mu_scale=0.01, alpha_max=2.0):
    mu_eff = mu * mu_scale
    alpha = max(0.0, (rho - m0) / max(mu_eff, 1e-8))
    return float(np.clip(alpha, 0.0, alpha_max))


def interpolate_ap(ap_curve, alpha):
    alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
    aps = [ap_curve[str(a)] for a in alphas_sorted]
    return float(np.interp(alpha, alphas_sorted, aps))


def main():
    print('=' * 80)
    print('详细分析：门控+固定α vs 门控+闭式解')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'\n总样本数: {len(all_samples)}')

    gate_type = 'size_viewpoint_and'
    alpha_open = 2.5

    # 计算各种方法的增益
    methods = {
        'baseline(α=0)': [],
        'fixed α=2.0': [],
        'fixed α=2.5': [],
        'closed_form(无门控)': [],
        '门控+闭式解(α_max=2.0)': [],
        '门控+固定α=2.5': [],
        'Oracle': [],
    }

    per_sample = []
    for s in all_samples:
        baseline = s['ap_curve']['0.0']
        gate = get_gate(s['target_cls'], gate_type)

        # 各种方法
        ap_0 = interpolate_ap(s['ap_curve'], 0.0)
        ap_2 = interpolate_ap(s['ap_curve'], 2.0)
        ap_25 = interpolate_ap(s['ap_curve'], 2.5)
        alpha_cf = compute_alpha_closed_form(s['m0'], s['mu'], rho=0.13, mu_scale=0.01, alpha_max=2.0)
        ap_cf = interpolate_ap(s['ap_curve'], alpha_cf)
        alpha_gated_cf = gate * alpha_cf
        ap_gated_cf = interpolate_ap(s['ap_curve'], alpha_gated_cf)
        alpha_gated_fixed = gate * alpha_open
        ap_gated_fixed = interpolate_ap(s['ap_curve'], alpha_gated_fixed)
        ap_best = s['best_ap']

        methods['baseline(α=0)'].append(ap_0 - baseline)
        methods['fixed α=2.0'].append(ap_2 - baseline)
        methods['fixed α=2.5'].append(ap_25 - baseline)
        methods['closed_form(无门控)'].append(ap_cf - baseline)
        methods['门控+闭式解(α_max=2.0)'].append(ap_gated_cf - baseline)
        methods['门控+固定α=2.5'].append(ap_gated_fixed - baseline)
        methods['Oracle'].append(ap_best - baseline)

        per_sample.append({
            'src': s['src'], 'target_cls': s['target_cls'], 'domain': s['target_domain'],
            'gate': gate, 'best_alpha': s['best_alpha'],
            'gated_fixed_gain': ap_gated_fixed - baseline,
            'gated_cf_gain': ap_gated_cf - baseline,
            'oracle_gain': ap_best - baseline,
        })

    # 总体结果
    print(f'\n{"=" * 80}')
    print('总体结果')
    print(f'{"=" * 80}')
    print(f'\n{"方法":<25} {"平均增益":>10} {"正例":>8} {"std":>8}')
    print('-' * 55)
    for name, gains in methods.items():
        mean_gain = np.mean(gains)
        pos = sum(g > 0 for g in gains)
        std = np.std(gains)
        print(f'{name:<25} {mean_gain:>+10.4f} {pos:>4}/{len(gains)} {std:>8.4f}')

    # 分域结果
    print(f'\n{"=" * 80}')
    print('分域结果')
    print(f'{"=" * 80}')
    for domain in ['COCO', 'VOC']:
        print(f'\n  --- {domain} ---')
        domain_indices = [i for i, s in enumerate(per_sample) if s['domain'] == domain]
        for name in ['门控+固定α=2.5', '门控+闭式解(α_max=2.0)', 'fixed α=2.5', 'Oracle']:
            gains = [methods[name][i] for i in domain_indices]
            mean_gain = np.mean(gains)
            pos = sum(g > 0 for g in gains)
            print(f'    {name:<25} {mean_gain:>+10.4f} ({pos}/{len(gains)})')

    # 逐类结果（门控开启的类 vs 关闭的类）
    print(f'\n{"=" * 80}')
    print('逐类结果（按门控分组）')
    print(f'{"=" * 80}')

    opened_classes = set()
    closed_classes = set()
    for s in per_sample:
        cls_norm = normalize_class(s['target_cls'])
        if s['gate'] > 0:
            opened_classes.add(cls_norm)
        else:
            closed_classes.add(cls_norm)

    print(f'\n  门控开启的类: {sorted(opened_classes)}')
    print(f'  门控关闭的类: {sorted(closed_classes)}')

    for cls_norm in sorted(opened_classes | closed_classes):
        cls_indices = [i for i, s in enumerate(per_sample) if normalize_class(s['target_cls']) == cls_norm]
        if not cls_indices:
            continue
        gate_val = per_sample[cls_indices[0]]['gate']
        gate_label = '开启' if gate_val > 0 else '关闭'
        gains_gated_fixed = [methods['门控+固定α=2.5'][i] for i in cls_indices]
        gains_oracle = [methods['Oracle'][i] for i in cls_indices]
        best_alphas = [per_sample[i]['best_alpha'] for i in cls_indices]
        print(f'\n  {cls_norm} ({gate_label}, {len(cls_indices)}个样本):')
        print(f'    门控+固定α=2.5: {np.mean(gains_gated_fixed):+.4f}')
        print(f'    Oracle: {np.mean(gains_oracle):+.4f}')
        print(f'    最优α: {best_alphas}')

    # 门控开启的类，α=2.5 vs 最优α的差距
    print(f'\n{"=" * 80}')
    print('门控开启的类：α=2.5 vs 最优α')
    print(f'{"=" * 80}')
    opened_indices = [i for i, s in enumerate(per_sample) if s['gate'] > 0]
    for i in opened_indices:
        s = per_sample[i]
        print(f'  {s["src"]}/{s["target_cls"]} ({s["domain"]}): '
              f'门控+α=2.5={s["gated_fixed_gain"]:+.4f}, '
              f'Oracle={s["oracle_gain"]:+.4f}, '
              f'最优α={s["best_alpha"]}')

    # 保存详细结果
    results = {
        'overall': {name: {'mean': float(np.mean(gains)), 'std': float(np.std(gains)),
                            'pos': int(sum(g > 0 for g in gains))}
                     for name, gains in methods.items()},
        'per_sample': per_sample,
        'best_method': '门控+固定α=2.5',
        'best_gain': float(np.mean(methods['门控+固定α=2.5'])),
    }
    results_file = DATA / 'detailed_gated_alpha_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  详细结果已保存: {results_file}')


if __name__ == '__main__':
    main()
