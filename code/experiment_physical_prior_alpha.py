# -*- coding: utf-8 -*-
"""experiment_physical_prior_alpha.py — 物理先验驱动的α缩放
门控决定on/off，α由物理先验（尺寸+视角差）决定
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
    else:
        return 1.0


def compute_physical_alpha(target_cls, alpha_base=2.5, gamma=0.3, size_ref=10.0, use_vp=True):
    """物理先验驱动的α缩放
    α = alpha_base * (size_ref / size) ^ gamma * (viewpoint_diff if use_vp else 1)
    """
    cls_norm = normalize_class(target_cls)
    size = OBJECT_SIZE.get(cls_norm, size_ref)
    vp = VIEWPOINT_DIFF.get(cls_norm, 0.5)
    size_factor = (size_ref / max(size, 0.1)) ** gamma
    vp_factor = vp if use_vp else 1.0
    alpha = alpha_base * size_factor * vp_factor
    return float(np.clip(alpha, 0.0, 6.0))


def interpolate_ap(ap_curve, alpha):
    alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
    aps = [ap_curve[str(a)] for a in alphas_sorted]
    return float(np.interp(alpha, alphas_sorted, aps))


def main():
    print('=' * 80)
    print('实验：物理先验驱动的α缩放')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'\n总样本数: {len(all_samples)}')

    # Oracle
    oracle_gains = [s['oracle_gain'] for s in all_samples]
    print(f'\nOracle平均增益: {np.mean(oracle_gains):+.4f}')

    # 基线：门控+固定α=2.5
    print(f'\n{"=" * 80}')
    print('基线：门控+固定α=2.5')
    print(f'{"=" * 80}')
    gate_type = 'size_viewpoint_and'
    baseline_gains = []
    for s in all_samples:
        gate = get_gate(s['target_cls'], gate_type)
        alpha_hat = gate * 2.5
        ap = interpolate_ap(s['ap_curve'], alpha_hat)
        baseline = s['ap_curve']['0.0']
        baseline_gains.append(ap - baseline)
    print(f'  平均增益: {np.mean(baseline_gains):+.4f}')
    print(f'  正例: {sum(g > 0 for g in baseline_gains)}/{len(baseline_gains)}')

    # 物理先验驱动的α缩放（网格搜索超参）
    print(f'\n{"=" * 80}')
    print('物理先验驱动的α缩放（网格搜索超参）')
    print(f'{"=" * 80}')

    alpha_bases = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    gammas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    use_vp_options = [True, False]

    best_result = None
    results = []

    for alpha_base in alpha_bases:
        for gamma in gammas:
            for use_vp in use_vp_options:
                gains = []
                alpha_values = []
                for s in all_samples:
                    gate = get_gate(s['target_cls'], gate_type)
                    if gate > 0:
                        alpha_hat = compute_physical_alpha(s['target_cls'], alpha_base, gamma, 10.0, use_vp)
                    else:
                        alpha_hat = 0.0
                    alpha_values.append(alpha_hat)
                    ap = interpolate_ap(s['ap_curve'], alpha_hat)
                    baseline = s['ap_curve']['0.0']
                    gains.append(ap - baseline)
                mean_gain = np.mean(gains)
                pos = sum(g > 0 for g in gains)
                alpha_std = np.std([a for a in alpha_values if a > 0])
                results.append({
                    'alpha_base': alpha_base, 'gamma': gamma, 'use_vp': use_vp,
                    'mean_gain': mean_gain, 'pos': pos, 'alpha_std': alpha_std,
                })
                if best_result is None or mean_gain > best_result['mean_gain']:
                    best_result = results[-1]

    # 按增益排序，输出前10
    results.sort(key=lambda x: x['mean_gain'], reverse=True)
    print(f'\n  前10个配置:')
    print(f'  {"alpha_base":>10} {"gamma":>6} {"use_vp":>6} {"增益":>10} {"正例":>8} {"α_std":>8}')
    print('  ' + '-' * 55)
    for r in results[:10]:
        print(f'  {r["alpha_base"]:>10.1f} {r["gamma"]:>6.1f} {str(r["use_vp"]):>6} '
              f'{r["mean_gain"]:>+10.4f} {r["pos"]:>4}/{len(all_samples)} {r["alpha_std"]:>8.4f}')

    print(f'\n  最好的配置:')
    print(f'    alpha_base={best_result["alpha_base"]}, gamma={best_result["gamma"]}, use_vp={best_result["use_vp"]}')
    print(f'    平均增益: {best_result["mean_gain"]:+.4f}')
    print(f'    正例: {best_result["pos"]}/{len(all_samples)}')
    print(f'    α_std: {best_result["alpha_std"]:.4f}')

    # 最好配置的逐类α值
    print(f'\n  最好配置的逐类α值:')
    for cls_norm in sorted(set(normalize_class(s['target_cls']) for s in all_samples)):
        gate = get_gate(cls_norm, gate_type)
        if gate > 0:
            alpha = compute_physical_alpha(cls_norm, best_result['alpha_base'], best_result['gamma'], 10.0, best_result['use_vp'])
        else:
            alpha = 0.0
        print(f'    {cls_norm}: α={alpha:.2f} (gate={"开" if gate > 0 else "关"})')

    # 分域结果
    print(f'\n  分域结果（最好配置）:')
    for domain in ['COCO', 'VOC']:
        domain_gains = []
        for s in all_samples:
            if s['target_domain'] != domain:
                continue
            gate = get_gate(s['target_cls'], gate_type)
            if gate > 0:
                alpha_hat = compute_physical_alpha(s['target_cls'], best_result['alpha_base'], best_result['gamma'], 10.0, best_result['use_vp'])
            else:
                alpha_hat = 0.0
            ap = interpolate_ap(s['ap_curve'], alpha_hat)
            baseline = s['ap_curve']['0.0']
            domain_gains.append(ap - baseline)
        print(f'    {domain}: {np.mean(domain_gains):+.4f} ({sum(g > 0 for g in domain_gains)}/{len(domain_gains)})')

    # 保存结果
    results_to_save = {
        'baseline_gated_fixed_2_5': float(np.mean(baseline_gains)),
        'best_physical_prior': best_result,
        'oracle': float(np.mean(oracle_gains)),
        'top10_configs': results[:10],
    }
    results_file = DATA / 'physical_prior_alpha_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    print(f'\n  结果已保存: {results_file}')


if __name__ == '__main__':
    main()
