# -*- coding: utf-8 -*-
"""retune_closed_form.py — 重新调参闭式解（ALPHA_MAX=6.0）
看看能否让闭式解的效果接近固定α=2.5
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


def compute_alpha_closed_form(m0, mu, rho, mu_scale, alpha_max=6.0):
    mu_eff = mu * mu_scale
    alpha = max(0.0, (rho - m0) / max(mu_eff, 1e-8))
    return float(np.clip(alpha, 0.0, alpha_max))


def interpolate_ap(ap_curve, alpha):
    alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
    aps = [ap_curve[str(a)] for a in alphas_sorted]
    return float(np.interp(alpha, alphas_sorted, aps))


def main():
    print('=' * 80)
    print('重新调参闭式解（ALPHA_MAX=6.0）')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'\n总样本数: {len(all_samples)}')

    # Oracle
    oracle_gains = [s['oracle_gain'] for s in all_samples]
    print(f'\nOracle平均增益: {np.mean(oracle_gains):+.4f}')

    # 基线：门控+固定α=2.5
    gate_type = 'size_viewpoint_and'
    baseline_gains = []
    for s in all_samples:
        gate = get_gate(s['target_cls'], gate_type)
        alpha_hat = gate * 2.5
        ap = interpolate_ap(s['ap_curve'], alpha_hat)
        baseline = s['ap_curve']['0.0']
        baseline_gains.append(ap - baseline)
    print(f'\n基线（门控+固定α=2.5）: {np.mean(baseline_gains):+.4f}')

    # 重新调参闭式解（无门控）
    print(f'\n{"=" * 80}')
    print('闭式解（无门控）网格搜索')
    print(f'{"=" * 80}')

    rhos = [0.05, 0.1, 0.13, 0.15, 0.2, 0.25, 0.3]
    mu_scales = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    best_cf = None
    cf_results = []
    for rho in rhos:
        for mu_scale in mu_scales:
            gains = []
            alpha_values = []
            for s in all_samples:
                alpha_hat = compute_alpha_closed_form(s['m0'], s['mu'], rho, mu_scale, alpha_max=6.0)
                alpha_values.append(alpha_hat)
                ap = interpolate_ap(s['ap_curve'], alpha_hat)
                baseline = s['ap_curve']['0.0']
                gains.append(ap - baseline)
            mean_gain = np.mean(gains)
            pos = sum(g > 0 for g in gains)
            alpha_std = np.std(alpha_values)
            alpha_zero_ratio = sum(a == 0 for a in alpha_values) / len(alpha_values)
            cf_results.append({
                'rho': rho, 'mu_scale': mu_scale,
                'mean_gain': mean_gain, 'pos': pos,
                'alpha_std': alpha_std, 'alpha_zero_ratio': alpha_zero_ratio,
            })
            if best_cf is None or mean_gain > best_cf['mean_gain']:
                best_cf = cf_results[-1]

    cf_results.sort(key=lambda x: x['mean_gain'], reverse=True)
    print(f'\n  前10个配置:')
    print(f'  {"rho":>6} {"mu_scale":>10} {"增益":>10} {"正例":>8} {"α_std":>8} {"α=0比例":>10}')
    print('  ' + '-' * 60)
    for r in cf_results[:10]:
        print(f'  {r["rho"]:>6.2f} {r["mu_scale"]:>10.3f} '
              f'{r["mean_gain"]:>+10.4f} {r["pos"]:>4}/{len(all_samples)} '
              f'{r["alpha_std"]:>8.4f} {r["alpha_zero_ratio"]*100:>9.1f}%')

    print(f'\n  最好的闭式解配置:')
    print(f'    rho={best_cf["rho"]}, mu_scale={best_cf["mu_scale"]}')
    print(f'    平均增益: {best_cf["mean_gain"]:+.4f}')
    print(f'    正例: {best_cf["pos"]}/{len(all_samples)}')

    # 重新调参闭式解（+门控）
    print(f'\n{"=" * 80}')
    print('闭式解+门控 网格搜索')
    print(f'{"=" * 80}')

    best_gated_cf = None
    gated_cf_results = []
    for rho in rhos:
        for mu_scale in mu_scales:
            gains = []
            alpha_values = []
            for s in all_samples:
                gate = get_gate(s['target_cls'], gate_type)
                alpha_cf = compute_alpha_closed_form(s['m0'], s['mu'], rho, mu_scale, alpha_max=6.0)
                alpha_hat = gate * alpha_cf
                alpha_values.append(alpha_hat)
                ap = interpolate_ap(s['ap_curve'], alpha_hat)
                baseline = s['ap_curve']['0.0']
                gains.append(ap - baseline)
            mean_gain = np.mean(gains)
            pos = sum(g > 0 for g in gains)
            alpha_std = np.std([a for a in alpha_values if a > 0])
            gated_cf_results.append({
                'rho': rho, 'mu_scale': mu_scale,
                'mean_gain': mean_gain, 'pos': pos, 'alpha_std': alpha_std,
            })
            if best_gated_cf is None or mean_gain > best_gated_cf['mean_gain']:
                best_gated_cf = gated_cf_results[-1]

    gated_cf_results.sort(key=lambda x: x['mean_gain'], reverse=True)
    print(f'\n  前10个配置:')
    print(f'  {"rho":>6} {"mu_scale":>10} {"增益":>10} {"正例":>8} {"α_std":>8}')
    print('  ' + '-' * 50)
    for r in gated_cf_results[:10]:
        print(f'  {r["rho"]:>6.2f} {r["mu_scale"]:>10.3f} '
              f'{r["mean_gain"]:>+10.4f} {r["pos"]:>4}/{len(all_samples)} '
              f'{r["alpha_std"]:>8.4f}')

    print(f'\n  最好的门控+闭式解配置:')
    print(f'    rho={best_gated_cf["rho"]}, mu_scale={best_gated_cf["mu_scale"]}')
    print(f'    平均增益: {best_gated_cf["mean_gain"]:+.4f}')
    print(f'    正例: {best_gated_cf["pos"]}/{len(all_samples)}')

    # 最好的门控+闭式解配置的逐类α值
    print(f'\n  最好配置的逐类α值:')
    for cls_norm in sorted(set(normalize_class(s['target_cls']) for s in all_samples)):
        gate = get_gate(cls_norm, gate_type)
        # 找一个该类的样本，计算α
        for s in all_samples:
            if normalize_class(s['target_cls']) == cls_norm:
                alpha_cf = compute_alpha_closed_form(s['m0'], s['mu'], best_gated_cf['rho'], best_gated_cf['mu_scale'], alpha_max=6.0)
                alpha_hat = gate * alpha_cf
                print(f'    {cls_norm}: α={alpha_hat:.2f} (gate={"开" if gate > 0 else "关"}, cf={alpha_cf:.2f})')
                break

    # 总结
    print(f'\n{"=" * 80}')
    print('总结')
    print(f'{"=" * 80}')
    print(f'\n  Oracle: {np.mean(oracle_gains):+.4f}')
    print(f'  门控+固定α=2.5: {np.mean(baseline_gains):+.4f}')
    print(f'  最好的闭式解（无门控）: {best_cf["mean_gain"]:+.4f}')
    print(f'  最好的门控+闭式解: {best_gated_cf["mean_gain"]:+.4f}')

    if best_gated_cf['mean_gain'] >= np.mean(baseline_gains) - 0.001:
        print(f'\n  ✅ 门控+闭式解的效果接近门控+固定α=2.5，可以保留理论创新度！')
    else:
        print(f'\n  ⚠️ 门控+闭式解的效果比门控+固定α=2.5差，需要考虑其他方案')

    # 保存结果
    results = {
        'oracle': float(np.mean(oracle_gains)),
        'baseline_gated_fixed_2_5': float(np.mean(baseline_gains)),
        'best_closed_form': best_cf,
        'best_gated_closed_form': best_gated_cf,
    }
    results_file = DATA / 'retuned_closed_form_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  结果已保存: {results_file}')


if __name__ == '__main__':
    main()
