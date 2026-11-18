# -*- coding: utf-8 -*-
"""analyze_9classes_results.py — 分析9个类别版本的实验结果
扩数据后（增加person和van），重新分析门控+固定α的效果
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from physical_priors import get_combined_gate, get_gate_info, normalize_class

DATA = Path(r'C:\Users\张三\Desktop\实验数据')


def interpolate_ap(ap_curve, alpha):
    alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
    aps = [ap_curve[str(a)] for a in alphas_sorted]
    return float(np.interp(alpha, alphas_sorted, aps))


def main():
    print('=' * 80)
    print('分析9个类别版本的实验结果（增加person和van）')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    if not samples_file.exists():
        print(f'\n错误：样本数据文件不存在: {samples_file}')
        print('请先运行 extend_alpha_grid.py 生成样本数据')
        return

    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'\n总样本数: {len(all_samples)}')

    # 统计类别
    classes = sorted(set(normalize_class(s['target_cls']) for s in all_samples))
    print(f'类别数: {len(classes)}')
    print(f'类别: {classes}')

    # 统计域
    domains = sorted(set(s['target_domain'] for s in all_samples))
    print(f'目标域: {domains}')

    # Oracle
    oracle_gains = [s['oracle_gain'] for s in all_samples]
    print(f'\nOracle平均增益: {np.mean(oracle_gains):+.4f}')
    print(f'Oracle正例: {sum(g > 0 for g in oracle_gains)}/{len(oracle_gains)}')

    # 固定α基线
    print(f'\n{"=" * 80}')
    print('固定α基线（无门控）')
    print(f'{"=" * 80}')
    fixed_alphas = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    best_fixed = None
    for alpha in fixed_alphas:
        gains = []
        for s in all_samples:
            ap = interpolate_ap(s['ap_curve'], alpha)
            baseline = s['ap_curve']['0.0']
            gains.append(ap - baseline)
        mean_gain = np.mean(gains)
        pos = sum(g > 0 for g in gains)
        print(f'  α={alpha:.1f}: 平均增益={mean_gain:+.4f}, 正例={pos}/{len(gains)}')
        if best_fixed is None or mean_gain > best_fixed['mean_gain']:
            best_fixed = {'alpha': alpha, 'mean_gain': mean_gain, 'pos': pos}

    print(f'\n  最好的固定α: α={best_fixed["alpha"]}, 增益={best_fixed["mean_gain"]:+.4f}')

    # 门控+固定α
    print(f'\n{"=" * 80}')
    print('门控+固定α')
    print(f'{"=" * 80}')

    gate_types = ['size', 'viewpoint', 'size_viewpoint_and', 'size_viewpoint_or']
    best_gated = None

    for gate_type in gate_types:
        print(f'\n  --- 门控类型: {gate_type} ---')
        for alpha in fixed_alphas:
            gains = []
            alpha_values = []
            for s in all_samples:
                gate = get_combined_gate(s['target_cls'], gate_type)
                alpha_hat = gate * alpha
                alpha_values.append(alpha_hat)
                ap = interpolate_ap(s['ap_curve'], alpha_hat)
                baseline = s['ap_curve']['0.0']
                gains.append(ap - baseline)
            mean_gain = np.mean(gains)
            pos = sum(g > 0 for g in gains)
            n_open = sum(a > 0 for a in alpha_values)
            if alpha == 2.5:  # 只打印α=2.5的结果
                print(f'    α={alpha:.1f}: 增益={mean_gain:+.4f}, 正例={pos}/{len(gains)}, 开启={n_open}')
            if best_gated is None or mean_gain > best_gated['mean_gain']:
                best_gated = {
                    'gate_type': gate_type, 'alpha': alpha,
                    'mean_gain': mean_gain, 'pos': pos, 'n_open': n_open,
                }

    print(f'\n  最好的门控+固定α:')
    print(f'    门控类型: {best_gated["gate_type"]}')
    print(f'    α={best_gated["alpha"]}')
    print(f'    平均增益: {best_gated["mean_gain"]:+.4f}')
    print(f'    正例: {best_gated["pos"]}/{len(all_samples)}')
    print(f'    开启样本数: {best_gated["n_open"]}')

    # 最好配置的逐类结果
    print(f'\n{"=" * 80}')
    print(f'最好配置（{best_gated["gate_type"]}门控 + α={best_gated["alpha"]}）的逐类结果')
    print(f'{"=" * 80}')
    print(f'\n{"类别":<12} {"样本数":<8} {"门控":<8} {"平均增益":<12} {"Oracle增益":<12}')
    print('-' * 60)
    for cls in classes:
        cls_samples = [s for s in all_samples if normalize_class(s['target_cls']) == cls]
        if not cls_samples:
            continue
        gate = get_combined_gate(cls, best_gated['gate_type'])
        gains = []
        oracle_g = []
        for s in cls_samples:
            alpha_hat = gate * best_gated['alpha']
            ap = interpolate_ap(s['ap_curve'], alpha_hat)
            baseline = s['ap_curve']['0.0']
            gains.append(ap - baseline)
            oracle_g.append(s['oracle_gain'])
        print(f'{cls:<12} {len(cls_samples):<8} {"开" if gate > 0 else "关":<8} '
              f'{np.mean(gains):+12.4f} {np.mean(oracle_g):+12.4f}')

    # 分域结果
    print(f'\n{"=" * 80}')
    print('分域结果（最好配置）')
    print(f'{"=" * 80}')
    for domain in domains:
        domain_gains = []
        for s in all_samples:
            if s['target_domain'] != domain:
                continue
            gate = get_combined_gate(s['target_cls'], best_gated['gate_type'])
            alpha_hat = gate * best_gated['alpha']
            ap = interpolate_ap(s['ap_curve'], alpha_hat)
            baseline = s['ap_curve']['0.0']
            domain_gains.append(ap - baseline)
        print(f'  {domain}: {np.mean(domain_gains):+.4f} ({sum(g > 0 for g in domain_gains)}/{len(domain_gains)})')

    # Headroom捕获率
    print(f'\n{"=" * 80}')
    print('Headroom捕获率')
    print(f'{"=" * 80}')
    oracle_mean = np.mean(oracle_gains)
    best_fixed_mean = best_fixed['mean_gain']
    best_gated_mean = best_gated['mean_gain']
    headroom = oracle_mean - best_fixed_mean
    captured = best_gated_mean - best_fixed_mean
    capture_rate = captured / headroom * 100 if headroom > 0 else 0
    print(f'  Oracle: {oracle_mean:+.4f}')
    print(f'  最好的固定α（无门控）: {best_fixed_mean:+.4f}')
    print(f'  最好的门控+固定α: {best_gated_mean:+.4f}')
    print(f'  Headroom（Oracle - 最好固定α）: {headroom:+.4f}')
    print(f'  门控捕获的headroom: {captured:+.4f}')
    print(f'  Headroom捕获率: {capture_rate:.1f}%')

    # 总结
    print(f'\n{"=" * 80}')
    print('总结')
    print(f'{"=" * 80}')
    print(f'\n  数据规模: {len(all_samples)}个样本, {len(classes)}个类别, {len(domains)}个目标域')
    print(f'  门控开启的类别: {[cls for cls in classes if get_combined_gate(cls, best_gated["gate_type"]) > 0]}')
    print(f'  门控关闭的类别: {[cls for cls in classes if get_combined_gate(cls, best_gated["gate_type"]) == 0]}')
    print(f'  最好的方法: {best_gated["gate_type"]}门控 + α={best_gated["alpha"]}')
    print(f'  平均增益: {best_gated_mean:+.4f}')
    print(f'  Headroom捕获率: {capture_rate:.1f}%')

    # 保存结果
    results = {
        'n_samples': len(all_samples),
        'n_classes': len(classes),
        'classes': classes,
        'domains': domains,
        'oracle_mean': float(oracle_mean),
        'best_fixed_alpha': best_fixed,
        'best_gated': best_gated,
        'headroom': float(headroom),
        'captured': float(captured),
        'capture_rate': float(capture_rate),
    }
    results_file = DATA / 'analysis_9classes_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  结果已保存: {results_file}')


if __name__ == '__main__':
    main()
