# -*- coding: utf-8 -*-
"""quantile_gate_analysis.py — 域内分位数门控 vs 硬门控对比分析
在三个检测器（GDINO-tiny, OWLv2, GDINO-base）上比较：
1. 硬门控（size<10 AND vp>0.5）
2. 域内分位数门控（baseline低于分位数阈值则开启）
3. 物理先验+分位数混合门控
"""
import json
import numpy as np
from pathlib import Path

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# 物理先验
OBJECT_SIZE = {'airplane': 30.0, 'boat': 15.0, 'car': 4.5, 'bus': 12.0,
               'truck': 12.0, 'motorcycle': 2.0, 'bicycle': 1.8, 'person': 1.7}
VIEWPOINT_DIFF = {'airplane': 0.3, 'boat': 0.4, 'car': 0.7, 'bus': 0.8,
                   'truck': 0.75, 'motorcycle': 0.6, 'bicycle': 0.5, 'person': 0.9}


def hard_gate(cls):
    """硬门控：size < 10 AND vp > 0.5"""
    return 1 if (OBJECT_SIZE.get(cls, 999) < 10.0 and VIEWPOINT_DIFF.get(cls, 0) > 0.5) else 0


def quantile_gate(baselines, cls, quantile=0.5):
    """域内分位数门控：baseline低于分位数阈值则开启"""
    threshold = np.quantile(list(baselines.values()), quantile)
    return 1 if baselines[cls] < threshold else 0


def mixed_gate(baselines, cls, quantile=0.5):
    """混合门控：物理先验开启 OR baseline低于分位数"""
    return 1 if (hard_gate(cls) == 1 or baselines[cls] < np.quantile(list(baselines.values()), quantile)) else 0


def evaluate_gate(results, gate_func, gate_name):
    """评估门控效果"""
    total_gain = 0
    total_oracle = 0
    on_gains = []
    off_gains = []
    correct_on = 0
    correct_off = 0
    false_positive = 0
    false_negative = 0

    for r in results:
        cls = r['target_cls']
        gate = gate_func(cls)
        gain_25 = r['gain_25']
        oracle = r['oracle_gain']

        if gate == 1:
            total_gain += gain_25
            on_gains.append(gain_25)
            if gain_25 > 0.001:
                correct_on += 1
            else:
                false_positive += 1
        else:
            off_gains.append(gain_25)
            if gain_25 <= 0.001:
                correct_off += 1
            else:
                false_negative += 1

        total_oracle += oracle

    n = len(results)
    avg_gain = total_gain / n
    avg_oracle = total_oracle / n

    # 如果不门控（全部用α=2.5）
    avg_no_gate = np.mean([r['gain_25'] for r in results])

    # 门控带来的改进
    improvement = avg_gain - avg_no_gate

    return {
        'gate_name': gate_name,
        'n_on': len(on_gains),
        'n_off': len(off_gains),
        'avg_gain': avg_gain,
        'avg_oracle': avg_oracle,
        'avg_no_gate': avg_no_gate,
        'improvement': improvement,
        'on_avg_gain': np.mean(on_gains) if on_gains else 0,
        'off_avg_gain': np.mean(off_gains) if off_gains else 0,
        'separation_ratio': (np.mean(on_gains) / abs(np.mean(off_gains))) if (on_gains and off_gains and np.mean(off_gains) != 0) else float('inf'),
        'correct_on': correct_on,
        'correct_off': correct_off,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'precision': correct_on / (correct_on + false_positive) if (correct_on + false_positive) > 0 else 0,
        'recall': correct_on / (correct_on + false_negative) if (correct_on + false_negative) > 0 else 0,
    }


def load_detector_results(detector_name):
    """加载指定检测器的结果"""
    if detector_name == 'gdino_tiny':
        # 从主实验数据中提取
        with open(DATA / 'zero_shot_gated_samples_extended_alpha.json', 'r') as f:
            data = json.load(f)
        results = []
        for sample in data:
            if sample['target_cls'] == 'person':
                continue  # 跳过person
            ap_curve = sample.get('ap_curve', {})
            if not ap_curve:
                continue
            baseline = float(sample.get('baseline_ap', ap_curve.get('0.0', 0)))
            best_alpha = sample.get('best_alpha', max(ap_curve.keys(), key=lambda k: float(ap_curve[k])))
            best_ap = float(sample.get('best_ap', ap_curve[str(best_alpha)]))
            gain_25 = float(ap_curve.get('2.5', baseline)) - baseline
            results.append({
                'target_cls': sample['target_cls'],
                'baseline_ap': baseline,
                'best_alpha': float(best_alpha),
                'oracle_gain': float(sample.get('oracle_gain', best_ap - baseline)),
                'gain_25': gain_25,
            })
        # 按类别去重（取平均）
        from collections import defaultdict
        cls_results = defaultdict(list)
        for r in results:
            cls_results[r['target_cls']].append(r)
        avg_results = []
        for cls, rs in cls_results.items():
            avg_results.append({
                'target_cls': cls,
                'baseline_ap': np.mean([r['baseline_ap'] for r in rs]),
                'best_alpha': np.mean([r['best_alpha'] for r in rs]),
                'oracle_gain': np.mean([r['oracle_gain'] for r in rs]),
                'gain_25': np.mean([r['gain_25'] for r in rs]),
            })
        return avg_results
    elif detector_name == 'owlv2':
        with open(DATA / 'owlv2_cross_detector_results.json', 'r') as f:
            return json.load(f)
    elif detector_name == 'gdino_base':
        with open(DATA / 'gdino_base_cross_detector_results.json', 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f'Unknown detector: {detector_name}')


def main():
    detectors = ['gdino_tiny', 'owlv2', 'gdino_base']
    detector_labels = {
        'gdino_tiny': 'GDINO-tiny',
        'owlv2': 'OWLv2',
        'gdino_base': 'GDINO-base',
    }

    all_results = {}
    for det in detectors:
        all_results[det] = load_detector_results(det)

    print('=' * 100)
    print('域内分位数门控 vs 硬门控对比分析')
    print('=' * 100)

    # 1. 各检测器的baseline分布
    print('\n' + '=' * 100)
    print('1. 各检测器的baseline分布')
    print('=' * 100)
    for det in detectors:
        results = all_results[det]
        baselines = [r['baseline_ap'] for r in results]
        print(f'\n{detector_labels[det]}:')
        print(f'  范围: [{min(baselines):.4f}, {max(baselines):.4f}]')
        print(f'  中位数: {np.median(baselines):.4f}')
        print(f'  均值: {np.mean(baselines):.4f}')
        print(f'  下四分位数: {np.quantile(baselines, 0.25):.4f}')
        print(f'  上四分位数: {np.quantile(baselines, 0.75):.4f}')
        print(f'  各类baseline:')
        for r in sorted(results, key=lambda x: x['baseline_ap']):
            print(f'    {r["target_cls"]:<12}: {r["baseline_ap"]:.4f} (gain_25={r["gain_25"]:+.4f})')

    # 2. 硬门控效果
    print('\n' + '=' * 100)
    print('2. 硬门控（size<10 AND vp>0.5）效果')
    print('=' * 100)
    for det in detectors:
        results = all_results[det]
        eval_result = evaluate_gate(results, lambda cls: hard_gate(cls), '硬门控')
        print(f'\n{detector_labels[det]}:')
        print(f'  开启类: {eval_result["n_on"]}个, 平均增益: {eval_result["on_avg_gain"]:+.4f}')
        print(f'  关闭类: {eval_result["n_off"]}个, 平均增益: {eval_result["off_avg_gain"]:+.4f}')
        print(f'  分离比: {eval_result["separation_ratio"]:.1f}x')
        print(f'  门控后平均增益: {eval_result["avg_gain"]:+.4f}')
        print(f'  不门控平均增益: {eval_result["avg_no_gate"]:+.4f}')
        print(f'  门控改进: {eval_result["improvement"]:+.4f}')
        print(f'  精确率: {eval_result["precision"]:.1%}, 召回率: {eval_result["recall"]:.1%}')
        print(f'  假阳性: {eval_result["false_positive"]}, 假阴性: {eval_result["false_negative"]}')

    # 3. 域内分位数门控效果（不同分位数）
    print('\n' + '=' * 100)
    print('3. 域内分位数门控效果（不同分位数阈值）')
    print('=' * 100)
    for quantile in [0.25, 0.4, 0.5, 0.6, 0.75]:
        print(f'\n--- 分位数阈值: {quantile} ---')
        for det in detectors:
            results = all_results[det]
            baselines = {r['target_cls']: r['baseline_ap'] for r in results}
            eval_result = evaluate_gate(
                results,
                lambda cls: quantile_gate(baselines, cls, quantile),
                f'分位数门控(q={quantile})'
            )
            print(f'  {detector_labels[det]:<12}: 开启{eval_result["n_on"]}个, '
                  f'门控增益{eval_result["avg_gain"]:+.4f}, '
                  f'不门控{eval_result["avg_no_gate"]:+.4f}, '
                  f'改进{eval_result["improvement"]:+.4f}, '
                  f'分离比{eval_result["separation_ratio"]:.1f}x, '
                  f'精确率{eval_result["precision"]:.1%}, 召回率{eval_result["recall"]:.1%}')

    # 4. 混合门控效果（物理先验 OR 分位数）
    print('\n' + '=' * 100)
    print('4. 混合门控效果（物理先验开启 OR baseline低于分位数）')
    print('=' * 100)
    for quantile in [0.25, 0.4, 0.5, 0.6, 0.75]:
        print(f'\n--- 分位数阈值: {quantile} ---')
        for det in detectors:
            results = all_results[det]
            baselines = {r['target_cls']: r['baseline_ap'] for r in results}
            eval_result = evaluate_gate(
                results,
                lambda cls: mixed_gate(baselines, cls, quantile),
                f'混合门控(q={quantile})'
            )
            print(f'  {detector_labels[det]:<12}: 开启{eval_result["n_on"]}个, '
                  f'门控增益{eval_result["avg_gain"]:+.4f}, '
                  f'不门控{eval_result["avg_no_gate"]:+.4f}, '
                  f'改进{eval_result["improvement"]:+.4f}, '
                  f'分离比{eval_result["separation_ratio"]:.1f}x, '
                  f'精确率{eval_result["precision"]:.1%}, 召回率{eval_result["recall"]:.1%}')

    # 5. 跨检测器稳定性对比
    print('\n' + '=' * 100)
    print('5. 跨检测器稳定性对比')
    print('=' * 100)

    # 硬门控
    hard_improvements = []
    for det in detectors:
        results = all_results[det]
        eval_result = evaluate_gate(results, lambda cls: hard_gate(cls), '硬门控')
        hard_improvements.append(eval_result['improvement'])
    print(f'\n硬门控:')
    print(f'  各检测器改进: {[f"{v:+.4f}" for v in hard_improvements]}')
    print(f'  改进均值: {np.mean(hard_improvements):+.4f}')
    print(f'  改进标准差: {np.std(hard_improvements):.4f}')
    print(f'  改进范围: [{min(hard_improvements):+.4f}, {max(hard_improvements):+.4f}]')

    # 分位数门控（q=0.5）
    quantile_improvements = []
    for det in detectors:
        results = all_results[det]
        baselines = {r['target_cls']: r['baseline_ap'] for r in results}
        eval_result = evaluate_gate(
            results,
            lambda cls: quantile_gate(baselines, cls, 0.5),
            '分位数门控(q=0.5)'
        )
        quantile_improvements.append(eval_result['improvement'])
    print(f'\n分位数门控(q=0.5):')
    print(f'  各检测器改进: {[f"{v:+.4f}" for v in quantile_improvements]}')
    print(f'  改进均值: {np.mean(quantile_improvements):+.4f}')
    print(f'  改进标准差: {np.std(quantile_improvements):.4f}')
    print(f'  改进范围: [{min(quantile_improvements):+.4f}, {max(quantile_improvements):+.4f}]')

    # 混合门控（q=0.5）
    mixed_improvements = []
    for det in detectors:
        results = all_results[det]
        baselines = {r['target_cls']: r['baseline_ap'] for r in results}
        eval_result = evaluate_gate(
            results,
            lambda cls: mixed_gate(baselines, cls, 0.5),
            '混合门控(q=0.5)'
        )
        mixed_improvements.append(eval_result['improvement'])
    print(f'\n混合门控(q=0.5):')
    print(f'  各检测器改进: {[f"{v:+.4f}" for v in mixed_improvements]}')
    print(f'  改进均值: {np.mean(mixed_improvements):+.4f}')
    print(f'  改进标准差: {np.std(mixed_improvements):.4f}')
    print(f'  改进范围: [{min(mixed_improvements):+.4f}, {max(mixed_improvements):+.4f}]')

    # 6. 详细对比表
    print('\n' + '=' * 100)
    print('6. 详细对比表（最佳分位数q=0.5）')
    print('=' * 100)
    print(f'\n{"检测器":<12} {"门控类型":<15} {"开启数":>6} {"门控增益":>10} {"不门控":>10} '
          f'{"改进":>10} {"分离比":>8} {"精确率":>8} {"召回率":>8}')
    print('-' * 100)

    for det in detectors:
        results = all_results[det]
        baselines = {r['target_cls']: r['baseline_ap'] for r in results}

        # 硬门控
        eval_hard = evaluate_gate(results, lambda cls: hard_gate(cls), '硬门控')
        print(f'{detector_labels[det]:<12} {"硬门控":<15} {eval_hard["n_on"]:>6} '
              f'{eval_hard["avg_gain"]:>+10.4f} {eval_hard["avg_no_gate"]:>+10.4f} '
              f'{eval_hard["improvement"]:>+10.4f} {eval_hard["separation_ratio"]:>8.1f} '
              f'{eval_hard["precision"]:>8.1%} {eval_hard["recall"]:>8.1%}')

        # 分位数门控
        eval_quant = evaluate_gate(
            results,
            lambda cls: quantile_gate(baselines, cls, 0.5),
            '分位数门控'
        )
        print(f'{"":<12} {"分位数门控":<15} {eval_quant["n_on"]:>6} '
              f'{eval_quant["avg_gain"]:>+10.4f} {eval_quant["avg_no_gate"]:>+10.4f} '
              f'{eval_quant["improvement"]:>+10.4f} {eval_quant["separation_ratio"]:>8.1f} '
              f'{eval_quant["precision"]:>8.1%} {eval_quant["recall"]:>8.1%}')

        # 混合门控
        eval_mixed = evaluate_gate(
            results,
            lambda cls: mixed_gate(baselines, cls, 0.5),
            '混合门控'
        )
        print(f'{"":<12} {"混合门控":<15} {eval_mixed["n_on"]:>6} '
              f'{eval_mixed["avg_gain"]:>+10.4f} {eval_mixed["avg_no_gate"]:>+10.4f} '
              f'{eval_mixed["improvement"]:>+10.4f} {eval_mixed["separation_ratio"]:>8.1f} '
              f'{eval_mixed["precision"]:>8.1%} {eval_mixed["recall"]:>8.1%}')
        print()


if __name__ == '__main__':
    main()
