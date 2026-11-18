# -*- coding: utf-8 -*-
"""analyze_source_domain_effect.py — 分析源域效应和car的多源域对照
这是决定person/van证据存废的关键分析
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from physical_priors import get_combined_gate, normalize_class

DATA = Path(r'C:\Users\张三\Desktop\实验数据')


def interpolate_ap(ap_curve, alpha):
    alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
    aps = [ap_curve[str(a)] for a in alphas_sorted]
    return float(np.interp(alpha, alphas_sorted, aps))


def main():
    print('=' * 80)
    print('源域效应分析：car的多源域对照')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'\n总样本数: {len(all_samples)}')

    # 检查所有类别
    print(f'\n--- 所有样本的源域/目标域/类别 ---')
    for i, s in enumerate(all_samples):
        print(f'  [{i+1}] src={s.get("src", "?")}, tgt={s["target_domain"]}, cls={s["target_cls"]}, '
              f'baseline={s["ap_curve"]["0.0"]:.4f}, best_alpha={s.get("best_alpha", "?")}, '
              f'oracle_gain={s["oracle_gain"]:+.4f}')

    # car的多源域对照
    print(f'\n{"=" * 80}')
    print('car的多源域对照（关键分析！）')
    print(f'{"=" * 80}')

    car_samples = [s for s in all_samples if normalize_class(s['target_cls']) == 'car']
    print(f'\ncar样本数: {len(car_samples)}')

    print(f'\n{"源域":<12} {"目标域":<8} {"基线AP":<10} {"α=2.5增益":<12} {"最优α":<8} {"Oracle增益":<12}')
    print('-' * 70)

    src_domains = sorted(set(s.get('src', '?') for s in car_samples))
    for src in src_domains:
        src_car = [s for s in car_samples if s.get('src', '?') == src]
        for s in src_car:
            ap_25 = interpolate_ap(s['ap_curve'], 2.5)
            baseline = s['ap_curve']['0.0']
            gain_25 = ap_25 - baseline
            print(f'{src:<12} {s["target_domain"]:<8} {baseline:<10.4f} {gain_25:+12.4f} '
                  f'{str(s.get("best_alpha", "?")):<8} {s["oracle_gain"]:+12.4f}')

    # 按源域统计car的平均增益
    print(f'\n--- 按源域统计car的平均增益（α=2.5）---')
    print(f'{"源域":<12} {"样本数":<8} {"平均基线":<10} {"平均增益":<12} {"平均Oracle":<12}')
    print('-' * 60)
    for src in src_domains:
        src_car = [s for s in car_samples if s.get('src', '?') == src]
        if not src_car:
            continue
        baselines = [s['ap_curve']['0.0'] for s in src_car]
        gains = [interpolate_ap(s['ap_curve'], 2.5) - s['ap_curve']['0.0'] for s in src_car]
        oracles = [s['oracle_gain'] for s in src_car]
        print(f'{src:<12} {len(src_car):<8} {np.mean(baselines):<10.4f} '
              f'{np.mean(gains):+12.4f} {np.mean(oracles):+12.4f}')

    # 统计检验：VisDrone源的car增益是否显著高于其他源
    print(f'\n--- 源域效应统计检验 ---')
    visdrone_car = [s for s in car_samples if s.get('src', '?') == 'VisDrone']
    other_car = [s for s in car_samples if s.get('src', '?') != 'VisDrone']
    if visdrone_car and other_car:
        visdrone_gains = [interpolate_ap(s['ap_curve'], 2.5) - s['ap_curve']['0.0'] for s in visdrone_car]
        other_gains = [interpolate_ap(s['ap_curve'], 2.5) - s['ap_curve']['0.0'] for s in other_car]
        print(f'  VisDrone源 car增益: {np.mean(visdrone_gains):+.4f} (n={len(visdrone_gains)})')
        print(f'  其他源 car增益: {np.mean(other_gains):+.4f} (n={len(other_gains)})')
        print(f'  差值: {np.mean(visdrone_gains) - np.mean(other_gains):+.4f}')
        if np.mean(visdrone_gains) > np.mean(other_gains) * 1.5:
            print(f'  ⚠️ VisDrone源的car增益显著高于其他源——源域效应可能是主因！')
            print(f'  ⚠️ person的大增益可能主要来自VisDrone源域，而不是类级先验！')
        else:
            print(f'  ✅ 各源域下car增益相近——类级先验是真的，person的证据成立！')

    # person的源域分析
    print(f'\n{"=" * 80}')
    print('person的源域分析')
    print(f'{"=" * 80}')
    person_samples = [s for s in all_samples if normalize_class(s['target_cls']) == 'person']
    print(f'\nperson样本数: {len(person_samples)}')
    for s in person_samples:
        ap_25 = interpolate_ap(s['ap_curve'], 2.5)
        baseline = s['ap_curve']['0.0']
        gain_25 = ap_25 - baseline
        print(f'  src={s.get("src", "?")}, tgt={s["target_domain"]}, '
              f'baseline={baseline:.4f}, α=2.5增益={gain_25:+.4f}, '
              f'最优α={s.get("best_alpha", "?")}, Oracle增益={s["oracle_gain"]:+.4f}')

    # van类为什么没有出现在结果里
    print(f'\n{"=" * 80}')
    print('van类检查')
    print(f'{"=" * 80}')
    van_samples = [s for s in all_samples if 'van' in s['target_cls'].lower()]
    print(f'\nvan样本数: {len(van_samples)}')
    if van_samples:
        for s in van_samples:
            print(f'  src={s.get("src", "?")}, tgt={s["target_domain"]}, cls={s["target_cls"]}')
    else:
        print(f'  ⚠️ 没有找到van样本！可能是因为：')
        print(f'  1. van在VOC里没有对应类别（VOC只有car, 没有van）')
        print(f'  2. 实验代码中van的映射有问题')
        print(f'  3. van在COCO目标域里有，但是在VOC目标域里被跳过了')

    # 检查COCO目标域里有没有van
    coco_van = [s for s in all_samples if s['target_domain'] == 'COCO' and 'van' in s['target_cls'].lower()]
    print(f'\n  COCO目标域里的van样本数: {len(coco_van)}')
    if coco_van:
        for s in coco_van:
            print(f'    src={s.get("src", "?")}, cls={s["target_cls"]}, baseline={s["ap_curve"]["0.0"]:.4f}')

    # 完整混淆矩阵
    print(f'\n{"=" * 80}')
    print('完整混淆矩阵（9类全报，含错误项）')
    print(f'{"=" * 80}')

    gate_type = 'size_viewpoint_and'
    alpha = 2.5

    print(f'\n门控类型: {gate_type}, α={alpha}')
    print(f'\n{"类别":<12} {"样本数":<8} {"门控":<8} {"有提升":<8} {"无提升":<8} {"平均增益":<12} {"判定":<10}')
    print('-' * 80)

    tp = fp = tn = fn = 0
    classes = sorted(set(normalize_class(s['target_cls']) for s in all_samples))
    for cls in classes:
        cls_samples = [s for s in all_samples if normalize_class(s['target_cls']) == cls]
        gate = get_combined_gate(cls, gate_type)
        gains = [interpolate_ap(s['ap_curve'], alpha) - s['ap_curve']['0.0'] for s in cls_samples]
        n_pos = sum(g > 0.001 for g in gains)  # 有提升（阈值0.001避免噪声）
        n_neg = len(gains) - n_pos
        mean_gain = np.mean(gains)

        if gate > 0 and n_pos > 0:
            verdict = 'TP（真阳性）'
            tp += len(cls_samples)
        elif gate > 0 and n_pos == 0:
            verdict = 'FP（假阳性）'
            fp += len(cls_samples)
        elif gate == 0 and n_pos == 0:
            verdict = 'TN（真阴性）'
            tn += len(cls_samples)
        else:
            verdict = 'FN（假阴性）'
            fn += len(cls_samples)

        print(f'{cls:<12} {len(cls_samples):<8} {"开" if gate > 0 else "关":<8} '
              f'{n_pos:<8} {n_neg:<8} {mean_gain:+12.4f} {verdict:<10}')

    print(f'\n混淆矩阵汇总:')
    print(f'  TP（门控开，有提升）: {tp}')
    print(f'  FP（门控开，无提升）: {fp}')
    print(f'  TN（门控关，无提升）: {tn}')
    print(f'  FN（门控关，有提升）: {fn}')
    if tp + fp > 0:
        print(f'  精确率（Precision）: {tp/(tp+fp)*100:.1f}%')
    if tp + fn > 0:
        print(f'  召回率（Recall）: {tp/(tp+fn)*100:.1f}%')

    # 保存结果
    results = {
        'car_multi_source': {
            'visdrone_mean_gain': float(np.mean([interpolate_ap(s['ap_curve'], 2.5) - s['ap_curve']['0.0'] for s in car_samples if s.get('src', '?') == 'VisDrone'])) if any(s.get('src', '?') == 'VisDrone' for s in car_samples) else None,
            'other_mean_gain': float(np.mean([interpolate_ap(s['ap_curve'], 2.5) - s['ap_curve']['0.0'] for s in car_samples if s.get('src', '?') != 'VisDrone'])) if any(s.get('src', '?') != 'VisDrone' for s in car_samples) else None,
        },
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
    }
    results_file = DATA / 'source_domain_effect_analysis.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {results_file}')


if __name__ == '__main__':
    main()
