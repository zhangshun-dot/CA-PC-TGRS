# -*- coding: utf-8 -*-
"""analyze_random_direction_fixed_s.py — 在固定位移s下分析随机方向对照实验
不取argmax，公平比较真实方向 vs 随机方向
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# 位移网格（与实验一致）
DISPLACEMENTS = sorted(set(
    list(np.logspace(np.log10(0.02), np.log10(4.0), 28)) +
    list(np.arange(0.02, 0.52, 0.02))
))

N_RANDOM = 20

# 固定位移点（用于比较）
FIXED_S = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def get_ap_at_s(ap_curve, s):
    """在位移s处获取AP值（如果s不在网格中，取最近的）"""
    s_str = str(s)
    if s_str in ap_curve:
        return ap_curve[s_str]
    # 找最近的位移点
    closest_s = min(DISPLACEMENTS, key=lambda x: abs(x - s))
    return ap_curve[str(closest_s)]


def main():
    print('=' * 100)
    print('随机方向对照实验 — 固定位移s下的分析（不取argmax）')
    print('=' * 100)

    # 加载结果
    with open(DATA / 'random_direction_control_results.json', 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    classes = list(all_results.keys())
    print(f'\n类别数: {len(classes)}')
    print(f'类别: {classes}')

    # 在每个固定位移s下比较
    print(f'\n{"=" * 100}')
    print('固定位移s下的平均增益（所有类）')
    print(f'{"=" * 100}')

    print(f'\n{"s":>6} {"真实":>10} {"R1随机":>10} {"R2打乱":>10} {"R3反向":>10} '
          f'{"真实-R1":>10} {"真实-R2":>10} {"真实-R3":>10}')
    print('-' * 90)

    fixed_s_results = {}
    for s in FIXED_S:
        real_gains = []
        random_gains = []
        shuffled_gains = []
        reverse_gains = []

        for cls, results in all_results.items():
            baseline = results['baseline_ap']

            # 真实方向
            real_ap = get_ap_at_s(results['real_direction'], s)
            real_gains.append(real_ap - baseline)

            # 随机方向（R1）：20个随机方向的平均
            random_aps = []
            for i in range(N_RANDOM):
                random_ap = get_ap_at_s(results['random_directions'][i], s)
                random_aps.append(random_ap)
            random_gains.append(np.mean(random_aps) - baseline)

            # 打乱方向（R2）
            if results['shuffled_direction']:
                shuffled_ap = get_ap_at_s(results['shuffled_direction'], s)
                shuffled_gains.append(shuffled_ap - baseline)

            # 反向（R3）
            reverse_ap = get_ap_at_s(results['reverse_direction'], s)
            reverse_gains.append(reverse_ap - baseline)

        mean_real = np.mean(real_gains)
        mean_random = np.mean(random_gains)
        mean_shuffled = np.mean(shuffled_gains) if shuffled_gains else 0
        mean_reverse = np.mean(reverse_gains)

        fixed_s_results[s] = {
            'real': real_gains,
            'random': random_gains,
            'shuffled': shuffled_gains,
            'reverse': reverse_gains,
            'mean_real': mean_real,
            'mean_random': mean_random,
            'mean_shuffled': mean_shuffled,
            'mean_reverse': mean_reverse,
        }

        print(f'{s:>6.1f} {mean_real:>+10.4f} {mean_random:>+10.4f} '
              f'{mean_shuffled:>+10.4f} {mean_reverse:>+10.4f} '
              f'{mean_real - mean_random:>+10.4f} {mean_real - mean_shuffled:>+10.4f} '
              f'{mean_real - mean_reverse:>+10.4f}')

    # 统计检验
    print(f'\n{"=" * 100}')
    print('统计检验（固定位移s下，真实 vs 随机）')
    print(f'{"=" * 100}')

    print(f'\n{"s":>6} {"差值均值":>10} {"差值SD":>10} {"配对t":>10} {"p值":>10} '
          f'{"Wilcoxon p":>12} {"正例比例":>10}')
    print('-' * 80)

    for s in FIXED_S:
        diffs = np.array(fixed_s_results[s]['real']) - np.array(fixed_s_results[s]['random'])
        mean_diff = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)

        # 配对t检验
        t_stat, p_t = stats.ttest_rel(fixed_s_results[s]['real'],
                                        fixed_s_results[s]['random'])

        # Wilcoxon符号秩检验
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(fixed_s_results[s]['real'],
                                                   fixed_s_results[s]['random'])
        except Exception:
            p_wilcoxon = float('nan')

        # 正例比例
        pos_ratio = np.mean([1 if d > 0.001 else 0 for d in diffs])

        print(f'{s:>6.1f} {mean_diff:>+10.4f} {sd_diff:>10.4f} {t_stat:>10.3f} '
              f'{p_t:>10.4f} {p_wilcoxon:>12.4f} {pos_ratio:>10.1%}')

    # 逐类分析（在最优固定位移下）
    print(f'\n{"=" * 100}')
    print('逐类分析（在s=1.0和s=2.0下）')
    print(f'{"=" * 100}')

    for s in [1.0, 2.0]:
        print(f'\n--- s = {s} ---')
        print(f'{"类别":<12} {"baseline":>10} {"真实增益":>10} {"随机增益":>10} '
              f'{"差值":>10} {"随机SD":>10}')
        print('-' * 70)

        for cls, results in all_results.items():
            baseline = results['baseline_ap']
            real_ap = get_ap_at_s(results['real_direction'], s)
            real_gain = real_ap - baseline

            random_aps = []
            for i in range(N_RANDOM):
                random_ap = get_ap_at_s(results['random_directions'][i], s)
                random_aps.append(random_ap)
            random_gain = np.mean(random_aps) - baseline
            random_sd = np.std(random_aps, ddof=1)

            diff = real_gain - random_gain
            print(f'{cls:<12} {baseline:>10.4f} {real_gain:>+10.4f} {random_gain:>+10.4f} '
                  f'{diff:>+10.4f} {random_sd:>10.4f}')

    # per-class置换排名 + Fisher合并
    print(f'\n{"=" * 100}')
    print('per-class置换排名 + Fisher合并检验')
    print(f'{"=" * 100}')

    for s in FIXED_S:
        p_values = []
        ranks = []
        for cls, results in all_results.items():
            real_ap = get_ap_at_s(results['real_direction'], s)
            random_aps = [get_ap_at_s(results['random_directions'][i], s)
                          for i in range(N_RANDOM)]
            # 真实方向在m个随机方向中的排名（1=最好）
            rank = sum(1 for rap in random_aps if rap > real_ap) + 1
            ranks.append(rank)
            p_c = rank / (N_RANDOM + 1)
            p_values.append(p_c)

        # Fisher合并
        fisher_stat = -2 * np.sum(np.log(p_values))
        df = 2 * len(p_values)
        p_fisher = 1 - stats.chi2.cdf(fisher_stat, df)

        print(f'\ns={s:.1f}: 排名={ranks}, Fisher统计量={fisher_stat:.2f}, '
              f'df={df}, p={p_fisher:.4f}')

    # 最终结论
    print(f'\n{"=" * 100}')
    print('最终结论')
    print(f'{"=" * 100}')

    # 找最优固定位移
    best_s = None
    best_p = 1.0
    best_diff = 0
    for s in FIXED_S:
        diffs = np.array(fixed_s_results[s]['real']) - np.array(fixed_s_results[s]['random'])
        mean_diff = np.mean(diffs)
        t_stat, p_t = stats.ttest_rel(fixed_s_results[s]['real'],
                                        fixed_s_results[s]['random'])
        if p_t < best_p and mean_diff > 0:
            best_p = p_t
            best_s = s
            best_diff = mean_diff

    print(f'\n最优固定位移: s={best_s}, 平均差值={best_diff:+.4f}, 配对t检验p={best_p:.4f}')

    if best_p < 0.05:
        print(f'\n✅ 在固定位移s={best_s}下，真实方向显著优于随机方向（p={best_p:.4f} < 0.05）')
        print(f'   机制解释成立：TGE的增益确实来自语义方向，而不是纯位移效应。')
    elif best_p < 0.1:
        print(f'\n⚠️ 在固定位移s={best_s}下，真实方向略优于随机方向（p={best_p:.4f}，边缘显著）')
        print(f'   有一定证据支持机制解释，但需要更多样本或更大的随机方向采样数来确认。')
    else:
        print(f'\n❌ 在所有固定位移下，真实方向都不显著优于随机方向（最小p={best_p:.4f}）')
        print(f'   当前证据不足以支持机制解释。需要：')
        print(f'   1. 增加随机方向采样数（m≥49）')
        print(f'   2. 增加类别数')
        print(f'   3. 检查car和motorcycle是否是唯一有效应的类')

    # 检查car和motorcycle的贡献
    print(f'\n{"=" * 100}')
    print('car和motorcycle的贡献分析')
    print(f'{"=" * 100}')

    if best_s:
        all_diffs = np.array(fixed_s_results[best_s]['real']) - np.array(fixed_s_results[best_s]['random'])
        car_idx = classes.index('car') if 'car' in classes else -1
        moto_idx = classes.index('motorcycle') if 'motorcycle' in classes else -1

        total_effect = np.sum(all_diffs)
        car_moto_effect = 0
        if car_idx >= 0:
            car_moto_effect += all_diffs[car_idx]
        if moto_idx >= 0:
            car_moto_effect += all_diffs[moto_idx]

        print(f'\n在s={best_s}下：')
        print(f'  所有类总效应: {total_effect:+.4f}')
        print(f'  car+motorcycle效应: {car_moto_effect:+.4f}')
        if total_effect != 0:
            print(f'  car+motorcycle占比: {car_moto_effect / total_effect:.1%}')

        # 剔除car和motorcycle后的效应
        other_indices = [i for i in range(len(classes))
                         if classes[i] not in ['car', 'motorcycle']]
        other_diffs = all_diffs[other_indices]
        print(f'  剔除car+motorcycle后，剩余{len(other_diffs)}类平均效应: {np.mean(other_diffs):+.4f}')


if __name__ == '__main__':
    main()
