# -*- coding: utf-8 -*-
"""analyze_random_direction.py — 分析随机方向对照实验结果"""
import json
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# 位移网格（与实验一致）
DISPLACEMENTS = sorted(set(
    list(np.logspace(np.log10(0.02), np.log10(4.0), 28)) +
    list(np.arange(0.02, 0.52, 0.02))
))

N_RANDOM = 20


def main():
    print('=' * 100)
    print('随机方向对照实验（R1/R2/R3）— 结果分析')
    print('验证TGE的增益是来自语义方向，还是纯位移效应')
    print('=' * 100)

    # 加载结果
    with open(DATA / 'random_direction_control_results.json', 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    print(f'\n类别数: {len(all_results)}')
    print(f'类别: {list(all_results.keys())}')

    # 逐类分析
    print(f'\n{"=" * 100}')
    print('逐类分析（最优位移处的AP50增益）')
    print(f'{"=" * 100}')

    real_gains = []
    random_gains = []
    shuffled_gains = []
    reverse_gains = []

    real_best_ss = []
    random_best_ss = []

    for cls, results in all_results.items():
        baseline = results['baseline_ap']

        # 真实方向
        real_aps = [results['real_direction'][str(s)] for s in DISPLACEMENTS]
        best_real_idx = np.argmax(real_aps)
        best_real_s = DISPLACEMENTS[best_real_idx]
        best_real_ap = real_aps[best_real_idx]
        real_gain = best_real_ap - baseline
        real_gains.append(real_gain)
        real_best_ss.append(best_real_s)

        # 随机方向（R1）
        random_best_aps = []
        random_best_ss_cls = []
        for i in range(N_RANDOM):
            random_aps = [results['random_directions'][i][str(s)] for s in DISPLACEMENTS]
            best_idx = np.argmax(random_aps)
            random_best_aps.append(random_aps[best_idx])
            random_best_ss_cls.append(DISPLACEMENTS[best_idx])
        mean_random_ap = np.mean(random_best_aps)
        random_gain = mean_random_ap - baseline
        random_gains.append(random_gain)
        random_best_ss.append(np.mean(random_best_ss_cls))

        # 打乱方向（R2）
        if results['shuffled_direction']:
            shuffled_aps = [results['shuffled_direction'][str(s)] for s in DISPLACEMENTS]
            best_shuffled_ap = np.max(shuffled_aps)
            shuffled_gain = best_shuffled_ap - baseline
            shuffled_gains.append(shuffled_gain)

        # 反向（R3）
        reverse_aps = [results['reverse_direction'][str(s)] for s in DISPLACEMENTS]
        best_reverse_ap = np.max(reverse_aps)
        reverse_gain = best_reverse_ap - baseline
        reverse_gains.append(reverse_gain)

        print(f'\n【{cls}】baseline={baseline:.4f}')
        print(f'  真实方向: 最优s={best_real_s:.3f}, 增益={real_gain:+.4f}')
        print(f'  随机方向(R1): 平均最优s={np.mean(random_best_ss_cls):.3f}, 平均增益={random_gain:+.4f}')
        if results['shuffled_direction']:
            print(f'  打乱方向(R2): 增益={shuffled_gain:+.4f}')
        print(f'  反向(R3): 增益={reverse_gain:+.4f}')
        print(f'  真实-随机差异: {real_gain - random_gain:+.4f}')

    # 汇总统计
    print(f'\n{"=" * 100}')
    print('汇总统计（所有类）')
    print(f'{"=" * 100}')

    print(f'\n{"方向类型":<18} {"平均增益":>10} {"中位数":>10} {"最小":>10} {"最大":>10} {"正例比例":>10}')
    print('-' * 75)

    def stats(gains):
        return (np.mean(gains), np.median(gains), np.min(gains), np.max(gains),
                np.mean([1 if g > 0.001 else 0 for g in gains]))

    for name, gains in [('真实方向', real_gains),
                          ('随机方向(R1)', random_gains),
                          ('打乱方向(R2)', shuffled_gains if shuffled_gains else [0]),
                          ('反向(R3)', reverse_gains)]:
        if gains:
            m, med, mn, mx, pos = stats(gains)
            print(f'{name:<18} {m:>+10.4f} {med:>+10.4f} {mn:>+10.4f} {mx:>+10.4f} {pos:>10.1%}')

    # 关键对比
    print(f'\n{"=" * 100}')
    print('关键对比')
    print(f'{"=" * 100}')

    mean_real = np.mean(real_gains)
    mean_random = np.mean(random_gains)
    mean_reverse = np.mean(reverse_gains)

    print(f'\n1. 真实方向 vs 随机方向(R1):')
    print(f'   真实方向平均增益: {mean_real:+.4f}')
    print(f'   随机方向平均增益: {mean_random:+.4f}')
    print(f'   差异: {mean_real - mean_random:+.4f}')
    print(f'   相对提升: {(mean_real - mean_random) / (abs(mean_random) + 1e-8):+.1%}')

    if shuffled_gains:
        mean_shuffled = np.mean(shuffled_gains)
        print(f'\n2. 真实方向 vs 打乱方向(R2):')
        print(f'   真实方向平均增益: {mean_real:+.4f}')
        print(f'   打乱方向平均增益: {mean_shuffled:+.4f}')
        print(f'   差异: {mean_real - mean_shuffled:+.4f}')

    print(f'\n3. 真实方向 vs 反向(R3):')
    print(f'   真实方向平均增益: {mean_real:+.4f}')
    print(f'   反向平均增益: {mean_reverse:+.4f}')
    print(f'   差异: {mean_real - mean_reverse:+.4f}')

    # 最优位移分析
    print(f'\n4. 最优位移分析:')
    print(f'   真实方向平均最优位移: {np.mean(real_best_ss):.3f}')
    print(f'   随机方向平均最优位移: {np.mean(random_best_ss):.3f}')
    print(f'   差异: {np.mean(real_best_ss) - np.mean(random_best_ss):.3f}')

    # 最终结论
    print(f'\n{"=" * 100}')
    print('最终结论')
    print(f'{"=" * 100}')

    diff_real_random = mean_real - mean_random

    if diff_real_random > 0.01:
        print(f'\n✅ 结论1：真实方向明显优于随机方向（差异={diff_real_random:.4f} > 0.01）')
        print(f'   语义方向有用，位移效应不是唯一因素。')
        print(f'   主方法的机制解释（"文本侧嵌入差异提供了有效校正方向"）成立。')
    elif diff_real_random > 0.003:
        print(f'\n⚠️ 结论1：真实方向略优于随机方向（差异={diff_real_random:.4f}，在0.003-0.01之间）')
        print(f'   语义方向有一定作用，但位移效应也贡献了相当一部分。')
        print(f'   需要更多样本或更精细的分析来确认。')
    else:
        print(f'\n❌ 结论1：真实方向不优于随机方向（差异={diff_real_random:.4f} < 0.003）')
        print(f'   TGE的增益可能主要来自纯位移效应（把原型推离原位），')
        print(f'   而不是语义方向本身。主方法的机制解释需要重新思考。')

    if shuffled_gains:
        diff_real_shuffled = mean_real - mean_shuffled
        if diff_real_shuffled > 0.01:
            print(f'\n✅ 结论2：真实方向明显优于打乱方向（差异={diff_real_shuffled:.4f}）')
            print(f'   存在类-方向匹配信息，不是随便一个真实方向都有用。')
        else:
            print(f'\n⚠️ 结论2：真实方向不优于打乱方向（差异={diff_real_shuffled:.4f}）')
            print(f'   只有"是真实类条件方向"有效，无类-算子匹配信息。')

    diff_real_reverse = mean_real - mean_reverse
    if diff_real_reverse > 0.01:
        print(f'\n✅ 结论3：真实方向明显优于反向（差异={diff_real_reverse:.4f}）')
        print(f'   方向信息有用，不是随便哪个方向都能拿到增益。')
    else:
        print(f'\n❌ 结论3：真实方向不优于反向（差异={diff_real_reverse:.4f}）')
        print(f'   方向信息可能为零，增益完全来自位移大小而非方向。')

    # 位移效应的贡献估计
    print(f'\n{"=" * 100}')
    print('位移效应 vs 语义方向的贡献分解')
    print(f'{"=" * 100}')

    # 随机方向的增益 = 纯位移效应的贡献
    # 真实方向的增益 - 随机方向的增益 = 语义方向的额外贡献
    displacement_effect = mean_random
    semantic_effect = mean_real - mean_random

    print(f'\n纯位移效应（随机方向的增益）: {displacement_effect:+.4f}')
    print(f'语义方向额外贡献（真实-随机）: {semantic_effect:+.4f}')
    print(f'总增益（真实方向）: {mean_real:+.4f}')

    if mean_real > 0:
        print(f'\n位移效应占比: {displacement_effect / mean_real:.1%}')
        print(f'语义方向占比: {semantic_effect / mean_real:.1%}')

    # 逐类的位移效应占比
    print(f'\n逐类的位移效应占比:')
    print(f'{"类别":<12} {"真实增益":>10} {"随机增益":>10} {"位移占比":>10} {"语义占比":>10}')
    print('-' * 60)
    for i, cls in enumerate(all_results.keys()):
        rg = real_gains[i]
        rng = random_gains[i]
        if rg > 0:
            disp_pct = rng / rg * 100
            sem_pct = (rg - rng) / rg * 100
        else:
            disp_pct = 0
            sem_pct = 0
        print(f'{cls:<12} {rg:>+10.4f} {rng:>+10.4f} {disp_pct:>9.1f}% {sem_pct:>9.1f}%')


if __name__ == '__main__':
    main()
