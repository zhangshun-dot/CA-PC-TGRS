# -*- coding: utf-8 -*-
"""analyze_alpha_slope.py — 分析α=2.0时AP的斜率，判断扩网格是否有收益"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')


def main():
    print('=' * 80)
    print('分析α=2.0时AP的斜率，判断扩网格是否有收益')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f'\n总样本数: {len(samples)}')

    # 分析每个样本的AP-α曲线在α=2.0处的斜率
    print(f'\n{"=" * 80}')
    print('每个样本的AP-α曲线在α=2.0处的斜率')
    print(f'{"=" * 80}')

    slopes = []
    still_rising = 0
    for i, s in enumerate(samples):
        ap_curve = s['ap_curve']
        alphas = sorted([float(a) for a in ap_curve.keys()])
        aps = [ap_curve[str(a)] for a in alphas]

        # 计算α=1.5到α=2.0的斜率
        if 1.5 in alphas and 2.0 in alphas:
            slope = (aps[-1] - aps[-2]) / (2.0 - 1.5)
        else:
            slope = 0.0

        slopes.append(slope)
        if slope > 0.001:  # 斜率大于0.001 AP/α，认为还在涨
            still_rising += 1

        print(f'  {i+1:2d}. {s["target_domain"]:5s} {s["target_cls"]:12s} '
              f'AP(1.5)={aps[-2]:.4f} AP(2.0)={aps[-1]:.4f} '
              f'斜率={slope:+.4f} {"⬆️还在涨" if slope > 0.001 else "➡️持平/下降"}')

    slopes = np.array(slopes)
    print(f'\n  斜率统计: mean={np.mean(slopes):+.4f}, median={np.median(slopes):+.4f}')
    print(f'  还在涨的样本: {still_rising}/{len(samples)} ({still_rising/len(samples)*100:.1f}%)')
    print(f'  斜率>0的样本: {sum(slopes > 0)}/{len(samples)}')
    print(f'  斜率>0.001的样本: {sum(slopes > 0.001)}/{len(samples)}')
    print(f'  斜率>0.01的样本: {sum(slopes > 0.01)}/{len(samples)}')

    # 估算扩到α=4.0的潜在收益
    print(f'\n{"=" * 80}')
    print('估算扩到α=4.0的潜在收益（线性外推）')
    print(f'{"=" * 80}')

    potential_gains = []
    for i, s in enumerate(samples):
        ap_curve = s['ap_curve']
        ap_2 = ap_curve['2.0']
        baseline = ap_curve['0.0']
        slope = slopes[i]
        # 线性外推到α=4.0
        ap_4_est = ap_2 + slope * (4.0 - 2.0)
        gain_2 = ap_2 - baseline
        gain_4_est = ap_4_est - baseline
        potential_gain = max(0, gain_4_est - gain_2)
        potential_gains.append(potential_gain)
        if potential_gain > 0.001:
            print(f'  {i+1:2d}. {s["target_domain"]:5s} {s["target_cls"]:12s} '
                  f'当前增益={gain_2:+.4f} 估算α=4增益={gain_4_est:+.4f} '
                  f'潜在收益={potential_gain:+.4f}')

    potential_gains = np.array(potential_gains)
    print(f'\n  潜在收益统计: mean={np.mean(potential_gains):+.4f}, '
          f'median={np.median(potential_gains):+.4f}, '
          f'sum={np.sum(potential_gains):+.4f}')
    print(f'  有潜在收益的样本: {sum(potential_gains > 0.001)}/{len(samples)}')
    print(f'  平均潜在收益（全部样本）: {np.mean(potential_gains):+.4f}')

    if np.mean(potential_gains) > 0.002:
        print(f'\n  ✅ 扩网格到α=4.0可能有显著收益（平均>{0.002}），建议重跑')
    else:
        print(f'\n  ⚠️  扩网格到α=4.0的潜在收益不大（平均<={0.002}），可能不值得重跑')


if __name__ == '__main__':
    main()
