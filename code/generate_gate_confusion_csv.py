# -*- coding: utf-8 -*-
"""generate_gate_confusion_csv.py — 根据最新实验数据生成gate_confusion.py需要的CSV
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from physical_priors import OBJECT_SIZE, VIEWPOINT_DIFF, get_combined_gate, normalize_class

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

def interpolate_ap(ap_curve, alpha):
    alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
    aps = [ap_curve[str(a)] for a in alphas_sorted]
    return float(np.interp(alpha, alphas_sorted, aps))

def main():
    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'总样本数: {len(all_samples)}')

    # 统一类别名
    for s in all_samples:
        s['target_cls'] = normalize_class(s['target_cls'])

    # 按类别统计
    classes = sorted(set(s['target_cls'] for s in all_samples))
    print(f'类别数: {len(classes)}')
    print(f'类别: {classes}')

    # 生成CSV
    csv_rows = []
    csv_rows.append('class,n,gate,n_pos,n_neg,mean_gain,size,vp')

    for cls in classes:
        cls_samples = [s for s in all_samples if s['target_cls'] == cls]
        n = len(cls_samples)

        # 门控决策
        gate = get_combined_gate(cls, 'size_viewpoint_and')

        # α=2.5下的增益
        gains = [interpolate_ap(s['ap_curve'], 2.5) - s['ap_curve']['0.0'] for s in cls_samples]
        n_pos = sum(g > 0.001 for g in gains)
        n_neg = n - n_pos
        mean_gain = float(np.mean(gains))

        # 物理先验
        size = OBJECT_SIZE.get(cls, 0)
        vp = VIEWPOINT_DIFF.get(cls, 0)

        csv_rows.append(f'{cls},{n},{gate},{n_pos},{n_neg},{mean_gain:.6f},{size},{vp}')
        print(f'  {cls}: n={n}, gate={gate}, n_pos={n_pos}, n_neg={n_neg}, '
              f'mean_gain={mean_gain:+.4f}, size={size}, vp={vp}')

    # 保存CSV
    csv_file = DATA / 'gate_confusion_input.csv'
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv_rows))
    print(f'\nCSV已保存到: {csv_file}')


if __name__ == '__main__':
    main()
