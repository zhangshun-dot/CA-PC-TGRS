# -*- coding: utf-8 -*-
"""generate_gates_csv.py — 生成gates.csv，用于类层面精确检验
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# 物理先验知识库
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
    """统一类别名称，COCO和VOC的不同写法映射到同一个"""
    mapping = {
        'aeroplane': 'airplane',
        'motorbike': 'motorcycle',
    }
    return mapping.get(cls, cls)

def main():
    print('=' * 80)
    print('生成gates.csv')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f'\n总样本数: {len(samples)}')

    # 生成gates.csv
    gates_csv = DATA / 'gates.csv'
    with open(gates_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('class,domain,oracle_gain,gate_size,gate_viewpoint,gate_size_viewpoint_and\n')

        for i, s in enumerate(samples):
            # 统一类别名称
            cls_norm = normalize_class(s['target_cls'])
            domain = s['target_domain']

            # Oracle增益 = best_alpha时的AP - alpha=0时的AP
            ap_curve = s['ap_curve']
            ap0 = ap_curve.get('0.0', ap_curve.get('0', 0.0))
            best_alpha = s['best_alpha']
            # 找到最接近best_alpha的网格点
            best_key = None
            best_diff = float('inf')
            for key in ap_curve.keys():
                try:
                    diff = abs(float(key) - best_alpha)
                    if diff < best_diff:
                        best_diff = diff
                        best_key = key
                except ValueError:
                    continue
            ap_best = ap_curve.get(best_key, ap0)
            oracle_gain = ap_best - ap0

            # 尺寸门控：size < 10m 开启
            size = OBJECT_SIZE.get(cls_norm, 10.0)
            gate_size = 1 if size < 10.0 else 0

            # 视角差门控：viewpoint_diff > 0.5 开启
            vp = VIEWPOINT_DIFF.get(cls_norm, 0.5)
            gate_viewpoint = 1 if vp > 0.5 else 0

            # 尺寸AND视角差门控
            gate_size_viewpoint_and = 1 if (gate_size == 1 and gate_viewpoint == 1) else 0

            f.write(f'{cls_norm},{domain},{oracle_gain:.6f},{gate_size},{gate_viewpoint},{gate_size_viewpoint_and}\n')

    print(f'\ngates.csv已生成: {gates_csv}')

    # 统计
    print('\n--- 门控统计 ---')
    import csv
    with open(gates_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for gate_col in ['gate_size', 'gate_viewpoint', 'gate_size_viewpoint_and']:
        opened = [r for r in rows if r[gate_col] == '1']
        closed = [r for r in rows if r[gate_col] == '0']
        opened_classes = set(r['class'] for r in opened)
        closed_classes = set(r['class'] for r in closed)
        print(f'\n{gate_col}:')
        print(f'  开启类: {sorted(opened_classes)} ({len(opened_classes)}个)')
        print(f'  关闭类: {sorted(closed_classes)} ({len(closed_classes)}个)')
        print(f'  开启单元: {len(opened)}, 关闭单元: {len(closed)}')

    # 检查类层面的一致性
    print('\n--- 类层面一致性检查 ---')
    from collections import defaultdict
    for gate_col in ['gate_size', 'gate_viewpoint', 'gate_size_viewpoint_and']:
        cls_labels = defaultdict(set)
        for r in rows:
            cls_labels[r['class']].add(r[gate_col])
        inconsistent = {c: labels for c, labels in cls_labels.items() if len(labels) > 1}
        if inconsistent:
            print(f'  {gate_col}: 不一致! {inconsistent}')
        else:
            print(f'  {gate_col}: 一致 ✓')


if __name__ == '__main__':
    main()
