# -*- coding: utf-8 -*-
"""analyze_x1_results.py — 分析X1跨算子复现实验结果"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
with open(DATA / 'x1_cross_operator_results.json', 'r', encoding='utf-8') as f:
    all_results = json.load(f)

MAIN_CLASSES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'truck']
OBJECT_SIZE = {'airplane': 30.0, 'bicycle': 1.8, 'boat': 15.0, 'bus': 12.0,
               'car': 4.5, 'motorcycle': 2.0, 'truck': 12.0, 'person': 1.7}
VIEWPOINT_DIFF = {'airplane': 0.3, 'bicycle': 0.5, 'boat': 0.4, 'bus': 0.8,
                   'car': 0.7, 'motorcycle': 0.6, 'truck': 0.75, 'person': 0.9}

def get_gate(cls):
    return 1 if (OBJECT_SIZE.get(cls, 999) < 10.0 and VIEWPOINT_DIFF.get(cls, 0) > 0.5) else 0

print('=' * 90)
print('X1跨算子复现实验 — 完整结果汇总')
print('=' * 90)

for op_name in ['TGE', 'FMS', 'DDC']:
    results = all_results[op_name]
    print('\n' + '─' * 90)
    print(f'算子: {op_name}')
    print('─' * 90)
    
    by_cls = defaultdict(list)
    for r in results:
        by_cls[r['target_cls']].append(r)
    
    print('\n按类别汇总（固定α=2.5）:')
    header = f'{"类别":<12} {"n":>4} {"gate":>5} {"mean_baseline":>14} {"mean_gain_25":>12} {"pos_ratio":>10} {"mean_oracle":>12}'
    print(header)
    print('-' * 80)
    
    class_gains = {}
    class_gates = {}
    for cls in MAIN_CLASSES:
        if cls in by_cls:
            rs = by_cls[cls]
            mean_baseline = np.mean([r['baseline_ap'] for r in rs])
            mean_gain = np.mean([r['gain_25'] for r in rs])
            mean_oracle = np.mean([r['oracle_gain'] for r in rs])
            pos_ratio = np.mean([1 if r['gain_25'] > 0.001 else 0 for r in rs])
            gate = get_gate(cls)
            class_gains[cls] = mean_gain
            class_gates[cls] = gate
            print(f'{cls:<12} {len(rs):>4} {gate:>5} {mean_baseline:>14.4f} '
                  f'{mean_gain:>+12.4f} {pos_ratio:>10.1%} {mean_oracle:>+12.4f}')
    
    on_gains = [class_gains[c] for c in class_gains if class_gates[c] == 1]
    off_gains = [class_gains[c] for c in class_gains if class_gates[c] == 0]
    
    if on_gains and off_gains:
        mean_on = np.mean(on_gains)
        mean_off = np.mean(off_gains)
        separation = mean_on / (np.abs(mean_off) + 1e-8) if mean_off != 0 else float('inf')
        print(f'\n门控分离度: 开启类均值={mean_on:+.4f}, 关闭类均值={mean_off:+.4f}, 分离比={separation:.1f}x')
        print(f'门控规则{"成立 ✅" if mean_on > mean_off else "不成立 ❌"}')

print('\n' + '=' * 90)
print('跨算子一致性分析')
print('=' * 90)
print(f'\n{"算子":<8} {"开启类均值":>12} {"关闭类均值":>12} {"分离比":>8} {"门控成立?"}')
print('-' * 60)

consistency_count = 0
for op_name in ['TGE', 'FMS', 'DDC']:
    results = all_results[op_name]
    by_cls = defaultdict(list)
    for r in results:
        by_cls[r['target_cls']].append(r)
    
    class_gains = {}
    class_gates = {}
    for cls in MAIN_CLASSES:
        if cls in by_cls:
            class_gains[cls] = np.mean([r['gain_25'] for r in by_cls[cls]])
            class_gates[cls] = get_gate(cls)
    
    on_gains = [class_gains[c] for c in class_gains if class_gates[c] == 1]
    off_gains = [class_gains[c] for c in class_gains if class_gates[c] == 0]
    
    if on_gains and off_gains:
        mean_on = np.mean(on_gains)
        mean_off = np.mean(off_gains)
        separation = mean_on / (np.abs(mean_off) + 1e-8) if mean_off != 0 else float('inf')
        holds = mean_on > mean_off
        if holds:
            consistency_count += 1
        print(f'{op_name:<8} {mean_on:>+12.4f} {mean_off:>+12.4f} {separation:>8.1f}x {"✅" if holds else "❌"}')

print(f'\n跨算子一致性: {consistency_count}/3 种算子上门控规则成立')

# 新发现：不同算子有不同的类级偏好
print('\n' + '=' * 90)
print('新发现：不同算子的类级偏好矩阵')
print('=' * 90)
print(f'\n{"类别":<12} {"gate":>5} {"TGE增益":>10} {"FMS增益":>10} {"DDC增益":>10} {"最优算子":>10}')
print('-' * 70)

for cls in MAIN_CLASSES:
    gains = {}
    for op_name in ['TGE', 'FMS', 'DDC']:
        results = all_results[op_name]
        cls_results = [r for r in results if r['target_cls'] == cls]
        if cls_results:
            gains[op_name] = np.mean([r['gain_25'] for r in cls_results])
        else:
            gains[op_name] = 0.0
    
    best_op = max(gains, key=gains.get)
    gate = get_gate(cls)
    print(f'{cls:<12} {gate:>5} {gains["TGE"]:>+10.4f} {gains["FMS"]:>+10.4f} {gains["DDC"]:>+10.4f} {best_op:>10}')
