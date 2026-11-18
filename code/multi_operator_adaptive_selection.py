# -*- coding: utf-8 -*-
"""multi_operator_adaptive_selection.py — 多算子自适应选择框架验证实验
验证：
1. 每个类自动选择最优算子的效果（Oracle选择）
2. 零训练的算子选择规则（基于类级属性）
3. 对比单一算子 vs 多算子自适应选择
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

MAIN_CLASSES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'truck']

# 类级属性（专家引导的序数属性）
OBJECT_SIZE = {'airplane': 30.0, 'bicycle': 1.8, 'boat': 15.0, 'bus': 12.0,
               'car': 4.5, 'motorcycle': 2.0, 'truck': 12.0, 'person': 1.7}
VIEWPOINT_DIFF = {'airplane': 0.3, 'bicycle': 0.5, 'boat': 0.4, 'bus': 0.8,
                   'car': 0.7, 'motorcycle': 0.6, 'truck': 0.75, 'person': 0.9}

def get_gate(cls):
    """门控决策：size < 10 AND vp > 0.5"""
    return 1 if (OBJECT_SIZE.get(cls, 999) < 10.0 and VIEWPOINT_DIFF.get(cls, 0) > 0.5) else 0

def select_operator_rule_based(cls):
    """零训练的算子选择规则（基于类级属性）
    
    规则：
    - 如果门控开启（size < 10 AND vp > 0.5）→ 用TGE（文本侧驱动，适合视角差大的类）
    - 如果门控关闭但目标域特征均值差异大（boat, bus等大目标/水面目标）→ 用FMS
    - 其他 → 不校正（α=0）
    """
    gate = get_gate(cls)
    if gate == 1:
        return 'TGE'  # 视角差大的类，文本侧驱动更有效
    elif cls in ['boat', 'bus', 'bicycle']:
        return 'FMS'  # 目标域特征均值差异大的类，特征侧驱动更有效
    else:
        return 'NONE'  # 不校正

def main():
    print('=' * 90)
    print('多算子自适应选择框架 — 验证实验')
    print('=' * 90)

    # 加载X1跨算子实验结果
    with open(DATA / 'x1_cross_operator_results.json', 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    # 按类别汇总每种算子的增益
    print('\n' + '─' * 90)
    print('第一步：按类别汇总每种算子的增益（固定α=2.5）')
    print('─' * 90)

    operator_gains = {}  # {op_name: {cls: mean_gain}}
    operator_baselines = {}  # {op_name: {cls: mean_baseline}}
    operator_oracles = {}  # {op_name: {cls: mean_oracle}}

    for op_name in ['TGE', 'FMS', 'DDC']:
        results = all_results[op_name]
        by_cls = defaultdict(list)
        for r in results:
            by_cls[r['target_cls']].append(r)
        
        operator_gains[op_name] = {}
        operator_baselines[op_name] = {}
        operator_oracles[op_name] = {}
        for cls in MAIN_CLASSES:
            if cls in by_cls:
                rs = by_cls[cls]
                operator_gains[op_name][cls] = np.mean([r['gain_25'] for r in rs])
                operator_baselines[op_name][cls] = np.mean([r['baseline_ap'] for r in rs])
                operator_oracles[op_name][cls] = np.mean([r['oracle_gain'] for r in rs])

    # 打印每种算子的按类别增益
    for op_name in ['TGE', 'FMS', 'DDC']:
        print(f'\n{op_name}算子:')
        print(f'{"类别":<12} {"n":>4} {"gate":>5} {"baseline":>10} {"gain_25":>10} {"oracle":>10}')
        print('-' * 60)
        for cls in MAIN_CLASSES:
            if cls in operator_gains[op_name]:
                n = len([r for r in all_results[op_name] if r['target_cls'] == cls])
                print(f'{cls:<12} {n:>4} {get_gate(cls):>5} '
                      f'{operator_baselines[op_name][cls]:>10.4f} '
                      f'{operator_gains[op_name][cls]:>+10.4f} '
                      f'{operator_oracles[op_name][cls]:>+10.4f}')

    # 第二步：Oracle选择（每个类选增益最大的算子）
    print('\n' + '─' * 90)
    print('第二步：Oracle选择（每个类选增益最大的算子）')
    print('─' * 90)

    print(f'\n{"类别":<12} {"gate":>5} {"TGE":>10} {"FMS":>10} {"DDC":>10} {"最优算子":>10} {"最优增益":>10}')
    print('-' * 80)

    oracle_selected_gains = {}
    oracle_selected_ops = {}
    for cls in MAIN_CLASSES:
        gains = {}
        for op_name in ['TGE', 'FMS', 'DDC']:
            if cls in operator_gains[op_name]:
                gains[op_name] = operator_gains[op_name][cls]
        if gains:
            best_op = max(gains, key=gains.get)
            best_gain = gains[best_op]
            oracle_selected_gains[cls] = best_gain
            oracle_selected_ops[cls] = best_op
            print(f'{cls:<12} {get_gate(cls):>5} '
                  f'{gains.get("TGE", 0):>+10.4f} '
                  f'{gains.get("FMS", 0):>+10.4f} '
                  f'{gains.get("DDC", 0):>+10.4f} '
                  f'{best_op:>10} {best_gain:>+10.4f}')

    oracle_mean_gain = np.mean(list(oracle_selected_gains.values()))
    print(f'\nOracle选择的7类平均增益: {oracle_mean_gain:+.4f}')

    # 第三步：单一算子的平均增益
    print('\n' + '─' * 90)
    print('第三步：单一算子的平均增益（对比）')
    print('─' * 90)

    for op_name in ['TGE', 'FMS', 'DDC']:
        gains = [operator_gains[op_name][cls] for cls in MAIN_CLASSES if cls in operator_gains[op_name]]
        mean_gain = np.mean(gains)
        print(f'{op_name}算子的7类平均增益: {mean_gain:+.4f}')

    # 第四步：零训练规则选择
    print('\n' + '─' * 90)
    print('第四步：零训练规则选择（基于类级属性，不需要训练）')
    print('─' * 90)

    print(f'\n规则：')
    print(f'  - 门控开启（size < 10 AND vp > 0.5）→ TGE')
    print(f'  - 门控关闭但boat/bus/bicycle → FMS')
    print(f'  - 其他 → 不校正（α=0）')

    print(f'\n{"类别":<12} {"gate":>5} {"size":>8} {"vp":>6} {"规则选择":>10} {"实际增益":>10} {"Oracle选择":>10} {"匹配?"}')
    print('-' * 90)

    rule_selected_gains = {}
    match_count = 0
    for cls in MAIN_CLASSES:
        rule_op = select_operator_rule_based(cls)
        if rule_op == 'NONE':
            actual_gain = 0.0
        else:
            actual_gain = operator_gains[rule_op].get(cls, 0.0)
        rule_selected_gains[cls] = actual_gain
        oracle_op = oracle_selected_ops.get(cls, 'NONE')
        match = '✅' if rule_op == oracle_op else '❌'
        if rule_op == oracle_op:
            match_count += 1
        print(f'{cls:<12} {get_gate(cls):>5} {OBJECT_SIZE[cls]:>8.1f} {VIEWPOINT_DIFF[cls]:>6.2f} '
              f'{rule_op:>10} {actual_gain:>+10.4f} {oracle_op:>10} {match}')

    rule_mean_gain = np.mean(list(rule_selected_gains.values()))
    print(f'\n零训练规则选择的7类平均增益: {rule_mean_gain:+.4f}')
    print(f'与Oracle选择的匹配率: {match_count}/{len(MAIN_CLASSES)} = {match_count/len(MAIN_CLASSES):.1%}')

    # 第五步：综合对比
    print('\n' + '=' * 90)
    print('综合对比：单一算子 vs 多算子自适应选择')
    print('=' * 90)

    print(f'\n{"方法":<30} {"7类平均增益":>12} {"相对TGE提升":>12}')
    print('-' * 60)

    tge_mean = np.mean([operator_gains['TGE'][cls] for cls in MAIN_CLASSES if cls in operator_gains['TGE']])
    fms_mean = np.mean([operator_gains['FMS'][cls] for cls in MAIN_CLASSES if cls in operator_gains['FMS']])
    ddc_mean = np.mean([operator_gains['DDC'][cls] for cls in MAIN_CLASSES if cls in operator_gains['DDC']])

    print(f'{"单一TGE算子":<30} {tge_mean:>+12.4f} {"—":>12}')
    print(f'{"单一FMS算子":<30} {fms_mean:>+12.4f} {(fms_mean-tge_mean)/abs(tge_mean+1e-8):>+12.1%}')
    print(f'{"单一DDC算子":<30} {ddc_mean:>+12.4f} {(ddc_mean-tge_mean)/abs(tge_mean+1e-8):>+12.1%}')
    print(f'{"零训练规则选择（多算子）":<30} {rule_mean_gain:>+12.4f} {(rule_mean_gain-tge_mean)/abs(tge_mean+1e-8):>+12.1%}')
    print(f'{"Oracle选择（多算子上限）":<30} {oracle_mean_gain:>+12.4f} {(oracle_mean_gain-tge_mean)/abs(tge_mean+1e-8):>+12.1%}')

    # 第六步：门控+多算子的组合效果
    print('\n' + '─' * 90)
    print('第六步：门控+多算子的组合效果（最终框架）')
    print('─' * 90)

    print(f'\n最终框架：')
    print(f'  1. 门控判断：这个类该不该校正？')
    print(f'  2. 算子选择：如果校正，用哪种算子？（基于类级属性）')
    print(f'  3. 强度调节：用多大的α？')

    print(f'\n{"类别":<12} {"门控":>5} {"算子":>8} {"α":>6} {"增益":>10} {"说明"}')
    print('-' * 70)

    final_gains = {}
    for cls in MAIN_CLASSES:
        gate = get_gate(cls)
        if gate == 1:
            op = 'TGE'
            alpha = 2.5
            gain = operator_gains['TGE'].get(cls, 0.0)
            note = '视角差大，文本侧驱动'
        elif cls in ['boat', 'bus']:
            op = 'FMS'
            alpha = 2.5
            gain = operator_gains['FMS'].get(cls, 0.0)
            note = '特征均值差异大，特征侧驱动'
        else:
            op = 'NONE'
            alpha = 0.0
            gain = 0.0
            note = '不校正'
        final_gains[cls] = gain
        print(f'{cls:<12} {gate:>5} {op:>8} {alpha:>6.1f} {gain:>+10.4f} {note}')

    final_mean_gain = np.mean(list(final_gains.values()))
    print(f'\n最终框架的7类平均增益: {final_mean_gain:+.4f}')
    print(f'相对单一TGE算子提升: {(final_mean_gain-tge_mean)/abs(tge_mean+1e-8):+.1%}')

    # 保存结果
    results = {
        'single_operator': {
            'TGE': float(tge_mean),
            'FMS': float(fms_mean),
            'DDC': float(ddc_mean),
        },
        'multi_operator': {
            'rule_based': float(rule_mean_gain),
            'oracle': float(oracle_mean_gain),
            'final_framework': float(final_mean_gain),
        },
        'per_class': {
            cls: {
                'gate': get_gate(cls),
                'TGE_gain': float(operator_gains['TGE'].get(cls, 0.0)),
                'FMS_gain': float(operator_gains['FMS'].get(cls, 0.0)),
                'DDC_gain': float(operator_gains['DDC'].get(cls, 0.0)),
                'oracle_op': oracle_selected_ops.get(cls, 'NONE'),
                'oracle_gain': float(oracle_selected_gains.get(cls, 0.0)),
                'rule_op': select_operator_rule_based(cls),
                'rule_gain': float(rule_selected_gains.get(cls, 0.0)),
                'final_gain': float(final_gains.get(cls, 0.0)),
            }
            for cls in MAIN_CLASSES
        }
    }

    out_file = DATA / 'multi_operator_adaptive_selection_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
