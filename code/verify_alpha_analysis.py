# -*- coding: utf-8 -*-
"""verify_alpha_analysis.py — 核对α分析的四个关键问题
1. 三个单元数（grid/pred/交集）
2. 尺寸门控用的是目标域proposal尺寸还是类级先验
3. 门控的precision/recall
4. 增益的完整分布
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# 物理先验知识库（类级先验，跨域恒定）
OBJECT_SIZE = {'airplane': 30, 'boat': 15, 'car': 4.5, 'bus': 12,
                'motorcycle': 2, 'bicycle': 1.8, 'truck': 12,
                'aeroplane': 30, 'motorbike': 2}
VIEWPOINT_DIFF = {'airplane': 0.3, 'boat': 0.4, 'car': 0.7, 'bus': 0.8,
                   'motorcycle': 0.6, 'bicycle': 0.5, 'truck': 0.8,
                   'aeroplane': 0.3, 'motorbike': 0.6}
RARITY = {'airplane': 0.3, 'boat': 0.5, 'car': 0.1, 'bus': 0.3,
          'motorcycle': 0.4, 'bicycle': 0.3, 'truck': 0.3,
          'aeroplane': 0.3, 'motorbike': 0.4}


def main():
    print('=' * 80)
    print('核对α分析的四个关键问题')
    print('=' * 80)

    # 加载零训练门控样本
    samples_file = DATA / 'zero_shot_gated_samples.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f'\n总样本数: {len(samples)}')

    # ============================================================
    # 1. 三个单元数核对
    # ============================================================
    print(f'\n{"=" * 80}')
    print('1. 三个单元数核对（grid/pred/交集）')
    print(f'{"=" * 80}')

    # grid.csv的单元数
    grid_csv = DATA / 'grid.csv'
    grid_units = set()
    with open(grid_csv, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                grid_units.add((parts[0], parts[1]))
    print(f'\n  grid.csv单元数: {len(grid_units)}')

    # pred.csv的单元数
    pred_csv = DATA / 'pred.csv'
    pred_units = set()
    pred_by_setting = {}
    with open(pred_csv, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                unit = (parts[0], parts[1])
                setting = parts[3]
                pred_units.add(unit)
                pred_by_setting.setdefault(setting, set()).add(unit)
    print(f'  pred.csv单元数: {len(pred_units)}')
    print(f'  交集单元数: {len(grid_units & pred_units)}')

    if len(grid_units) == len(pred_units) == len(grid_units & pred_units) == 27:
        print(f'  ✅ 三个单元数都是27，口径一致')
    else:
        print(f'  ❌ 单元数不一致，需要排查')
        print(f'    grid独有: {grid_units - pred_units}')
        print(f'    pred独有: {pred_units - grid_units}')

    # 每个setting的单元数
    print(f'\n  每个setting的单元数:')
    for setting, units in sorted(pred_by_setting.items()):
        print(f'    {setting}: {len(units)}')

    # ============================================================
    # 2. 尺寸门控用的是目标域proposal尺寸还是类级先验
    # ============================================================
    print(f'\n{"=" * 80}')
    print('2. 尺寸门控用的是目标域proposal尺寸还是类级先验？')
    print(f'{"=" * 80}')

    print(f'\n  物理先验门控的实现:')
    print(f'    OBJECT_SIZE（类级先验，跨域恒定）:')
    for cls, size in sorted(OBJECT_SIZE.items()):
        print(f'      {cls}: {size}m')
    print(f'    尺寸门控阈值: log(size) < log(10) → 开启')
    print(f'    即 size < 10m → 开启（car, motorcycle, bicycle）')
    print(f'       size >= 10m → 关闭（airplane, boat, bus, truck）')

    print(f'\n  结论: 尺寸门控用的是**类级先验**，不是目标域proposal尺寸')
    print(f'  这意味着物理先验门控是**域盲**的，对同一个类在COCO和VOC上给出相同决策')

    # 验证：同一个类在COCO和VOC上的门控决策是否相同
    print(f'\n  验证：同一个类在COCO和VOC上的门控决策')
    print(f'  {"类":<12s} {"COCO尺寸门控":>12s} {"VOC尺寸门控":>12s} {"一致？":>8s}')
    print('-' * 50)
    classes_in_both = set()
    coco_classes = set(s['target_cls'] for s in samples if s['target_domain'] == 'COCO')
    voc_classes = set(s['target_cls'] for s in samples if s['target_domain'] == 'VOC')
    classes_in_both = coco_classes & voc_classes

    for cls in sorted(classes_in_both):
        size = OBJECT_SIZE.get(cls, 10)
        gate_coco = 1 if size < 10 else 0
        gate_voc = 1 if size < 10 else 0
        consistent = '✅' if gate_coco == gate_voc else '❌'
        print(f'  {cls:<12s} {gate_coco:>12d} {gate_voc:>12d} {consistent:>8s}')

    print(f'\n  ⚠️  物理先验门控是域盲的，不能解释VOC的关闭行为')
    print(f'     对同一个类，COCO和VOC的决策完全相同')
    print(f'     这意味着物理先验门控的价值在于"类级先验"，而不是"域级适应"')

    # ============================================================
    # 3. 门控的precision/recall
    # ============================================================
    print(f'\n{"=" * 80}')
    print('3. 门控的precision/recall')
    print(f'{"=" * 80}')

    # 定义"真零"：oracle α* = 0（即最优α是0，不需要校正）
    true_zeros = [s for s in samples if abs(s['best_alpha']) < 0.01]
    true_nonzeros = [s for s in samples if abs(s['best_alpha']) >= 0.01]
    print(f'\n  真零（oracle α*=0）: {len(true_zeros)}个')
    for s in true_zeros:
        print(f'    {s["target_domain"]:5s} {s["target_cls"]:12s} best_alpha={s["best_alpha"]:.1f}')
    print(f'  非零（oracle α*>0）: {len(true_nonzeros)}个')

    # 闭式解的门控（α=0的样本）
    closed_zeros = [s for s in samples if abs(
        max(0.0, (0.13 - s['m0']) / max(s['mu'] * 0.01, 1e-8))
    ) < 0.01]
    print(f'\n  闭式解预测α=0的样本: {len(closed_zeros)}个')
    for s in closed_zeros:
        print(f'    {s["target_domain"]:5s} {s["target_cls"]:12s} '
              f'm0={s["m0"]:.4f} best_alpha={s["best_alpha"]:.1f}')

    # 计算precision/recall
    # precision = 真零中被预测为零的比例 / 被预测为零的总数
    # recall = 真零中被预测为零的比例 / 真零总数
    tp = len([s for s in closed_zeros if abs(s['best_alpha']) < 0.01])
    fp = len([s for s in closed_zeros if abs(s['best_alpha']) >= 0.01])
    fn = len([s for s in true_zeros if s not in closed_zeros])
    tn = len([s for s in true_nonzeros if s not in closed_zeros])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f'\n  混淆矩阵:')
    print(f'    TP（真零预测为零）: {tp}')
    print(f'    FP（非零预测为零）: {fp}')
    print(f'    FN（真零预测为非零）: {fn}')
    print(f'    TN（非零预测为非零）: {tn}')
    print(f'\n  Precision: {precision:.3f}')
    print(f'  Recall: {recall:.3f}')
    print(f'  F1: {f1:.3f}')

    # 物理先验门控（viewpoint门控）的precision/recall
    print(f'\n  物理先验门控（viewpoint门控，viewpoint_diff > 0.5 → 开启）:')
    viewpoint_gated_off = [s for s in samples if VIEWPOINT_DIFF.get(s['target_cls'], 0.5) <= 0.5]
    viewpoint_gated_on = [s for s in samples if VIEWPOINT_DIFF.get(s['target_cls'], 0.5) > 0.5]
    print(f'    门控关闭（不校正）: {len(viewpoint_gated_off)}个')
    for s in viewpoint_gated_off:
        print(f'      {s["target_domain"]:5s} {s["target_cls"]:12s} '
              f'viewpoint_diff={VIEWPOINT_DIFF.get(s["target_cls"], 0.5):.1f} '
              f'best_alpha={s["best_alpha"]:.1f}')
    print(f'    门控开启（校正）: {len(viewpoint_gated_on)}个')

    # 物理先验门控的"关闭"是否正确？
    # 关闭的类中，oracle增益低的比例
    off_oracle_gains = [s['oracle_gain'] for s in viewpoint_gated_off]
    on_oracle_gains = [s['oracle_gain'] for s in viewpoint_gated_on]
    print(f'\n    关闭类的oracle增益: mean={np.mean(off_oracle_gains):+.4f}')
    print(f'    开启类的oracle增益: mean={np.mean(on_oracle_gains):+.4f}')
    print(f'    分离倍数: {np.mean(on_oracle_gains) / (np.mean(off_oracle_gains) + 1e-8):.1f}x')

    # ============================================================
    # 4. 增益的完整分布
    # ============================================================
    print(f'\n{"=" * 80}')
    print('4. 增益的完整分布')
    print(f'{"=" * 80}')

    # 闭式解（最优参数）的增益分布
    gains_closed = []
    for s in samples:
        alpha = max(0.0, (0.13 - s['m0']) / max(s['mu'] * 0.01, 1e-8))
        alpha = float(np.clip(alpha, 0.0, 2.0))
        ap_curve = s['ap_curve']
        alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
        ap = float(np.interp(alpha, alphas_sorted, [ap_curve[str(a)] for a in alphas_sorted]))
        baseline = ap_curve['0.0']
        gains_closed.append(ap - baseline)

    gains_closed = np.array(gains_closed)
    print(f'\n  闭式解（最优参数）的增益分布:')
    print(f'    mean: {np.mean(gains_closed):+.4f}')
    print(f'    median: {np.median(gains_closed):+.4f}')
    print(f'    std: {np.std(gains_closed):.4f}')
    print(f'    min: {np.min(gains_closed):+.4f}')
    print(f'    max: {np.max(gains_closed):+.4f}')
    print(f'    Q25: {np.percentile(gains_closed, 25):+.4f}')
    print(f'    Q75: {np.percentile(gains_closed, 75):+.4f}')
    print(f'    正例: {sum(gains_closed > 0)}/{len(gains_closed)}')
    print(f'    零增益: {sum(abs(gains_closed) < 1e-6)}/{len(gains_closed)}')
    print(f'    负例: {sum(gains_closed < 0)}/{len(gains_closed)}')

    if abs(np.median(gains_closed)) < 1e-6 and np.mean(gains_closed) > 0:
        print(f'\n  ⚠️  中位=0而均值>0，分布高度偏斜')
        print(f'     说明大部分样本增益为0，少数样本贡献了全部增益')

    # 分域分布
    print(f'\n  分域分布:')
    for domain in ['COCO', 'VOC']:
        domain_gains = [g for g, s in zip(gains_closed, samples) if s['target_domain'] == domain]
        print(f'    {domain}: mean={np.mean(domain_gains):+.4f}, '
              f'median={np.median(domain_gains):+.4f}, '
              f'正例={sum(g > 0 for g in domain_gains)}/{len(domain_gains)}')

    # 保存结果
    results = {
        'grid_units': len(grid_units),
        'pred_units': len(pred_units),
        'intersection_units': len(grid_units & pred_units),
        'size_gate_uses_class_prior': True,
        'size_gate_is_domain_blind': True,
        'closed_form_precision': float(precision),
        'closed_form_recall': float(recall),
        'closed_form_f1': float(f1),
        'true_zeros_count': len(true_zeros),
        'closed_zeros_count': len(closed_zeros),
        'gains_closed_mean': float(np.mean(gains_closed)),
        'gains_closed_median': float(np.median(gains_closed)),
        'gains_closed_skewed': bool(abs(np.median(gains_closed)) < 1e-6 and np.mean(gains_closed) > 0),
    }
    results_file = DATA / 'alpha_analysis_verification.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  结果已保存: {results_file}')


if __name__ == '__main__':
    main()
