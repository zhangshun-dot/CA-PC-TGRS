# -*- coding: utf-8 -*-
"""analyze_untagged_factors.py — 分析无标签指标和Oracleα的相关性
目标：找到不需要目标域标注、但能预测基线AP/Oracleα的指标
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

def load_coco_data():
    """从新因子验证结果中加载COCO目标域数据"""
    coco_file = DATA / 'new_factor_validation_results.json'
    with open(coco_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data.get('all_factor_data', []):
        samples.append({
            'src': item.get('domain', 'unknown'),
            'cls': item.get('src_cls', 'unknown'),
            'target': 'COCO',
            'sv': item['box_size_variance'],
            'tc': item['text_confusability'],
            'tds': item.get('text_domain_sensitivity', 0),
            'dcv': item.get('det_conf_variance', 0),
            'dcm': item.get('det_conf_mean', 0),
            'n_boxes': item.get('n_boxes', 0),
            'bsm': item.get('box_size_mean', 0),
            'oracle_alpha': item['optimal_alpha'],
            'baseline_ap': item.get('optimal_ap', 0),
        })
    return samples


def load_voc_data():
    """从VOC目标域SOTA对比结果中加载VOC目标域数据"""
    voc_file = DATA / 'voc_sota_comparison_results.json'
    with open(voc_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for src_name in data:
        for cls in data[src_name]:
            item = data[src_name][cls]
            samples.append({
                'src': src_name,
                'cls': cls,
                'target': 'VOC',
                'sv': item['box_size_variance'],
                'tc': item['text_confusability'],
                'oracle_alpha': item['oracle_alpha'],
                'baseline_ap': item.get('Baseline', 0),
            })
    return samples


def analyze_correlation(samples, factor_name, target_name='oracle_alpha'):
    """分析单个因子和目标的相关性"""
    x = np.array([s[factor_name] for s in samples if factor_name in s and s[factor_name] is not None])
    y = np.array([s[target_name] for s in samples if factor_name in s and s[factor_name] is not None])

    if len(x) < 3:
        return None

    rho, p = stats.spearmanr(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    return {
        'factor': factor_name,
        'n': len(x),
        'spearman_rho': rho,
        'spearman_p': p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'x_mean': x.mean(),
        'x_std': x.std(),
        'y_mean': y.mean(),
    }


def main():
    print('=' * 80)
    print('分析无标签指标和Oracleα的相关性')
    print('=' * 80)

    # 加载数据
    coco_samples = load_coco_data()
    voc_samples = load_voc_data()
    all_samples = coco_samples + voc_samples

    print(f'\nCOCO样本数: {len(coco_samples)}')
    print(f'VOC样本数: {len(voc_samples)}')
    print(f'合并样本数: {len(all_samples)}')

    # 要分析的无标签因子
    factors = ['sv', 'tc', 'tds', 'dcv', 'dcm', 'n_boxes', 'bsm', 'baseline_ap']
    factor_names = {
        'sv': '候选框尺寸方差',
        'tc': '文本混淆度',
        'tds': '文本域敏感度',
        'dcv': '检测器置信度方差',
        'dcm': '检测器置信度均值',
        'n_boxes': '候选框数量',
        'bsm': '候选框尺寸均值',
        'baseline_ap': '基线AP（需要标注，参考用）',
    }

    # 分别在COCO、VOC、合并样本上分析
    for dataset_name, samples in [('COCO', coco_samples), ('VOC', voc_samples), ('COCO+VOC', all_samples)]:
        print(f'\n{"=" * 80}')
        print(f'数据集: {dataset_name} (N={len(samples)})')
        print(f'{"=" * 80}')
        print(f'{"因子":<20s} {"Spearman ρ":>12s} {"p值":>10s} {"Pearson r":>12s} {"显著性":>8s}')
        print('-' * 70)

        results = []
        for factor in factors:
            result = analyze_correlation(samples, factor)
            if result:
                sig = '***' if result['spearman_p'] < 0.001 else ('**' if result['spearman_p'] < 0.01 else ('*' if result['spearman_p'] < 0.05 else ''))
                print(f'{factor_names[factor]:<20s} {result["spearman_rho"]:>12.4f} {result["spearman_p"]:>10.4f} {result["pearson_r"]:>12.4f} {sig:>8s}')
                results.append(result)

        # 按|ρ|排序
        results.sort(key=lambda x: abs(x['spearman_rho']), reverse=True)
        print(f'\n按|Spearman ρ|排序:')
        for i, r in enumerate(results[:5]):
            print(f'  {i+1}. {factor_names[r["factor"]]}: ρ={r["spearman_rho"]:.4f}, p={r["spearman_p"]:.4f}')

    # 分析：哪些无标签因子能预测基线AP？
    print(f'\n{"=" * 80}')
    print('无标签因子 vs 基线AP（找到能估计基线AP的无标签指标）')
    print(f'{"=" * 80}')

    for dataset_name, samples in [('COCO', coco_samples), ('VOC', voc_samples), ('COCO+VOC', all_samples)]:
        print(f'\n--- {dataset_name} ---')
        for factor in factors:
            if factor == 'baseline_ap':
                continue
            result = analyze_correlation(samples, factor, 'baseline_ap')
            if result:
                sig = '***' if result['spearman_p'] < 0.001 else ('**' if result['spearman_p'] < 0.01 else ('*' if result['spearman_p'] < 0.05 else ''))
                print(f'  {factor_names[factor]:<20s}: ρ={result["spearman_rho"]:.4f}, p={result["spearman_p"]:.4f} {sig}')

    # 多因子回归分析
    print(f'\n{"=" * 80}')
    print('多因子回归分析（用无标签因子预测Oracleα）')
    print(f'{"=" * 80}')

    # 在COCO+VOC合并样本上，用逐步回归选择最佳因子组合
    from itertools import combinations

    available_factors = ['sv', 'tc', 'n_boxes', 'dcm', 'bsm']
    best_combo = None
    best_rho = -1

    print(f'\n可用无标签因子: {[factor_names[f] for f in available_factors]}')
    print(f'\n逐步回归结果（按Spearman ρ排序）:')

    all_results = []
    for k in range(1, len(available_factors) + 1):
        for combo in combinations(available_factors, k):
            # 收集有效样本
            valid_samples = []
            for s in all_samples:
                if all(f in s and s[f] is not None for f in combo):
                    valid_samples.append(s)

            if len(valid_samples) < 10:
                continue

            X = np.column_stack([[s[f] for s in valid_samples] for f in combo])
            y = np.array([s['oracle_alpha'] for s in valid_samples])

            # 标准化
            X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            X_norm = np.column_stack([X_norm, np.ones(len(valid_samples))])

            # OLS回归
            w, _, _, _ = np.linalg.lstsq(X_norm, y, rcond=None)
            y_pred = X_norm @ w

            rho, p = stats.spearmanr(y, y_pred)
            mae = np.mean(np.abs(y - y_pred))

            all_results.append({
                'combo': combo,
                'n': len(valid_samples),
                'rho': rho,
                'p': p,
                'mae': mae,
                'weights': w,
            })

    all_results.sort(key=lambda x: x['rho'], reverse=True)
    for i, r in enumerate(all_results[:10]):
        combo_names = [factor_names[f] for f in r['combo']]
        print(f'  {i+1}. {" + ".join(combo_names):<40s} N={r["n"]:2d} ρ={r["rho"]:.4f} MAE={r["mae"]:.4f}')

    # 最佳组合的详细结果
    if all_results:
        best = all_results[0]
        print(f'\n最佳组合: {" + ".join([factor_names[f] for f in best["combo"]])}')
        print(f'  N={best["n"]}, ρ={best["rho"]:.4f}, MAE={best["mae"]:.4f}')
        print(f'  权重: {best["weights"]}')

    # LODO验证：用COCO训练，VOC测试
    print(f'\n{"=" * 80}')
    print('LODO验证：用COCO训练，VOC测试（真正的跨目标域泛化）')
    print(f'{"=" * 80}')

    if all_results:
        best_combo = all_results[0]['combo']
    else:
        best_combo = ('sv', 'tc')

    # 用COCO训练
    coco_valid = [s for s in coco_samples if all(f in s and s[f] is not None for f in best_combo)]
    X_train = np.column_stack([[s[f] for s in coco_valid] for f in best_combo])
    y_train = np.array([s['oracle_alpha'] for s in coco_valid])
    X_train_norm = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
    X_train_norm = np.column_stack([X_train_norm, np.ones(len(coco_valid))])
    w_train, _, _, _ = np.linalg.lstsq(X_train_norm, y_train, rcond=None)

    # 用VOC测试
    voc_valid = [s for s in voc_samples if all(f in s and s[f] is not None for f in best_combo)]
    X_test = np.column_stack([[s[f] for s in voc_valid] for f in best_combo])
    y_test = np.array([s['oracle_alpha'] for s in voc_valid])
    X_test_norm = (X_test - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
    X_test_norm = np.column_stack([X_test_norm, np.ones(len(voc_valid))])
    y_pred = X_test_norm @ w_train
    y_pred = np.clip(y_pred, 0, 2)

    rho_test, p_test = stats.spearmanr(y_test, y_pred)
    mae_test = np.mean(np.abs(y_test - y_pred))

    print(f'\n因子组合: {" + ".join([factor_names[f] for f in best_combo])}')
    print(f'COCO训练: N={len(coco_valid)}, ρ={stats.spearmanr(y_train, X_train_norm @ w_train)[0]:.4f}')
    print(f'VOC测试:  N={len(voc_valid)}, ρ={rho_test:.4f}, MAE={mae_test:.4f}')
    print(f'\nVOC逐类预测:')
    for i, s in enumerate(voc_valid):
        print(f'  {s["src"]}/{s["cls"]}: 真实α={y_test[i]:.2f}, 预测α={y_pred[i]:.2f}, 误差={y_pred[i]-y_test[i]:+.2f}')

    print(f'\n结论: 如果VOC测试ρ>0且MAE小，说明该因子组合有跨目标域泛化能力')


if __name__ == '__main__':
    main()
