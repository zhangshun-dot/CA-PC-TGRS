# -*- coding: utf-8 -*-
"""任务2：修复Oracle实现率计算bug
任务4：三因子（尺寸方差+文本混淆度+n_boxes）联合验证
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# ============================================================
# 工具函数
# ============================================================
def standardize(values):
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-10:
        return np.zeros_like(values), mean, std
    return (values - mean) / std, mean, std


def linear_regression_predict(X_train, y_train, X_test):
    X_train_aug = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_aug = np.column_stack([X_test, np.ones(len(X_test))])
    try:
        coef, _, _, _ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
        return X_test_aug @ coef
    except Exception:
        return np.full(len(X_test), np.mean(y_train))


def evaluate_prediction(y_true, y_pred):
    rho, p = stats.spearmanr(y_true, y_pred)
    r, _ = stats.pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {'spearman_rho': float(rho), 'spearman_p': float(p),
            'pearson_r': float(r), 'mae': float(mae), 'rmse': float(rmse)}


# ============================================================
# 任务2：修复Oracle实现率计算bug
# ============================================================
def fix_oracle_realization():
    print("=" * 60)
    print("任务2：修复Oracle实现率计算bug")
    print("=" * 60)

    with open(DATA / 'comprehensive_factor_experiment_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    per_class = data['task1_adaptive_correction']['per_class']
    print(f"\n总样本数: {len(per_class)}")

    # 排除oracle增益接近0的类别
    threshold = 0.001
    valid = [r for r in per_class if abs(r['gain_oracle_vs_fixed']) > threshold]
    excluded = [r for r in per_class if abs(r['gain_oracle_vs_fixed']) <= threshold]

    print(f"排除oracle增益<={threshold}的类别: {len(excluded)}个")
    for r in excluded:
        print(f"  {r['domain']} {r['src_cls']}→{r['tgt_cls']}: oracle增益={r['gain_oracle_vs_fixed']:+.6f}")

    print(f"\n有效样本数: {len(valid)}")

    # 重新计算实现率
    total_gain_adaptive = sum(r['gain_adaptive_vs_fixed'] for r in valid)
    total_gain_oracle = sum(r['gain_oracle_vs_fixed'] for r in valid)
    realization_ratio = total_gain_adaptive / total_gain_oracle if total_gain_oracle != 0 else 0

    mean_ap_fixed = np.mean([r['ap_fixed_1.0'] for r in valid])
    mean_ap_adaptive = np.mean([r['ap_adaptive'] for r in valid])
    mean_ap_oracle = np.mean([r['ap_oracle'] for r in valid])
    n_positive = sum(1 for r in valid if r['gain_adaptive_vs_fixed'] > 0)

    print(f"\n修复后结果（有效样本{len(valid)}个）:")
    print(f"  固定α=1.0  平均AP: {mean_ap_fixed:.4f}")
    print(f"  双因子自适应 平均AP: {mean_ap_adaptive:.4f} (增益: {mean_ap_adaptive-mean_ap_fixed:+.4f})")
    print(f"  Oracle最优  平均AP: {mean_ap_oracle:.4f} (增益: {mean_ap_oracle-mean_ap_fixed:+.4f})")
    print(f"  正增益类别: {n_positive}/{len(valid)} ({n_positive/len(valid)*100:.1f}%)")
    print(f"  Oracle实现率: {realization_ratio*100:.1f}%")

    # 逐类别详细
    print(f"\n逐类别详细（有效样本）:")
    print(f"{'域对':<10} {'类别':<20} {'固定AP':>8} {'自适应AP':>10} {'OracleAP':>10} {'自适应增益':>10} {'Oracle增益':>10} {'实现率':>8}")
    print("-" * 95)
    for r in valid:
        cls = f"{r['src_cls']}→{r['tgt_cls']}"
        ratio = r['gain_adaptive_vs_fixed'] / r['gain_oracle_vs_fixed'] * 100 if r['gain_oracle_vs_fixed'] != 0 else 0
        print(f"{r['domain']:<10} {cls:<20} {r['ap_fixed_1.0']:>8.4f} {r['ap_adaptive']:>10.4f} {r['ap_oracle']:>10.4f} {r['gain_adaptive_vs_fixed']:>+10.4f} {r['gain_oracle_vs_fixed']:>+10.4f} {ratio:>7.1f}%")

    return {
        'n_total': len(per_class),
        'n_valid': len(valid),
        'n_excluded': len(excluded),
        'excluded_classes': [f"{r['domain']} {r['src_cls']}→{r['tgt_cls']}" for r in excluded],
        'mean_ap_fixed': float(mean_ap_fixed),
        'mean_ap_adaptive': float(mean_ap_adaptive),
        'mean_ap_oracle': float(mean_ap_oracle),
        'mean_gain_adaptive': float(mean_ap_adaptive - mean_ap_fixed),
        'mean_gain_oracle': float(mean_ap_oracle - mean_ap_fixed),
        'n_positive': n_positive,
        'oracle_realization_ratio': float(realization_ratio),
    }


# ============================================================
# 任务4：三因子（尺寸方差+文本混淆度+n_boxes）联合验证
# ============================================================
def three_factor_validation():
    print("\n" + "=" * 60)
    print("任务4：三因子（尺寸方差+文本混淆度+n_boxes）联合验证")
    print("=" * 60)

    with open(DATA / 'new_factor_validation_results.json', 'r', encoding='utf-8') as f:
        factor_data = json.load(f)['all_factor_data']

    n = len(factor_data)
    size_var = np.array([d['box_size_variance'] for d in factor_data])
    text_conf = np.array([d['text_confusability'] for d in factor_data])
    n_boxes = np.array([d['n_boxes'] for d in factor_data])
    alphas = np.array([d['optimal_alpha'] for d in factor_data])

    print(f"\n样本数: {n}")
    print(f"n_boxes统计: mean={np.mean(n_boxes):.1f}, std={np.std(n_boxes):.1f}, range=[{np.min(n_boxes)}, {np.max(n_boxes)}]")
    print(f"n_boxes vs 最优α: Spearman ρ = {stats.spearmanr(n_boxes, alphas)[0]:.4f}")

    # 标准化
    size_var_std, _, _ = standardize(size_var)
    text_conf_std, _, _ = standardize(text_conf)
    n_boxes_std, _, _ = standardize(n_boxes)

    # 因子组合
    factor_configs = {
        'size_only': ('仅尺寸方差', [size_var_std]),
        'text_only': ('仅文本混淆度', [text_conf_std]),
        'nboxes_only': ('仅n_boxes', [n_boxes_std]),
        'size+text': ('尺寸+文本（双因子）', [size_var_std, text_conf_std]),
        'size+nboxes': ('尺寸+n_boxes', [size_var_std, n_boxes_std]),
        'text+nboxes': ('文本+n_boxes', [text_conf_std, n_boxes_std]),
        'size+text+nboxes': ('三因子联合', [size_var_std, text_conf_std, n_boxes_std]),
    }

    # 5折交叉验证
    print(f"\n5折交叉验证（5次重复）:")
    print(f"{'配置':<25} {'Spearman ρ':>12} {'标准差':>10} {'MAE':>8}")
    print("-" * 60)
    n_folds = 5
    n_repeats = 5
    cv_results = {}

    for config_key, (config_name, factors) in factor_configs.items():
        X = np.column_stack(factors)
        all_rho = []
        all_mae = []
        for repeat in range(n_repeats):
            np.random.seed(repeat * 100 + 42)
            indices = np.random.permutation(n)
            fold_size = n // n_folds
            y_pred_cv = np.zeros(n)
            for fold in range(n_folds):
                start = fold * fold_size
                end = start + fold_size if fold < n_folds - 1 else n
                test_idx = indices[start:end]
                train_idx = np.setdiff1d(indices, test_idx)
                y_pred_cv[test_idx] = linear_regression_predict(X[train_idx], alphas[train_idx], X[test_idx])
            rho, _ = stats.spearmanr(alphas, y_pred_cv)
            mae = np.mean(np.abs(alphas - y_pred_cv))
            all_rho.append(rho)
            all_mae.append(mae)
        mean_rho = np.mean(all_rho)
        std_rho = np.std(all_rho)
        mean_mae = np.mean(all_mae)
        cv_results[config_key] = {
            'name': config_name,
            'mean_rho': float(mean_rho),
            'std_rho': float(std_rho),
            'mean_mae': float(mean_mae),
        }
        print(f"{config_name:<25} {mean_rho:>12.4f} {std_rho:>10.4f} {mean_mae:>8.4f}")

    # LOO验证
    print(f"\nLOO验证:")
    print(f"{'配置':<25} {'Spearman ρ':>12} {'MAE':>8}")
    print("-" * 50)
    loo_results = {}
    for config_key, (config_name, factors) in factor_configs.items():
        X = np.column_stack(factors)
        y_pred_loo = np.zeros(n)
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(alphas, i)
            y_pred_loo[i] = linear_regression_predict(X_train, y_train, X[i:i+1])[0]
        metrics = evaluate_prediction(alphas, y_pred_loo)
        loo_results[config_key] = metrics
        print(f"{config_name:<25} {metrics['spearman_rho']:>12.4f} {metrics['mae']:>8.4f}")

    # 总结
    print(f"\n总结:")
    best_cv = max(cv_results.items(), key=lambda x: x[1]['mean_rho'])
    print(f"  5折CV最佳: {best_cv[1]['name']} (ρ={best_cv[1]['mean_rho']:.4f}±{best_cv[1]['std_rho']:.4f})")

    # 对比双因子 vs 三因子
    two_factor_cv = cv_results['size+text']
    three_factor_cv = cv_results['size+text+nboxes']
    print(f"\n  双因子（尺寸+文本）: ρ={two_factor_cv['mean_rho']:.4f}±{two_factor_cv['std_rho']:.4f}")
    print(f"  三因子（尺寸+文本+n_boxes）: ρ={three_factor_cv['mean_rho']:.4f}±{three_factor_cv['std_rho']:.4f}")
    diff = three_factor_cv['mean_rho'] - two_factor_cv['mean_rho']
    print(f"  差异: {diff:+.4f} ({'提升' if diff > 0 else '下降'})")

    return {
        'cv_5fold': cv_results,
        'loo': loo_results,
    }


# ============================================================
# 主函数
# ============================================================
def main():
    # 任务2
    task2_results = fix_oracle_realization()

    # 任务4
    task4_results = three_factor_validation()

    # 保存
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    final_results = {
        'task2_fixed_oracle_realization': convert(task2_results),
        'task4_three_factor_validation': convert(task4_results),
        'note': '任务2：修复Oracle实现率bug；任务4：三因子（尺寸方差+文本混淆度+n_boxes）联合验证',
    }
    out_file = DATA / 'followup_experiment_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_file}")


if __name__ == '__main__':
    main()
