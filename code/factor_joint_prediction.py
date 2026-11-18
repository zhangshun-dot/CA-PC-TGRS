# -*- coding: utf-8 -*-
"""三因子联合预测 + LOO验证
因子1：候选框尺寸方差（ρ=-0.724）
因子2：文本混淆度（ρ=+0.620）
因子3：物理先验（如果有数据）
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
from scipy import stats
from pathlib import Path

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
RESULT_FILE = DATA / 'new_factor_validation_results.json'
OUT_FILE = DATA / 'factor_joint_prediction_results.json'

# 物理先验（人工赋值，基于类别固有属性）
# 尺度因子：大目标=1.0，中目标=0.5，小目标=0.2
# 形态复杂度：简单形状=0.2，复杂形状=0.8
PHYSICAL_PRIOR = {
    'airplane': {'scale': 0.9, 'shape': 0.6},
    'plane': {'scale': 0.9, 'shape': 0.6},
    'ship': {'scale': 0.85, 'shape': 0.5},
    'boat': {'scale': 0.4, 'shape': 0.4},
    'vehicle': {'scale': 0.5, 'shape': 0.4},
    'car': {'scale': 0.5, 'shape': 0.4},
    'truck': {'scale': 0.6, 'shape': 0.5},
    'bus': {'scale': 0.65, 'shape': 0.5},
    'motorcycle': {'scale': 0.2, 'shape': 0.7},
    'motor': {'scale': 0.2, 'shape': 0.7},
    'bicycle': {'scale': 0.15, 'shape': 0.6},
    'small-vehicle': {'scale': 0.3, 'shape': 0.4},
    'large-vehicle': {'scale': 0.7, 'shape': 0.5},
}


def get_physical_prior(src_cls, tgt_cls):
    """获取物理先验值（尺度+形态的综合）"""
    src = PHYSICAL_PRIOR.get(src_cls, {})
    tgt = PHYSICAL_PRIOR.get(tgt_cls, {})
    if not src and not tgt:
        return 0.5
    scale = src.get('scale', tgt.get('scale', 0.5))
    shape = src.get('shape', tgt.get('shape', 0.5))
    # 物理先验综合分：尺度越大、形态越简单，需要的校正越强
    return 0.6 * scale + 0.4 * (1.0 - shape)


def standardize(values):
    """Z-score标准化"""
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-10:
        return np.zeros_like(values), mean, std
    return (values - mean) / std, mean, std


def linear_regression_predict(X_train, y_train, X_test):
    """简单线性回归预测（带截距）"""
    # 添加截距项
    X_train_aug = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_aug = np.column_stack([X_test, np.ones(len(X_test))])
    # 最小二乘解
    try:
        coef, _, _, _ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)
        y_pred = X_test_aug @ coef
        return y_pred
    except Exception:
        return np.full(len(X_test), np.mean(y_train))


def evaluate_prediction(y_true, y_pred):
    """评估预测性能"""
    # Spearman相关
    rho, p = stats.spearmanr(y_true, y_pred)
    # Pearson相关
    r, _ = stats.pearsonr(y_true, y_pred)
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {
        'spearman_rho': float(rho),
        'spearman_p': float(p),
        'pearson_r': float(r),
        'mae': float(mae),
        'rmse': float(rmse),
    }


def main():
    print("=" * 60)
    print("三因子联合预测 + LOO验证")
    print("=" * 60)

    # 1. 读取数据
    with open(RESULT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_data = data['all_factor_data']
    print(f"\n样本数: {len(all_data)}")

    # 2. 提取因子
    size_var = np.array([d['box_size_variance'] for d in all_data])
    text_conf = np.array([d['text_confusability'] for d in all_data])
    physical = np.array([get_physical_prior(d['src_cls'], d['tgt_cls']) for d in all_data])
    alphas = np.array([d['optimal_alpha'] for d in all_data])

    print(f"\n因子统计:")
    print(f"  候选框尺寸方差: mean={np.mean(size_var):.2f}, std={np.std(size_var):.2f}")
    print(f"  文本混淆度: mean={np.mean(text_conf):.4f}, std={np.std(text_conf):.4f}")
    print(f"  物理先验: mean={np.mean(physical):.4f}, std={np.std(physical):.4f}")
    print(f"  最优α: mean={np.mean(alphas):.4f}, std={np.std(alphas):.4f}")

    # 3. 标准化因子
    size_var_std, _, _ = standardize(size_var)
    text_conf_std, _, _ = standardize(text_conf)
    physical_std, _, _ = standardize(physical)

    # 4. 定义因子组合
    factor_configs = {
        'size_var_only': ('仅候选框尺寸方差', [size_var_std]),
        'text_conf_only': ('仅文本混淆度', [text_conf_std]),
        'physical_only': ('仅物理先验', [physical_std]),
        'size+text': ('尺寸方差 + 文本混淆度', [size_var_std, text_conf_std]),
        'size+physical': ('尺寸方差 + 物理先验', [size_var_std, physical_std]),
        'text+physical': ('文本混淆度 + 物理先验', [text_conf_std, physical_std]),
        'size+text+physical': ('三因子联合', [size_var_std, text_conf_std, physical_std]),
    }

    # 5. 全数据拟合 + 预测
    print(f"\n{'='*60}")
    print("全数据拟合 + 预测（训练集=测试集，上界）")
    print(f"{'='*60}")
    print(f"{'配置':<25} {'Spearman ρ':>12} {'p值':>10} {'Pearson r':>12} {'MAE':>8} {'RMSE':>8}")
    print("-" * 80)

    full_results = {}
    for config_key, (config_name, factors) in factor_configs.items():
        X = np.column_stack(factors)
        y_pred = linear_regression_predict(X, alphas, X)
        metrics = evaluate_prediction(alphas, y_pred)
        full_results[config_key] = metrics
        print(f"{config_name:<25} {metrics['spearman_rho']:>12.4f} {metrics['spearman_p']:>10.4f} {metrics['pearson_r']:>12.4f} {metrics['mae']:>8.4f} {metrics['rmse']:>8.4f}")

    # 6. LOO验证
    print(f"\n{'='*60}")
    print("留一法（LOO）验证")
    print(f"{'='*60}")
    print(f"{'配置':<25} {'Spearman ρ':>12} {'p值':>10} {'Pearson r':>12} {'MAE':>8} {'RMSE':>8}")
    print("-" * 80)

    loo_results = {}
    n = len(all_data)
    for config_key, (config_name, factors) in factor_configs.items():
        X = np.column_stack(factors)
        y_pred_loo = np.zeros(n)
        for i in range(n):
            # 留出第i个样本
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(alphas, i)
            X_test = X[i:i+1]
            # 预测
            y_pred_loo[i] = linear_regression_predict(X_train, y_train, X_test)[0]
        metrics = evaluate_prediction(alphas, y_pred_loo)
        loo_results[config_key] = metrics
        print(f"{config_name:<25} {metrics['spearman_rho']:>12.4f} {metrics['spearman_p']:>10.4f} {metrics['pearson_r']:>12.4f} {metrics['mae']:>8.4f} {metrics['rmse']:>8.4f}")

    # 7. 三因子LOO的详细预测
    print(f"\n{'='*60}")
    print("三因子联合 LOO 详细预测")
    print(f"{'='*60}")
    X_three = np.column_stack([size_var_std, text_conf_std, physical_std])
    y_pred_three_loo = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X_three, i, axis=0)
        y_train = np.delete(alphas, i)
        X_test = X_three[i:i+1]
        y_pred_three_loo[i] = linear_regression_predict(X_train, y_train, X_test)[0]

    print(f"{'域对':<10} {'源类别':<15} {'目标类别':<15} {'真实α':>8} {'预测α':>8} {'误差':>8}")
    print("-" * 75)
    for i, d in enumerate(all_data):
        err = y_pred_three_loo[i] - alphas[i]
        print(f"{d['domain']:<10} {d['src_cls']:<15} {d['tgt_cls']:<15} {alphas[i]:>8.2f} {y_pred_three_loo[i]:>8.2f} {err:>8.2f}")

    # 8. 保存结果
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
        'n_samples': n,
        'factor_configs': {k: v[0] for k, v in factor_configs.items()},
        'full_data_results': convert(full_results),
        'loo_results': convert(loo_results),
        'three_factor_loo_predictions': [
            {
                'domain': all_data[i]['domain'],
                'src_cls': all_data[i]['src_cls'],
                'tgt_cls': all_data[i]['tgt_cls'],
                'true_alpha': float(alphas[i]),
                'pred_alpha': float(y_pred_three_loo[i]),
                'error': float(y_pred_three_loo[i] - alphas[i]),
            }
            for i in range(n)
        ],
        'note': '三因子联合预测 + LOO验证（候选框尺寸方差 + 文本混淆度 + 物理先验）',
    }
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT_FILE}")

    # 9. 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    best_loo = max(loo_results.items(), key=lambda x: abs(x[1]['spearman_rho']))
    print(f"LOO验证最佳配置: {factor_configs[best_loo[0]][0]}")
    print(f"  Spearman ρ = {best_loo[1]['spearman_rho']:.4f}, p = {best_loo[1]['spearman_p']:.4f}")
    print(f"  MAE = {best_loo[1]['mae']:.4f}, RMSE = {best_loo[1]['rmse']:.4f}")

    # 对比单一因子 vs 多因子
    print(f"\n单一因子 vs 多因子（LOO）:")
    for key in ['size_var_only', 'text_conf_only', 'physical_only', 'size+text', 'size+text+physical']:
        r = loo_results[key]
        print(f"  {factor_configs[key][0]:<25}: ρ={r['spearman_rho']:.4f}, MAE={r['mae']:.4f}")


if __name__ == '__main__':
    main()
