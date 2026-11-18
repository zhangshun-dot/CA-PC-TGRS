# -*- coding: utf-8 -*-
"""综合实验：双因子自适应校正 + 改进物理先验 + 5折交叉验证
任务1：双因子（尺寸方差 + 文本混淆度）自适应校正实际AP对比
任务2：改进物理先验赋值（四因子：尺度/形态/类内变异/类间混淆）
任务3：5折交叉验证（代替单一LOO，更稳定）
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from scipy import stats
from pathlib import Path
import pickle
import tge_probe as T
from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate,
)

device = 'cpu'
OUT_FILE = DATA / 'comprehensive_factor_experiment_results.json'

# ============================================================
# 任务2：改进物理先验赋值（四因子）
# ============================================================
# 尺度因子：大目标=1.0，中目标=0.5，小目标=0.2
# 形态复杂度：简单形状=0.2，复杂形状=0.8
# 类内变异：低变异=0.2，高变异=0.8
# 类间混淆：低混淆=0.2，高混淆=0.8
IMPROVED_PHYSICAL_PRIOR = {
    'airplane': {'scale': 0.9, 'shape': 0.6, 'intra_var': 0.3, 'inter_conf': 0.4},
    'plane': {'scale': 0.9, 'shape': 0.6, 'intra_var': 0.3, 'inter_conf': 0.4},
    'ship': {'scale': 0.85, 'shape': 0.5, 'intra_var': 0.4, 'inter_conf': 0.3},
    'boat': {'scale': 0.4, 'shape': 0.4, 'intra_var': 0.5, 'inter_conf': 0.5},
    'vehicle': {'scale': 0.5, 'shape': 0.4, 'intra_var': 0.5, 'inter_conf': 0.6},
    'car': {'scale': 0.5, 'shape': 0.4, 'intra_var': 0.4, 'inter_conf': 0.6},
    'truck': {'scale': 0.6, 'shape': 0.5, 'intra_var': 0.4, 'inter_conf': 0.5},
    'bus': {'scale': 0.65, 'shape': 0.5, 'intra_var': 0.3, 'inter_conf': 0.4},
    'motorcycle': {'scale': 0.2, 'shape': 0.7, 'intra_var': 0.6, 'inter_conf': 0.5},
    'motor': {'scale': 0.2, 'shape': 0.7, 'intra_var': 0.6, 'inter_conf': 0.5},
    'bicycle': {'scale': 0.15, 'shape': 0.6, 'intra_var': 0.5, 'inter_conf': 0.6},
    'small-vehicle': {'scale': 0.3, 'shape': 0.4, 'intra_var': 0.5, 'inter_conf': 0.6},
    'large-vehicle': {'scale': 0.7, 'shape': 0.5, 'intra_var': 0.4, 'inter_conf': 0.5},
}


def get_improved_physical_prior(src_cls, tgt_cls):
    """获取改进的物理先验综合分"""
    src = IMPROVED_PHYSICAL_PRIOR.get(src_cls, {})
    tgt = IMPROVED_PHYSICAL_PRIOR.get(tgt_cls, {})
    p = src if src else tgt
    if not p:
        return 0.5
    # 综合分：尺度越大、形态越简单、类内变异越高、类间混淆越高 → 需要的校正越强
    return (0.35 * p['scale'] + 0.25 * (1.0 - p['shape']) +
            0.20 * p['intra_var'] + 0.20 * p['inter_conf'])


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


def find_nearest_alpha(pred_alpha):
    """找到最接近的α扫描值"""
    alphas = np.array(ALPHAS)
    idx = np.argmin(np.abs(alphas - pred_alpha))
    return ALPHAS[idx]


# ============================================================
# 任务1：双因子自适应校正实际实验
# ============================================================
def run_adaptive_correction_experiment():
    """跑双因子自适应校正的实际AP对比"""
    print("\n" + "=" * 60)
    print("任务1：双因子自适应校正实际AP对比")
    print("=" * 60)

    # 读取已有数据
    with open(DATA / 'new_factor_validation_results.json', 'r', encoding='utf-8') as f:
        factor_data = json.load(f)['all_factor_data']

    # 用全数据拟合双因子模型
    size_var = np.array([d['box_size_variance'] for d in factor_data])
    text_conf = np.array([d['text_confusability'] for d in factor_data])
    alphas_true = np.array([d['optimal_alpha'] for d in factor_data])

    size_var_std, _, _ = standardize(size_var)
    text_conf_std, _, _ = standardize(text_conf)
    X_two = np.column_stack([size_var_std, text_conf_std])

    # 全数据拟合
    y_pred_full = linear_regression_predict(X_two, alphas_true, X_two)

    print(f"\n双因子模型（全数据拟合）:")
    print(f"  尺寸方差系数 + 文本混淆度系数")
    metrics = evaluate_prediction(alphas_true, y_pred_full)
    print(f"  Spearman ρ = {metrics['spearman_rho']:.4f}, MAE = {metrics['mae']:.4f}")

    # 现在重新跑每个类别的AP，用：
    # 1. 固定α=1.0（默认）
    # 2. 双因子预测α（自适应）
    # 3. oracle最优α
    print(f"\n重新跑每个类别的AP（固定α=1.0 vs 双因子自适应 vs oracle）...")

    all_results = []
    for source_name in SOURCE_LOADERS.keys():
        print(f"\n--- 域对: {source_name} ---")
        load_func, targets = SOURCE_LOADERS[source_name]

        # 加载源域
        source_data = {}
        for eval_name, coco_cls, src_cls in targets:
            data = load_func(src_cls, n=50)
            if data is not None:
                source_data[src_cls] = data

        # 加载目标域
        target_data = {}
        valid_targets = []
        for eval_name, coco_cls, src_cls in targets:
            if src_cls not in source_data:
                continue
            data = load_target_coco(coco_cls, n=200)
            if data is not None:
                target_data[coco_cls] = data
                valid_targets.append((eval_name, coco_cls, src_cls))

        if len(valid_targets) < 2:
            continue

        # 构建USC
        all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
        all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])

        from usc_experiment_v12 import USC
        usc = USC(n_components=K_PCA)
        usc.fit(all_src, all_tgt)

        all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}
        protos_gpu = {s: torch.tensor(p, dtype=torch.float32, device=device) for s, p in all_protos.items()}

        # 文本偏移
        text_deltas = {}
        for eval_name, coco_cls, src_cls in valid_targets:
            s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
            d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
            text_deltas[src_cls] = d_emb - s_emb

        # 对每个类别跑三种配置
        for eval_name, coco_cls, src_cls in valid_targets:
            # 找到这个类别的因子数据
            factor_entry = None
            for d in factor_data:
                if d['domain'] == source_name and d['src_cls'] == src_cls and d['tgt_cls'] == coco_cls:
                    factor_entry = d
                    break
            if factor_entry is None:
                continue

            # 双因子预测α
            idx = factor_data.index(factor_entry)
            pred_alpha = y_pred_full[idx]
            adaptive_alpha = find_nearest_alpha(pred_alpha)
            oracle_alpha = factor_entry['optimal_alpha']

            # 跑三种配置的AP
            aps = {}
            for alpha_name, alpha_val in [('fixed_1.0', 1.0), ('adaptive', adaptive_alpha), ('oracle', oracle_alpha)]:
                p_corr_gpu = usc.correct_prototype(protos_gpu[src_cls], alpha_val)
                p_corr = p_corr_gpu.cpu().numpy() if hasattr(p_corr_gpu, 'cpu') else p_corr_gpu
                neg_raw = build_neg_proto(all_protos, src_cls)
                if neg_raw is not None:
                    neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
                    neg_corr_gpu = usc.correct_prototype(neg_gpu, alpha_val)
                    neg = neg_corr_gpu.cpu().numpy() if hasattr(neg_corr_gpu, 'cpu') else neg_corr_gpu
                else:
                    neg = None
                delta = text_deltas[src_cls]
                p_hat = p_corr + 1.0 * delta
                p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
                ap = evaluate(target_data[coco_cls], p_hat, neg)
                aps[alpha_name] = ap

            result = {
                'domain': source_name,
                'src_cls': src_cls,
                'tgt_cls': coco_cls,
                'oracle_alpha': oracle_alpha,
                'pred_alpha_raw': float(pred_alpha),
                'adaptive_alpha': adaptive_alpha,
                'ap_fixed_1.0': aps['fixed_1.0'],
                'ap_adaptive': aps['adaptive'],
                'ap_oracle': aps['oracle'],
                'gain_adaptive_vs_fixed': aps['adaptive'] - aps['fixed_1.0'],
                'gain_oracle_vs_fixed': aps['oracle'] - aps['fixed_1.0'],
                'oracle_realization_ratio': (aps['adaptive'] - aps['fixed_1.0']) / (aps['oracle'] - aps['fixed_1.0'] + 1e-8),
            }
            all_results.append(result)
            print(f"  {src_cls}→{coco_cls}: 固定={aps['fixed_1.0']:.4f}, 自适应(α={adaptive_alpha})={aps['adaptive']:.4f}, oracle(α={oracle_alpha})={aps['oracle']:.4f}")

    # 汇总
    print(f"\n{'='*60}")
    print("任务1汇总")
    print(f"{'='*60}")
    n = len(all_results)
    mean_fixed = np.mean([r['ap_fixed_1.0'] for r in all_results])
    mean_adaptive = np.mean([r['ap_adaptive'] for r in all_results])
    mean_oracle = np.mean([r['ap_oracle'] for r in all_results])
    mean_gain_adaptive = np.mean([r['gain_adaptive_vs_fixed'] for r in all_results])
    mean_gain_oracle = np.mean([r['gain_oracle_vs_fixed'] for r in all_results])
    mean_realization = np.mean([r['oracle_realization_ratio'] for r in all_results])
    n_positive = sum(1 for r in all_results if r['gain_adaptive_vs_fixed'] > 0)

    print(f"样本数: {n}")
    print(f"固定α=1.0  平均AP: {mean_fixed:.4f}")
    print(f"双因子自适应 平均AP: {mean_adaptive:.4f} (增益: {mean_gain_adaptive:+.4f})")
    print(f"Oracle最优  平均AP: {mean_oracle:.4f} (增益: {mean_gain_oracle:+.4f})")
    print(f"自适应正增益类别数: {n_positive}/{n} ({n_positive/n*100:.1f}%)")
    print(f"Oracle实现率: {mean_realization*100:.1f}%")

    return {
        'per_class': all_results,
        'summary': {
            'n_samples': n,
            'mean_ap_fixed': float(mean_fixed),
            'mean_ap_adaptive': float(mean_adaptive),
            'mean_ap_oracle': float(mean_oracle),
            'mean_gain_adaptive': float(mean_gain_adaptive),
            'mean_gain_oracle': float(mean_gain_oracle),
            'n_positive': n_positive,
            'oracle_realization_ratio': float(mean_realization),
        }
    }


# ============================================================
# 任务2+3：改进物理先验 + 5折交叉验证
# ============================================================
def run_improved_prior_and_cv():
    """改进物理先验 + 5折交叉验证"""
    print("\n" + "=" * 60)
    print("任务2+3：改进物理先验 + 5折交叉验证")
    print("=" * 60)

    with open(DATA / 'new_factor_validation_results.json', 'r', encoding='utf-8') as f:
        factor_data = json.load(f)['all_factor_data']

    n = len(factor_data)
    size_var = np.array([d['box_size_variance'] for d in factor_data])
    text_conf = np.array([d['text_confusability'] for d in factor_data])
    physical_improved = np.array([get_improved_physical_prior(d['src_cls'], d['tgt_cls']) for d in factor_data])
    alphas = np.array([d['optimal_alpha'] for d in factor_data])

    size_var_std, _, _ = standardize(size_var)
    text_conf_std, _, _ = standardize(text_conf)
    physical_std, _, _ = standardize(physical_improved)

    print(f"\n改进物理先验统计: mean={np.mean(physical_improved):.4f}, std={np.std(physical_improved):.4f}")
    print(f"改进物理先验 vs 最优α: Spearman ρ = {stats.spearmanr(physical_improved, alphas)[0]:.4f}")

    # 因子组合
    factor_configs = {
        'size_only': ('仅尺寸方差', [size_var_std]),
        'text_only': ('仅文本混淆度', [text_conf_std]),
        'physical_improved_only': ('仅改进物理先验', [physical_std]),
        'size+text': ('尺寸+文本（双因子）', [size_var_std, text_conf_std]),
        'size+physical': ('尺寸+改进物理先验', [size_var_std, physical_std]),
        'text+physical': ('文本+改进物理先验', [text_conf_std, physical_std]),
        'size+text+physical': ('三因子联合', [size_var_std, text_conf_std, physical_std]),
    }

    # 5折交叉验证
    print(f"\n5折交叉验证（5次重复，每次随机划分）:")
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
            'all_rho': [float(r) for r in all_rho],
        }
        print(f"  {config_name:<25}: ρ={mean_rho:.4f}±{std_rho:.4f}, MAE={mean_mae:.4f}")

    # LOO对比
    print(f"\nLOO验证（对比）:")
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
        print(f"  {config_name:<25}: ρ={metrics['spearman_rho']:.4f}, MAE={metrics['mae']:.4f}")

    return {
        'cv_5fold': cv_results,
        'loo': loo_results,
        'physical_prior_values': {f"{d['src_cls']}→{d['tgt_cls']}": float(get_improved_physical_prior(d['src_cls'], d['tgt_cls'])) for d in factor_data},
    }


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("综合实验：双因子自适应校正 + 改进物理先验 + 5折CV")
    print("=" * 60)

    # 任务1
    task1_results = run_adaptive_correction_experiment()

    # 任务2+3
    task23_results = run_improved_prior_and_cv()

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
        'task1_adaptive_correction': convert(task1_results),
        'task23_improved_prior_cv': convert(task23_results),
        'note': '综合实验：双因子自适应校正实际AP + 改进物理先验四因子 + 5折交叉验证',
    }
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT_FILE}")

    # 最终总结
    print(f"\n{'='*60}")
    print("最终总结")
    print(f"{'='*60}")
    s = task1_results['summary']
    print(f"任务1（双因子自适应校正）:")
    print(f"  固定α=1.0  平均AP: {s['mean_ap_fixed']:.4f}")
    print(f"  双因子自适应 平均AP: {s['mean_ap_adaptive']:.4f} (增益: {s['mean_gain_adaptive']:+.4f})")
    print(f"  Oracle最优  平均AP: {s['mean_ap_oracle']:.4f} (增益: {s['mean_gain_oracle']:+.4f})")
    print(f"  正增益类别: {s['n_positive']}/{s['n_samples']} ({s['n_positive']/s['n_samples']*100:.1f}%)")
    print(f"  Oracle实现率: {s['oracle_realization_ratio']*100:.1f}%")

    print(f"\n任务2+3（改进物理先验 + 5折CV）:")
    best_cv = max(task23_results['cv_5fold'].items(), key=lambda x: x[1]['mean_rho'])
    print(f"  5折CV最佳: {best_cv[1]['name']} (ρ={best_cv[1]['mean_rho']:.4f}±{best_cv[1]['std_rho']:.4f})")


if __name__ == '__main__':
    main()
