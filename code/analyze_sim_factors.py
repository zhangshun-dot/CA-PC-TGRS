# -*- coding: utf-8 -*-
"""analyze_sim_factors.py — 深入分析sim_pos_mean和sim_pos_std作为零训练预测因子
1. z-score合并检验（判断是否主要在做数据集判别）
2. 多因子回归（sim_pos_mean + sim_pos_std）
3. 两级模型（gate + 回归）
4. 和baseline_ap对比
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA = Path(r'C:\Users\张三\Desktop\实验数据')


def load_samples():
    """加载κ_c分析的样本数据"""
    with open(DATA / 'kappa_analysis_samples.json', 'r', encoding='utf-8') as f:
        samples = json.load(f)
    return samples


def zscore_test(samples, factor):
    """z-score合并检验"""
    values_raw = np.array([s[factor] for s in samples])
    alphas = np.array([s['best_alpha'] for s in samples])

    # 原始值
    rho_raw, p_raw = stats.spearmanr(values_raw, alphas)

    # 数据集内z-score
    values_z = values_raw.copy()
    for domain in ['COCO', 'VOC']:
        mask = np.array([s['target_domain'] == domain for s in samples])
        if mask.sum() > 1:
            values_z[mask] = (values_raw[mask] - values_raw[mask].mean()) / (values_raw[mask].std() + 1e-8)
    rho_z, p_z = stats.spearmanr(values_z, alphas)

    return rho_raw, p_raw, rho_z, p_z


def multi_factor_regression(samples):
    """多因子回归：sim_pos_mean + sim_pos_std"""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    factors = ['sim_pos_mean', 'sim_pos_std']
    X = np.array([[s[f] for f in factors] for s in samples])
    y = np.array([s['best_alpha'] for s in samples])
    y_binary = (y > 0).astype(int)

    # 标准化
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    loo = LeaveOneOut()

    # OLS回归
    y_pred_ols = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X_norm):
        model = LinearRegression()
        model.fit(X_norm[train_idx], y[train_idx])
        y_pred_ols[test_idx] = model.predict(X_norm[test_idx])

    ss_res = np.sum((y - y_pred_ols) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_ols = 1 - ss_res / ss_tot
    rho_ols, p_ols = stats.spearmanr(y_pred_ols, y)

    # 全数据拟合系数
    model_full = LinearRegression()
    model_full.fit(X_norm, y)

    # 两级模型：gate + 回归
    y_pred_two = np.zeros_like(y)
    for train_idx, test_idx in loo.split(X_norm):
        # 第一级：gate
        gate = LogisticRegression(random_state=42)
        gate.fit(X_norm[train_idx], y_binary[train_idx])
        gate_pred = gate.predict(X_norm[test_idx])

        # 第二级：只在需要校正的样本上回归
        need_correction = y_binary[train_idx] > 0
        if need_correction.sum() > 1:
            reg = LinearRegression()
            reg.fit(X_norm[train_idx][need_correction], y[train_idx][need_correction])
            reg_pred = reg.predict(X_norm[test_idx])
        else:
            reg_pred = np.ones(len(test_idx)) * y[train_idx].mean()

        y_pred_two[test_idx] = gate_pred * np.maximum(0, reg_pred)

    ss_res_two = np.sum((y - y_pred_two) ** 2)
    r2_two = 1 - ss_res_two / ss_tot
    rho_two, p_two = stats.spearmanr(y_pred_two, y)

    return {
        'ols_r2': r2_ols,
        'ols_rho': rho_ols,
        'ols_p': p_ols,
        'two_r2': r2_two,
        'two_rho': rho_two,
        'two_p': p_two,
        'coef': dict(zip(factors, model_full.coef_)),
        'intercept': model_full.intercept_,
    }


def compare_with_baseline(samples):
    """和baseline_ap对比"""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    alphas = np.array([s['best_alpha'] for s in samples])
    loo = LeaveOneOut()

    # baseline_ap单因子
    X_ap = np.array([[s['baseline_ap']] for s in samples])
    scaler_ap = StandardScaler()
    X_ap_norm = scaler_ap.fit_transform(X_ap)

    y_pred_ap = np.zeros_like(alphas)
    for train_idx, test_idx in loo.split(X_ap_norm):
        model = LinearRegression()
        model.fit(X_ap_norm[train_idx], alphas[train_idx])
        y_pred_ap[test_idx] = model.predict(X_ap_norm[test_idx])

    ss_res = np.sum((alphas - y_pred_ap) ** 2)
    ss_tot = np.sum((alphas - alphas.mean()) ** 2)
    r2_ap = 1 - ss_res / ss_tot
    rho_ap, _ = stats.spearmanr(y_pred_ap, alphas)

    # sim_pos_mean + sim_pos_std
    factors = ['sim_pos_mean', 'sim_pos_std']
    X_sim = np.array([[s[f] for f in factors] for s in samples])
    scaler_sim = StandardScaler()
    X_sim_norm = scaler_sim.fit_transform(X_sim)

    y_pred_sim = np.zeros_like(alphas)
    for train_idx, test_idx in loo.split(X_sim_norm):
        model = LinearRegression()
        model.fit(X_sim_norm[train_idx], alphas[train_idx])
        y_pred_sim[test_idx] = model.predict(X_sim_norm[test_idx])

    ss_res_sim = np.sum((alphas - y_pred_sim) ** 2)
    r2_sim = 1 - ss_res_sim / ss_tot
    rho_sim, _ = stats.spearmanr(y_pred_sim, alphas)

    # 组合：baseline_ap + sim_pos_mean + sim_pos_std
    X_all = np.array([[s['baseline_ap'], s['sim_pos_mean'], s['sim_pos_std']] for s in samples])
    scaler_all = StandardScaler()
    X_all_norm = scaler_all.fit_transform(X_all)

    y_pred_all = np.zeros_like(alphas)
    for train_idx, test_idx in loo.split(X_all_norm):
        model = LinearRegression()
        model.fit(X_all_norm[train_idx], alphas[train_idx])
        y_pred_all[test_idx] = model.predict(X_all_norm[test_idx])

    ss_res_all = np.sum((alphas - y_pred_all) ** 2)
    r2_all = 1 - ss_res_all / ss_tot
    rho_all, _ = stats.spearmanr(y_pred_all, alphas)

    return {
        'baseline_ap': {'r2': r2_ap, 'rho': rho_ap},
        'sim_factors': {'r2': r2_sim, 'rho': rho_sim},
        'all_combined': {'r2': r2_all, 'rho': rho_all},
    }


def main():
    print('=' * 80)
    print('深入分析sim_pos_mean和sim_pos_std作为零训练预测因子')
    print('=' * 80)

    samples = load_samples()
    print(f'\n总样本数: {len(samples)}')
    print(f'  COCO: {sum(1 for s in samples if s["target_domain"] == "COCO")}')
    print(f'  VOC: {sum(1 for s in samples if s["target_domain"] == "VOC")}')

    # 1. z-score合并检验
    print(f'\n{"=" * 80}')
    print('1. z-score合并检验（判断是否主要在做数据集判别）')
    print(f'{"=" * 80}')

    for factor in ['sim_pos_mean', 'sim_pos_std', 'baseline_ap', 'kappa']:
        rho_raw, p_raw, rho_z, p_z = zscore_test(samples, factor)
        diff = rho_raw - rho_z
        print(f'\n  {factor}:')
        print(f'    raw: ρ={rho_raw:.4f} (p={p_raw:.4f})')
        print(f'    z-score: ρ={rho_z:.4f} (p={p_z:.4f})')
        print(f'    diff: {diff:.4f}')
        if abs(diff) > 0.2:
            print(f'    → 主要在做数据集判别')
        elif abs(rho_z) > 0.3 and p_z < 0.05:
            print(f'    → 有显著类级别解释力！')
        else:
            print(f'    → 类级别解释力弱')

    # 2. 多因子回归
    print(f'\n{"=" * 80}')
    print('2. 多因子回归：sim_pos_mean + sim_pos_std')
    print(f'{"=" * 80}')

    reg_results = multi_factor_regression(samples)
    print(f'\n  OLS回归:')
    print(f'    LOCO R²={reg_results["ols_r2"]:.4f}')
    print(f'    Spearman ρ={reg_results["ols_rho"]:.4f} (p={reg_results["ols_p"]:.4f})')
    print(f'    系数: {reg_results["coef"]}')
    print(f'    截距: {reg_results["intercept"]:.4f}')

    print(f'\n  两级模型（gate + 回归）:')
    print(f'    LOCO R²={reg_results["two_r2"]:.4f}')
    print(f'    Spearman ρ={reg_results["two_rho"]:.4f} (p={reg_results["two_p"]:.4f})')

    # 3. 和baseline_ap对比
    print(f'\n{"=" * 80}')
    print('3. 和baseline_ap对比')
    print(f'{"=" * 80}')

    compare_results = compare_with_baseline(samples)
    print(f'\n  {"模型":<30s} {"LOCO R²":>10s} {"Spearman ρ":>12s}')
    print('-' * 55)
    print(f'  {"baseline_ap（需GT）":<30s} {compare_results["baseline_ap"]["r2"]:>10.4f} {compare_results["baseline_ap"]["rho"]:>12.4f}')
    print(f'  {"sim_pos_mean+sim_pos_std（零训练）":<30s} {compare_results["sim_factors"]["r2"]:>10.4f} {compare_results["sim_factors"]["rho"]:>12.4f}')
    print(f'  {"三者组合":<30s} {compare_results["all_combined"]["r2"]:>10.4f} {compare_results["all_combined"]["rho"]:>12.4f}')

    # 保存结果
    output = {
        'zscore_test': {f: {'rho_raw': zscore_test(samples, f)[0], 'rho_z': zscore_test(samples, f)[2]} for f in ['sim_pos_mean', 'sim_pos_std', 'baseline_ap', 'kappa']},
        'regression': reg_results,
        'comparison': compare_results,
    }
    with open(DATA / 'sim_factors_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\n{"=" * 80}')
    print('核心结论：')
    sim_r2 = compare_results['sim_factors']['r2']
    ap_r2 = compare_results['baseline_ap']['r2']
    if sim_r2 > ap_r2 * 0.7:
        print(f'  sim_pos_mean+sim_pos_std（零训练）的R²达到baseline_ap（需GT）的{sim_r2/ap_r2*100:.1f}%')
        print(f'  这是一个真正的零训练预测因子！')
    else:
        print(f'  sim_pos_mean+sim_pos_std的R²只有baseline_ap的{sim_r2/ap_r2*100:.1f}%，还需要改进')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
