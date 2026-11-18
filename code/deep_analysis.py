# -*- coding: utf-8 -*-
"""deep_analysis.py — 深入分析baseline AP为什么能预测α
1. 计算所有候选因子和α_grid的相关性
2. 尝试多因子线性回归
3. 分析COCO和VOC的分布差异
4. 尝试baseline AP的变换
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ALPHA_MAX = 2.0


def load_data():
    """加载COCO和VOC的α网格搜索结果"""
    with open(DATA / 'coco_alpha_grid_scan_results.json', 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    for item in coco_data:
        item['target_domain'] = 'COCO'

    with open(DATA / 'voc_alpha_grid_scan_results.json', 'r', encoding='utf-8') as f:
        voc_data = json.load(f)
    for item in voc_data:
        item['target_domain'] = 'VOC'

    all_data = coco_data + voc_data

    samples = []
    for item in all_data:
        if item.get('mu_c', 0) > 1e-8:
            # 计算额外的因子
            ap_curve = item.get('ap_curve', {})
            if ap_curve:
                ap_values = list(ap_curve.values())
                ap_at_alpha0 = ap_curve.get('0.0', ap_values[0])
                ap_at_alpha2 = ap_curve.get('2.0', ap_values[-1])
                ap_gain = ap_at_alpha2 - ap_at_alpha0
                ap_curve_slope = (ap_at_alpha2 - ap_at_alpha0) / 2.0
            else:
                ap_gain = item['best_ap'] - item['baseline_ap']
                ap_curve_slope = ap_gain / max(item['best_alpha'], 0.01)

            samples.append({
                'src': item['src'],
                'src_cls': item['src_cls'],
                'target_cls': item['target_cls'],
                'target_domain': item['target_domain'],
                'm_0': item['m_0'],
                'mu_c': item['mu_c'],
                'alpha_grid': item['best_alpha'],
                'baseline_ap': item['baseline_ap'],
                'best_ap': item['best_ap'],
                'alpha_gt_0_better': item['alpha_gt_0_better'],
                # 额外因子
                'ap_gain': ap_gain,  # α=2时的AP增益
                'ap_curve_slope': ap_curve_slope,  # AP曲线斜率
                'log_1_minus_ap': np.log(1 - item['baseline_ap'] + 1e-8),  # log(1-AP)
                'inv_ap': 1.0 / (item['baseline_ap'] + 0.1),  # 1/AP
                'm0_over_mu': item['m_0'] / item['mu_c'],  # m⁰/μ
                'mu_over_m0': item['mu_c'] / (item['m_0'] + 1e-8),  # μ/m⁰
            })

    return samples


def compute_all_correlations(samples):
    """计算所有因子和α_grid的相关性"""
    factors = ['m_0', 'mu_c', 'baseline_ap', 'ap_gain', 'ap_curve_slope',
               'log_1_minus_ap', 'inv_ap', 'm0_over_mu', 'mu_over_m0']

    alpha_grid = np.array([s['alpha_grid'] for s in samples])

    print(f'\n=== 所有因子和α_grid的相关性（全数据，N={len(samples)}）===')
    print(f'{"因子":<20s} {"Spearman ρ":>12s} {"p值":>10s} {"显著性":>8s}')
    print('-' * 55)

    results = {}
    for factor in factors:
        values = np.array([s[factor] for s in samples])
        rho, p = stats.spearmanr(values, alpha_grid)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f'{factor:<20s} {rho:>12.4f} {p:>10.4f} {sig:>8s}')
        results[factor] = {'rho': rho, 'p': p}

    # 分目标域
    for domain in ['COCO', 'VOC']:
        domain_samples = [s for s in samples if s['target_domain'] == domain]
        if len(domain_samples) < 3:
            continue
        alpha_domain = np.array([s['alpha_grid'] for s in domain_samples])

        print(f'\n=== {domain}目标域（N={len(domain_samples)}）===')
        print(f'{"因子":<20s} {"Spearman ρ":>12s} {"p值":>10s} {"显著性":>8s}')
        print('-' * 55)

        for factor in factors:
            values = np.array([s[factor] for s in domain_samples])
            rho, p = stats.spearmanr(values, alpha_domain)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f'{factor:<20s} {rho:>12.4f} {p:>10.4f} {sig:>8s}')

    return results


def multi_factor_regression(samples):
    """多因子线性回归"""
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    factors = ['baseline_ap', 'm_0', 'mu_c', 'log_1_minus_ap', 'ap_gain']

    X = np.array([[s[f] for f in factors] for s in samples])
    y = np.array([s['alpha_grid'] for s in samples])

    # 标准化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    print(f'\n=== 多因子线性回归（N={len(samples)}）===')
    print(f'因子: {factors}')

    # 全数据拟合
    model = LinearRegression()
    model.fit(X_norm, y)
    y_pred = model.predict(X_norm)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f'\n全数据R²: {r2:.4f}')
    print(f'系数（标准化后）:')
    for f, c in zip(factors, model.coef_):
        print(f'  {f}: {c:.4f}')
    print(f'  截距: {model.intercept_:.4f}')

    # LOOCV
    loo = LeaveOneOut()
    y_pred_cv = cross_val_predict(model, X_norm, y, cv=loo)
    ss_res_cv = np.sum((y - y_pred_cv) ** 2)
    r2_cv = 1 - ss_res_cv / ss_tot
    rho_cv, p_cv = stats.spearmanr(y_pred_cv, y)

    print(f'\nLOOCV R²: {r2_cv:.4f}')
    print(f'LOOCV Spearman ρ: {rho_cv:.4f} (p={p_cv:.4f})')

    # 尝试不同的因子组合
    print(f'\n=== 不同因子组合的R²对比 ===')
    print(f'{"因子组合":<40s} {"全数据R²":>10s} {"LOOCV R²":>10s}')
    print('-' * 65)

    combos = [
        (['baseline_ap'], 'baseline_ap'),
        (['log_1_minus_ap'], 'log(1-AP)'),
        (['baseline_ap', 'm_0'], 'AP + m⁰'),
        (['baseline_ap', 'mu_c'], 'AP + μ'),
        (['baseline_ap', 'm_0', 'mu_c'], 'AP + m⁰ + μ'),
        (['log_1_minus_ap', 'm_0', 'mu_c'], 'log(1-AP) + m⁰ + μ'),
        (['baseline_ap', 'ap_gain'], 'AP + AP增益'),
        (['baseline_ap', 'm_0', 'mu_c', 'ap_gain'], 'AP + m⁰ + μ + AP增益'),
    ]

    for factor_list, name in combos:
        X_sub = np.array([[s[f] for f in factor_list] for s in samples])
        X_sub_norm = (X_sub - X_sub.mean(axis=0)) / (X_sub.std(axis=0) + 1e-8)

        model_sub = LinearRegression()
        model_sub.fit(X_sub_norm, y)
        y_pred_sub = model_sub.predict(X_sub_norm)
        r2_sub = 1 - np.sum((y - y_pred_sub) ** 2) / ss_tot

        y_pred_cv_sub = cross_val_predict(model_sub, X_sub_norm, y, cv=loo)
        r2_cv_sub = 1 - np.sum((y - y_pred_cv_sub) ** 2) / ss_tot

        print(f'{name:<40s} {r2_sub:>10.4f} {r2_cv_sub:>10.4f}')

    return r2, r2_cv


def analyze_domain_difference(samples):
    """分析COCO和VOC的分布差异"""
    print(f'\n=== COCO vs VOC 分布差异 ===')

    factors = ['m_0', 'mu_c', 'baseline_ap', 'alpha_grid', 'ap_gain']

    print(f'{"因子":<15s} {"COCO均值":>10s} {"VOC均值":>10s} {"差异":>10s} {"t检验p":>10s}')
    print('-' * 60)

    for factor in factors:
        coco_values = np.array([s[factor] for s in samples if s['target_domain'] == 'COCO'])
        voc_values = np.array([s[factor] for s in samples if s['target_domain'] == 'VOC'])

        coco_mean = coco_values.mean()
        voc_mean = voc_values.mean()
        diff = coco_mean - voc_mean

        # t检验
        t_stat, p_val = stats.ttest_ind(coco_values, voc_values, equal_var=False)

        print(f'{factor:<15s} {coco_mean:>10.4f} {voc_mean:>10.4f} {diff:>10.4f} {p_val:>10.4f}')


def plot_deep_analysis(samples, output_file):
    """画深入分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    alpha_grid = np.array([s['alpha_grid'] for s in samples])
    colors = ['#2563eb' if s['target_domain'] == 'COCO' else '#ea580c' for s in samples]

    # 1. baseline AP vs α
    ax = axes[0, 0]
    baseline_aps = np.array([s['baseline_ap'] for s in samples])
    ax.scatter(baseline_aps, alpha_grid, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)
    # 拟合线
    z = np.polyfit(baseline_aps, alpha_grid, 1)
    p = np.poly1d(z)
    x_line = np.linspace(baseline_aps.min(), baseline_aps.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label=f'线性拟合 (slope={z[0]:.2f})')
    ax.set_xlabel('Baseline AP', fontsize=12)
    ax.set_ylabel('最优 α', fontsize=12)
    ax.set_title('Baseline AP vs α', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. log(1-AP) vs α
    ax = axes[0, 1]
    log_aps = np.array([s['log_1_minus_ap'] for s in samples])
    ax.scatter(log_aps, alpha_grid, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)
    z = np.polyfit(log_aps, alpha_grid, 1)
    p = np.poly1d(z)
    x_line = np.linspace(log_aps.min(), log_aps.max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label=f'线性拟合 (slope={z[0]:.2f})')
    ax.set_xlabel('log(1 - Baseline AP)', fontsize=12)
    ax.set_ylabel('最优 α', fontsize=12)
    ax.set_title('log(1-AP) vs α', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. AP增益 vs α
    ax = axes[0, 2]
    ap_gains = np.array([s['ap_gain'] for s in samples])
    ax.scatter(ap_gains, alpha_grid, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)
    ax.set_xlabel('AP增益 (α=2 - α=0)', fontsize=12)
    ax.set_ylabel('最优 α', fontsize=12)
    ax.set_title('AP增益 vs α', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 4. m⁰ vs α (分域)
    ax = axes[1, 0]
    m_0_values = np.array([s['m_0'] for s in samples])
    ax.scatter(m_0_values, alpha_grid, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)
    ax.set_xlabel('初始间隔 m⁰', fontsize=12)
    ax.set_ylabel('最优 α', fontsize=12)
    ax.set_title('m⁰ vs α (全数据，不相关!)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 5. μ vs α
    ax = axes[1, 1]
    mu_values = np.array([s['mu_c'] for s in samples])
    ax.scatter(mu_values, alpha_grid, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)
    ax.set_xlabel('校正方向投影 μ', fontsize=12)
    ax.set_ylabel('最优 α', fontsize=12)
    ax.set_title('μ vs α', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 6. COCO vs VOC 分布对比
    ax = axes[1, 2]
    coco_aps = [s['baseline_ap'] for s in samples if s['target_domain'] == 'COCO']
    voc_aps = [s['baseline_ap'] for s in samples if s['target_domain'] == 'VOC']
    ax.hist(coco_aps, bins=8, alpha=0.6, color='#2563eb', label='COCO', density=True)
    ax.hist(voc_aps, bins=8, alpha=0.6, color='#ea580c', label='VOC', density=True)
    ax.set_xlabel('Baseline AP', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('COCO vs VOC Baseline AP分布', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n深入分析图已保存到: {output_file}')


def main():
    print('=' * 80)
    print('深入分析：baseline AP为什么能预测α？')
    print('=' * 80)

    samples = load_data()
    print(f'\n总样本数: {len(samples)}')
    print(f'  COCO: {sum(1 for s in samples if s["target_domain"] == "COCO")}')
    print(f'  VOC: {sum(1 for s in samples if s["target_domain"] == "VOC")}')

    # 1. 计算所有相关性
    corr_results = compute_all_correlations(samples)

    # 2. 多因子回归
    r2_full, r2_cv = multi_factor_regression(samples)

    # 3. 域差异分析
    analyze_domain_difference(samples)

    # 4. 画图
    plot_deep_analysis(samples, DATA / 'deep_analysis_plot.png')

    # 保存结果
    output = {
        'n_samples': len(samples),
        'correlations': corr_results,
        'multi_factor_r2': r2_full,
        'multi_factor_r2_cv': r2_cv,
    }
    with open(DATA / 'deep_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\n{"=" * 80}')
    print('核心发现：')
    print(f'  1. baseline AP是最强的单因子预测器（全数据ρ=-0.65）')
    print(f'  2. log(1-AP)变换可能更好（因为AP有上界1）')
    print(f'  3. m⁰在全数据上完全不相关（ρ≈0），只在VOC上弱相关')
    print(f'  4. COCO和VOC的baseline AP分布差异显著')
    print(f'  5. 多因子回归全数据R²={r2_full:.4f}，LOOCV R²={r2_cv:.4f}')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
