# -*- coding: utf-8 -*-
"""fit_rho_v2.py — 改进版拟合：加缩放因子c + 截断 + 重新检查m^0
α_c = clip(c * (ρ - m_c^(0))_+ / μ_c, 0, α_max)
拟合2个参数：ρ和c
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

ALPHA_MAX = 2.0  # α的上界


def load_data():
    """加载VOC α网格搜索结果"""
    result_file = DATA / 'voc_alpha_grid_scan_results.json'
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        if item['mu_c'] > 1e-8:
            samples.append({
                'src': item['src'],
                'src_cls': item['src_cls'],
                'target_cls': item['target_cls'],
                'm_0': item['m_0'],
                'mu_c': item['mu_c'],
                'alpha_grid': item['best_alpha'],
                'baseline_ap': item['baseline_ap'],
                'alpha_gt_0_better': item['alpha_gt_0_better'],
            })
    return samples


def hinge_model_v2(m_0, mu_c, rho, c):
    """改进版闭式族：α = clip(c * (ρ - m_0)_+ / μ, 0, α_max)"""
    raw = c * np.maximum(0, rho - m_0) / mu_c
    return np.clip(raw, 0, ALPHA_MAX)


def fit_rho_c(samples):
    """拟合ρ和c（2个参数），用网格搜索"""
    m_0_values = np.array([s['m_0'] for s in samples])
    mu_c_values = np.array([s['mu_c'] for s in samples])
    alpha_grid = np.array([s['alpha_grid'] for s in samples])

    # 网格搜索
    rho_grid = np.linspace(0, 0.5, 101)
    c_grid = np.linspace(0.01, 1.0, 100)

    best_rho = None
    best_c = None
    best_mse = float('inf')

    for rho in rho_grid:
        for c in c_grid:
            alpha_pred = hinge_model_v2(m_0_values, mu_c_values, rho, c)
            mse = np.mean((alpha_pred - alpha_grid) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_rho = rho
                best_c = c

    # 计算R²
    alpha_pred = hinge_model_v2(m_0_values, mu_c_values, best_rho, best_c)
    ss_res = np.sum((alpha_grid - alpha_pred) ** 2)
    ss_tot = np.sum((alpha_grid - alpha_grid.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    rho_spearman, p_spearman = stats.spearmanr(alpha_pred, alpha_grid)

    return {
        'rho': best_rho,
        'c': best_c,
        'mse': best_mse,
        'r_squared': r_squared,
        'spearman_rho': rho_spearman,
        'spearman_p': p_spearman,
        'alpha_pred': alpha_pred,
        'n_samples': len(samples),
    }


def loco_cv_v2(samples):
    """留一交叉验证（2个参数）"""
    n = len(samples)
    alpha_pred_loco = np.zeros(n)

    for i in range(n):
        train_samples = [s for j, s in enumerate(samples) if j != i]

        m_0_train = np.array([s['m_0'] for s in train_samples])
        mu_c_train = np.array([s['mu_c'] for s in train_samples])
        alpha_train = np.array([s['alpha_grid'] for s in train_samples])

        rho_grid = np.linspace(0, 0.5, 101)
        c_grid = np.linspace(0.01, 1.0, 100)
        best_rho = None
        best_c = None
        best_mse = float('inf')
        for rho in rho_grid:
            for c in c_grid:
                alpha_pred = hinge_model_v2(m_0_train, mu_c_train, rho, c)
                mse = np.mean((alpha_pred - alpha_train) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_rho = rho
                    best_c = c

        alpha_pred_loco[i] = hinge_model_v2(samples[i]['m_0'], samples[i]['mu_c'], best_rho, best_c)

    alpha_grid = np.array([s['alpha_grid'] for s in samples])
    ss_res = np.sum((alpha_grid - alpha_pred_loco) ** 2)
    ss_tot = np.sum((alpha_grid - alpha_grid.mean()) ** 2)
    r_squared_loco = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    rho_spearman_loco, p_spearman_loco = stats.spearmanr(alpha_pred_loco, alpha_grid)

    return {
        'r_squared_loco': r_squared_loco,
        'spearman_rho_loco': rho_spearman_loco,
        'spearman_p_loco': p_spearman_loco,
        'alpha_pred_loco': alpha_pred_loco,
    }


def plot_results(samples, fit_result, output_file):
    """画结果图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：α^grid vs m^0
    ax = axes[0]
    m_0_values = np.array([s['m_0'] for s in samples])
    alpha_grid = np.array([s['alpha_grid'] for s in samples])
    colors = ['#2563eb' if s['alpha_gt_0_better'] else '#9ca3af' for s in samples]

    ax.scatter(m_0_values, alpha_grid, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)

    # 画拟合曲线（用平均μ）
    rho = fit_result['rho']
    c = fit_result['c']
    mu_c_mean = np.mean([s['mu_c'] for s in samples])
    m_0_range = np.linspace(0, max(m_0_values) * 1.1, 100)
    alpha_curve = hinge_model_v2(m_0_range, mu_c_mean, rho, c)
    ax.plot(m_0_range, alpha_curve, 'r-', linewidth=2,
            label=f'拟合 (ρ={rho:.4f}, c={c:.4f}, μ̄={mu_c_mean:.4f})')

    ax.axvline(x=rho, color='red', linestyle='--', alpha=0.5, label=f'ρ={rho:.4f}')

    ax.set_xlabel('初始间隔 m_c^(0)', fontsize=12)
    ax.set_ylabel('网格搜索最优 α_c^grid', fontsize=12)
    ax.set_title('Hinge形状验证：α vs m^(0)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 右图：预测 vs 真实
    ax = axes[1]
    alpha_pred = fit_result['alpha_pred']
    ax.scatter(alpha_grid, alpha_pred, c='#2563eb', s=80, alpha=0.8, edgecolors='white', linewidth=1.5)

    max_val = max(max(alpha_grid), max(alpha_pred)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='y=x')

    ax.set_xlabel('网格搜索最优 α^grid', fontsize=12)
    ax.set_ylabel('闭式解预测 α^pred', fontsize=12)
    ax.set_title(f'预测 vs 真实 (R²={fit_result["r_squared"]:.3f}, ρ={fit_result["spearman_rho"]:.3f})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'结果图已保存到: {output_file}')


def main():
    print('=' * 80)
    print('改进版拟合：加缩放因子c + 截断')
    print('α_c = clip(c * (ρ - m_c^(0))_+ / μ_c, 0, 2.0)')
    print('=' * 80)

    samples = load_data()
    print(f'\n加载样本数: {len(samples)}')

    # 拟合ρ和c
    print(f'\n--- 拟合ρ和c（2个参数）---')
    fit_result = fit_rho_c(samples)
    print(f'  最优ρ: {fit_result["rho"]:.4f}')
    print(f'  最优c: {fit_result["c"]:.4f}')
    print(f'  MSE: {fit_result["mse"]:.4f}')
    print(f'  R²: {fit_result["r_squared"]:.4f}')
    print(f'  Spearman ρ: {fit_result["spearman_rho"]:.4f} (p={fit_result["spearman_p"]:.4f})')

    # LOCO-CV
    print(f'\n--- 留一交叉验证（LOCO-CV）---')
    loco_result = loco_cv_v2(samples)
    print(f'  LOCO-CV R²: {loco_result["r_squared_loco"]:.4f}')
    print(f'  LOCO-CV Spearman ρ: {loco_result["spearman_rho_loco"]:.4f} (p={loco_result["spearman_p_loco"]:.4f})')

    # 画图
    print(f'\n--- 画图 ---')
    plot_results(samples, fit_result, DATA / 'hinge_fit_v2.png')

    # 保存结果
    output = {
        'rho': fit_result['rho'],
        'c': fit_result['c'],
        'mse': fit_result['mse'],
        'r_squared': fit_result['r_squared'],
        'spearman_rho': fit_result['spearman_rho'],
        'spearman_p': fit_result['spearman_p'],
        'loco_r_squared': loco_result['r_squared_loco'],
        'loco_spearman_rho': loco_result['spearman_rho_loco'],
        'loco_spearman_p': loco_result['spearman_p_loco'],
        'n_samples': len(samples),
        'alpha_max': ALPHA_MAX,
        'samples': [
            {
                'src': s['src'],
                'src_cls': s['src_cls'],
                'target_cls': s['target_cls'],
                'm_0': s['m_0'],
                'mu_c': s['mu_c'],
                'alpha_grid': s['alpha_grid'],
                'alpha_pred': float(fit_result['alpha_pred'][i]),
                'alpha_pred_loco': float(loco_result['alpha_pred_loco'][i]),
                'residual': float(s['alpha_grid'] - fit_result['alpha_pred'][i]),
            }
            for i, s in enumerate(samples)
        ]
    }

    with open(DATA / 'rho_fit_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n拟合结果已保存到: {DATA / "rho_fit_v2_results.json"}')

    # 打印逐样本预测
    print(f'\n--- 逐样本预测 ---')
    print(f'{"源域":<10s} {"类别":<15s} {"m^0":>8s} {"μ":>8s} {"α_grid":>8s} {"α_pred":>8s} {"残差":>8s}')
    print('-' * 75)
    for i, s in enumerate(samples):
        print(f'{s["src"]:<10s} {s["src_cls"]:<15s} {s["m_0"]:>8.4f} {s["mu_c"]:>8.4f} '
              f'{s["alpha_grid"]:>8.2f} {fit_result["alpha_pred"][i]:>8.4f} '
              f'{s["alpha_grid"] - fit_result["alpha_pred"][i]:>8.4f}')

    print(f'\n{"=" * 80}')
    print('结论：')
    print(f'  1. 最优ρ={fit_result["rho"]:.4f}, c={fit_result["c"]:.4f}')
    print(f'  2. 全数据R²={fit_result["r_squared"]:.4f}, LOCO-CV R²={loco_result["r_squared_loco"]:.4f}')
    print(f'  3. Spearman ρ={fit_result["spearman_rho"]:.4f} (p={fit_result["spearman_p"]:.4f})')
    if fit_result['r_squared'] > 0.3:
        print(f'  4. ✅ R²>0.3，闭式解有一定解释力')
    else:
        print(f'  4. ⚠️ R²较低，可能需要：(a)重新检查m^0/μ定义 (b)加SV/TC因子 (c)增加样本量')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
