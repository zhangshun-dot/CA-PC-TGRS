# -*- coding: utf-8 -*-
"""fit_combined.py — 合并COCO+VOC数据重新拟合参数化闭式族
α_c = clip(c * (ρ - m_c^(0))_+ / μ_c, 0, α_max)
同时尝试用baseline AP替代m^0
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
    # COCO
    with open(DATA / 'coco_alpha_grid_scan_results.json', 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    for item in coco_data:
        item['target_domain'] = 'COCO'

    # VOC
    with open(DATA / 'voc_alpha_grid_scan_results.json', 'r', encoding='utf-8') as f:
        voc_data = json.load(f)
    for item in voc_data:
        item['target_domain'] = 'VOC'

    all_data = coco_data + voc_data

    # 过滤有效样本
    samples = []
    for item in all_data:
        if item.get('mu_c', 0) > 1e-8:
            samples.append({
                'src': item['src'],
                'src_cls': item['src_cls'],
                'target_cls': item['target_cls'],
                'target_domain': item['target_domain'],
                'm_0': item['m_0'],
                'mu_c': item['mu_c'],
                'alpha_grid': item['best_alpha'],
                'baseline_ap': item['baseline_ap'],
                'alpha_gt_0_better': item['alpha_gt_0_better'],
            })

    return samples


def hinge_model(x, mu, rho, c):
    """闭式族：α = clip(c * (ρ - x)_+ / μ, 0, α_max)"""
    raw = c * np.maximum(0, rho - x) / mu
    return np.clip(raw, 0, ALPHA_MAX)


def fit_model(samples, x_key, rho_range=(-1.0, 1.0)):
    """拟合ρ和c"""
    x_values = np.array([s[x_key] for s in samples])
    mu_values = np.array([s['mu_c'] for s in samples])
    alpha_grid = np.array([s['alpha_grid'] for s in samples])

    rho_grid = np.linspace(rho_range[0], rho_range[1], 201)
    c_grid = np.linspace(0.01, 2.0, 200)

    best_rho = None
    best_c = None
    best_mse = float('inf')

    for rho in rho_grid:
        for c in c_grid:
            alpha_pred = hinge_model(x_values, mu_values, rho, c)
            mse = np.mean((alpha_pred - alpha_grid) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_rho = rho
                best_c = c

    alpha_pred = hinge_model(x_values, mu_values, best_rho, best_c)
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
        'x_key': x_key,
    }


def loco_cv(samples, x_key, rho_range=(-1.0, 1.0)):
    """留一交叉验证"""
    n = len(samples)
    alpha_pred_loco = np.zeros(n)

    for i in range(n):
        train_samples = [s for j, s in enumerate(samples) if j != i]

        x_train = np.array([s[x_key] for s in train_samples])
        mu_train = np.array([s['mu_c'] for s in train_samples])
        alpha_train = np.array([s['alpha_grid'] for s in train_samples])

        rho_grid = np.linspace(rho_range[0], rho_range[1], 101)
        c_grid = np.linspace(0.01, 2.0, 100)
        best_rho = None
        best_c = None
        best_mse = float('inf')
        for rho in rho_grid:
            for c in c_grid:
                alpha_pred = hinge_model(x_train, mu_train, rho, c)
                mse = np.mean((alpha_pred - alpha_train) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_rho = rho
                    best_c = c

        alpha_pred_loco[i] = hinge_model(samples[i][x_key], samples[i]['mu_c'], best_rho, best_c)

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


def plot_results(samples, fit_m0, fit_baseline, output_file):
    """画结果对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    alpha_grid = np.array([s['alpha_grid'] for s in samples])
    colors = ['#2563eb' if s['target_domain'] == 'COCO' else '#ea580c' for s in samples]

    # 左上：m^0 vs α_grid
    ax = axes[0, 0]
    m_0_values = np.array([s['m_0'] for s in samples])
    ax.scatter(m_0_values, alpha_grid, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)
    ax.set_xlabel('初始间隔 m_c^(0)', fontsize=12)
    ax.set_ylabel('网格搜索最优 α_c^grid', fontsize=12)
    ax.set_title(f'm^0 vs α (蓝=COCO, 橙=VOC)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 右上：baseline AP vs α_grid
    ax = axes[0, 1]
    baseline_aps = np.array([s['baseline_ap'] for s in samples])
    ax.scatter(baseline_aps, alpha_grid, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)
    ax.set_xlabel('Baseline AP', fontsize=12)
    ax.set_ylabel('网格搜索最优 α_c^grid', fontsize=12)
    ax.set_title(f'Baseline AP vs α (蓝=COCO, 橙=VOC)', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 左下：m^0模型预测 vs 真实
    ax = axes[1, 0]
    alpha_pred_m0 = fit_m0['alpha_pred']
    ax.scatter(alpha_grid, alpha_pred_m0, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)
    max_val = max(max(alpha_grid), max(alpha_pred_m0)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='y=x')
    ax.set_xlabel('网格搜索最优 α^grid', fontsize=12)
    ax.set_ylabel('闭式解预测 α^pred', fontsize=12)
    ax.set_title(f'm^0模型: R²={fit_m0["r_squared"]:.3f}, ρ={fit_m0["spearman_rho"]:.3f}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 右下：baseline AP模型预测 vs 真实
    ax = axes[1, 1]
    alpha_pred_bl = fit_baseline['alpha_pred']
    ax.scatter(alpha_grid, alpha_pred_bl, c=colors, s=80, alpha=0.8, edgecolors='white', linewidth=1.5)
    max_val = max(max(alpha_grid), max(alpha_pred_bl)) * 1.1
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='y=x')
    ax.set_xlabel('网格搜索最优 α^grid', fontsize=12)
    ax.set_ylabel('闭式解预测 α^pred', fontsize=12)
    ax.set_title(f'baseline AP模型: R²={fit_baseline["r_squared"]:.3f}, ρ={fit_baseline["spearman_rho"]:.3f}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'结果图已保存到: {output_file}')


def main():
    print('=' * 80)
    print('合并COCO+VOC数据重新拟合参数化闭式族')
    print('=' * 80)

    samples = load_data()
    print(f'\n总样本数: {len(samples)}')
    print(f'  COCO目标域: {sum(1 for s in samples if s["target_domain"] == "COCO")}')
    print(f'  VOC目标域: {sum(1 for s in samples if s["target_domain"] == "VOC")}')

    # 整体相关性
    print(f'\n--- 整体相关性分析 ---')
    m_0_values = np.array([s['m_0'] for s in samples])
    baseline_aps = np.array([s['baseline_ap'] for s in samples])
    alpha_grid = np.array([s['alpha_grid'] for s in samples])

    rho_m0, p_m0 = stats.spearmanr(m_0_values, alpha_grid)
    rho_bl, p_bl = stats.spearmanr(baseline_aps, alpha_grid)

    print(f'  m^0 vs α_grid: Spearman ρ={rho_m0:.4f}, p={p_m0:.4f}')
    print(f'  baseline AP vs α_grid: Spearman ρ={rho_bl:.4f}, p={p_bl:.4f}')

    # 用m^0拟合
    print(f'\n--- 用m^0拟合（2个参数）---')
    fit_m0 = fit_model(samples, 'm_0', rho_range=(0, 0.5))
    print(f'  最优ρ: {fit_m0["rho"]:.4f}')
    print(f'  最优c: {fit_m0["c"]:.4f}')
    print(f'  R²: {fit_m0["r_squared"]:.4f}')
    print(f'  Spearman ρ: {fit_m0["spearman_rho"]:.4f} (p={fit_m0["spearman_p"]:.4f})')

    # 用baseline AP拟合
    print(f'\n--- 用baseline AP拟合（2个参数）---')
    fit_bl = fit_model(samples, 'baseline_ap', rho_range=(0, 1.0))
    print(f'  最优ρ: {fit_bl["rho"]:.4f}')
    print(f'  最优c: {fit_bl["c"]:.4f}')
    print(f'  R²: {fit_bl["r_squared"]:.4f}')
    print(f'  Spearman ρ: {fit_bl["spearman_rho"]:.4f} (p={fit_bl["spearman_p"]:.4f})')

    # LOCO-CV
    print(f'\n--- 留一交叉验证（LOCO-CV）---')
    loco_m0 = loco_cv(samples, 'm_0', rho_range=(0, 0.5))
    loco_bl = loco_cv(samples, 'baseline_ap', rho_range=(0, 1.0))
    print(f'  m^0模型 LOCO-CV R²: {loco_m0["r_squared_loco"]:.4f}, ρ={loco_m0["spearman_rho_loco"]:.4f}')
    print(f'  baseline AP模型 LOCO-CV R²: {loco_bl["r_squared_loco"]:.4f}, ρ={loco_bl["spearman_rho_loco"]:.4f}')

    # 画图
    print(f'\n--- 画图 ---')
    plot_results(samples, fit_m0, fit_bl, DATA / 'combined_fit_plot.png')

    # 保存结果
    output = {
        'n_samples': len(samples),
        'n_coco': sum(1 for s in samples if s['target_domain'] == 'COCO'),
        'n_voc': sum(1 for s in samples if s['target_domain'] == 'VOC'),
        'overall_correlation': {
            'm0_vs_alpha': {'rho': rho_m0, 'p': p_m0},
            'baseline_vs_alpha': {'rho': rho_bl, 'p': p_bl},
        },
        'fit_m0': {
            'rho': fit_m0['rho'],
            'c': fit_m0['c'],
            'r_squared': fit_m0['r_squared'],
            'spearman_rho': fit_m0['spearman_rho'],
            'spearman_p': fit_m0['spearman_p'],
            'loco_r_squared': loco_m0['r_squared_loco'],
            'loco_spearman_rho': loco_m0['spearman_rho_loco'],
        },
        'fit_baseline': {
            'rho': fit_bl['rho'],
            'c': fit_bl['c'],
            'r_squared': fit_bl['r_squared'],
            'spearman_rho': fit_bl['spearman_rho'],
            'spearman_p': fit_bl['spearman_p'],
            'loco_r_squared': loco_bl['r_squared_loco'],
            'loco_spearman_rho': loco_bl['spearman_rho_loco'],
        },
    }

    with open(DATA / 'combined_fit_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {DATA / "combined_fit_results.json"}')

    print(f'\n{"=" * 80}')
    print('结论：')
    print(f'  1. 总样本数: {len(samples)} (COCO {output["n_coco"]} + VOC {output["n_voc"]})')
    print(f'  2. m^0模型: R²={fit_m0["r_squared"]:.4f}, LOCO-CV R²={loco_m0["r_squared_loco"]:.4f}')
    print(f'  3. baseline AP模型: R²={fit_bl["r_squared"]:.4f}, LOCO-CV R²={loco_bl["r_squared_loco"]:.4f}')
    if fit_bl['r_squared'] > fit_m0['r_squared']:
        print(f'  4. ✅ baseline AP模型拟合效果更好')
        print(f'     说明baseline AP是跨目标域更稳定的预测因子')
    else:
        print(f'  4. m^0模型拟合效果更好')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
