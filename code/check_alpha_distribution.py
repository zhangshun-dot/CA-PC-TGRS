# -*- coding: utf-8 -*-
"""check_alpha_distribution.py — 检查α̂分布和整张网格增益"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def compute_alpha_closed_form(m0, mu, rho=0.13, mu_scale=0.01):
    mu_eff = mu * mu_scale
    alpha = max(0.0, (rho - m0) / max(mu_eff, 1e-8))
    return float(np.clip(alpha, 0.0, 2.0))


def main():
    print('=' * 80)
    print('检查α̂分布和整张网格增益')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f'\n总样本数: {len(samples)}')

    # 1. α̂分布统计
    print(f'\n{"=" * 80}')
    print('1. α̂分布统计（最优参数ρ=0.13, mu_scale=0.01）')
    print(f'{"=" * 80}')

    alphas_pred = []
    for s in samples:
        alpha = compute_alpha_closed_form(s['m0'], s['mu'], rho=0.13, mu_scale=0.01)
        alphas_pred.append(alpha)
    alphas_pred = np.array(alphas_pred)

    print(f'\n  min:    {np.min(alphas_pred):.4f}')
    print(f'  Q25:    {np.percentile(alphas_pred, 25):.4f}')
    print(f'  median: {np.median(alphas_pred):.4f}')
    print(f'  Q75:    {np.percentile(alphas_pred, 75):.4f}')
    print(f'  max:    {np.max(alphas_pred):.4f}')
    print(f'  mean:   {np.mean(alphas_pred):.4f}')
    print(f'  std:    {np.std(alphas_pred):.4f}')
    print(f'  α=0的比例:   {np.mean(alphas_pred == 0):.1%} ({sum(alphas_pred == 0)}/{len(alphas_pred)})')
    print(f'  α=2.0的比例: {np.mean(alphas_pred >= 1.999):.1%} ({sum(alphas_pred >= 1.999)}/{len(alphas_pred)})')
    print(f'  0<α<2的比例: {np.mean((alphas_pred > 0) & (alphas_pred < 1.999)):.1%}')

    p_zero = np.mean(alphas_pred == 0)
    expected_std = 2.0 * np.sqrt(p_zero * (1 - p_zero))
    print(f'\n  反推验证：若取值只有{{0, 2.0}}，p={p_zero:.3f}')
    print(f'    期望std = 2 * sqrt({p_zero:.3f} * {1-p_zero:.3f}) = {expected_std:.4f}')
    print(f'    实际std = {np.std(alphas_pred):.4f}')
    print(f'    差异 = {abs(np.std(alphas_pred) - expected_std):.6f}')
    if abs(np.std(alphas_pred) - expected_std) < 0.001:
        print(f'    ✅ 完全吻合！α̂确实是二值的（0/2.0）')
    else:
        print(f'    ⚠️  不完全吻合，α̂可能有连续值')

    print(f'\n  每个样本的α̂:')
    for i, s in enumerate(samples):
        print(f'    {i+1:2d}. {s["target_domain"]:5s} {s["target_cls"]:12s} '
              f'm0={s["m0"]:.4f} mu={s["mu"]:.4f} alpha={alphas_pred[i]:.4f}')

    # 2. 整张网格增益
    print(f'\n{"=" * 80}')
    print('2. (rho, mu_scale)整张网格的增益')
    print(f'{"=" * 80}')

    rho_list = [0.05, 0.07, 0.09, 0.10, 0.11, 0.13, 0.15, 0.20]
    mu_scale_list = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    header = 'rho/mu_scale'
    print(f'\n{header:>12s}', end='')
    for ms in mu_scale_list:
        print(f'{ms:>8.2f}', end='')
    print()
    print('-' * (12 + 8 * len(mu_scale_list)))

    grid_results = {}
    for rho in rho_list:
        print(f'{rho:>12.2f}', end='')
        for mu_scale in mu_scale_list:
            gains = []
            for s in samples:
                alpha = compute_alpha_closed_form(s['m0'], s['mu'], rho=rho, mu_scale=mu_scale)
                ap_curve = s['ap_curve']
                alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
                alpha_clipped = np.clip(alpha, 0.0, 2.0)
                ap = float(np.interp(alpha_clipped, alphas_sorted, [ap_curve[str(a)] for a in alphas_sorted]))
                baseline = ap_curve['0.0']
                gains.append(ap - baseline)
            mean_gain = np.mean(gains)
            grid_results[(rho, mu_scale)] = mean_gain
            print(f'{mean_gain:>+8.4f}', end='')
        print()

    best_key = max(grid_results.keys(), key=lambda k: grid_results[k])
    best_gain = grid_results[best_key]
    print(f'\n  最优参数: rho={best_key[0]:.2f}, mu_scale={best_key[1]:.2f}, 平均增益={best_gain:+.4f}')

    print(f'\n  mu_scale=0.01是否在网格边界？')
    print(f'    网格范围: [{min(mu_scale_list)}, {max(mu_scale_list)}]')
    if best_key[1] == min(mu_scale_list):
        print(f'    ⚠️  最优mu_scale={best_key[1]:.2f}在左边界，真实最优可能在外面（更小的mu_scale）')
    else:
        print(f'    ✅ 最优mu_scale={best_key[1]:.2f}不在边界')

    all_gains = list(grid_results.values())
    print(f'\n  全网格增益范围: [{min(all_gains):+.4f}, {max(all_gains):+.4f}]')
    print(f'  全网格增益均值: {np.mean(all_gains):+.4f}')
    print(f'  全网格增益标准差: {np.std(all_gains):.4f}')
    if max(all_gains) - min(all_gains) < 0.005:
        print(f'  ⚠️  全网格增益波动很小（<0.005），"最优参数"可能是搜索噪声')
    else:
        print(f'  ✅ 全网格增益波动较大，参数选择有意义')

    # 3. 生成CSV
    print(f'\n{"=" * 80}')
    print('3. 生成adaptive_ablation.py需要的CSV')
    print(f'{"=" * 80}')

    grid_csv = DATA / 'grid.csv'
    with open(grid_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('dataset,class,alpha,ap50\n')
        for s in samples:
            for alpha_str, ap in s['ap_curve'].items():
                f.write(f'{s["target_domain"]},{s["target_cls"]},{alpha_str},{ap:.6f}\n')
    print(f'\n  已生成: {grid_csv}')

    pred_csv = DATA / 'pred.csv'
    with open(pred_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('dataset,class,alpha_hat,setting\n')
        for s in samples:
            alpha_opt = compute_alpha_closed_form(s['m0'], s['mu'], rho=0.13, mu_scale=0.01)
            f.write(f'{s["target_domain"]},{s["target_cls"]},{alpha_opt:.6f},closed_form_opt\n')
            alpha_old = compute_alpha_closed_form(s['m0'], s['mu'], rho=0.2, mu_scale=0.1)
            f.write(f'{s["target_domain"]},{s["target_cls"]},{alpha_old:.6f},closed_form_old\n')
            f.write(f'{s["target_domain"]},{s["target_cls"]},{s["best_alpha"]:.6f},oracle\n')
    print(f'  已生成: {pred_csv}')

    print(f'\n  运行adaptive_ablation.py:')
    print(f'    python adaptive_ablation.py --grid {grid_csv} --pred {pred_csv} --out {DATA / "adaptive_report.md"}')


if __name__ == '__main__':
    main()
