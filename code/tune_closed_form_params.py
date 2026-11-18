# -*- coding: utf-8 -*-
"""tune_closed_form_params.py — 调整闭式解参数
目标：让α̂有真正的变化，而不是都被clip到2.0
扫描不同的ρ和mu缩放因子，找到最优组合
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
ALPHA_MAX = 2.0
ALPHA_MIN = 0.0


def compute_alpha_closed_form(m0, mu, rho=0.2, mu_scale=0.1):
    """理论闭式解：α = (ρ - m^0)_+ / (mu * mu_scale)"""
    mu_eff = mu * mu_scale
    alpha = max(0.0, (rho - m0) / max(mu_eff, 1e-8))
    return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))


def evaluate_params(samples, rho, mu_scale):
    """评估一组参数的效果"""
    alphas_pred = []
    alphas_oracle = []
    gains = []

    for s in samples:
        alpha_pred = compute_alpha_closed_form(s['m0'], s['mu'], rho=rho, mu_scale=mu_scale)
        alphas_pred.append(alpha_pred)
        alphas_oracle.append(s['best_alpha'])

        ap_curve = s['ap_curve']
        alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
        alpha_clipped = np.clip(alpha_pred, ALPHAS[0], ALPHAS[-1])
        ap = float(np.interp(alpha_clipped, alphas_sorted, [ap_curve[str(a)] for a in alphas_sorted]))
        baseline = ap_curve['0.0']
        gains.append(ap - baseline)

    alphas_pred = np.array(alphas_pred)
    alphas_oracle = np.array(alphas_oracle)
    gains = np.array(gains)

    # 统计
    alpha_std = np.std(alphas_pred)
    alpha_zero_ratio = np.mean(alphas_pred == 0)
    alpha_max_ratio = np.mean(alphas_pred >= ALPHA_MAX - 1e-6)

    # 相关性
    if alpha_std > 1e-8:
        rho_spearman, p_spearman = stats.spearmanr(alphas_pred, alphas_oracle)
    else:
        rho_spearman, p_spearman = 0.0, 1.0

    # 增益
    mean_gain = np.mean(gains)
    n_positive = sum(1 for g in gains if g > 0)

    # Oracle实现率
    oracle_mean = np.mean([s['oracle_gain'] for s in samples])
    oracle_realization = mean_gain / (oracle_mean + 1e-8) * 100

    return {
        'rho': rho, 'mu_scale': mu_scale,
        'alpha_std': alpha_std,
        'alpha_zero_ratio': alpha_zero_ratio,
        'alpha_max_ratio': alpha_max_ratio,
        'spearman_rho': rho_spearman,
        'spearman_p': p_spearman,
        'mean_gain': mean_gain,
        'n_positive': n_positive,
        'oracle_realization': oracle_realization,
    }


def main():
    print('=' * 80)
    print('闭式解参数扫描：找到让α̂有真正变化的参数组合')
    print('=' * 80)

    # 加载样本
    samples_file = DATA / 'zero_shot_gated_samples.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    print(f'\n总样本数: {len(samples)}')

    # 打印m0和mu的分布
    m0_list = np.array([s['m0'] for s in samples])
    mu_list = np.array([s['mu'] for s in samples])
    print(f'\nm0分布: min={np.min(m0_list):.4f}, max={np.max(m0_list):.4f}, '
          f'mean={np.mean(m0_list):.4f}, median={np.median(m0_list):.4f}')
    print(f'mu分布: min={np.min(mu_list):.4f}, max={np.max(mu_list):.4f}, '
          f'mean={np.mean(mu_list):.4f}, median={np.median(mu_list):.4f}')

    # 参数扫描
    print(f'\n{"=" * 80}')
    print('参数扫描结果')
    print(f'{"=" * 80}')
    print(f'\n{"ρ":>8s} {"mu_scale":>10s} {"α_std":>8s} {"α=0%":>8s} {"α=2%":>8s} '
          f'{"ρ_spear":>10s} {"平均增益":>10s} {"正例":>8s} {"Oracle%":>8s}')
    print('-' * 100)

    results = []
    # ρ在m0的范围内扫描
    for rho in [0.05, 0.07, 0.09, 0.10, 0.11, 0.13, 0.15, 0.20]:
        # mu_scale扫描
        for mu_scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            result = evaluate_params(samples, rho, mu_scale)
            results.append(result)

            # 只打印有意义的结果（α_std > 0.1 且 α_zero_ratio在[0.1, 0.9]之间）
            if result['alpha_std'] > 0.1 and 0.1 < result['alpha_zero_ratio'] < 0.9:
                print(f'{rho:>8.2f} {mu_scale:>10.2f} {result["alpha_std"]:>8.4f} '
                      f'{result["alpha_zero_ratio"]:>7.1%} {result["alpha_max_ratio"]:>7.1%} '
                      f'{result["spearman_rho"]:>10.4f} {result["mean_gain"]:>+10.4f} '
                      f'{result["n_positive"]:>4d}/{len(samples):<4d} '
                      f'{result["oracle_realization"]:>7.1f}%')

    # 找到最优参数（按平均增益排序，同时要求α_std > 0.1）
    print(f'\n{"=" * 80}')
    print('Top 10 参数组合（按平均增益排序，要求α_std > 0.1）')
    print(f'{"=" * 80}')

    valid_results = [r for r in results if r['alpha_std'] > 0.1]
    valid_results.sort(key=lambda x: x['mean_gain'], reverse=True)

    print(f'\n{"排名":>4s} {"ρ":>8s} {"mu_scale":>10s} {"α_std":>8s} {"α=0%":>8s} '
          f'{"ρ_spear":>10s} {"平均增益":>10s} {"正例":>8s} {"Oracle%":>8s}')
    print('-' * 100)

    for i, r in enumerate(valid_results[:10]):
        print(f'{i+1:>4d} {r["rho"]:>8.2f} {r["mu_scale"]:>10.2f} {r["alpha_std"]:>8.4f} '
              f'{r["alpha_zero_ratio"]:>7.1%} {r["spearman_rho"]:>10.4f} '
              f'{r["mean_gain"]:>+10.4f} {r["n_positive"]:>4d}/{len(samples):<4d} '
              f'{r["oracle_realization"]:>7.1f}%')

    # 找到最优参数（按Spearman ρ排序）
    print(f'\n{"=" * 80}')
    print('Top 10 参数组合（按Spearman ρ排序，要求α_std > 0.1）')
    print(f'{"=" * 80}')

    valid_results.sort(key=lambda x: x['spearman_rho'], reverse=True)

    print(f'\n{"排名":>4s} {"ρ":>8s} {"mu_scale":>10s} {"α_std":>8s} {"α=0%":>8s} '
          f'{"ρ_spear":>10s} {"平均增益":>10s} {"正例":>8s} {"Oracle%":>8s}')
    print('-' * 100)

    for i, r in enumerate(valid_results[:10]):
        print(f'{i+1:>4d} {r["rho"]:>8.2f} {r["mu_scale"]:>10.2f} {r["alpha_std"]:>8.4f} '
              f'{r["alpha_zero_ratio"]:>7.1%} {r["spearman_rho"]:>10.4f} '
              f'{r["mean_gain"]:>+10.4f} {r["n_positive"]:>4d}/{len(samples):<4d} '
              f'{r["oracle_realization"]:>7.1f}%')

    # 保存最优参数
    if valid_results:
        best_by_gain = valid_results[0]
        best_by_rho = max(valid_results, key=lambda x: x['spearman_rho'])

        print(f'\n{"=" * 80}')
        print('推荐参数')
        print(f'{"=" * 80}')
        print(f'\n按平均增益最优: ρ={best_by_gain["rho"]:.2f}, mu_scale={best_by_gain["mu_scale"]:.2f}')
        print(f'  α_std={best_by_gain["alpha_std"]:.4f}, α=0比例={best_by_gain["alpha_zero_ratio"]:.1%}')
        print(f'  Spearman ρ={best_by_gain["spearman_rho"]:.4f}')
        print(f'  平均增益={best_by_gain["mean_gain"]:+.4f}, Oracle实现率={best_by_gain["oracle_realization"]:.1f}%')

        print(f'\n按Spearman ρ最优: ρ={best_by_rho["rho"]:.2f}, mu_scale={best_by_rho["mu_scale"]:.2f}')
        print(f'  α_std={best_by_rho["alpha_std"]:.4f}, α=0比例={best_by_rho["alpha_zero_ratio"]:.1%}')
        print(f'  Spearman ρ={best_by_rho["spearman_rho"]:.4f}')
        print(f'  平均增益={best_by_rho["mean_gain"]:+.4f}, Oracle实现率={best_by_rho["oracle_realization"]:.1f}%')

        # 保存结果
        with open(DATA / 'closed_form_params_tuning.json', 'w', encoding='utf-8') as f:
            json.dump({
                'best_by_gain': best_by_gain,
                'best_by_rho': best_by_rho,
                'all_results': valid_results,
            }, f, indent=2, ensure_ascii=False)

        print(f'\n结果已保存到: {DATA / "closed_form_params_tuning.json"}')


if __name__ == '__main__':
    main()
