# -*- coding: utf-8 -*-
"""scale_match_experiment.py — 双因子scale-match实验"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import math
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def spearman_corr(x, y):
    """自己实现Spearman秩相关系数"""
    def rank_with_ties(arr):
        sorted_idx = np.argsort(arr)
        ranks = np.empty_like(sorted_idx, dtype=float)
        i = 0
        while i < len(arr):
            j = i
            while j < len(arr) and arr[sorted_idx[j]] == arr[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg_rank
            i = j
        return ranks
    x_rank = rank_with_ties(x)
    y_rank = rank_with_ties(y)
    x_mean = np.mean(x_rank)
    y_mean = np.mean(y_rank)
    numerator = np.sum((x_rank - x_mean) * (y_rank - y_mean))
    denominator = np.sqrt(np.sum((x_rank - x_mean)**2) * np.sum((y_rank - y_mean)**2))
    if denominator < 1e-12:
        return 0.0, 1.0
    rho = numerator / denominator
    n = len(x)
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t = rho * np.sqrt((n - 2) / (1 - rho**2))
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / np.sqrt(2))))
    return float(rho), float(p)


def compute_alpha_closed_form(m0, mu, rho=0.13, mu_scale=0.01):
    mu_eff = mu * mu_scale
    alpha = max(0.0, (rho - m0) / max(mu_eff, 1e-8))
    return float(np.clip(alpha, 0.0, 2.0))


def compute_alpha_dual_factor(s, sv_weight=-0.5, tc_weight=0.3, bias=1.0):
    sv = s.get('sv', 0.0)
    tc = s.get('tc', 0.0)
    sv_norm = (sv - 0.5) / 2.0
    tc_norm = (tc - 0.5) / 0.5
    alpha = sv_weight * sv_norm + tc_weight * tc_norm + bias
    return float(np.clip(alpha, 0.0, 2.0))


def get_ap_at_alpha(s, alpha):
    ap_curve = s['ap_curve']
    alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
    alpha_clipped = np.clip(alpha, 0.0, 2.0)
    ap = float(np.interp(alpha_clipped, alphas_sorted,
                          [ap_curve[str(a)] for a in alphas_sorted]))
    return ap


def main():
    print('=' * 80)
    print('双因子scale-match实验')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f'\n总样本数: {len(samples)}')

    # 1. 计算各种方法的α̂
    print(f'\n{"=" * 80}')
    print('1. 计算各种方法的α̂')
    print(f'{"=" * 80}')

    alphas_closed = np.array([
        compute_alpha_closed_form(s['m0'], s['mu'], rho=0.13, mu_scale=0.01)
        for s in samples
    ])
    alphas_dual = np.array([compute_alpha_dual_factor(s) for s in samples])

    print(f'\n  闭式解α̂: mean={np.mean(alphas_closed):.4f}, std={np.std(alphas_closed):.4f}')
    print(f'  双因子α̂: mean={np.mean(alphas_dual):.4f}, std={np.std(alphas_dual):.4f}')

    # 2. Scale-match
    print(f'\n{"=" * 80}')
    print('2. Scale-match：把双因子的α̂线性重标定')
    print(f'{"=" * 80}')

    dual_mean = np.mean(alphas_dual)
    dual_std = np.std(alphas_dual)
    closed_mean = np.mean(alphas_closed)
    closed_std = np.std(alphas_closed)

    if dual_std > 1e-8:
        alphas_dual_zscore = (alphas_dual - dual_mean) / dual_std * closed_std + closed_mean
    else:
        alphas_dual_zscore = np.full_like(alphas_dual, closed_mean)
    alphas_dual_zscore = np.clip(alphas_dual_zscore, 0.0, 2.0)

    dual_min, dual_max = np.min(alphas_dual), np.max(alphas_dual)
    closed_min, closed_max = np.min(alphas_closed), np.max(alphas_closed)
    if dual_max - dual_min > 1e-8:
        alphas_dual_minmax = (alphas_dual - dual_min) / (dual_max - dual_min) * (closed_max - closed_min) + closed_min
    else:
        alphas_dual_minmax = np.full_like(alphas_dual, closed_mean)
    alphas_dual_minmax = np.clip(alphas_dual_minmax, 0.0, 2.0)

    alphas_dual_mean = alphas_dual + (closed_mean - dual_mean)
    alphas_dual_mean = np.clip(alphas_dual_mean, 0.0, 2.0)

    print(f'\n  z-score匹配后: mean={np.mean(alphas_dual_zscore):.4f}, std={np.std(alphas_dual_zscore):.4f}')
    print(f'  min-max匹配后: mean={np.mean(alphas_dual_minmax):.4f}, std={np.std(alphas_dual_minmax):.4f}')
    print(f'  均值匹配后:   mean={np.mean(alphas_dual_mean):.4f}, std={np.std(alphas_dual_mean):.4f}')

    # 3. 计算增益
    print(f'\n{"=" * 80}')
    print('3. 计算各种方法的增益')
    print(f'{"=" * 80}')

    def compute_gains(alphas_pred):
        gains = []
        for i, s in enumerate(samples):
            ap = get_ap_at_alpha(s, alphas_pred[i])
            baseline = s['ap_curve']['0.0']
            gains.append(ap - baseline)
        return np.array(gains)

    gains_closed = compute_gains(alphas_closed)
    gains_dual = compute_gains(alphas_dual)
    gains_dual_zscore = compute_gains(alphas_dual_zscore)
    gains_dual_minmax = compute_gains(alphas_dual_minmax)
    gains_dual_mean = compute_gains(alphas_dual_mean)
    gains_oracle = np.array([max(s['ap_curve'].values()) - s['ap_curve']['0.0'] for s in samples])

    best_const_gain = -np.inf
    best_const_alpha = 0.0
    for alpha in ALPHAS:
        gains_const = compute_gains(np.full(len(samples), alpha))
        if np.mean(gains_const) > best_const_gain:
            best_const_gain = np.mean(gains_const)
            best_const_alpha = alpha

    headroom = np.mean(gains_oracle) - best_const_gain

    print(f'\n  {"方法":<30s} {"平均增益":>10s} {"正例":>8s} {"vs常数":>10s} {"headroom":>10s}')
    print('-' * 75)

    def print_row(name, gains):
        mean_g = np.mean(gains)
        pos = sum(gains > 0)
        vs_const = mean_g - best_const_gain
        cap = vs_const / headroom * 100 if abs(headroom) > 1e-12 else 0
        print(f'  {name:<30s} {mean_g:>+10.4f} {pos:>4d}/{len(gains):<3d} {vs_const:>+10.4f} {cap:>9.1f}%')

    print_row(f'常数α={best_const_alpha:.1f}（基线）', compute_gains(np.full(len(samples), best_const_alpha)))
    print_row('双因子（原始）', gains_dual)
    print_row('双因子（z-score匹配）', gains_dual_zscore)
    print_row('双因子（min-max匹配）', gains_dual_minmax)
    print_row('双因子（均值匹配）', gains_dual_mean)
    print_row('闭式解（最优参数）', gains_closed)
    print_row('Oracle', gains_oracle)

    # 4. 排序相关性
    print(f'\n{"=" * 80}')
    print('4. 排序相关性分析')
    print(f'{"=" * 80}')

    alphas_oracle = np.array([s['best_alpha'] for s in samples])
    rho_dual, p_dual = spearman_corr(alphas_dual, alphas_oracle)
    rho_closed, p_closed = spearman_corr(alphas_closed, alphas_oracle)

    print(f'\n  双因子α̂ vs Oracle α*: Spearman ρ={rho_dual:.4f}, p={p_dual:.4f}')
    print(f'  闭式解α̂ vs Oracle α*: Spearman ρ={rho_closed:.4f}, p={p_closed:.4f}')

    # 5. 结论
    print(f'\n{"=" * 80}')
    print('5. 结论')
    print(f'{"=" * 80}')

    dual_original_gain = np.mean(gains_dual)
    dual_best_scaled_gain = max(np.mean(gains_dual_zscore), np.mean(gains_dual_minmax), np.mean(gains_dual_mean))
    closed_gain = np.mean(gains_closed)

    print(f'\n  双因子原始增益: {dual_original_gain:+.4f}')
    print(f'  双因子最佳scale-match增益: {dual_best_scaled_gain:+.4f}')
    print(f'  闭式解增益: {closed_gain:+.4f}')
    print(f'  Oracle增益: {np.mean(gains_oracle):+.4f}')

    if dual_best_scaled_gain >= closed_gain * 0.9:
        print(f'\n  ✅ 双因子scale-match后追上来了！')
        print(f'     结论：「因子决定排序，闭式解决定尺度」')
    elif dual_best_scaled_gain > dual_original_gain * 1.5:
        print(f'\n  ⚠️  双因子scale-match后有提升，但仍不如闭式解')
        print(f'     结论：双因子的排序有部分信息，但尺度是主要问题')
    else:
        print(f'\n  ❌ 双因子scale-match后也追不上')
        print(f'     结论：双因子作为α生成器彻底判死')

    results = {
        'dual_original_gain': float(dual_original_gain),
        'dual_zscore_gain': float(np.mean(gains_dual_zscore)),
        'dual_minmax_gain': float(np.mean(gains_dual_minmax)),
        'dual_mean_gain': float(np.mean(gains_dual_mean)),
        'dual_best_scaled_gain': float(dual_best_scaled_gain),
        'closed_gain': float(closed_gain),
        'oracle_gain': float(np.mean(gains_oracle)),
        'best_const_alpha': float(best_const_alpha),
        'best_const_gain': float(best_const_gain),
        'rho_dual_vs_oracle': float(rho_dual),
        'p_dual_vs_oracle': float(p_dual),
        'rho_closed_vs_oracle': float(rho_closed),
        'p_closed_vs_oracle': float(p_closed),
    }
    results_file = DATA / 'scale_match_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  结果已保存: {results_file}')


if __name__ == '__main__':
    main()
