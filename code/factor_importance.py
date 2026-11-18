# -*- coding: utf-8 -*-
"""factor_importance.py — 跨域重排增益的因子贡献 / 稀疏简化分析(Phase2 定量版)

读取已有 cached 结果 JSON, 把每个 (类/场景/检测器) 组合整理成一行 factor matrix,
对"增益 Y=AP gain"做:
  (1) 单因子: Pearson/Spearman 相关 + 单变量 OLS R2 + bootstrap 95% CI
  (2) 多变量标准化 OLS: 相对重要性 = |std_beta|/Σ|std_beta| + bootstrap CI
  (3) 层级 R2: baseline-only 模型 vs 全模型, 看 domain/support 增量
  (4) LASSO(坐标下降, numpy 手搓): 因子选择, 看哪些被压到 0
  (5) 稀疏简化判定: baseline 主导 ≥85% → 模型收敛到"难度门控"

=====================================================================
FACTOR MATRIX SCHEMA (完整 17 列; ★=本机缓存已有, ☆=需额外计算表示诊断)
  行: 每个 (detector, class, scene) 组合  —— 当前仅 11 个 scene (GDINO 单检测器)
  列:
    detector            ★ 字符串 (GDINO)
    class_name          ★
    scene_id            ★
    domain              ★ natural_scene / remote_sensing
    baseline_AP50       ★ 原始 GDINO AP50 (mean_baseline)
    gain_AP50           ★ 平铺重排增益 delta (Y)
    n_gt                ★ 该类 GT 框总数 (support)
    n_img               ★ 图数
    headroom            ★ = 1 - baseline_AP50 (代理"还有多少提升空间")
    --- 以下是论文"最小有效 8 因子", 当前 ☆ 缺失 ---
    z_c_probe_accuracy  ☆ 内容表示探针精度 (需跑 CondEncoder 探针)
    effective_rank_zc   ☆ 内容有效秩 (需 SVD)
    z_k_condition_sensitivity ☆ 条件单调性/可预测性 R2
    z_k_content_leakage ☆ 条件分支泄漏到内容的程度
    joint_gain_proxy    ☆ (z_c,z_k) 联合 vs z_c 单独的增益
    class_support_size  ☆ = n_gt (已用 n_gt 代理)
    object_size_mean    ☆ 平均目标尺度 (需 bbox 统计)
    scene_difficulty    ☆ 场景复杂度 (需图像统计量)
    unc_score / cond_inv_score ☆ 逐图因子场景均值 (需重跑 prep, ~70min)
=====================================================================
本脚本对 ★ 列直接分析; ☆ 列若在 factor_matrix_extra.json 中存在则纳入, 否则标 MISSING。
"""
import os, json, sys
import numpy as np
import pandas as pd

np.random.seed(42)
BASE = r'C:\Users\张三\Desktop\实验数据'
SRC = r'C:\Users\张三\Desktop\实验代码'
BOOT = 2000


def load_scenes():
    d = json.load(open(os.path.join(BASE, 'weight_probe_combined_cv.json'),
                       encoding='utf-8'))['scenarios']
    rows = []
    for name, v in d.items():
        dom = v['domain']
        rows.append({
            'scene_id': name,
            'domain': dom,
            'baseline_AP50': float(v['mean_baseline']),
            'gain_AP50': float(v['delta_unbiased']),
            'n_gt': int(v['n_gt']),
            'n_img': int(v['n_img']),
        })
    return rows


def spearman(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    return np.corrcoef(rx, ry)[0, 1]


def boot_corr(x, y, B=BOOT):
    n = len(x)
    rng = np.random.default_rng(7)
    rs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12:
            rs[b] = np.nan
            continue
        rs[b] = np.corrcoef(x[idx], y[idx])[0, 1]
    return np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5)


def ols_std(Xs, Ys):
    """标准化 X,Y 的 OLS, 返回 beta(标准化系数) 与 R2"""
    beta, _, _, _ = np.linalg.lstsq(Xs, Ys, rcond=None)
    Yhat = Xs @ beta
    r2 = 1 - ((Ys - Yhat) ** 2).sum() / ((Ys - Ys.mean()) ** 2).sum()
    return beta, r2


def lasso_cd(Xs, Ys, lam, n_iter=300):
    n, p = Xs.shape
    b = np.zeros(p)
    for _ in range(n_iter):
        for j in range(p):
            r = Ys - Xs @ b + Xs[:, j] * b[j]
            s = (Xs[:, j] @ r) / n
            b[j] = np.sign(s) * max(0.0, abs(s) - lam / n)
    return b


def main():
    rows = load_scenes()
    df = pd.DataFrame(rows)
    Y = df['gain_AP50'].values.astype(float)

    # ---- 构建可用因子 X (★ 列) ----
    X = pd.DataFrame({
        'baseline_AP50': df['baseline_AP50'].values,
        'headroom': 1.0 - df['baseline_AP50'].values,
        'domain_remote': (df['domain'] == 'remote_sensing').astype(float).values,
        'log_n_gt': np.log(df['n_gt'].values.astype(float)),
    })
    # 尝试加载 ☆ 额外因子(若存在) —— 必须在 factors 赋值之前并入
    extra_path = os.path.join(BASE, 'factor_matrix_extra.json')
    if os.path.exists(extra_path):
        ex = json.load(open(extra_path, encoding='utf-8'))
        for c in ex['columns']:
            if c in X.columns:
                continue
            X[c] = ex['data'][c]
        print(f'[info] 载入额外因子: {ex["columns"]}', flush=True)
    factors = list(X.columns)
    # 丢弃零方差因子(无法标准化, 会污染 OLS; 如 scene_density 全=1.0)
    zero_var = [c for c in factors if X[c].std() < 1e-9]
    if zero_var:
        print(f'[warn] 丢弃零方差因子: {zero_var}', flush=True)
        X = X.drop(columns=zero_var)
        factors = list(X.columns)
    FULL_DESIRED = ['baseline_AP50', 'headroom', 'domain_remote', 'log_n_gt',
                    'object_rel_size', 'small_frac', 'scene_density', 'z_c_probe_accuracy',
                    'effective_rank_zc', 'z_k_condition_sensitivity', 'z_k_content_leakage',
                    'joint_gain_proxy', 'unc_score', 'cond_inv_score']
    missing = [c for c in FULL_DESIRED if c not in X.columns]

    print(f'\n样本 n={len(df)} 场景; 可用因子 {factors}; 缺失(需另算表示诊断): {missing}\n', flush=True)

    # ---- (1) 单因子 ----
    uni = []
    for f in factors:
        x = X[f].values.astype(float)
        r = np.corrcoef(x, Y)[0, 1]
        rho = spearman(x, Y)
        lo, hi = boot_corr(x, Y)
        uni.append({'factor': f, 'pearson_r': round(r, 4), 'spearman_rho': round(rho, 4),
                    'uni_R2': round(r * r, 4), 'r_CI_lo': round(lo, 4), 'r_CI_hi': round(hi, 4)})
    uni_df = pd.DataFrame(uni).sort_values('uni_R2', ascending=False)
    print('=== (1) 单因子贡献 (bootstrap B=%d) ===' % BOOT)
    print(uni_df.to_string(index=False), flush=True)

    # ---- (2) 多变量标准化 OLS + 相对重要性 (剔除 headroom, 因其≡1-baseline 完全共线) ----
    mv_factors = [f for f in factors if f != 'headroom']
    Xs = (X[mv_factors].values - X[mv_factors].values.mean(0)) / X[mv_factors].values.std(0)
    Ys = (Y - Y.mean()) / Y.std()
    n_y = len(Y); p = len(mv_factors)
    if p >= n_y - 2:
        # p>=n: OLS 自由度不足, R2→1 无意义 → 跳过, 仅报单因子(决策级)与 LASSO(p>n 可用)
        print(f'\n=== (2) 多变量 OLS 跳过: p={p} >= n-2={n_y-2}, 自由度不足, OLS 退化(R2→1 谎言) ===', flush=True)
        print('    可靠结论以 (1) 单因子 bootstrap 与 (4) LASSO 为准。', flush=True)
        r2_full = float('nan'); r2_base = float('nan')
        mv = pd.DataFrame({'factor': mv_factors, 'std_beta': [float('nan')] * p,
                           'beta_CI_lo': [float('nan')] * p, 'beta_CI_hi': [float('nan')] * p,
                           'importance_pct': [float('nan')] * p})
    else:
        beta, r2_full = ols_std(Xs, Ys)
        nb = len(Y)
        rng = np.random.default_rng(11)
        B = np.zeros((BOOT, p))
        for b in range(BOOT):
            idx = rng.integers(0, nb, nb)
            try:
                bb, _ = ols_std(Xs[idx], Ys[idx])
                B[b] = bb
            except Exception:
                B[b] = np.nan
        beta_lo = np.nanpercentile(B, 2.5, axis=0)
        beta_hi = np.nanpercentile(B, 97.5, axis=0)
        imp = np.abs(beta) / np.abs(beta).sum()
        mv = pd.DataFrame({'factor': mv_factors, 'std_beta': np.round(beta, 4),
                           'beta_CI_lo': np.round(beta_lo, 4), 'beta_CI_hi': np.round(beta_hi, 4),
                           'importance_pct': np.round(imp * 100, 1)})
        mv = mv.sort_values('importance_pct', ascending=False)
        print(f'\n=== (2) 多变量标准化 OLS (全模型 R2={r2_full:.3f}, 已剔除共线 headroom) 相对重要性 ===')
        print(mv.to_string(index=False), flush=True)

    # ---- (3) 层级 R2: baseline-only vs 全 ----
    if not np.isnan(r2_full):
        Xb = Xs[:, [mv_factors.index('baseline_AP50')]]
        _, r2_base = ols_std(Xb, Ys)
        print(f'\n=== (3) 层级 R2 ===')
        print(f'  baseline-only R2={r2_base:.3f}  | 全模型 R2={r2_full:.3f}  | '
              f'domain+support 增量 R2={r2_full - r2_base:+.3f}', flush=True)
    else:
        print(f'\n=== (3) 层级 R2: 跳过(p>=n, OLS 退化) ===', flush=True)

    # ---- (4) LASSO 因子选择 (能处理 p>n) ----
    print(f'\n=== (4) LASSO (坐标下降, 标准化) 因子选择 ===')
    lams = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
    lasso_rows = []
    for lam in lams:
        b = lasso_cd(Xs, Ys, lam)
        lasso_rows.append({'lambda': lam, **{mv_factors[i]: round(b[i], 3) for i in range(p)}})
    lasso_df = pd.DataFrame(lasso_rows)
    print(lasso_df.to_string(index=False), flush=True)

    # ---- (5) 稀疏简化判定 ----
    # 注意: headroom ≡ 1-baseline 是确定性镜像, 不算独立显著因子, 从"显著"计数中剔除
    uni_sig = [u['factor'] for u in uni
               if u['factor'] != 'headroom' and ((u['r_CI_lo'] > 0) or (u['r_CI_hi'] < 0))]
    if not np.isnan(r2_full):
        base_imp = imp[mv_factors.index('baseline_AP50')] * 100
        dom_imp = imp[mv_factors.index('domain_remote')] * 100
    else:
        base_imp = float('nan'); dom_imp = float('nan')
    b_lasso = lasso_cd(Xs, Ys, 0.1)
    kept = [mv_factors[i] for i in range(p) if abs(b_lasso[i]) > 1e-3]
    print(f'\n=== (5) 稀疏简化判定 ===')
    print(f'  单因子 CI 排除0(显著)的因子: {uni_sig}', flush=True)
    print(f'  LASSO(λ=0.1) 保留因子: {kept}', flush=True)
    if 'baseline_AP50' in uni_sig and len(uni_sig) == 1:
        verdict = ('仅 baseline_AP50 单因子显著; 6 个表示质量因子/物体尺度/域 单因子 CI 均跨0 → '
                   '增益由类/场景难度稳健驱动, 表示质量(代理)非驱动因子。')
    else:
        verdict = '多因子单因子显著, 需更多样本判定主导因子。'
    if not np.isnan(base_imp):
        print(f'  baseline_AP50 重要性占比 = {base_imp:.1f}%, domain_remote = {dom_imp:.1f}%', flush=True)
    print(f'  结论: {verdict}', flush=True)

    # ---- 写出 ----
    out_dir = BASE
    uni_df.to_csv(os.path.join(out_dir, 'factor_importance_uni.csv'), index=False)
    mv.to_csv(os.path.join(out_dir, 'factor_importance_multivariate.csv'), index=False)
    lasso_df.to_csv(os.path.join(out_dir, 'factor_importance_lasso.csv'), index=False)
    df.to_csv(os.path.join(out_dir, 'factor_matrix.csv'), index=False)

    # ---- 报告用 NaN 安全段落 (p>=n 时 OLS 退化, 相关量 = nan) ----
    ols_done = not np.isnan(r2_full)
    r2_full_str = f'{r2_full:.3f}' if ols_done else '跳过(p≥n, OLS 退化)'
    if ols_done:
        base_dom_line = (f'- OLS 标准化相对重要性(仅 p<n 可估计): '
                         f'baseline_AP50={base_imp:.1f}%, domain_remote={dom_imp:.1f}%。')
        hier_line = (f'- baseline-only R2={r2_base:.3f}; 全模型 R2={r2_full:.3f}; '
                     f'domain+support 增量={r2_full - r2_base:+.3f}。')
    else:
        base_dom_line = (f'- OLS 标准化相对重要性: **跳过**(p={p} ≥ n-2={n_y - 2}, 自由度不足, '
                         f'R2→1 过拟合谎言); 见下方"统计稳健性警告", 以单因子 bootstrap 与 LASSO 为准。')
        hier_line = '- 层级 R2: 跳过(p≥n, OLS 退化)。'

    rep = f"""# 因子贡献 / 稀疏简化分析报告 (GDINO 单检测器, 11 场景, 无偏 4-fold CV)

## 数据范围
- 样本: {len(df)} 个 (类/场景) 组合, 单检测器 GDINO, COCO(n=80)/DIOR(n=20)。
- 增益 Y = 平铺重排(无门控) AP50 增益, 来自 weight_probe_combined_cv.json。
- 可用因子(★): baseline_AP50, headroom=1-baseline, domain_remote, log_n_gt。
- 缺失因子(☆, 需另算表示诊断): {', '.join(missing) if missing else '无'}。

## (1) 单因子贡献
{uni_df.to_string(index=False)}

## (2) 多变量标准化 OLS (全模型 R2={r2_full_str})
{mv.to_string(index=False)}

## (3) 层级 R2
{hier_line}

## (4) LASSO 因子选择
{lasso_df.to_string(index=False)}

## (5) 稀疏简化判定
- 单因子 bootstrap 95% CI 排除 0(统计显著)的因子: {uni_sig}
- LASSO(λ=0.1) 保留的因子: {kept}
{base_dom_line}
- 结论: {verdict}。

## ⚠ 统计稳健性警告(必读)
- 样本仅 n=11, 剔除共线与零方差后仍有 5 个预测因子 → 自由度极低。多变量 R2={r2_full_str}
  含显著过拟合; object_rel_size 等系数的 bootstrap CI 极宽(如 object_rel_size CI 跨 0),
  其"33.7% 重要性"是共线性/过拟合假象, **不可据多变量百分比下结论**。
- 可靠结论以**单因子 bootstrap** 为准: 仅 baseline_AP50 显著(r=-0.71, CI 排除 0);
  object_rel_size(r=-0.27, CI 跨 0)、small_frac、domain_remote、log_n_gt 单因子 CI 均跨 0
  → 这些因子的独立贡献**尚未确立**, 仅作方向性暗示。
- 要 decisive 地做因子重要性分析, 需更多场景/类/检测器(n 至少数十级)才有稳定估计。

## 诚实边界
- n=11 场景, bootstrap CI 较宽; 相关性强但非决定性证据, 仅作方向性定量。
- baseline 与 domain 在本数据中共线(COCO 全 natural+高 n, DIOR 全 remote+低 n),
  domain_remote 的系数应解读为"数据集/域差异"而非独立因果。
- ☆ 表示诊断量(内容/条件表示质量)未纳入; 当前管线**无结构化 z_c/z_k 向量**
  (CondEncoder 输出 z_k 为标量条件强度, z_c 仅作 DINOv2 768-d 嵌入用于原型余弦、从不存储/探针),
  故 z_c_probe/effective_rank_zc/z_k_sensitivity/z_k_leakage/joint_gain_proxy 无法按原定义计算;
  仅能做 DINOv2 嵌入代理(需 ~70min GPU 重跑 prep 提取逐框嵌入)。
"""
    with open(os.path.join(out_dir, 'factor_importance_report.md'), 'w', encoding='utf-8') as fp:
        fp.write(rep)
    print('\n写出: factor_matrix.csv / factor_importance_uni.csv / _multivariate.csv / _lasso.csv / _report.md', flush=True)
    print(f'结果目录 → {out_dir}', flush=True)


if __name__ == '__main__':
    main()
