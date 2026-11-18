# -*- coding: utf-8 -*-
"""diagnose_alpha.py — α*诊断脚本
检查：
1. 触界比例（α*=0或α*=2的比例）
2. α*可辨识性（AP峰值±0.5内的平台宽度）
3. 数据集内相关 + Fisher z检验
4. OLS vs 两部分模型的LOCO对比（z-score合并与原始值合并）
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import csv
from pathlib import Path
import numpy as np
from scipy import stats

DATA = Path(r'C:\Users\张三\Desktop\实验数据')


def load_data():
    """加载alpha_star汇总表和网格搜索长表"""
    # 加载alpha_star汇总表
    with open(DATA / 'alpha_star_summary.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        alpha_star_data = list(reader)

    # 加载网格搜索长表
    with open(DATA / 'alpha_grid_search.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        grid_data = list(reader)

    # 转换数值类型
    for row in alpha_star_data:
        row['alpha_star'] = float(row['alpha_star'])
        row['ap50_star'] = float(row['ap50_star'])
        row['ap50_alpha0'] = float(row['ap50_alpha0'])
        row['m0'] = float(row['m0'])
        row['mu'] = float(row['mu'])

    for row in grid_data:
        row['alpha'] = float(row['alpha'])
        row['ap50'] = float(row['ap50'])
        row['m0'] = float(row['m0'])
        row['mu'] = float(row['mu'])

    return alpha_star_data, grid_data


def check_boundary(alpha_star_data):
    """检查触界比例"""
    print('\n' + '=' * 80)
    print('1. 触界比例检查')
    print('=' * 80)

    n_total = len(alpha_star_data)
    n_zero = sum(1 for r in alpha_star_data if r['alpha_star'] == 0.0)
    n_max = sum(1 for r in alpha_star_data if r['alpha_star'] == 2.0)
    n_interior = n_total - n_zero - n_max

    print(f'\n总类别数: {n_total}')
    print(f'α*=0（触下界）: {n_zero} ({n_zero/n_total*100:.1f}%)')
    print(f'α*=2（触上界）: {n_max} ({n_max/n_total*100:.1f}%)')
    print(f'内部点: {n_interior} ({n_interior/n_total*100:.1f}%)')

    # 分数据集
    for dataset in ['COCO', 'VOC']:
        subset = [r for r in alpha_star_data if r['dataset'] == dataset]
        n = len(subset)
        n_z = sum(1 for r in subset if r['alpha_star'] == 0.0)
        n_m = sum(1 for r in subset if r['alpha_star'] == 2.0)
        print(f'\n  {dataset} (N={n}):')
        print(f'    α*=0: {n_z} ({n_z/n*100:.1f}%)')
        print(f'    α*=2: {n_m} ({n_m/n*100:.1f}%)')

    print(f'\n结论: α*在0处被截断，主要发生在VOC高基线类别。')
    print(f'这意味着应该用两级模型（gate+回归），而不是单级OLS回归。')


def check_identifiability(grid_data):
    """检查α*可辨识性（AP峰值±0.005内的平台宽度）"""
    print('\n' + '=' * 80)
    print('2. α*可辨识性检查')
    print('=' * 80)

    # 按类别分组
    classes = list(set(r['class'] for r in grid_data))
    classes.sort()

    print(f'\n{"类别":<25s} {"α*":>6s} {"峰值AP":>10s} {"平台宽度":>10s} {"可辨识":>8s}')
    print('-' * 65)

    identifiable_count = 0
    for cls in classes:
        subset = [r for r in grid_data if r['class'] == cls]
        subset.sort(key=lambda x: x['alpha'])

        alphas = [r['alpha'] for r in subset]
        aps = [r['ap50'] for r in subset]

        if not aps:
            continue

        peak_ap = max(aps)
        peak_idx = aps.index(peak_ap)
        alpha_star = alphas[peak_idx]

        # 计算平台宽度：AP在峰值±0.005内的α范围
        threshold = 0.005
        plateau_alphas = [alphas[i] for i in range(len(aps)) if abs(aps[i] - peak_ap) <= threshold]
        plateau_width = max(plateau_alphas) - min(plateau_alphas) if plateau_alphas else 0

        identifiable = plateau_width < 0.5  # 平台宽度小于0.5才算可辨识
        if identifiable:
            identifiable_count += 1

        print(f'{cls:<25s} {alpha_star:>6.2f} {peak_ap:>10.4f} {plateau_width:>10.2f} {"是" if identifiable else "否":>8s}')

    print(f'\n可辨识类别数: {identifiable_count}/{len(classes)} ({identifiable_count/len(classes)*100:.1f}%)')
    print(f'平台宽度大的类别，α*估计不可靠，应该谨慎解读。')


def check_within_dataset_correlation(alpha_star_data):
    """检查数据集内相关 + Fisher z检验"""
    print('\n' + '=' * 80)
    print('3. 数据集内相关 + Fisher z检验')
    print('=' * 80)

    factors = ['m0', 'ap50_alpha0']

    for factor in factors:
        print(f'\n--- 因子: {factor} ---')

        for dataset in ['COCO', 'VOC']:
            subset = [r for r in alpha_star_data if r['dataset'] == dataset]
            if len(subset) < 3:
                continue

            x = np.array([r[factor] for r in subset])
            y = np.array([r['alpha_star'] for r in subset])

            rho, p = stats.spearmanr(x, y)
            print(f'  {dataset} (N={len(subset)}): ρ={rho:.4f}, p={p:.4f}')

        # Fisher z检验
        coco_subset = [r for r in alpha_star_data if r['dataset'] == 'COCO']
        voc_subset = [r for r in alpha_star_data if r['dataset'] == 'VOC']

        if len(coco_subset) >= 3 and len(voc_subset) >= 3:
            x_coco = np.array([r[factor] for r in coco_subset])
            y_coco = np.array([r['alpha_star'] for r in coco_subset])
            rho_coco, _ = stats.spearmanr(x_coco, y_coco)

            x_voc = np.array([r[factor] for r in voc_subset])
            y_voc = np.array([r['alpha_star'] for r in voc_subset])
            rho_voc, _ = stats.spearmanr(x_voc, y_voc)

            # Fisher z变换
            z_coco = np.arctanh(rho_coco)
            z_voc = np.arctanh(rho_voc)
            se = np.sqrt(1/(len(coco_subset)-3) + 1/(len(voc_subset)-3))
            z_stat = (z_coco - z_voc) / se
            p_fisher = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            print(f'\n  Fisher z检验: z={z_stat:.4f}, p={p_fisher:.4f}')
            if p_fisher > 0.05:
                print(f'  → 两个数据集的相关系数无显著差异，不能说"COCO强、VOC弱"')
            else:
                print(f'  → 两个数据集的相关系数有显著差异')


def compare_models(alpha_star_data):
    """OLS vs 两部分模型的LOCO对比"""
    print('\n' + '=' * 80)
    print('4. OLS vs 两部分模型的LOCO对比')
    print('=' * 80)

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler

    factors = ['m0', 'ap50_alpha0']

    for factor in factors:
        print(f'\n--- 因子: {factor} ---')

        # 准备数据
        X = np.array([[r[factor]] for r in alpha_star_data])
        y = np.array([r['alpha_star'] for r in alpha_star_data])
        y_binary = (y > 0).astype(int)  # gate标签：是否需要校正

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
        rho_ols, _ = stats.spearmanr(y_pred_ols, y)

        # 两部分模型：gate + 回归
        y_pred_two = np.zeros_like(y)
        for train_idx, test_idx in loo.split(X_norm):
            # 第一级：gate（逻辑回归）
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

            # 合并
            y_pred_two[test_idx] = gate_pred * np.maximum(0, reg_pred)

        ss_res_two = np.sum((y - y_pred_two) ** 2)
        r2_two = 1 - ss_res_two / ss_tot
        rho_two, _ = stats.spearmanr(y_pred_two, y)

        print(f'\n  原始值合并:')
        print(f'    OLS回归: LOCO R²={r2_ols:.4f}, Spearman ρ={rho_ols:.4f}')
        print(f'    两部分模型: LOCO R²={r2_two:.4f}, Spearman ρ={rho_two:.4f}')

        # z-score合并（数据集内标准化）
        X_zscore = X.copy()
        for dataset in ['COCO', 'VOC']:
            mask = np.array([r['dataset'] == dataset for r in alpha_star_data])
            if mask.sum() > 1:
                X_zscore[mask] = (X[mask] - X[mask].mean()) / (X[mask].std() + 1e-8)

        # OLS on z-score
        y_pred_ols_z = np.zeros_like(y)
        for train_idx, test_idx in loo.split(X_zscore):
            model = LinearRegression()
            model.fit(X_zscore[train_idx], y[train_idx])
            y_pred_ols_z[test_idx] = model.predict(X_zscore[test_idx])

        ss_res_z = np.sum((y - y_pred_ols_z) ** 2)
        r2_ols_z = 1 - ss_res_z / ss_tot
        rho_ols_z, _ = stats.spearmanr(y_pred_ols_z, y)

        print(f'\n  数据集内z-score合并:')
        print(f'    OLS回归: LOCO R²={r2_ols_z:.4f}, Spearman ρ={rho_ols_z:.4f}')

        # 判读
        print(f'\n  判读:')
        if r2_ols > r2_ols_z + 0.1:
            print(f'    raw表远高于z-score表 → 该因子主要在做数据集判别')
        else:
            print(f'    raw表和z-score表接近 → 该因子有一定类级别解释力')


def main():
    print('=' * 80)
    print('α*诊断脚本')
    print('=' * 80)

    alpha_star_data, grid_data = load_data()
    print(f'\n加载数据:')
    print(f'  alpha_star汇总: {len(alpha_star_data)} 个类别')
    print(f'  网格搜索长表: {len(grid_data)} 行')

    # 1. 触界比例
    check_boundary(alpha_star_data)

    # 2. 可辨识性
    check_identifiability(grid_data)

    # 3. 数据集内相关
    check_within_dataset_correlation(alpha_star_data)

    # 4. 模型对比
    compare_models(alpha_star_data)

    print('\n' + '=' * 80)
    print('诊断完成')
    print('=' * 80)


if __name__ == '__main__':
    main()
