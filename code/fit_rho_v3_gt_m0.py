# -*- coding: utf-8 -*-
"""fit_rho_v3.py — 用GT标签重新计算m^0 + 拟合
理论中的初始间隔应该用GT标签计算：
m_c^(0) = mean(p_c^T f_i - max_{k!=c} p_k^T f_i) for i: y_i=c
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


def compute_m0_with_gt(target_data, proto_c, proto_neg_list):
    """用GT标签计算初始间隔m_c^(0)
    m_c^(0) = mean(p_c^T f_i - max(p_neg^T f_i)) for i: GT是c类
    """
    features = target_data['features']
    per_img = target_data['per_img']

    # 收集所有GT是c类的proposal
    pos_margins = []
    for p in per_img:
        gts = p['gt']  # GT框列表
        if len(gts) == 0:
            continue
        crop_idx = p['crop_idx']
        if len(crop_idx) == 0:
            continue
        f_boxes = features[crop_idx]

        # 对每个GT框，找最匹配的proposal
        for gt in gts:
            # 计算每个proposal和GT框的IoU
            best_iou = 0
            best_f = None
            for i, box in enumerate(p['box']):
                iou = compute_iou(box, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_f = f_boxes[i]
            if best_f is not None and best_iou > 0.5:
                # 计算正类别得分
                s_pos = best_f @ proto_c
                # 计算最大竞争类别得分
                s_neg = max([best_f @ pn for pn in proto_neg_list])
                margin = s_pos - s_neg
                pos_margins.append(margin)

    if len(pos_margins) == 0:
        return None, 0

    m_0_gt = float(np.mean(pos_margins))
    return m_0_gt, len(pos_margins)


def compute_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def load_data_with_gt_m0():
    """加载VOC α网格搜索结果，并用GT标签重新计算m^0"""
    # 先加载原始结果（获取alpha_grid等）
    result_file = DATA / 'voc_alpha_grid_scan_results.json'
    with open(result_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 导入需要的模块
    from sota_comparison import (
        SOURCE_LOADERS, build_proto_tge,
        T, SRC_TMPL_OVERHEAD, TGT_TMPL
    )
    from voc_sota_comparison import load_target_voc

    samples = []
    for item in raw_data:
        if item['mu_c'] <= 1e-8:
            continue

        src_name = item['src']
        src_cls = item['src_cls']
        target_cls = item['target_cls']

        print(f'  重新计算 {src_name}/{src_cls} -> VOC/{target_cls} 的m^0(GT)...')

        # 加载源域数据
        src_loader = SOURCE_LOADERS[src_name][0]
        src_data = src_loader(src_cls, n=150)
        if src_data is None:
            continue
        src_proto = src_data['prototype']
        src_features = src_data['features']

        # 加载VOC目标域数据
        target_data = load_target_voc(target_cls, n=200)
        if target_data is None:
            continue

        # 构造负原型列表（用源域特征中离正原型最远的3个）
        sims = src_features @ src_proto
        neg_indices = np.argsort(sims)[:3]
        proto_neg_list = [src_features[i] / (np.linalg.norm(src_features[i]) + 1e-8)
                          for i in neg_indices]

        # 用GT标签计算m^0
        m_0_gt, n_pos = compute_m0_with_gt(target_data, src_proto, proto_neg_list)
        if m_0_gt is None:
            print(f'    警告: 没有找到GT匹配的proposal')
            continue

        samples.append({
            'src': src_name,
            'src_cls': src_cls,
            'target_cls': target_cls,
            'm_0': item['m_0'],  # 原来的置信度m^0
            'm_0_gt': m_0_gt,  # 用GT标签计算的m^0
            'n_pos': n_pos,
            'mu_c': item['mu_c'],
            'alpha_grid': item['best_alpha'],
            'baseline_ap': item['baseline_ap'],
            'alpha_gt_0_better': item['alpha_gt_0_better'],
        })
        print(f'    m^0(置信度)={item["m_0"]:.4f}, m^0(GT)={m_0_gt:.4f}, n_pos={n_pos}')

    return samples


def hinge_model(m_0, mu_c, rho, c):
    """闭式族：α = clip(c * (ρ - m_0)_+ / μ, 0, α_max)"""
    raw = c * np.maximum(0, rho - m_0) / mu_c
    return np.clip(raw, 0, ALPHA_MAX)


def fit_rho_c(samples, m0_key='m_0_gt'):
    """拟合ρ和c"""
    m_0_values = np.array([s[m0_key] for s in samples])
    mu_c_values = np.array([s['mu_c'] for s in samples])
    alpha_grid = np.array([s['alpha_grid'] for s in samples])

    rho_grid = np.linspace(-1.0, 1.0, 201)  # GT的m^0可能是负的
    c_grid = np.linspace(0.01, 2.0, 200)

    best_rho = None
    best_c = None
    best_mse = float('inf')

    for rho in rho_grid:
        for c in c_grid:
            alpha_pred = hinge_model(m_0_values, mu_c_values, rho, c)
            mse = np.mean((alpha_pred - alpha_grid) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_rho = rho
                best_c = c

    alpha_pred = hinge_model(m_0_values, mu_c_values, best_rho, best_c)
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
        'm0_key': m0_key,
    }


def main():
    print('=' * 80)
    print('用GT标签重新计算m^0 + 拟合')
    print('=' * 80)

    # 加载数据并用GT标签重新计算m^0
    print(f'\n--- 用GT标签重新计算m^0 ---')
    samples = load_data_with_gt_m0()
    print(f'\n有效样本数: {len(samples)}')

    if len(samples) < 5:
        print('样本太少，退出')
        return

    # 对比两种m^0
    print(f'\n--- 对比两种m^0 ---')
    m0_conf = np.array([s['m_0'] for s in samples])
    m0_gt = np.array([s['m_0_gt'] for s in samples])
    alpha_grid = np.array([s['alpha_grid'] for s in samples])

    rho_conf, p_conf = stats.spearmanr(m0_conf, alpha_grid)
    rho_gt, p_gt = stats.spearmanr(m0_gt, alpha_grid)

    print(f'  置信度m^0 vs α_grid: Spearman ρ={rho_conf:.4f}, p={p_conf:.4f}')
    print(f'  GT m^0 vs α_grid: Spearman ρ={rho_gt:.4f}, p={p_gt:.4f}')

    # 用GT m^0拟合
    print(f'\n--- 用GT m^0拟合（2个参数）---')
    fit_gt = fit_rho_c(samples, m0_key='m_0_gt')
    print(f'  最优ρ: {fit_gt["rho"]:.4f}')
    print(f'  最优c: {fit_gt["c"]:.4f}')
    print(f'  R²: {fit_gt["r_squared"]:.4f}')
    print(f'  Spearman ρ: {fit_gt["spearman_rho"]:.4f} (p={fit_gt["spearman_p"]:.4f})')

    # 用置信度m^0拟合（对比）
    print(f'\n--- 用置信度m^0拟合（对比）---')
    fit_conf = fit_rho_c(samples, m0_key='m_0')
    print(f'  最优ρ: {fit_conf["rho"]:.4f}')
    print(f'  最优c: {fit_conf["c"]:.4f}')
    print(f'  R²: {fit_conf["r_squared"]:.4f}')
    print(f'  Spearman ρ: {fit_conf["spearman_rho"]:.4f} (p={fit_conf["spearman_p"]:.4f})')

    # 保存结果
    output = {
        'n_samples': len(samples),
        'm0_conf_spearman': rho_conf,
        'm0_conf_p': p_conf,
        'm0_gt_spearman': rho_gt,
        'm0_gt_p': p_gt,
        'fit_gt': {
            'rho': fit_gt['rho'],
            'c': fit_gt['c'],
            'r_squared': fit_gt['r_squared'],
            'spearman_rho': fit_gt['spearman_rho'],
            'spearman_p': fit_gt['spearman_p'],
        },
        'fit_conf': {
            'rho': fit_conf['rho'],
            'c': fit_conf['c'],
            'r_squared': fit_conf['r_squared'],
            'spearman_rho': fit_conf['spearman_rho'],
            'spearman_p': fit_conf['spearman_p'],
        },
        'samples': [
            {
                'src': s['src'],
                'src_cls': s['src_cls'],
                'target_cls': s['target_cls'],
                'm_0_conf': s['m_0'],
                'm_0_gt': s['m_0_gt'],
                'n_pos': s['n_pos'],
                'mu_c': s['mu_c'],
                'alpha_grid': s['alpha_grid'],
                'baseline_ap': s['baseline_ap'],
            }
            for s in samples
        ]
    }

    with open(DATA / 'rho_fit_v3_gt_m0.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {DATA / "rho_fit_v3_gt_m0.json"}')

    # 打印逐样本对比
    print(f'\n--- 逐样本对比 ---')
    print(f'{"源域":<10s} {"类别":<15s} {"m0_conf":>9s} {"m0_gt":>9s} {"α_grid":>8s} {"baselineAP":>11s}')
    print('-' * 80)
    for s in samples:
        print(f'{s["src"]:<10s} {s["src_cls"]:<15s} {s["m_0"]:>9.4f} {s["m_0_gt"]:>9.4f} '
              f'{s["alpha_grid"]:>8.2f} {s["baseline_ap"]:>11.4f}')

    print(f'\n{"=" * 80}')
    print('结论：')
    if fit_gt['r_squared'] > fit_conf['r_squared']:
        print(f'  ✅ GT m^0的拟合效果更好 (R²={fit_gt["r_squared"]:.4f} vs {fit_conf["r_squared"]:.4f})')
        print(f'  说明理论中的"初始间隔"应该用GT标签计算')
        print(f'  下一步：改进零训练m^0估计方法，使其更接近GT m^0')
    else:
        print(f'  ⚠️ 置信度m^0的拟合效果更好 (R²={fit_conf["r_squared"]:.4f} vs {fit_gt["r_squared"]:.4f})')
        print(f'  可能是GT m^0的计算方式有问题，或者样本量太小')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
