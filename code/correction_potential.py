# -*- coding: utf-8 -*-
"""correction_potential.py — 寻找零训练的"校正潜力"估计量
目标：找到一个零训练可算的因子，能近似ap_gain（α=2时AP增益）
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

from sota_comparison import (
    SOURCE_LOADERS,
    DIOR_TARGETS, NWPU_TARGETS, DOTA_TARGETS, VISDRONE_TARGETS,
    build_proto_tge, evaluate_with_features,
    T, SRC_TMPL_OVERHEAD, TGT_TMPL
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

REMOTE_TARGETS = {
    'DIOR': DIOR_TARGETS,
    'NWPU': NWPU_TARGETS,
    'DOTA': DOTA_TARGETS,
    'VisDrone': VISDRONE_TARGETS,
}


def compute_zero_shot_factors(target_data, src_proto, proto_neg):
    """计算零训练可算的"校正潜力"候选因子"""
    features = target_data['features']
    per_img = target_data['per_img']

    # 收集所有候选框的检测器分数
    all_det_scores = []
    for p in per_img:
        if 'gd' in p and len(p['gd']) > 0:
            all_det_scores.extend(p['gd'].tolist() if hasattr(p['gd'], 'tolist') else list(p['gd']))
    all_det_scores = np.array(all_det_scores)

    # 计算原型相似度
    sim_pos = features @ src_proto
    sim_neg = features @ proto_neg
    sim_margin = sim_pos - sim_neg

    factors = {}

    # 1. 检测器分数相关
    if len(all_det_scores) > 0:
        factors['det_score_mean'] = float(np.mean(all_det_scores))
        factors['det_score_std'] = float(np.std(all_det_scores))
        factors['det_score_low_ratio'] = float(np.mean(all_det_scores < 0.1))  # 低置信度比例
        factors['det_score_entropy'] = float(-np.sum(all_det_scores * np.log(all_det_scores + 1e-8)) / len(all_det_scores))
        factors['det_score_max'] = float(np.max(all_det_scores))
        factors['det_score_median'] = float(np.median(all_det_scores))
    else:
        for k in ['det_score_mean', 'det_score_std', 'det_score_low_ratio', 'det_score_entropy', 'det_score_max', 'det_score_median']:
            factors[k] = 0.0

    # 2. 原型相似度相关
    factors['sim_pos_mean'] = float(np.mean(sim_pos))
    factors['sim_pos_std'] = float(np.std(sim_pos))
    factors['sim_neg_mean'] = float(np.mean(sim_neg))
    factors['sim_margin_mean'] = float(np.mean(sim_margin))
    factors['sim_margin_std'] = float(np.std(sim_margin))
    factors['sim_margin_median'] = float(np.median(sim_margin))
    factors['sim_pos_low_ratio'] = float(np.mean(sim_pos < 0.2))  # 正原型相似度低的比例

    # 3. 校正潜力估计
    # 假设：检测器分数低 + 原型相似度低 → 校正潜力大
    factors['correction_potential_v1'] = factors['det_score_low_ratio'] * (1 - factors['sim_pos_mean'])
    factors['correction_potential_v2'] = (1 - factors['det_score_mean']) * (1 - factors['sim_pos_mean'])
    factors['correction_potential_v3'] = factors['det_score_std'] * factors['sim_margin_std']

    # 4. 候选框数量
    factors['n_boxes'] = int(len(all_det_scores))
    factors['n_boxes_log'] = float(np.log(len(all_det_scores) + 1))

    return factors


def process_all_samples():
    """处理所有样本，计算零训练因子"""
    all_samples = []

    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            print(f'  处理 {src_name}/{src_cls} -> COCO/{coco_cls}...')

            # 加载源域数据
            src_loader = SOURCE_LOADERS[src_name][0]
            src_data = src_loader(src_cls, n=150)
            if src_data is None:
                continue
            src_proto = src_data['prototype']
            src_features = src_data['features']

            # 构造负原型
            sims = src_features @ src_proto
            neg_idx = np.argmin(sims)
            proto_neg = src_features[neg_idx]
            proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

            # 加载COCO目标域数据
            from sota_comparison import load_target_coco
            target_data = load_target_coco(coco_cls, n=200)
            if target_data is None:
                continue

            # 计算零训练因子
            factors = compute_zero_shot_factors(target_data, src_proto, proto_neg)

            # 计算ap_gain（α=2时AP增益）
            proto_alpha2 = build_proto_tge(src_proto, src_cls, coco_cls, beta=2.0)
            ap_alpha0 = evaluate_with_features(target_data, src_proto, neg_proto=proto_neg)
            ap_alpha2 = evaluate_with_features(target_data, proto_alpha2, neg_proto=proto_neg)
            ap_gain = ap_alpha2 - ap_alpha0

            # 找最优α
            best_alpha = 0
            best_ap = ap_alpha0
            for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                proto_a = build_proto_tge(src_proto, src_cls, coco_cls, beta=alpha)
                ap_a = evaluate_with_features(target_data, proto_a, neg_proto=proto_neg)
                if ap_a > best_ap:
                    best_ap = ap_a
                    best_alpha = alpha

            factors['src'] = src_name
            factors['src_cls'] = src_cls
            factors['target_cls'] = coco_cls
            factors['target_domain'] = 'COCO'
            factors['baseline_ap'] = float(ap_alpha0)
            factors['ap_gain'] = float(ap_gain)
            factors['best_alpha'] = float(best_alpha)
            factors['best_ap'] = float(best_ap)

            all_samples.append(factors)
            print(f'    baseline AP={ap_alpha0:.4f}, ap_gain={ap_gain:.4f}, best α={best_alpha}')

    return all_samples


def analyze_correlations(samples):
    """分析所有零训练因子和ap_gain/最优α的相关性"""
    factor_names = [k for k in samples[0].keys()
                    if k not in ['src', 'src_cls', 'target_cls', 'target_domain',
                                 'baseline_ap', 'ap_gain', 'best_alpha', 'best_ap']]

    print(f'\n=== 零训练因子 vs ap_gain（校正潜力）相关性（N={len(samples)}）===')
    print(f'{"因子":<30s} {"vs ap_gain ρ":>14s} {"p值":>10s} {"vs α ρ":>10s} {"p值":>10s}')
    print('-' * 80)

    results = {}
    ap_gains = np.array([s['ap_gain'] for s in samples])
    alphas = np.array([s['best_alpha'] for s in samples])

    for factor in factor_names:
        values = np.array([s[factor] for s in samples])
        rho_gain, p_gain = stats.spearmanr(values, ap_gains)
        rho_alpha, p_alpha = stats.spearmanr(values, alphas)
        sig_gain = '***' if p_gain < 0.001 else ('**' if p_gain < 0.01 else ('*' if p_gain < 0.05 else ''))
        print(f'{factor:<30s} {rho_gain:>14.4f} {p_gain:>10.4f}{sig_gain} {rho_alpha:>10.4f} {p_alpha:>10.4f}')
        results[factor] = {'rho_gain': rho_gain, 'p_gain': p_gain, 'rho_alpha': rho_alpha, 'p_alpha': p_alpha}

    return results


def main():
    print('=' * 80)
    print('寻找零训练的"校正潜力"估计量')
    print('=' * 80)

    # 处理所有样本
    print(f'\n--- 处理所有COCO目标域样本 ---')
    samples = process_all_samples()
    print(f'\n总样本数: {len(samples)}')

    # 保存原始数据
    with open(DATA / 'correction_potential_samples.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # 分析相关性
    corr_results = analyze_correlations(samples)

    # 保存结果
    output = {
        'n_samples': len(samples),
        'correlations': corr_results,
    }
    with open(DATA / 'correction_potential_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\n{"=" * 80}')
    print('核心发现：')
    # 找出和ap_gain相关性最强的3个因子
    sorted_factors = sorted(corr_results.items(), key=lambda x: abs(x[1]['rho_gain']), reverse=True)
    print(f'  和ap_gain相关性最强的零训练因子：')
    for i, (factor, res) in enumerate(sorted_factors[:5]):
        print(f'    {i+1}. {factor}: ρ={res["rho_gain"]:.4f} (p={res["p_gain"]:.4f})')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
