# -*- coding: utf-8 -*-
"""compute_kappa.py — 计算κ_c = 类内proposal分数落在决策阈值±δ带内的比例
这是零训练可算、跨数据集可比的因子，直接连到机制
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats

from sota_comparison import (
    SOURCE_LOADERS,
    DIOR_TARGETS, NWPU_TARGETS, DOTA_TARGETS, VISDRONE_TARGETS,
    build_proto_tge, evaluate_with_features,
    T, SRC_TMPL_OVERHEAD, TGT_TMPL
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

REMOTE_TARGETS = {
    'DIOR': DIOR_TARGETS,
    'NWPU': NWPU_TARGETS,
    'DOTA': DOTA_TARGETS,
    'VisDrone': VISDRONE_TARGETS,
}

# 决策阈值（检测器分数的预过滤阈值）
DECISION_THRESHOLD = 0.03
# 带宽度δ
DELTA = 0.02


def compute_kappa(target_data, delta=DELTA):
    """计算κ_c = 类内proposal分数落在决策阈值±δ带内的比例"""
    per_img = target_data['per_img']

    all_scores = []
    for p in per_img:
        if 'gd' in p and len(p['gd']) > 0:
            scores = p['gd']
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            all_scores.extend(scores)

    all_scores = np.array(all_scores)

    if len(all_scores) == 0:
        return 0.0, 0, all_scores

    # 落在阈值±δ带内的比例
    in_band = np.abs(all_scores - DECISION_THRESHOLD) <= delta
    kappa = float(np.mean(in_band))

    # 其他相关统计量
    n_band = int(in_band.sum())
    n_total = len(all_scores)

    return kappa, n_band, all_scores


def compute_additional_factors(target_data, src_proto, proto_neg):
    """计算其他零训练因子"""
    features = target_data['features']
    per_img = target_data['per_img']

    # 收集所有检测器分数
    all_det_scores = []
    for p in per_img:
        if 'gd' in p and len(p['gd']) > 0:
            scores = p['gd']
            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            all_det_scores.extend(scores)
    all_det_scores = np.array(all_det_scores)

    # 原型相似度
    sim_pos = features @ src_proto
    sim_neg = features @ proto_neg
    sim_margin = sim_pos - sim_neg

    factors = {}

    # κ_c：阈值附近的概率质量
    kappa, n_band, _ = compute_kappa(target_data)
    factors['kappa'] = kappa
    factors['kappa_n'] = n_band

    # 检测器分数分布
    if len(all_det_scores) > 0:
        factors['det_score_mean'] = float(np.mean(all_det_scores))
        factors['det_score_std'] = float(np.std(all_det_scores))
        factors['det_score_low_ratio'] = float(np.mean(all_det_scores < 0.05))
        factors['det_score_median'] = float(np.median(all_det_scores))
        # 分数低于阈值的比例
        factors['det_below_threshold'] = float(np.mean(all_det_scores < DECISION_THRESHOLD))
        # 分数在阈值以下的标准差（不确定性）
        below = all_det_scores[all_det_scores < DECISION_THRESHOLD]
        factors['det_below_std'] = float(np.std(below)) if len(below) > 1 else 0.0
    else:
        for k in ['det_score_mean', 'det_score_std', 'det_score_low_ratio', 'det_score_median',
                  'det_below_threshold', 'det_below_std']:
            factors[k] = 0.0

    # 原型相似度分布
    factors['sim_pos_mean'] = float(np.mean(sim_pos))
    factors['sim_pos_std'] = float(np.std(sim_pos))
    factors['sim_margin_mean'] = float(np.mean(sim_margin))
    factors['sim_margin_std'] = float(np.std(sim_margin))

    # 原型相似度在阈值附近的比例（用0作为阈值，因为相似度可能正负）
    factors['sim_near_zero_ratio'] = float(np.mean(np.abs(sim_margin) < 0.1))

    return factors


def process_all_samples():
    """处理所有COCO和VOC样本"""
    all_samples = []

    # COCO目标域
    from sota_comparison import load_target_coco
    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            print(f'  处理 {src_name}/{src_cls} -> COCO/{coco_cls}...')

            src_loader = SOURCE_LOADERS[src_name][0]
            src_data = src_loader(src_cls, n=150)
            if src_data is None:
                continue
            src_proto = src_data['prototype']
            src_features = src_data['features']

            sims = src_features @ src_proto
            neg_idx = np.argmin(sims)
            proto_neg = src_features[neg_idx]
            proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

            target_data = load_target_coco(coco_cls, n=200)
            if target_data is None:
                continue

            factors = compute_additional_factors(target_data, src_proto, proto_neg)

            # 计算最优α
            best_alpha = 0
            best_ap = evaluate_with_features(target_data, src_proto, neg_proto=proto_neg)
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
            factors['baseline_ap'] = float(best_ap - (best_ap - evaluate_with_features(target_data, src_proto, neg_proto=proto_neg)))
            factors['baseline_ap'] = float(evaluate_with_features(target_data, src_proto, neg_proto=proto_neg))
            factors['best_alpha'] = float(best_alpha)
            factors['best_ap'] = float(best_ap)

            all_samples.append(factors)
            print(f'    κ={factors["kappa"]:.4f}, baseline AP={factors["baseline_ap"]:.4f}, best α={best_alpha}')

    # VOC目标域
    from voc_sota_comparison import load_target_voc
    COCO2VOC = {
        'airplane': 'aeroplane', 'boat': 'boat', 'car': 'car',
        'bus': 'bus', 'motorcycle': 'motorbike', 'bicycle': 'bicycle',
    }

    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            if coco_cls not in COCO2VOC:
                continue
            voc_cls = COCO2VOC[coco_cls]
            print(f'  处理 {src_name}/{src_cls} -> VOC/{voc_cls}...')

            src_loader = SOURCE_LOADERS[src_name][0]
            src_data = src_loader(src_cls, n=150)
            if src_data is None:
                continue
            src_proto = src_data['prototype']
            src_features = src_data['features']

            sims = src_features @ src_proto
            neg_idx = np.argmin(sims)
            proto_neg = src_features[neg_idx]
            proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

            target_data = load_target_voc(voc_cls, n=200)
            if target_data is None:
                continue

            factors = compute_additional_factors(target_data, src_proto, proto_neg)

            # 计算最优α
            best_alpha = 0
            baseline_ap = evaluate_with_features(target_data, src_proto, neg_proto=proto_neg)
            best_ap = baseline_ap
            for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                proto_a = build_proto_tge(src_proto, src_cls, voc_cls, beta=alpha)
                ap_a = evaluate_with_features(target_data, proto_a, neg_proto=proto_neg)
                if ap_a > best_ap:
                    best_ap = ap_a
                    best_alpha = alpha

            factors['src'] = src_name
            factors['src_cls'] = src_cls
            factors['target_cls'] = voc_cls
            factors['target_domain'] = 'VOC'
            factors['baseline_ap'] = float(baseline_ap)
            factors['best_alpha'] = float(best_alpha)
            factors['best_ap'] = float(best_ap)

            all_samples.append(factors)
            print(f'    κ={factors["kappa"]:.4f}, baseline AP={baseline_ap:.4f}, best α={best_alpha}')

    return all_samples


def analyze_kappa(samples):
    """分析κ_c和α*的相关性"""
    print(f'\n{"=" * 80}')
    print('κ_c 和其他零训练因子 vs α* 相关性分析')
    print(f'{"=" * 80}')

    factor_names = ['kappa', 'det_score_mean', 'det_score_std', 'det_score_low_ratio',
                    'det_below_threshold', 'det_below_std', 'sim_pos_mean', 'sim_pos_std',
                    'sim_margin_mean', 'sim_margin_std', 'sim_near_zero_ratio', 'baseline_ap']

    alphas = np.array([s['best_alpha'] for s in samples])

    # 全数据
    print(f'\n--- 全数据 (N={len(samples)}) ---')
    print(f'{"因子":<25s} {"Spearman ρ":>12s} {"p值":>10s} {"显著性":>8s}')
    print('-' * 60)

    results = {}
    for factor in factor_names:
        values = np.array([s[factor] for s in samples])
        rho, p = stats.spearmanr(values, alphas)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f'{factor:<25s} {rho:>12.4f} {p:>10.4f}{sig}')
        results[factor] = {'rho': rho, 'p': p}

    # 分数据集
    for domain in ['COCO', 'VOC']:
        subset = [s for s in samples if s['target_domain'] == domain]
        if len(subset) < 3:
            continue
        alphas_sub = np.array([s['best_alpha'] for s in subset])
        print(f'\n--- {domain} (N={len(subset)}) ---')
        print(f'{"因子":<25s} {"Spearman ρ":>12s} {"p值":>10s}')
        print('-' * 55)
        for factor in factor_names:
            values = np.array([s[factor] for s in subset])
            rho, p = stats.spearmanr(values, alphas_sub)
            print(f'{factor:<25s} {rho:>12.4f} {p:>10.4f}')

    # z-score合并检验
    print(f'\n--- 数据集内z-score合并检验 ---')
    for factor in ['kappa', 'det_below_threshold', 'sim_near_zero_ratio']:
        # 原始值
        values_raw = np.array([s[factor] for s in samples])
        rho_raw, _ = stats.spearmanr(values_raw, alphas)

        # z-score
        values_z = values_raw.copy()
        for domain in ['COCO', 'VOC']:
            mask = np.array([s['target_domain'] == domain for s in samples])
            if mask.sum() > 1:
                values_z[mask] = (values_raw[mask] - values_raw[mask].mean()) / (values_raw[mask].std() + 1e-8)
        rho_z, _ = stats.spearmanr(values_z, alphas)

        print(f'  {factor}: raw ρ={rho_raw:.4f}, z-score ρ={rho_z:.4f}, diff={rho_raw - rho_z:.4f}')
        if abs(rho_raw - rho_z) > 0.2:
            print(f'    → 主要在做数据集判别')
        else:
            print(f'    → 有一定类级别解释力')

    return results


def main():
    print('=' * 80)
    print('计算κ_c = 类内proposal分数落在决策阈值±δ带内的比例')
    print('=' * 80)

    samples = process_all_samples()
    print(f'\n总样本数: {len(samples)}')

    # 保存
    with open(DATA / 'kappa_analysis_samples.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # 分析
    results = analyze_kappa(samples)

    # 保存结果
    with open(DATA / 'kappa_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n{"=" * 80}')
    print('核心发现：')
    sorted_factors = sorted(results.items(), key=lambda x: abs(x[1]['rho']), reverse=True)
    print(f'  和α*相关性最强的零训练因子：')
    for i, (factor, res) in enumerate(sorted_factors[:5]):
        print(f'    {i+1}. {factor}: ρ={res["rho"]:.4f} (p={res["p"]:.4f})')
    print(f'{"=" * 80}')


if __name__ == '__main__':
    main()
