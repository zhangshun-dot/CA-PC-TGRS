# -*- coding: utf-8 -*-
"""extend_alpha_grid.py — 扩α网格到4/6重搜
不改方法、不重训练，只是把α网格扩大，看看主表数字能不能往上走。
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
    T, SRC_TMPL_OVERHEAD, TGT_TMPL,
    compute_text_confusability,
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

REMOTE_TARGETS = {
    'DIOR': DIOR_TARGETS,
    'NWPU': NWPU_TARGETS,
    'DOTA': DOTA_TARGETS,
    'VisDrone': VISDRONE_TARGETS,
}

# COCO到VOC类别映射（增加person和van）
COCO2VOC = {'airplane': 'aeroplane', 'boat': 'boat', 'car': 'car',
             'bus': 'bus', 'motorcycle': 'motorbike', 'bicycle': 'bicycle',
             'person': 'person', 'van': 'car'}  # van在VOC里映射到car

# 扩α网格到4/6
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
ALPHA_MAX = 6.0
ALPHA_MIN = 0.0

ALL_CLASSES = ['airplane', 'boat', 'car', 'bus', 'motorcycle', 'bicycle', 'truck',
               'aeroplane', 'motorbike', 'ship', 'vehicle', 'plane', 'small-vehicle', 'large-vehicle']


def compute_sv(target_data):
    per_img = target_data['per_img']
    all_areas = []
    for p in per_img:
        if 'box' in p and len(p['box']) > 0:
            boxes = p['box']
            for box in boxes:
                if len(box) >= 4:
                    area = max(0, (box[2] - box[0])) * max(0, (box[3] - box[1]))
                    if area > 0:
                        all_areas.append(area)
    if len(all_areas) < 2:
        return 0.0
    all_areas = np.array(all_areas)
    log_areas = np.log(all_areas + 1e-8)
    return float(np.var(log_areas))


def compute_tc(target_cls):
    return float(compute_text_confusability(target_cls, ALL_CLASSES))


def compute_sim_factors(target_data, src_proto):
    features = target_data['features']
    sim_pos = features @ src_proto
    return {
        'sim_pos_mean': float(np.mean(sim_pos)),
        'sim_pos_std': float(np.std(sim_pos)),
    }


def compute_m0(target_data, src_proto):
    features = target_data['features']
    sim_pos = features @ src_proto
    top1 = np.median(sim_pos)
    top2 = np.percentile(sim_pos, 25)
    m0 = float(top1 - top2)
    return max(0.0, m0)


def compute_mu(target_data, src_proto, src_cls, target_cls):
    features = target_data['features']
    s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], target_cls).cpu().numpy()
    delta = d_emb - s_emb
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    mu = float(np.mean(features @ delta))
    return abs(mu) + 1e-6


def compute_alpha_closed_form(m0, mu, rho=0.13, mu_scale=0.01):
    mu_eff = mu * mu_scale
    alpha = max(0.0, (rho - m0) / max(mu_eff, 1e-8))
    return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))


def process_sample(src_name, src_cls, target_cls, target_domain, target_loader_func):
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']

    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

    target_data = target_loader_func(target_cls, n=200)
    if target_data is None:
        return None

    sim_factors = compute_sim_factors(target_data, src_proto)
    sv = compute_sv(target_data)
    tc = compute_tc(target_cls)
    m0 = compute_m0(target_data, src_proto)
    mu = compute_mu(target_data, src_proto, src_cls, target_cls)

    # α网格搜索（扩到6.0）
    ap_curve = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            proto = src_proto
        else:
            proto = build_proto_tge(src_proto, src_cls, target_cls, beta=alpha)
        ap = evaluate_with_features(target_data, proto, neg_proto=proto_neg)
        ap_curve[str(alpha)] = float(ap)
        print(f'    α={alpha:.1f}: AP={ap:.4f}')

    best_alpha = ALPHAS[0]
    best_ap = ap_curve[str(ALPHAS[0])]
    for alpha in ALPHAS:
        if ap_curve[str(alpha)] > best_ap:
            best_ap = ap_curve[str(alpha)]
            best_alpha = alpha

    baseline_ap = ap_curve['0.0']

    return {
        'src': src_name, 'src_cls': src_cls, 'target_cls': target_cls,
        'target_domain': target_domain,
        'sv': sv, 'tc': tc, 'm0': m0, 'mu': mu,
        'sim_pos_mean': sim_factors['sim_pos_mean'],
        'sim_pos_std': sim_factors['sim_pos_std'],
        'best_alpha': float(best_alpha), 'best_ap': float(best_ap),
        'baseline_ap': float(baseline_ap),
        'oracle_gain': float(best_ap - baseline_ap),
        'ap_curve': ap_curve,
    }


def main():
    print('=' * 80)
    print('扩α网格到6.0重搜实验')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    if samples_file.exists():
        print(f'\n加载已保存的样本数据: {samples_file}')
        with open(samples_file, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        print(f'加载了 {len(all_samples)} 个样本')
    else:
        all_samples = []
        from sota_comparison import load_target_coco
        print(f'\n--- COCO目标域 ---')
        for src_name, class_list in REMOTE_TARGETS.items():
            for coco_cls, src_cls, src_display in class_list:
                print(f'\n  处理 {src_name}/{src_cls} -> COCO/{coco_cls}...')
                result = process_sample(src_name, src_cls, coco_cls, 'COCO', load_target_coco)
                if result:
                    all_samples.append(result)

        from voc_sota_comparison import load_target_voc
        COCO2VOC = {'airplane': 'aeroplane', 'boat': 'boat', 'car': 'car',
                     'bus': 'bus', 'motorcycle': 'motorbike', 'bicycle': 'bicycle',
                     'person': 'person', 'van': 'car'}  # van在VOC里映射到car
        print(f'\n--- VOC目标域 ---')
        for src_name, class_list in REMOTE_TARGETS.items():
            for coco_cls, src_cls, src_display in class_list:
                if coco_cls not in COCO2VOC:
                    continue
                voc_cls = COCO2VOC[coco_cls]
                print(f'\n  处理 {src_name}/{src_cls} -> VOC/{voc_cls}...')
                result = process_sample(src_name, src_cls, voc_cls, 'VOC', load_target_voc)
                if result:
                    all_samples.append(result)

        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f'\n总样本数: {len(all_samples)}')

    # 分析结果
    print(f'\n{"=" * 80}')
    print('分析结果：α=2.0 vs 最优α')
    print(f'{"=" * 80}')

    gains_at_2 = []
    gains_at_best = []
    best_alphas = []
    for s in all_samples:
        ap_2 = s['ap_curve'].get('2.0', s['ap_curve'].get('2'))
        baseline = s['ap_curve']['0.0']
        gains_at_2.append(ap_2 - baseline)
        gains_at_best.append(s['oracle_gain'])
        best_alphas.append(s['best_alpha'])

    print(f'\n  α=2.0时平均增益: {np.mean(gains_at_2):+.4f}')
    print(f'  最优α时平均增益: {np.mean(gains_at_best):+.4f}')
    print(f'  提升: {np.mean(gains_at_best) - np.mean(gains_at_2):+.4f}')
    print(f'  最优α分布:')
    for alpha in ALPHAS:
        count = sum(1 for a in best_alphas if abs(a - alpha) < 0.01)
        if count > 0:
            print(f'    α={alpha:.1f}: {count}个样本 ({count/len(best_alphas)*100:.1f}%)')

    # 闭式解+物理先验门控的效果（用最优参数ρ=0.13, mu_scale=0.01）
    print(f'\n{"=" * 80}')
    print('闭式解+物理先验门控的效果（用扩网格后的ap_curve）')
    print(f'{"=" * 80}')

    # 物理先验门控（viewpoint门控）
    VIEWPOINT_DIFF = {'airplane': 0.3, 'boat': 0.4, 'car': 0.7, 'bus': 0.8,
                       'motorcycle': 0.6, 'bicycle': 0.5, 'truck': 0.8,
                       'aeroplane': 0.3, 'motorbike': 0.6}

    gains_closed = []
    gains_gated = []
    for s in all_samples:
        # 闭式解
        alpha_closed = compute_alpha_closed_form(s['m0'], s['mu'], rho=0.13, mu_scale=0.01)
        ap_curve = s['ap_curve']
        alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
        ap_closed = float(np.interp(alpha_closed, alphas_sorted, [ap_curve[str(a)] for a in alphas_sorted]))
        baseline = ap_curve['0.0']
        gains_closed.append(ap_closed - baseline)

        # 物理先验门控（viewpoint门控）
        target_cls = s['target_cls']
        viewpoint_diff = VIEWPOINT_DIFF.get(target_cls, 0.5)
        gate = 1.0 if viewpoint_diff > 0.5 else 0.0
        alpha_gated = gate * alpha_closed
        ap_gated = float(np.interp(alpha_gated, alphas_sorted, [ap_curve[str(a)] for a in alphas_sorted]))
        gains_gated.append(ap_gated - baseline)

    print(f'\n  闭式解（无门控）: 平均增益={np.mean(gains_closed):+.4f}, 正例={sum(g > 0 for g in gains_closed)}/{len(gains_closed)}')
    print(f'  物理先验门控+闭式解: 平均增益={np.mean(gains_gated):+.4f}, 正例={sum(g > 0 for g in gains_gated)}/{len(gains_gated)}')
    print(f'  Oracle: 平均增益={np.mean(gains_at_best):+.4f}, 正例={sum(g > 0 for g in gains_at_best)}/{len(gains_at_best)}')

    # 分域结果
    print(f'\n  分域结果（物理先验门控+闭式解）:')
    for domain in ['COCO', 'VOC']:
        domain_gains = [g for g, s in zip(gains_gated, all_samples) if s['target_domain'] == domain]
        print(f'    {domain}: 平均增益={np.mean(domain_gains):+.4f}, 正例={sum(g > 0 for g in domain_gains)}/{len(domain_gains)}')

    # 保存结果
    results = {
        'gains_at_2_mean': float(np.mean(gains_at_2)),
        'gains_at_best_mean': float(np.mean(gains_at_best)),
        'improvement': float(np.mean(gains_at_best) - np.mean(gains_at_2)),
        'gains_closed_mean': float(np.mean(gains_closed)),
        'gains_gated_mean': float(np.mean(gains_gated)),
        'best_alphas': [float(a) for a in best_alphas],
    }
    results_file = DATA / 'extended_alpha_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n  结果已保存: {results_file}')


if __name__ == '__main__':
    main()
