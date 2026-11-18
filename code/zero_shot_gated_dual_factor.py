# -*- coding: utf-8 -*-
"""zero_shot_gated_dual_factor.py — 真正零训练的门控双因子
完全不需要拟合权重、不需要搜索温度、不需要比较结构：
1. 门控：硬阈值（sim_pos_mean < 域内中位数 → gate=1，否则0）
2. 双因子：理论闭式解 α = (ρ - m^0)_+ / μ
3. 结构：乘法结构（理论分析决定）
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

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
ALPHA_MAX = 2.0
ALPHA_MIN = 0.0

ALL_CLASSES = ['airplane', 'boat', 'car', 'bus', 'motorcycle', 'bicycle', 'truck',
               'aeroplane', 'motorbike', 'ship', 'vehicle', 'plane', 'small-vehicle', 'large-vehicle']


def compute_sv(target_data):
    """计算sv = 候选框尺寸方差（log面积的方差）"""
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
    """计算m^0 = 初始间隔（top1-top2置信度中位数），零训练可算"""
    features = target_data['features']
    # 这里简化：用源域原型打分，计算top1-top2
    # 实际应该用所有类别的原型打分，这里只有一个类别的原型
    # 所以用sim_pos的分布来近似：m^0 = median(sim_pos) - percentile(sim_pos, 25)
    sim_pos = features @ src_proto
    top1 = np.median(sim_pos)
    top2 = np.percentile(sim_pos, 25)  # 近似top2
    m0 = float(top1 - top2)
    return max(0.0, m0)


def compute_mu(target_data, src_proto, src_cls, target_cls):
    """计算μ = 校正效率（原型移动方向与特征的平均内积），零训练可算"""
    features = target_data['features']
    # 计算外推方向
    s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], target_cls).cpu().numpy()
    delta = d_emb - s_emb
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    # 校正效率 = 特征与外推方向的平均内积
    mu = float(np.mean(features @ delta))
    return abs(mu) + 1e-6  # 避免除零


def compute_gate_hard(sim_pos_mean, domain_median):
    """硬阈值门控：零训练，不需要温度"""
    return 1.0 if sim_pos_mean < domain_median else 0.0


def compute_alpha_closed_form(m0, mu, rho=0.2):
    """理论闭式解：α = (ρ - m^0)_+ / μ，零训练，不需要拟合权重"""
    alpha = max(0.0, (rho - m0) / mu)
    return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))


def compute_alpha_dual_factor(sv, tc, sv_mean=3.0, sv_std=1.0, tc_mean=0.3, tc_std=0.1,
                               sv_weight=-0.5, tc_weight=0.3, bias=1.0):
    """双因子固定权重（零训练），先归一化再预测"""
    # z-score归一化（用全局统计量，零训练可预先计算）
    sv_norm = (sv - sv_mean) / (sv_std + 1e-8)
    tc_norm = (tc - tc_mean) / (tc_std + 1e-8)
    alpha = sv_weight * sv_norm + tc_weight * tc_norm + bias
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

    # α网格搜索（找oracle）
    ap_curve = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            proto = src_proto
        else:
            proto = build_proto_tge(src_proto, src_cls, target_cls, beta=alpha)
        ap = evaluate_with_features(target_data, proto, neg_proto=proto_neg)
        ap_curve[str(alpha)] = float(ap)

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


def evaluate_method(samples, method_name, alpha_func):
    """评估一种方法"""
    n = len(samples)
    y_true = np.array([s['best_alpha'] for s in samples])
    y_pred = np.zeros(n)
    gains = []
    gates = []

    for i, sample in enumerate(samples):
        alpha, gate = alpha_func(sample, samples)
        y_pred[i] = alpha
        gates.append(gate)

        # 计算实际AP增益
        ap_curve = sample['ap_curve']
        alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
        alpha_clipped = np.clip(alpha, ALPHAS[0], ALPHAS[-1])
        ap = float(np.interp(alpha_clipped, alphas_sorted, [ap_curve[str(a)] for a in alphas_sorted]))
        baseline = ap_curve['0.0']
        gains.append(ap - baseline)

    rho, p = stats.spearmanr(y_pred, y_true)

    return {
        'method': method_name,
        'rho': float(rho), 'p': float(p),
        'mean_gain': float(np.mean(gains)),
        'median_gain': float(np.median(gains)),
        'n_positive': int(sum(1 for g in gains if g > 0)),
        'n_total': n,
        'oracle_realization': float(np.mean(gains) / (np.mean([s['oracle_gain'] for s in samples]) + 1e-8) * 100),
        'mean_gate': float(np.mean(gates)),
        'gains': gains,
    }


def main():
    print('=' * 80)
    print('真正零训练的门控双因子实验')
    print('=' * 80)

    # 加载或处理样本
    samples_file = DATA / 'zero_shot_gated_samples.json'
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
                print(f'  处理 {src_name}/{src_cls} -> COCO/{coco_cls}...')
                result = process_sample(src_name, src_cls, coco_cls, 'COCO', load_target_coco)
                if result:
                    all_samples.append(result)

        from voc_sota_comparison import load_target_voc
        COCO2VOC = {'airplane': 'aeroplane', 'boat': 'boat', 'car': 'car',
                     'bus': 'bus', 'motorcycle': 'motorbike', 'bicycle': 'bicycle'}
        print(f'\n--- VOC目标域 ---')
        for src_name, class_list in REMOTE_TARGETS.items():
            for coco_cls, src_cls, src_display in class_list:
                if coco_cls not in COCO2VOC:
                    continue
                voc_cls = COCO2VOC[coco_cls]
                print(f'  处理 {src_name}/{src_cls} -> VOC/{voc_cls}...')
                result = process_sample(src_name, src_cls, voc_cls, 'VOC', load_target_voc)
                if result:
                    all_samples.append(result)

        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f'\n总样本数: {len(all_samples)}')

    # 计算域内中位数
    domain_medians = {}
    for domain in ['COCO', 'VOC']:
        domain_samples = [s for s in all_samples if s['target_domain'] == domain]
        if domain_samples:
            domain_medians[domain] = np.median([s['sim_pos_mean'] for s in domain_samples])
        else:
            domain_medians[domain] = 0.5

    print(f'\n域内中位数:')
    for domain, median in domain_medians.items():
        print(f'  {domain}: {median:.4f}')

    # 定义各种方法
    methods = []

    # 方法1：纯双因子固定权重（无门控）
    def method_dual_fixed(sample, all_samples):
        alpha = compute_alpha_dual_factor(sample['sv'], sample['tc'])
        return alpha, 1.0
    methods.append(('纯双因子(固定权重)', method_dual_fixed))

    # 方法2：理论闭式解（无门控）
    def method_closed_form(sample, all_samples):
        alpha = compute_alpha_closed_form(sample['m0'], sample['mu'], rho=0.2)
        return alpha, 1.0
    methods.append(('理论闭式解(无门控)', method_closed_form))

    # 方法3：硬阈值门控 + 双因子固定权重
    def method_gated_dual(sample, all_samples):
        domain = sample['target_domain']
        gate = compute_gate_hard(sample['sim_pos_mean'], domain_medians[domain])
        alpha_dual = compute_alpha_dual_factor(sample['sv'], sample['tc'])
        alpha = gate * alpha_dual
        return alpha, gate
    methods.append(('门控+双因子(固定权重)', method_gated_dual))

    # 方法4：硬阈值门控 + 理论闭式解
    def method_gated_closed(sample, all_samples):
        domain = sample['target_domain']
        gate = compute_gate_hard(sample['sim_pos_mean'], domain_medians[domain])
        alpha_closed = compute_alpha_closed_form(sample['m0'], sample['mu'], rho=0.2)
        alpha = gate * alpha_closed
        return alpha, gate
    methods.append(('门控+闭式解(零训练)', method_gated_closed))

    # 方法5：Oracle（上界）
    def method_oracle(sample, all_samples):
        return sample['best_alpha'], 1.0
    methods.append(('Oracle(上界)', method_oracle))

    # 评估所有方法
    print(f'\n{"=" * 80}')
    print('所有方法对比')
    print(f'{"=" * 80}')
    print(f'\n{"方法":<25s} {"ρ":>8s} {"平均增益":>10s} {"中位增益":>10s} {"正例":>8s} {"Oracle%":>8s} {"平均门控":>8s}')
    print('-' * 90)

    results = []
    for method_name, method_func in methods:
        result = evaluate_method(all_samples, method_name, method_func)
        results.append(result)
        print(f'{method_name:<25s} {result["rho"]:>8.4f} {result["mean_gain"]:>+10.4f} '
              f'{result["median_gain"]:>+10.4f} {result["n_positive"]:>4d}/{result["n_total"]:<4d} '
              f'{result["oracle_realization"]:>7.1f}% {result["mean_gate"]:>8.4f}')

    # 分域详细结果（重点看零训练门控+闭式解）
    print(f'\n{"=" * 80}')
    print('分域详细结果（重点：零训练门控+闭式解）')
    print(f'{"=" * 80}')

    zero_shot_result = [r for r in results if '零训练' in r['method']][0]

    for domain in ['COCO', 'VOC']:
        domain_idx = [i for i, s in enumerate(all_samples) if s['target_domain'] == domain]
        domain_samples = [all_samples[i] for i in domain_idx]
        gains = [zero_shot_result['gains'][i] for i in domain_idx]

        # 纯双因子的增益
        dual_result = [r for r in results if '纯双因子' in r['method']][0]
        gains_dual = [dual_result['gains'][i] for i in domain_idx]

        print(f'\n  {domain} (N={len(domain_idx)}):')
        print(f'    零训练门控+闭式解: 平均增益={np.mean(gains):+.4f}, '
              f'中位增益={np.median(gains):+.4f}, '
              f'正例={sum(1 for g in gains if g > 0)}/{len(gains)}')
        print(f'    纯双因子(固定权重): 平均增益={np.mean(gains_dual):+.4f}, '
              f'正例={sum(1 for g in gains_dual if g > 0)}/{len(gains_dual)}')
        print(f'    Oracle: 平均增益={np.mean([s["oracle_gain"] for s in domain_samples]):+.4f}')

    # 门控决策分析
    print(f'\n{"=" * 80}')
    print('门控决策分析')
    print(f'{"=" * 80}')

    for domain in ['COCO', 'VOC']:
        domain_samples = [s for s in all_samples if s['target_domain'] == domain]
        median = domain_medians[domain]
        gated_on = [s for s in domain_samples if s['sim_pos_mean'] < median]
        gated_off = [s for s in domain_samples if s['sim_pos_mean'] >= median]

        print(f'\n  {domain}:')
        print(f'    门控开启（sim_pos_mean < {median:.4f}）: {len(gated_on)}/{len(domain_samples)}')
        if gated_on:
            print(f'      平均Oracle增益: {np.mean([s["oracle_gain"] for s in gated_on]):+.4f}')
        if gated_off:
            print(f'    门控关闭（sim_pos_mean >= {median:.4f}）: {len(gated_off)}/{len(domain_samples)}')
            print(f'      平均Oracle增益: {np.mean([s["oracle_gain"] for s in gated_off]):+.4f}')

    # 保存结果
    with open(DATA / 'zero_shot_gated_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'results': [{k: v for k, v in r.items() if k != 'gains'} for r in results],
            'domain_medians': domain_medians,
        }, f, indent=2, ensure_ascii=False)

    print(f'\n结果已保存到: {DATA / "zero_shot_gated_results.json"}')


if __name__ == '__main__':
    main()
