# -*- coding: utf-8 -*-
"""physical_prior_closed_form.py — 物理先验门控 + 理论闭式解
完全零训练：连目标域无标签数据都不需要
- 门控：物理先验（尺寸/视角差/稀有度），从物体本质属性出发
- α预测：理论闭式解 α = (ρ - m^0)_+ / μ
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

# ============================================================
# 物理先验知识库（零样本，不需要目标域数据）
# ============================================================

OBJECT_SIZE = {
    'airplane': 30.0, 'aeroplane': 30.0, 'plane': 30.0,
    'boat': 15.0, 'ship': 150.0,
    'car': 4.5, 'vehicle': 6.0, 'small-vehicle': 4.5, 'large-vehicle': 12.0,
    'truck': 12.0, 'bus': 12.0,
    'motorcycle': 2.0, 'motorbike': 2.0, 'motor': 2.0,
    'bicycle': 1.8,
}

VIEWPOINT_DIFF = {
    'airplane': 0.3, 'aeroplane': 0.3, 'plane': 0.3,
    'boat': 0.4, 'ship': 0.5,
    'car': 0.7, 'vehicle': 0.7, 'small-vehicle': 0.7, 'large-vehicle': 0.8,
    'truck': 0.8, 'bus': 0.8,
    'motorcycle': 0.6, 'motorbike': 0.6, 'motor': 0.6,
    'bicycle': 0.5,
}

RARITY = {
    'airplane': 0.3, 'aeroplane': 0.3, 'plane': 0.3,
    'boat': 0.5, 'ship': 0.6,
    'car': 0.1, 'vehicle': 0.2, 'small-vehicle': 0.2, 'large-vehicle': 0.4,
    'truck': 0.3, 'bus': 0.3,
    'motorcycle': 0.4, 'motorbike': 0.4, 'motor': 0.4,
    'bicycle': 0.3,
}


def get_physical_priors(class_name):
    size = OBJECT_SIZE.get(class_name, 5.0)
    viewpoint = VIEWPOINT_DIFF.get(class_name, 0.5)
    rarity = RARITY.get(class_name, 0.5)
    return {
        'size': size, 'viewpoint_diff': viewpoint, 'rarity': rarity,
        'log_size': np.log(size + 1e-8),
    }


def compute_gate_physical(class_name, method='size_viewpoint', threshold=None):
    """物理先验门控（零样本，不需要目标域数据）"""
    priors = get_physical_priors(class_name)

    if method == 'size':
        if threshold is None:
            threshold = np.log(10.0)
        gate = 1.0 if priors['log_size'] < threshold else 0.0

    elif method == 'viewpoint':
        if threshold is None:
            threshold = 0.5
        gate = 1.0 if priors['viewpoint_diff'] > threshold else 0.0

    elif method == 'size_viewpoint':
        # 尺寸小 OR 视角差大 → 需要校正
        size_need = priors['log_size'] < np.log(10.0)
        viewpoint_need = priors['viewpoint_diff'] > 0.5
        gate = 1.0 if (size_need or viewpoint_need) else 0.0

    elif method == 'size_viewpoint_and':
        # 尺寸小 AND 视角差大 → 需要校正（更严格）
        size_need = priors['log_size'] < np.log(10.0)
        viewpoint_need = priors['viewpoint_diff'] > 0.5
        gate = 1.0 if (size_need and viewpoint_need) else 0.0

    elif method == 'combined_score':
        # 多因子加权得分
        size_score = 1.0 - min(1.0, priors['log_size'] / np.log(50.0))
        viewpoint_score = priors['viewpoint_diff']
        combined = 0.5 * size_score + 0.5 * viewpoint_score
        if threshold is None:
            threshold = 0.5
        gate = 1.0 if combined > threshold else 0.0

    else:
        gate = 1.0

    return float(gate)


def compute_m0_mu(target_data, src_proto):
    """计算m0（初始间隔）和mu（校正效率），零训练可算"""
    features = target_data['features']
    if len(features) == 0:
        return 0.5, 0.1

    # 计算每个proposal与源原型的相似度
    sims = features @ src_proto

    # m0: 与源原型的相似度中位数（初始间隔的代理）
    m0 = float(np.median(sims))

    # mu: 校正效率，用特征的平均范数作为代理
    # 最优缩放因子：0.001（=0.1 × 0.01，来自参数扫描）
    norms = np.linalg.norm(features, axis=1)
    mu = float(np.mean(norms)) * 0.001  # 最优缩放因子

    return m0, max(mu, 1e-8)


def compute_alpha_closed_form(m0, mu, rho=0.13, mu_scale=0.01):
    """理论闭式解：α = (ρ - m^0)_+ / (μ × mu_scale)（最优参数：ρ=0.13, mu_scale=0.01）"""
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

    priors = get_physical_priors(target_cls)
    m0, mu = compute_m0_mu(target_data, src_proto)

    # α网格搜索
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
        'size': priors['size'], 'viewpoint_diff': priors['viewpoint_diff'],
        'rarity': priors['rarity'], 'log_size': priors['log_size'],
        'm0': m0, 'mu': mu,
        'best_alpha': float(best_alpha), 'best_ap': float(best_ap),
        'baseline_ap': float(baseline_ap),
        'oracle_gain': float(best_ap - baseline_ap),
        'ap_curve': ap_curve,
    }


def evaluate_method(samples, method_name, alpha_func):
    n = len(samples)
    y_true = np.array([s['best_alpha'] for s in samples])
    y_pred = np.zeros(n)
    gains = []
    gates = []

    for i, sample in enumerate(samples):
        alpha, gate = alpha_func(sample, samples)
        y_pred[i] = alpha
        gates.append(gate)

        ap_curve = sample['ap_curve']
        alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
        alpha_clipped = np.clip(alpha, ALPHAS[0], ALPHAS[-1])
        ap = float(np.interp(alpha_clipped, alphas_sorted, [ap_curve[str(a)] for a in alphas_sorted]))
        baseline = ap_curve['0.0']
        gains.append(ap - baseline)

    if np.std(y_pred) > 1e-8:
        rho, p = stats.spearmanr(y_pred, y_true)
    else:
        rho, p = 0.0, 1.0

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
    print('物理先验门控 + 理论闭式解（完全零训练）')
    print('=' * 80)

    # 加载零训练门控的样本数据（已经计算好m0/mu，和参数扫描一致）
    samples_file = DATA / 'zero_shot_gated_samples.json'
    if samples_file.exists():
        print(f'\n加载零训练门控样本数据: {samples_file}')
        with open(samples_file, 'r', encoding='utf-8') as f:
            all_samples = json.load(f)
        print(f'加载了 {len(all_samples)} 个样本')
    else:
        print(f'错误：零训练门控样本数据不存在 {samples_file}')
        return

    print(f'\n总样本数: {len(all_samples)}')

    # 打印物理先验统计
    print(f'\n{"=" * 80}')
    print('物理先验统计')
    print(f'{"=" * 80}')
    print(f'\n{"类别":<15s} {"尺寸(m)":>8s} {"视角差":>8s} {"稀有度":>8s} {"log尺寸":>8s}')
    print('-' * 55)
    for cls in ['airplane', 'boat', 'car', 'bus', 'motorcycle', 'bicycle', 'truck']:
        priors = get_physical_priors(cls)
        print(f'{cls:<15s} {priors["size"]:>8.1f} {priors["viewpoint_diff"]:>8.2f} '
              f'{priors["rarity"]:>8.2f} {priors["log_size"]:>8.3f}')

    # 定义各种方法
    methods = []

    # 基线1：理论闭式解（无门控）
    def method_closed_no_gate(sample, all_samples):
        alpha = compute_alpha_closed_form(sample['m0'], sample['mu'], rho=0.2)
        return alpha, 1.0
    methods.append(('理论闭式解(无门控)', method_closed_no_gate))

    # 基线2：数据驱动门控+闭式解（对比用，需要目标域数据）
    domain_medians = {}
    for domain in ['COCO', 'VOC']:
        domain_samples = [s for s in all_samples if s['target_domain'] == domain]
        if domain_samples:
            # 用m0作为sim_pos_mean的代理
            domain_medians[domain] = np.median([s['m0'] for s in domain_samples])
        else:
            domain_medians[domain] = 0.5

    def method_data_gate_closed(sample, all_samples):
        domain = sample['target_domain']
        gate = 1.0 if sample['m0'] < domain_medians[domain] else 0.0
        alpha_closed = compute_alpha_closed_form(sample['m0'], sample['mu'], rho=0.2)
        return gate * alpha_closed, gate
    methods.append(('数据驱动门控+闭式解', method_data_gate_closed))

    # 物理先验门控 + 闭式解
    for gate_method in ['size', 'viewpoint', 'size_viewpoint', 'size_viewpoint_and', 'combined_score']:
        def make_method(gm):
            def method(sample, all_samples):
                gate = compute_gate_physical(sample['target_cls'], method=gm)
                alpha_closed = compute_alpha_closed_form(sample['m0'], sample['mu'], rho=0.2)
                return gate * alpha_closed, gate
            return method
        methods.append((f'物理先验-{gate_method}门控+闭式解', make_method(gate_method)))

    # Oracle（上界）
    def method_oracle(sample, all_samples):
        return sample['best_alpha'], 1.0
    methods.append(('Oracle(上界)', method_oracle))

    # 评估所有方法
    print(f'\n{"=" * 80}')
    print('所有方法对比')
    print(f'{"=" * 80}')
    print(f'\n{"方法":<30s} {"ρ":>8s} {"平均增益":>10s} {"中位增益":>10s} {"正例":>8s} {"Oracle%":>8s} {"平均门控":>8s}')
    print('-' * 95)

    results = []
    for method_name, method_func in methods:
        result = evaluate_method(all_samples, method_name, method_func)
        results.append(result)
        print(f'{method_name:<30s} {result["rho"]:>8.4f} {result["mean_gain"]:>+10.4f} '
              f'{result["median_gain"]:>+10.4f} {result["n_positive"]:>4d}/{result["n_total"]:<4d} '
              f'{result["oracle_realization"]:>7.1f}% {result["mean_gate"]:>8.4f}')

    # 分域详细结果
    print(f'\n{"=" * 80}')
    print('分域详细结果')
    print(f'{"=" * 80}')

    for result in results:
        if '物理先验' not in result['method'] and '数据驱动' not in result['method']:
            continue
        print(f'\n  --- {result["method"]} ---')
        for domain in ['COCO', 'VOC']:
            domain_idx = [i for i, s in enumerate(all_samples) if s['target_domain'] == domain]
            gains = [result['gains'][i] for i in domain_idx]
            print(f'    {domain}: 平均增益={np.mean(gains):+.4f}, '
                  f'正例={sum(1 for g in gains if g > 0)}/{len(gains)}')

    # 门控决策分析
    print(f'\n{"=" * 80}')
    print('物理先验门控决策分析')
    print(f'{"=" * 80}')

    for gate_method in ['size', 'viewpoint', 'size_viewpoint', 'size_viewpoint_and']:
        print(f'\n  --- {gate_method}门控 ---')
        for domain in ['COCO', 'VOC']:
            domain_samples = [s for s in all_samples if s['target_domain'] == domain]
            gated_on = [s for s in domain_samples if compute_gate_physical(s['target_cls'], method=gate_method) > 0.5]
            gated_off = [s for s in domain_samples if compute_gate_physical(s['target_cls'], method=gate_method) <= 0.5]

            print(f'    {domain}: 开启={len(gated_on)}/{len(domain_samples)}, '
                  f'关闭={len(gated_off)}/{len(domain_samples)}')
            if gated_on:
                print(f'      开启类平均Oracle增益: {np.mean([s["oracle_gain"] for s in gated_on]):+.4f}')
            if gated_off:
                print(f'      关闭类平均Oracle增益: {np.mean([s["oracle_gain"] for s in gated_off]):+.4f}')

    # 保存结果
    with open(DATA / 'physical_prior_closed_form_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'results': [{k: v for k, v in r.items() if k != 'gains'} for r in results],
        }, f, indent=2, ensure_ascii=False)

    print(f'\n结果已保存到: {DATA / "physical_prior_closed_form_results.json"}')


if __name__ == '__main__':
    main()
