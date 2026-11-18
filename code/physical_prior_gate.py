# -*- coding: utf-8 -*-
"""physical_prior_gate.py — 物理先验门控
从物体本质属性出发判断"该不该校正"，完全零样本（不需要目标域数据）

物理先验因子：
1. 物体绝对尺寸（米）：小物体需要校正，大物体不需要
2. 视角差（俯视→平视）：视角差异大的需要校正
3. 稀有度（预训练频率）：稀有类别需要校正
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

# 物体绝对尺寸（典型长度，米）
# 来源：COCO/ImageNet类别常识 + 遥感目标典型尺寸
OBJECT_SIZE = {
    'airplane': 30.0,      # 客机约30米
    'aeroplane': 30.0,     # 同airplane
    'plane': 30.0,         # 同airplane
    'boat': 15.0,          # 船只约15米
    'ship': 150.0,         # 大型船舶约150米
    'car': 4.5,            # 轿车约4.5米
    'vehicle': 6.0,        # 通用车辆约6米
    'small-vehicle': 4.5,  # 小型车辆
    'large-vehicle': 12.0, # 大型车辆（卡车/巴士）
    'truck': 12.0,         # 卡车约12米
    'bus': 12.0,           # 巴士约12米
    'motorcycle': 2.0,     # 摩托车约2米
    'motorbike': 2.0,      # 同motorcycle
    'bicycle': 1.8,        # 自行车约1.8米
    'motor': 2.0,          # 同motorcycle
}

# 视角差（俯视→平视，0-1）
# 0 = 视角差异小（物体从上面和从侧面看差不多）
# 1 = 视角差异大（物体从上面和从侧面看完全不同）
VIEWPOINT_DIFF = {
    'airplane': 0.3,       # 飞机俯视和平视差异中等（机翼形状）
    'aeroplane': 0.3,
    'plane': 0.3,
    'boat': 0.4,           # 船只俯视和平视差异中等
    'ship': 0.5,           # 大型船舶差异较大
    'car': 0.7,            # 轿车俯视和平视差异大（车顶vs车身）
    'vehicle': 0.7,
    'small-vehicle': 0.7,
    'large-vehicle': 0.8,  # 卡车/巴士差异更大
    'truck': 0.8,
    'bus': 0.8,
    'motorcycle': 0.6,     # 摩托车差异中等
    'motorbike': 0.6,
    'bicycle': 0.5,        # 自行车差异中等
    'motor': 0.6,
}

# 稀有度（COCO/ImageNet预训练中的相对频率，0-1）
# 0 = 非常常见（预训练中大量出现）
# 1 = 非常稀有（预训练中很少出现）
RARITY = {
    'airplane': 0.3,       # 飞机较常见
    'aeroplane': 0.3,
    'plane': 0.3,
    'boat': 0.5,           # 船只中等稀有
    'ship': 0.6,           # 大型船舶较稀有
    'car': 0.1,            # 轿车非常常见
    'vehicle': 0.2,        # 车辆较常见
    'small-vehicle': 0.2,
    'large-vehicle': 0.4,  # 大型车辆中等稀有
    'truck': 0.3,          # 卡车较常见
    'bus': 0.3,            # 巴士较常见
    'motorcycle': 0.4,     # 摩托车中等稀有
    'motorbike': 0.4,
    'bicycle': 0.3,        # 自行车较常见
    'motor': 0.4,
}


def get_physical_priors(class_name):
    """获取类别的物理先验（零样本）"""
    size = OBJECT_SIZE.get(class_name, 5.0)
    viewpoint = VIEWPOINT_DIFF.get(class_name, 0.5)
    rarity = RARITY.get(class_name, 0.5)
    return {
        'size': size,
        'viewpoint_diff': viewpoint,
        'rarity': rarity,
        'log_size': np.log(size + 1e-8),
    }


def compute_gate_physical(class_name, method='size', threshold=None):
    """物理先验门控（零样本，不需要目标域数据）

    method:
      'size': 物体尺寸门控（小物体需要校正）
      'viewpoint': 视角差门控（视角差异大需要校正）
      'rarity': 稀有度门控（稀有类别需要校正）
      'combined': 多因子加权门控
    """
    priors = get_physical_priors(class_name)

    if method == 'size':
        # 小物体需要校正：log(size) < threshold → gate=1
        if threshold is None:
            threshold = np.log(10.0)  # 默认阈值：10米
        gate = 1.0 if priors['log_size'] < threshold else 0.0

    elif method == 'viewpoint':
        # 视角差异大需要校正：viewpoint_diff > threshold → gate=1
        if threshold is None:
            threshold = 0.5
        gate = 1.0 if priors['viewpoint_diff'] > threshold else 0.0

    elif method == 'rarity':
        # 稀有类别需要校正：rarity > threshold → gate=1
        if threshold is None:
            threshold = 0.4
        gate = 1.0 if priors['rarity'] > threshold else 0.0

    elif method == 'combined':
        # 多因子加权：尺寸(0.4) + 视角(0.3) + 稀有度(0.3)
        # 归一化到[0,1]
        size_score = 1.0 - min(1.0, priors['log_size'] / np.log(50.0))  # 越小越需要校正
        viewpoint_score = priors['viewpoint_diff']
        rarity_score = priors['rarity']

        combined_score = 0.4 * size_score + 0.3 * viewpoint_score + 0.3 * rarity_score
        if threshold is None:
            threshold = 0.5
        gate = 1.0 if combined_score > threshold else 0.0

    else:
        gate = 1.0  # 默认全开

    return float(gate)


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


def compute_alpha_dual_factor(sv, tc, sv_mean=3.0, sv_std=1.0, tc_mean=0.3, tc_std=0.1,
                               sv_weight=-0.5, tc_weight=0.3, bias=1.0):
    """双因子固定权重（零训练），先归一化再预测"""
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

    sv = compute_sv(target_data)
    tc = compute_tc(target_cls)
    priors = get_physical_priors(target_cls)

    # sim_pos_mean（用于数据驱动门控对比）
    features = target_data['features']
    sim_pos_mean = float(np.mean(features @ src_proto))

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
        'sv': sv, 'tc': tc,
        'size': priors['size'], 'viewpoint_diff': priors['viewpoint_diff'],
        'rarity': priors['rarity'], 'log_size': priors['log_size'],
        'sim_pos_mean': sim_pos_mean,
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
    print('物理先验门控实验（零样本，不需要目标域数据）')
    print('=' * 80)

    # 加载或处理样本
    samples_file = DATA / 'physical_prior_gate_samples.json'
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

    # 基线：纯双因子（无门控）
    def method_dual(sample, all_samples):
        alpha = compute_alpha_dual_factor(sample['sv'], sample['tc'])
        return alpha, 1.0
    methods.append(('纯双因子(无门控)', method_dual))

    # 数据驱动门控（对比用）
    domain_medians = {}
    for domain in ['COCO', 'VOC']:
        domain_samples = [s for s in all_samples if s['target_domain'] == domain]
        if domain_samples:
            domain_medians[domain] = np.median([s['sim_pos_mean'] for s in domain_samples])
        else:
            domain_medians[domain] = 0.5

    def method_data_gate(sample, all_samples):
        domain = sample['target_domain']
        gate = 1.0 if sample['sim_pos_mean'] < domain_medians[domain] else 0.0
        alpha_dual = compute_alpha_dual_factor(sample['sv'], sample['tc'])
        return gate * alpha_dual, gate
    methods.append(('数据驱动门控+双因子', method_data_gate))

    # 物理先验门控：尺寸
    def method_size_gate(sample, all_samples):
        gate = compute_gate_physical(sample['target_cls'], method='size')
        alpha_dual = compute_alpha_dual_factor(sample['sv'], sample['tc'])
        return gate * alpha_dual, gate
    methods.append(('物理先验-尺寸门控', method_size_gate))

    # 物理先验门控：视角差
    def method_viewpoint_gate(sample, all_samples):
        gate = compute_gate_physical(sample['target_cls'], method='viewpoint')
        alpha_dual = compute_alpha_dual_factor(sample['sv'], sample['tc'])
        return gate * alpha_dual, gate
    methods.append(('物理先验-视角差门控', method_viewpoint_gate))

    # 物理先验门控：稀有度
    def method_rarity_gate(sample, all_samples):
        gate = compute_gate_physical(sample['target_cls'], method='rarity')
        alpha_dual = compute_alpha_dual_factor(sample['sv'], sample['tc'])
        return gate * alpha_dual, gate
    methods.append(('物理先验-稀有度门控', method_rarity_gate))

    # 物理先验门控：多因子组合
    def method_combined_gate(sample, all_samples):
        gate = compute_gate_physical(sample['target_cls'], method='combined')
        alpha_dual = compute_alpha_dual_factor(sample['sv'], sample['tc'])
        return gate * alpha_dual, gate
    methods.append(('物理先验-组合门控', method_combined_gate))

    # Oracle（上界）
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

    for gate_method in ['size', 'viewpoint', 'rarity', 'combined']:
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
    with open(DATA / 'physical_prior_gate_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'results': [{k: v for k, v in r.items() if k != 'gains'} for r in results],
        }, f, indent=2, ensure_ascii=False)

    print(f'\n结果已保存到: {DATA / "physical_prior_gate_results.json"}')


if __name__ == '__main__':
    main()
