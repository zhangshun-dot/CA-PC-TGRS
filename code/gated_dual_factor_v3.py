# -*- coding: utf-8 -*-
"""gated_dual_factor_v3.py — 门控双因子改进版
尝试多个改进方向：
1. 多因子门控（sim_pos_mean + sim_pos_std）
2. 更平滑的门控函数（softplus/tanh）
3. 门控+双因子加法结构
4. 域内分位数门控
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

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


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def softplus(x):
    """softplus函数，比sigmoid更平滑"""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def tanh_gate(x):
    """tanh门控，输出[0,1]"""
    return (np.tanh(x) + 1) / 2


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


def compute_gate_v1(sim_pos_mean, domain_median, temperature=0.05):
    """原始门控：sigmoid((median - mean) / temp)"""
    x = (domain_median - sim_pos_mean) / temperature
    return float(sigmoid(x))


def compute_gate_v2(sim_pos_mean, sim_pos_std, domain_median_mean, domain_median_std,
                     temperature=0.05, std_weight=0.3):
    """多因子门控：mean + std共同决定
    std大 → 特征离散度大 → 原型不稳定 → 门控应该更保守（降低）
    """
    x_mean = (domain_median_mean - sim_pos_mean) / temperature
    x_std = (sim_pos_std - domain_median_std) / (temperature + 1e-8)
    x = x_mean - std_weight * x_std  # std大则降低门控
    return float(sigmoid(x))


def compute_gate_v3(sim_pos_mean, domain_median, temperature=0.05):
    """softplus门控，更平滑"""
    x = (domain_median - sim_pos_mean) / temperature
    # softplus归一化到[0,1]
    gate = softplus(x) / (softplus(x) + softplus(-x) + 1e-8)
    return float(gate)


def compute_gate_v4(sim_pos_mean, domain_quantile, temperature=0.05):
    """域内分位数门控：用分位数而不是绝对中位数"""
    x = (domain_quantile - sim_pos_mean) / temperature
    return float(sigmoid(x))


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
        'sim_pos_mean': sim_factors['sim_pos_mean'],
        'sim_pos_std': sim_factors['sim_pos_std'],
        'best_alpha': float(best_alpha), 'best_ap': float(best_ap),
        'baseline_ap': float(baseline_ap),
        'oracle_gain': float(best_ap - baseline_ap),
        'ap_curve': ap_curve,
    }


def train_and_evaluate(samples, gate_type='v1', structure='multiply', temperature=0.05):
    """训练门控+双因子模型
    gate_type: v1(原始sigmoid), v2(多因子), v3(softplus), v4(分位数)
    structure: multiply(α=gate*α_dual), add(α=gate*α_max+(1-gate)*α_dual)
    """
    n = len(samples)

    # 计算域内统计量
    domain_stats = {}
    for domain in ['COCO', 'VOC']:
        domain_samples = [s for s in samples if s['target_domain'] == domain]
        if domain_samples:
            domain_stats[domain] = {
                'median_mean': np.median([s['sim_pos_mean'] for s in domain_samples]),
                'median_std': np.median([s['sim_pos_std'] for s in domain_samples]),
                'q30_mean': np.percentile([s['sim_pos_mean'] for s in domain_samples], 30),
                'q70_mean': np.percentile([s['sim_pos_mean'] for s in domain_samples], 70),
            }
        else:
            domain_stats[domain] = {'median_mean': 0.5, 'median_std': 0.05,
                                      'q30_mean': 0.5, 'q70_mean': 0.5}

    loo = LeaveOneOut()
    y_true = np.array([s['best_alpha'] for s in samples])
    y_pred = np.zeros(n)
    y_pred_no_gate = np.zeros(n)
    gate_values = np.zeros(n)

    for train_idx, test_idx in loo.split(range(n)):
        train_samples = [samples[i] for i in train_idx]
        test_sample = samples[test_idx[0]]

        # 训练双因子模型
        train_need = [s for s in train_samples if s['best_alpha'] > 0]
        if len(train_need) >= 3:
            X_train = np.array([[s['sv'], s['tc']] for s in train_need])
            y_train = np.array([s['best_alpha'] for s in train_need])
            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            model = LinearRegression()
            model.fit(X_train_norm, y_train)
            X_test = np.array([[test_sample['sv'], test_sample['tc']]])
            X_test_norm = scaler.transform(X_test)
            alpha_dual = float(np.clip(model.predict(X_test_norm)[0], ALPHA_MIN, ALPHA_MAX))
        else:
            alpha_dual = 1.0

        y_pred_no_gate[test_idx[0]] = alpha_dual

        # 计算门控
        domain = test_sample['target_domain']
        ds = domain_stats[domain]

        if gate_type == 'v1':
            gate = compute_gate_v1(test_sample['sim_pos_mean'], ds['median_mean'], temperature)
        elif gate_type == 'v2':
            gate = compute_gate_v2(test_sample['sim_pos_mean'], test_sample['sim_pos_std'],
                                    ds['median_mean'], ds['median_std'], temperature)
        elif gate_type == 'v3':
            gate = compute_gate_v3(test_sample['sim_pos_mean'], ds['median_mean'], temperature)
        elif gate_type == 'v4':
            gate = compute_gate_v4(test_sample['sim_pos_mean'], ds['q30_mean'], temperature)
        else:
            gate = compute_gate_v1(test_sample['sim_pos_mean'], ds['median_mean'], temperature)

        gate_values[test_idx[0]] = gate

        # 结构
        if structure == 'multiply':
            alpha_gated = gate * alpha_dual
        elif structure == 'add':
            # α = gate * α_max + (1-gate) * α_dual
            # gate=1 → α=α_max（全开），gate=0 → α=α_dual（双因子）
            alpha_gated = gate * ALPHA_MAX + (1 - gate) * alpha_dual
        else:
            alpha_gated = gate * alpha_dual

        alpha_gated = float(np.clip(alpha_gated, ALPHA_MIN, ALPHA_MAX))
        y_pred[test_idx[0]] = alpha_gated

    # 评估
    rho_gated, p_gated = stats.spearmanr(y_pred, y_true)
    rho_no_gate, p_no_gate = stats.spearmanr(y_pred_no_gate, y_true)

    # 计算实际AP增益
    gains_gated = []
    gains_no_gate = []
    gains_oracle = []

    for i, sample in enumerate(samples):
        ap_curve = sample['ap_curve']
        alphas_sorted = sorted([float(a) for a in ap_curve.keys()])

        def interp_ap(alpha):
            alpha = np.clip(alpha, ALPHAS[0], ALPHAS[-1])
            return float(np.interp(alpha, alphas_sorted, [ap_curve[str(a)] for a in alphas_sorted]))

        baseline = ap_curve['0.0']
        gains_gated.append(interp_ap(y_pred[i]) - baseline)
        gains_no_gate.append(interp_ap(y_pred_no_gate[i]) - baseline)
        gains_oracle.append(sample['oracle_gain'])

    return {
        'gate_type': gate_type, 'structure': structure, 'temperature': temperature,
        'rho_gated': float(rho_gated), 'p_gated': float(p_gated),
        'rho_no_gate': float(rho_no_gate), 'p_no_gate': float(p_no_gate),
        'mean_gain_gated': float(np.mean(gains_gated)),
        'mean_gain_no_gate': float(np.mean(gains_no_gate)),
        'mean_gain_oracle': float(np.mean(gains_oracle)),
        'median_gain_gated': float(np.median(gains_gated)),
        'n_positive_gated': int(sum(1 for g in gains_gated if g > 0)),
        'n_positive_no_gate': int(sum(1 for g in gains_no_gate if g > 0)),
        'n_total': n,
        'oracle_realization_gated': float(np.mean(gains_gated) / (np.mean(gains_oracle) + 1e-8) * 100),
        'oracle_realization_no_gate': float(np.mean(gains_no_gate) / (np.mean(gains_oracle) + 1e-8) * 100),
        'mean_gate': float(np.mean(gate_values)),
        'gains_gated': gains_gated,
        'gains_no_gate': gains_no_gate,
    }


def main():
    print('=' * 80)
    print('门控双因子改进版实验')
    print('=' * 80)

    # 加载或处理样本
    samples_file = DATA / 'gated_dual_factor_samples.json'
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

    # 实验配置网格
    gate_types = ['v1', 'v2', 'v3', 'v4']
    structures = ['multiply', 'add']
    temperatures = [0.02, 0.05, 0.1, 0.2]

    print(f'\n{"=" * 80}')
    print('实验网格搜索')
    print(f'{"=" * 80}')
    print(f'\n{"门控":<6s} {"结构":<10s} {"温度":>6s} {"ρ":>8s} {"增益":>10s} {"正例":>8s} {"Oracle%":>8s} {"中位增益":>10s}')
    print('-' * 80)

    all_results = []
    best_result = None
    best_gain = -float('inf')

    for gate_type in gate_types:
        for structure in structures:
            for temp in temperatures:
                result = train_and_evaluate(all_samples, gate_type=gate_type,
                                             structure=structure, temperature=temp)
                all_results.append(result)

                print(f'{gate_type:<6s} {structure:<10s} {temp:>6.2f} {result["rho_gated"]:>8.4f} '
                      f'{result["mean_gain_gated"]:>+10.4f} {result["n_positive_gated"]:>4d}/{result["n_total"]:<4d} '
                      f'{result["oracle_realization_gated"]:>7.1f}% {result["median_gain_gated"]:>+10.4f}')

                if result['mean_gain_gated'] > best_gain:
                    best_gain = result['mean_gain_gated']
                    best_result = result

    # 最佳配置详细结果
    print(f'\n{"=" * 80}')
    print(f'最佳配置: 门控={best_result["gate_type"]}, 结构={best_result["structure"]}, '
          f'温度={best_result["temperature"]}')
    print(f'{"=" * 80}')
    print(f'  ρ: {best_result["rho_gated"]:.4f} (无门控: {best_result["rho_no_gate"]:.4f})')
    print(f'  平均增益: {best_result["mean_gain_gated"]:+.4f} (无门控: {best_result["mean_gain_no_gate"]:+.4f})')
    print(f'  中位增益: {best_result["median_gain_gated"]:+.4f}')
    print(f'  正例: {best_result["n_positive_gated"]}/{best_result["n_total"]} '
          f'(无门控: {best_result["n_positive_no_gate"]}/{best_result["n_total"]})')
    print(f'  Oracle实现率: {best_result["oracle_realization_gated"]:.1f}% '
          f'(无门控: {best_result["oracle_realization_no_gate"]:.1f}%)')
    print(f'  平均门控值: {best_result["mean_gate"]:.4f}')

    # 分域详细结果
    for domain in ['COCO', 'VOC']:
        domain_idx = [i for i, s in enumerate(all_samples) if s['target_domain'] == domain]
        gains_gated = [best_result['gains_gated'][i] for i in domain_idx]
        gains_no_gate = [best_result['gains_no_gate'][i] for i in domain_idx]

        print(f'\n  {domain} (N={len(domain_idx)}):')
        print(f'    门控+双因子: 平均增益={np.mean(gains_gated):+.4f}, '
              f'中位增益={np.median(gains_gated):+.4f}, '
              f'正例={sum(1 for g in gains_gated if g > 0)}/{len(gains_gated)}')
        print(f'    纯双因子:     平均增益={np.mean(gains_no_gate):+.4f}, '
              f'正例={sum(1 for g in gains_no_gate if g > 0)}/{len(gains_no_gate)}')

    # 保存结果
    with open(DATA / 'gated_dual_factor_v3_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_result': {k: v for k, v in best_result.items() if k not in ['gains_gated', 'gains_no_gate']},
            'all_results': [{k: v for k, v in r.items() if k not in ['gains_gated', 'gains_no_gate']}
                             for r in all_results],
        }, f, indent=2, ensure_ascii=False)

    print(f'\n结果已保存到: {DATA / "gated_dual_factor_v3_results.json"}')


if __name__ == '__main__':
    main()
