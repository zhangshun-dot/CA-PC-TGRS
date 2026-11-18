# -*- coding: utf-8 -*-
"""gated_dual_factor.py — 门控+双因子两级结构
α_c = gate(域内sim_pos_mean分位数) × (w1·SV_c + w2·TC_c + b)

门控设计：
- 用 sim_pos_mean 在域内的分位数作为门控
- gate = sigmoid((域内中位数 - sim_pos_mean) / temperature)
- sim_pos_mean低 → 对齐差 → 需要校正 → gate高
- 域内分位数比绝对阈值更稳，跨数据集不用重标定
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
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

# 所有类别列表
ALL_CLASSES = ['airplane', 'boat', 'car', 'bus', 'motorcycle', 'bicycle', 'truck',
               'aeroplane', 'motorbike', 'ship', 'vehicle', 'plane', 'small-vehicle', 'large-vehicle']


def sigmoid(x):
    """sigmoid函数，数值稳定"""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def compute_sv(target_data):
    """计算sv = 候选框尺寸方差（log面积的方差）"""
    per_img = target_data['per_img']
    all_areas = []
    for p in per_img:
        if 'box' in p and len(p['box']) > 0:
            boxes = p['box']  # 字段名是box，不是boxes
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
    """计算tc = 文本混淆度"""
    return float(compute_text_confusability(target_cls, ALL_CLASSES))


def compute_sim_factors(target_data, src_proto):
    """计算sim_pos_mean和sim_pos_std"""
    features = target_data['features']
    sim_pos = features @ src_proto
    return {
        'sim_pos_mean': float(np.mean(sim_pos)),
        'sim_pos_std': float(np.std(sim_pos)),
    }


def compute_gate(sim_pos_mean, domain_median, temperature=0.05):
    """计算门控值
    gate = sigmoid((domain_median - sim_pos_mean) / temperature)
    sim_pos_mean低 → 对齐差 → 需要校正 → gate高
    """
    x = (domain_median - sim_pos_mean) / temperature
    return float(sigmoid(x))


def process_sample(src_name, src_cls, target_cls, target_domain, target_loader_func):
    """处理单个样本"""
    # 加载源域
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']

    # 负原型
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

    # 加载目标域
    target_data = target_loader_func(target_cls, n=200)
    if target_data is None:
        return None

    # 计算所有因子
    sim_factors = compute_sim_factors(target_data, src_proto)
    sv = compute_sv(target_data)
    tc = compute_tc(target_cls)

    # α网格搜索（找oracle）
    ap_curve = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            proto = src_proto
        else:
            proto = build_proto_tge(src_proto, src_cls, target_cls, beta=alpha)
        ap = evaluate_with_features(target_data, proto, neg_proto=proto_neg)
        ap_curve[str(alpha)] = float(ap)

    # 找最优α
    best_alpha = ALPHAS[0]
    best_ap = ap_curve[str(ALPHAS[0])]
    for alpha in ALPHAS:
        if ap_curve[str(alpha)] > best_ap:
            best_ap = ap_curve[str(alpha)]
            best_alpha = alpha

    baseline_ap = ap_curve['0.0']

    result = {
        'src': src_name,
        'src_cls': src_cls,
        'target_cls': target_cls,
        'target_domain': target_domain,
        'sv': sv,
        'tc': tc,
        'sim_pos_mean': sim_factors['sim_pos_mean'],
        'sim_pos_std': sim_factors['sim_pos_std'],
        'best_alpha': float(best_alpha),
        'best_ap': float(best_ap),
        'baseline_ap': float(baseline_ap),
        'oracle_gain': float(best_ap - baseline_ap),
        'ap_curve': ap_curve,
    }

    return result


def train_and_evaluate(samples, temperature=0.05, use_gate=True):
    """训练门控+双因子模型，用LOOCV评估"""
    n = len(samples)

    # 计算每个域的sim_pos_mean中位数
    domain_medians = {}
    for domain in ['COCO', 'VOC']:
        domain_samples = [s for s in samples if s['target_domain'] == domain]
        if domain_samples:
            domain_medians[domain] = np.median([s['sim_pos_mean'] for s in domain_samples])
        else:
            domain_medians[domain] = 0.5

    # LOOCV
    loo = LeaveOneOut()
    y_true = np.array([s['best_alpha'] for s in samples])
    y_pred = np.zeros(n)
    y_pred_no_gate = np.zeros(n)

    for train_idx, test_idx in loo.split(range(n)):
        train_samples = [samples[i] for i in train_idx]
        test_sample = samples[test_idx[0]]

        # 训练双因子模型（只用需要校正的样本，即best_alpha > 0）
        train_need = [s for s in train_samples if s['best_alpha'] > 0]
        if len(train_need) >= 3:
            X_train = np.array([[s['sv'], s['tc']] for s in train_need])
            y_train = np.array([s['best_alpha'] for s in train_need])

            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)

            model = LinearRegression()
            model.fit(X_train_norm, y_train)

            # 预测测试样本
            X_test = np.array([[test_sample['sv'], test_sample['tc']]])
            X_test_norm = scaler.transform(X_test)
            alpha_dual = model.predict(X_test_norm)[0]
            alpha_dual = float(np.clip(alpha_dual, ALPHA_MIN, ALPHA_MAX))
        else:
            alpha_dual = 1.0  # fallback

        y_pred_no_gate[test_idx[0]] = alpha_dual

        # 计算门控
        if use_gate:
            domain = test_sample['target_domain']
            domain_median = domain_medians[domain]
            gate = compute_gate(test_sample['sim_pos_mean'], domain_median, temperature)
            alpha_gated = gate * alpha_dual
        else:
            alpha_gated = alpha_dual

        y_pred[test_idx[0]] = alpha_gated

    # 评估
    rho_gated, p_gated = stats.spearmanr(y_pred, y_true)
    rho_no_gate, p_no_gate = stats.spearmanr(y_pred_no_gate, y_true)

    # 计算实际AP增益（用预测的α跑评估）
    # 这里我们用ap_curve插值来近似
    gains_gated = []
    gains_no_gate = []
    gains_oracle = []

    for i, sample in enumerate(samples):
        ap_curve = sample['ap_curve']
        alphas = sorted([float(a) for a in ap_curve.keys()])

        # 插值函数
        def interp_ap(alpha):
            alpha = np.clip(alpha, ALPHAS[0], ALPHAS[-1])
            return float(np.interp(alpha, alphas, [ap_curve[str(a)] for a in alphas]))

        baseline = ap_curve['0.0']
        gain_gated = interp_ap(y_pred[i]) - baseline
        gain_no_gate = interp_ap(y_pred_no_gate[i]) - baseline
        gain_oracle = sample['oracle_gain']

        gains_gated.append(gain_gated)
        gains_no_gate.append(gain_no_gate)
        gains_oracle.append(gain_oracle)

    results = {
        'temperature': temperature,
        'use_gate': use_gate,
        'rho_gated': float(rho_gated),
        'p_gated': float(p_gated),
        'rho_no_gate': float(rho_no_gate),
        'p_no_gate': float(p_no_gate),
        'mean_gain_gated': float(np.mean(gains_gated)),
        'mean_gain_no_gate': float(np.mean(gains_no_gate)),
        'mean_gain_oracle': float(np.mean(gains_oracle)),
        'n_positive_gated': int(sum(1 for g in gains_gated if g > 0)),
        'n_positive_no_gate': int(sum(1 for g in gains_no_gate if g > 0)),
        'n_total': n,
        'oracle_realization_gated': float(np.mean(gains_gated) / (np.mean(gains_oracle) + 1e-8) * 100),
        'oracle_realization_no_gate': float(np.mean(gains_no_gate) / (np.mean(gains_oracle) + 1e-8) * 100),
    }

    return results, y_pred, y_pred_no_gate


def main():
    print('=' * 80)
    print('门控+双因子两级结构实验')
    print('=' * 80)

    all_samples = []

    # COCO目标域
    from sota_comparison import load_target_coco
    print(f'\n--- COCO目标域 ---')
    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            print(f'  处理 {src_name}/{src_cls} -> COCO/{coco_cls}...')
            result = process_sample(src_name, src_cls, coco_cls, 'COCO', load_target_coco)
            if result:
                all_samples.append(result)
                print(f'    sv={result["sv"]:.4f}, tc={result["tc"]:.4f}, '
                      f'sim_mean={result["sim_pos_mean"]:.4f}, '
                      f'best α={result["best_alpha"]:.1f}, oracle gain={result["oracle_gain"]:+.4f}')

    # VOC目标域
    from voc_sota_comparison import load_target_voc
    COCO2VOC = {
        'airplane': 'aeroplane', 'boat': 'boat', 'car': 'car',
        'bus': 'bus', 'motorcycle': 'motorbike', 'bicycle': 'bicycle',
    }
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
                print(f'    sv={result["sv"]:.4f}, tc={result["tc"]:.4f}, '
                      f'sim_mean={result["sim_pos_mean"]:.4f}, '
                      f'best α={result["best_alpha"]:.1f}, oracle gain={result["oracle_gain"]:+.4f}')

    print(f'\n总样本数: {len(all_samples)}')

    # 保存原始数据
    with open(DATA / 'gated_dual_factor_samples.json', 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    # 温度网格搜索
    print(f'\n{"=" * 80}')
    print('温度网格搜索（门控+双因子 vs 纯双因子）')
    print(f'{"=" * 80}')
    print(f'\n{"温度":>8s} {"门控ρ":>10s} {"无门控ρ":>10s} {"门控增益":>10s} {"无门控增益":>10s} {"门控正例":>10s} {"无门控正例":>10s}')
    print('-' * 80)

    best_temp = None
    best_gain = -float('inf')
    all_results = []

    for temp in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]:
        results, _, _ = train_and_evaluate(all_samples, temperature=temp, use_gate=True)
        results_no_gate, _, _ = train_and_evaluate(all_samples, temperature=temp, use_gate=False)

        print(f'{temp:>8.2f} {results["rho_gated"]:>10.4f} {results_no_gate["rho_no_gate"]:>10.4f} '
              f'{results["mean_gain_gated"]:>+10.4f} {results_no_gate["mean_gain_no_gate"]:>+10.4f} '
              f'{results["n_positive_gated"]:>5d}/{results["n_total"]:<5d} '
              f'{results_no_gate["n_positive_no_gate"]:>5d}/{results_no_gate["n_total"]:<5d}')

        all_results.append(results)

        if results['mean_gain_gated'] > best_gain:
            best_gain = results['mean_gain_gain'] if 'mean_gain_gain' in results else results['mean_gain_gated']
            best_temp = temp

    # 最佳温度的详细结果
    print(f'\n{"=" * 80}')
    print(f'最佳温度: {best_temp}')
    print(f'{"=" * 80}')

    best_results, y_pred_gated, y_pred_no_gate = train_and_evaluate(
        all_samples, temperature=best_temp, use_gate=True)

    # 分域详细结果
    for domain in ['COCO', 'VOC']:
        domain_idx = [i for i, s in enumerate(all_samples) if s['target_domain'] == domain]
        domain_samples = [all_samples[i] for i in domain_idx]

        gains_gated = []
        gains_no_gate = []
        gains_oracle = []

        for i, sample in zip(domain_idx, domain_samples):
            ap_curve = sample['ap_curve']
            alphas = sorted([float(a) for a in ap_curve.keys()])

            def interp_ap(alpha):
                alpha = np.clip(alpha, ALPHAS[0], ALPHAS[-1])
                return float(np.interp(alpha, alphas, [ap_curve[str(a)] for a in alphas]))

            baseline = ap_curve['0.0']
            gains_gated.append(interp_ap(y_pred_gated[i]) - baseline)
            gains_no_gate.append(interp_ap(y_pred_no_gate[i]) - baseline)
            gains_oracle.append(sample['oracle_gain'])

        print(f'\n  {domain} (N={len(domain_idx)}):')
        print(f'    门控+双因子: 平均增益={np.mean(gains_gated):+.4f}, '
              f'正例={sum(1 for g in gains_gated if g > 0)}/{len(gains_gated)}, '
              f'Oracle实现率={np.mean(gains_gated)/(np.mean(gains_oracle)+1e-8)*100:.1f}%')
        print(f'    纯双因子:     平均增益={np.mean(gains_no_gate):+.4f}, '
              f'正例={sum(1 for g in gains_no_gate if g > 0)}/{len(gains_no_gate)}, '
              f'Oracle实现率={np.mean(gains_no_gate)/(np.mean(gains_oracle)+1e-8)*100:.1f}%')
        print(f'    Oracle:       平均增益={np.mean(gains_oracle):+.4f}')

    # 保存结果
    with open(DATA / 'gated_dual_factor_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_temperature': best_temp,
            'best_results': best_results,
            'all_temperatures': all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f'\n结果已保存到: {DATA / "gated_dual_factor_results.json"}')


if __name__ == '__main__':
    main()
