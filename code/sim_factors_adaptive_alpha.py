# -*- coding: utf-8 -*-
"""sim_factors_adaptive_alpha.py — 用sim_pos_mean + sim_pos_std作为预测因子，重新跑自适应α实验
1. 训练sim_pos_mean + sim_pos_std线性回归模型
2. 在COCO目标域上跑自适应α实验
3. 在VOC目标域上跑自适应α实验
4. 和原来的双因子（SV+TC）对比
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

ALPHA_MAX = 2.0
ALPHA_MIN = 0.0


def compute_sim_factors(target_data, src_proto):
    """计算sim_pos_mean和sim_pos_std"""
    features = target_data['features']
    sim_pos = features @ src_proto
    return {
        'sim_pos_mean': float(np.mean(sim_pos)),
        'sim_pos_std': float(np.std(sim_pos)),
    }


def train_sim_model(samples):
    """用sim_pos_mean + sim_pos_std训练线性回归模型"""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    X = np.array([[s['sim_pos_mean'], s['sim_pos_std']] for s in samples])
    y = np.array([s['best_alpha'] for s in samples])

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_norm, y)

    # 训练集表现
    y_pred = model.predict(X_norm)
    y_pred_clipped = np.clip(y_pred, ALPHA_MIN, ALPHA_MAX)
    rho, p = stats.spearmanr(y_pred_clipped, y)

    print(f'  训练集: Spearman ρ={rho:.4f} (p={p:.4f})')
    print(f'  系数: sim_pos_mean={model.coef_[0]:.4f}, sim_pos_std={model.coef_[1]:.4f}')
    print(f'  截距: {model.intercept_:.4f}')

    return model, scaler


def predict_alpha(model, scaler, sim_factors):
    """用模型预测α"""
    X = np.array([[sim_factors['sim_pos_mean'], sim_factors['sim_pos_std']]])
    X_norm = scaler.transform(X)
    alpha = model.predict(X_norm)[0]
    return float(np.clip(alpha, ALPHA_MIN, ALPHA_MAX))


def run_coco_experiment(model, scaler):
    """在COCO目标域上跑自适应α实验"""
    from sota_comparison import load_target_coco

    print(f'\n--- COCO目标域实验 ---')

    results = []
    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            print(f'  处理 {src_name}/{src_cls} -> COCO/{coco_cls}...')

            # 加载源域
            src_loader = SOURCE_LOADERS[src_name][0]
            src_data = src_loader(src_cls, n=150)
            if src_data is None:
                continue
            src_proto = src_data['prototype']
            src_features = src_data['features']

            # 负原型
            sims = src_features @ src_proto
            neg_idx = np.argmin(sims)
            proto_neg = src_features[neg_idx]
            proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

            # 加载目标域
            target_data = load_target_coco(coco_cls, n=200)
            if target_data is None:
                continue

            # 计算sim因子
            sim_factors = compute_sim_factors(target_data, src_proto)

            # 预测α
            alpha_pred = predict_alpha(model, scaler, sim_factors)

            # 评估
            baseline_ap = evaluate_with_features(target_data, src_proto, neg_proto=proto_neg)
            proto_adaptive = build_proto_tge(src_proto, src_cls, coco_cls, beta=alpha_pred)
            adaptive_ap = evaluate_with_features(target_data, proto_adaptive, neg_proto=proto_neg)

            # Oracle最优α
            best_alpha = 0
            best_ap = baseline_ap
            for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                proto_a = build_proto_tge(src_proto, src_cls, coco_cls, beta=alpha)
                ap_a = evaluate_with_features(target_data, proto_a, neg_proto=proto_neg)
                if ap_a > best_ap:
                    best_ap = ap_a
                    best_alpha = alpha

            result = {
                'src': src_name,
                'src_cls': src_cls,
                'target_cls': coco_cls,
                'target_domain': 'COCO',
                'sim_pos_mean': sim_factors['sim_pos_mean'],
                'sim_pos_std': sim_factors['sim_pos_std'],
                'alpha_pred': alpha_pred,
                'alpha_oracle': best_alpha,
                'baseline_ap': float(baseline_ap),
                'adaptive_ap': float(adaptive_ap),
                'oracle_ap': float(best_ap),
                'adaptive_gain': float(adaptive_ap - baseline_ap),
                'oracle_gain': float(best_ap - baseline_ap),
            }
            results.append(result)
            print(f'    α_pred={alpha_pred:.3f}, α_oracle={best_alpha:.1f}, '
                  f'baseline={baseline_ap:.4f}, adaptive={adaptive_ap:.4f} (+{adaptive_ap-baseline_ap:.4f}), '
                  f'oracle={best_ap:.4f} (+{best_ap-baseline_ap:.4f})')

    return results


def run_voc_experiment(model, scaler):
    """在VOC目标域上跑自适应α实验"""
    from voc_sota_comparison import load_target_voc

    print(f'\n--- VOC目标域实验 ---')

    COCO2VOC = {
        'airplane': 'aeroplane', 'boat': 'boat', 'car': 'car',
        'bus': 'bus', 'motorcycle': 'motorbike', 'bicycle': 'bicycle',
    }

    results = []
    for src_name, class_list in REMOTE_TARGETS.items():
        for coco_cls, src_cls, src_display in class_list:
            if coco_cls not in COCO2VOC:
                continue
            voc_cls = COCO2VOC[coco_cls]
            print(f'  处理 {src_name}/{src_cls} -> VOC/{voc_cls}...')

            # 加载源域
            src_loader = SOURCE_LOADERS[src_name][0]
            src_data = src_loader(src_cls, n=150)
            if src_data is None:
                continue
            src_proto = src_data['prototype']
            src_features = src_data['features']

            # 负原型
            sims = src_features @ src_proto
            neg_idx = np.argmin(sims)
            proto_neg = src_features[neg_idx]
            proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

            # 加载目标域
            target_data = load_target_voc(voc_cls, n=200)
            if target_data is None:
                continue

            # 计算sim因子
            sim_factors = compute_sim_factors(target_data, src_proto)

            # 预测α
            alpha_pred = predict_alpha(model, scaler, sim_factors)

            # 评估
            baseline_ap = evaluate_with_features(target_data, src_proto, neg_proto=proto_neg)
            proto_adaptive = build_proto_tge(src_proto, src_cls, voc_cls, beta=alpha_pred)
            adaptive_ap = evaluate_with_features(target_data, proto_adaptive, neg_proto=proto_neg)

            # Oracle最优α
            best_alpha = 0
            best_ap = baseline_ap
            for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                proto_a = build_proto_tge(src_proto, src_cls, voc_cls, beta=alpha)
                ap_a = evaluate_with_features(target_data, proto_a, neg_proto=proto_neg)
                if ap_a > best_ap:
                    best_ap = ap_a
                    best_alpha = alpha

            result = {
                'src': src_name,
                'src_cls': src_cls,
                'target_cls': voc_cls,
                'target_domain': 'VOC',
                'sim_pos_mean': sim_factors['sim_pos_mean'],
                'sim_pos_std': sim_factors['sim_pos_std'],
                'alpha_pred': alpha_pred,
                'alpha_oracle': best_alpha,
                'baseline_ap': float(baseline_ap),
                'adaptive_ap': float(adaptive_ap),
                'oracle_ap': float(best_ap),
                'adaptive_gain': float(adaptive_ap - baseline_ap),
                'oracle_gain': float(best_ap - baseline_ap),
            }
            results.append(result)
            print(f'    α_pred={alpha_pred:.3f}, α_oracle={best_alpha:.1f}, '
                  f'baseline={baseline_ap:.4f}, adaptive={adaptive_ap:.4f} ({adaptive_ap-baseline_ap:+.4f}), '
                  f'oracle={best_ap:.4f} ({best_ap-baseline_ap:+.4f})')

    return results


def main():
    print('=' * 80)
    print('用sim_pos_mean + sim_pos_std重新跑自适应α实验')
    print('=' * 80)

    # 加载κ_c分析的样本数据（包含sim因子和最优α）
    with open(DATA / 'kappa_analysis_samples.json', 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    print(f'\n总样本数: {len(all_samples)}')

    # 训练模型（用所有样本）
    print(f'\n--- 训练sim_pos_mean + sim_pos_std模型 ---')
    model, scaler = train_sim_model(all_samples)

    # 保存模型参数
    model_params = {
        'coef': {'sim_pos_mean': float(model.coef_[0]), 'sim_pos_std': float(model.coef_[1])},
        'intercept': float(model.intercept_),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
    }
    with open(DATA / 'sim_factors_model_params.json', 'w', encoding='utf-8') as f:
        json.dump(model_params, f, indent=2, ensure_ascii=False)

    # COCO目标域实验
    coco_results = run_coco_experiment(model, scaler)

    # VOC目标域实验
    voc_results = run_voc_experiment(model, scaler)

    # 汇总结果
    all_results = coco_results + voc_results

    print(f'\n{"=" * 80}')
    print('汇总结果')
    print(f'{"=" * 80}')

    for domain, results in [('COCO', coco_results), ('VOC', voc_results), ('全部', all_results)]:
        if not results:
            continue
        n = len(results)
        mean_baseline = np.mean([r['baseline_ap'] for r in results])
        mean_adaptive = np.mean([r['adaptive_ap'] for r in results])
        mean_oracle = np.mean([r['oracle_ap'] for r in results])
        mean_adaptive_gain = np.mean([r['adaptive_gain'] for r in results])
        mean_oracle_gain = np.mean([r['oracle_gain'] for r in results])
        n_positive = sum(1 for r in results if r['adaptive_gain'] > 0)
        oracle_realization = mean_adaptive_gain / (mean_oracle_gain + 1e-8) * 100

        # α预测和oracle的相关性
        alpha_preds = [r['alpha_pred'] for r in results]
        alpha_oracles = [r['alpha_oracle'] for r in results]
        rho_alpha, p_alpha = stats.spearmanr(alpha_preds, alpha_oracles)

        print(f'\n  {domain} (N={n}):')
        print(f'    Baseline AP: {mean_baseline:.4f}')
        print(f'    Adaptive AP: {mean_adaptive:.4f} ({mean_adaptive_gain:+.4f})')
        print(f'    Oracle AP:   {mean_oracle:.4f} ({mean_oracle_gain:+.4f})')
        print(f'    正增益类别: {n_positive}/{n} ({n_positive/n*100:.1f}%)')
        print(f'    Oracle实现率: {oracle_realization:.1f}%')
        print(f'    α预测 vs Oracle Spearman ρ: {rho_alpha:.4f} (p={p_alpha:.4f})')

    # 保存结果
    with open(DATA / 'sim_factors_adaptive_alpha_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'model_params': model_params,
            'coco_results': coco_results,
            'voc_results': voc_results,
        }, f, indent=2, ensure_ascii=False)

    print(f'\n结果已保存到: {DATA / "sim_factors_adaptive_alpha_results.json"}')


if __name__ == '__main__':
    main()
