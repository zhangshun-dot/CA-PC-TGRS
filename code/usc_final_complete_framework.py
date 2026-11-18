# -*- coding: utf-8 -*-
"""第二篇最终完整实验：传感器感知γ选择 + det_stats自适应α + 软门控 + 传感器感知模板 + 完整消融
对比配置：
1. Baseline (原始检测器)
2. 原始USC (γ=0, α=固定)
3. 秩1仿射USC固定 (全局γ=-2.0, α=固定)
4. 传感器感知γ选择 (卫星γ=-2.0, 无人机γ=-1.0, α=固定)
5. + det_stats自适应α
6. + 软门控
7. + 传感器感知模板
8. 完整框架 (所有模块)
9. Oracle逐类最优(γ,α)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate, evaluate_baseline,
)
import tge_probe as T
from usc_rank1_affine_gpu_v2 import AffineUSC, load_domain_data

OUT = DATA / 'usc_final_complete_framework_results.json'

# 传感器感知配置
SENSOR_CONFIG = {
    'DIOR': {'gamma': -2.0, 'template': 'an overhead image of {}'},
    'NWPU': {'gamma': -2.0, 'template': 'an overhead image of {}'},
    'DOTA': {'gamma': -2.0, 'template': 'an overhead image of {}'},
    'VisDrone': {'gamma': -1.0, 'template': 'a drone view of {}'},
}

# 软门控配置
T_OFFSET = 0.2644
TAU_M = 0.08
TAU_D = 0.08


def extract_det_stats(target_data, coco_cls):
    """从目标域检测结果中提取det_stats特征"""
    data = target_data[coco_cls]
    all_scores = []
    for img in data['per_img']:
        if 'gd' in img and len(img['gd']) > 0:
            scores = img['gd']
            if hasattr(scores, 'tolist'):
                all_scores.extend(scores.tolist())
            else:
                all_scores.extend(list(scores))
    n_boxes = len(all_scores)
    scores_arr = np.array(all_scores) if n_boxes > 0 else np.array([0.0])
    return {
        'det_mean': float(np.mean(scores_arr)),
        'det_var': float(np.var(scores_arr)),
        'n_boxes': n_boxes,
        'det_std': float(np.std(scores_arr)),
    }


def sigmoid(x):
    """数值稳定的sigmoid"""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def compute_soft_gate(delta_norm, g2, t_offset=T_OFFSET, tau_m=TAU_M, tau_d=TAU_D):
    """计算软门控值 G_soft = s_m * s_d"""
    s_m = sigmoid((delta_norm - t_offset) / tau_m)
    s_d = sigmoid(g2 / tau_d)
    return float(s_m * s_d)


def compute_g2(domain_data, src_cls, coco_cls, template):
    """计算g2方向门控值：文本外推方向与域偏移的一致性"""
    usc = domain_data['usc']
    protos_gpu = domain_data['protos_gpu']
    # 文本外推方向
    s_emb = T.text_avg([template], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    text_dir = d_emb - s_emb
    text_dir = text_dir / (np.linalg.norm(text_dir) + 1e-8)
    # 域偏移方向（在原始特征空间）
    domain_shift = usc.domain_shift_ @ usc.components_  # 从低维映射回部分空间
    domain_shift_full = np.zeros(512)
    domain_shift_full[:len(domain_shift)] = domain_shift
    domain_shift_full = domain_shift_full / (np.linalg.norm(domain_shift_full) + 1e-8)
    # 余弦相似度
    g2 = float(np.dot(text_dir, domain_shift_full))
    return g2


def evaluate_config(domain_data, src_cls, coco_cls, gamma, alpha, template,
                    use_soft_gate=False, delta_norm=0.0, g2=0.0):
    """评估单个类别在指定配置下的AP"""
    usc = domain_data['usc']
    protos_gpu = domain_data['protos_gpu']
    all_protos = domain_data['all_protos']
    target_data = domain_data['target_data']

    # 软门控：有效α = G_soft * α
    effective_alpha = alpha
    if use_soft_gate:
        g_soft = compute_soft_gate(delta_norm, g2)
        effective_alpha = g_soft * alpha

    # 校正正原型
    p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, effective_alpha)
    p_corr = p_corr_gpu.cpu().numpy()

    # 负原型校正
    neg_raw = build_neg_proto(all_protos, src_cls)
    if neg_raw is not None:
        neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
        neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, gamma, effective_alpha)
        neg = neg_corr_gpu.cpu().numpy()
    else:
        neg = None

    # 文本外推（用指定模板）
    s_emb = T.text_avg([template], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = p_corr + 1.0 * delta
    p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)

    ap = evaluate(target_data[coco_cls], p_hat, neg)
    return float(ap)


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"\n域对: {source_names}")
    print(f"传感器配置: {SENSOR_CONFIG}")

    # 阶段1：加载所有域对数据
    print(f"\n{'='*60}")
    print("阶段1：加载所有域对数据")
    print(f"{'='*60}")
    all_domain_data = {}
    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data(src)
        if data is not None:
            all_domain_data[src] = data
            print(f"  完成，有效类别: {len(data['valid_targets'])}, delta_norm={data['usc'].domain_shift_norm_:.4f}")

    # 阶段2：提取det_stats，训练自适应α模型
    print(f"\n{'='*60}")
    print("阶段2：提取det_stats + 训练自适应α模型（全局γ=-2.0）")
    print(f"{'='*60}")

    all_class_data = []
    feature_names = ['det_mean', 'det_var', 'n_boxes']

    for src in all_domain_data:
        domain_data = all_domain_data[src]
        valid_targets = domain_data['valid_targets']
        delta_norm = domain_data['usc'].domain_shift_norm_

        for eval_name, coco_cls, src_cls in valid_targets:
            det_stats = extract_det_stats(domain_data['target_data'], coco_cls)
            # 计算g2（用默认overhead模板）
            g2_default = compute_g2(domain_data, src_cls, coco_cls, SRC_TMPL_OVERHEAD)
            # 计算g2（用传感器感知模板）
            sensor_template = SENSOR_CONFIG[src]['template']
            g2_sensor = compute_g2(domain_data, src_cls, coco_cls, sensor_template)

            # 扫描α（γ=-2.0固定，用默认模板）找Oracle α
            best_ap = -1
            best_alpha = 0.0
            sweep = {}
            for alpha in ALPHAS:
                ap = evaluate_config(domain_data, src_cls, coco_cls, -2.0, alpha, SRC_TMPL_OVERHEAD)
                sweep[alpha] = ap
                if ap > best_ap:
                    best_ap = ap
                    best_alpha = alpha

            class_info = {
                'domain': src,
                'class': coco_cls,
                'src_class': src_cls,
                'det_stats': det_stats,
                'delta_norm': delta_norm,
                'g2_default': g2_default,
                'g2_sensor': g2_sensor,
                'oracle_alpha_gamma2': best_alpha,
                'oracle_ap_gamma2': best_ap,
                'sweep_gamma2': sweep,
            }
            all_class_data.append(class_info)

    n = len(all_class_data)
    print(f"总类别数: {n}")

    # 训练det_stats自适应α模型（LOO）
    X = np.array([[c['det_stats'][f] for f in feature_names] for c in all_class_data])
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    y_alpha = np.array([c['oracle_alpha_gamma2'] for c in all_class_data])

    pred_alpha = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X_norm, i, axis=0)
        y_train = np.delete(y_alpha, i)
        X_test = X_norm[i:i+1]
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred_alpha[i] = model.predict(X_test)[0]

    alpha_array = np.array(ALPHAS)
    pred_alpha_discrete = np.array([alpha_array[np.argmin(np.abs(alpha_array - a))] for a in pred_alpha])

    for i, c in enumerate(all_class_data):
        c['pred_alpha'] = float(pred_alpha_discrete[i])

    # 相关性
    print(f"\ndet_stats与最优α相关性:")
    for f in feature_names:
        vals = np.array([c['det_stats'][f] for c in all_class_data])
        rho, p = spearmanr(vals, y_alpha)
        print(f"  {f:<12}: ρ={rho:+.3f} (p={p:.4f})")

    # 阶段3：完整框架评估 + 消融
    print(f"\n{'='*60}")
    print("阶段3：完整框架评估 + 消融实验")
    print(f"{'='*60}")

    # 配置列表
    configs = [
        ('C1_原始USC', {'gamma': 0.0, 'alpha': 0.5, 'template': SRC_TMPL_OVERHEAD,
                         'sensor_gamma': False, 'adaptive_alpha': False,
                         'soft_gate': False, 'sensor_template': False}),
        ('C2_仿射固定全局', {'gamma': -2.0, 'alpha': 0.0, 'template': SRC_TMPL_OVERHEAD,
                             'sensor_gamma': False, 'adaptive_alpha': False,
                             'soft_gate': False, 'sensor_template': False}),
        ('C3_传感器感知γ', {'gamma': None, 'alpha': 0.0, 'template': SRC_TMPL_OVERHEAD,
                            'sensor_gamma': True, 'adaptive_alpha': False,
                            'soft_gate': False, 'sensor_template': False}),
        ('C4_+自适应α', {'gamma': None, 'alpha': None, 'template': SRC_TMPL_OVERHEAD,
                         'sensor_gamma': True, 'adaptive_alpha': True,
                         'soft_gate': False, 'sensor_template': False}),
        ('C5_+软门控', {'gamma': None, 'alpha': None, 'template': SRC_TMPL_OVERHEAD,
                        'sensor_gamma': True, 'adaptive_alpha': True,
                        'soft_gate': True, 'sensor_template': False}),
        ('C6_+传感器模板', {'gamma': None, 'alpha': None, 'template': None,
                            'sensor_gamma': True, 'adaptive_alpha': True,
                            'soft_gate': True, 'sensor_template': True}),
        ('C7_完整框架', {'gamma': None, 'alpha': None, 'template': None,
                         'sensor_gamma': True, 'adaptive_alpha': True,
                         'soft_gate': True, 'sensor_template': True}),
    ]

    results_by_config = {}
    for config_name, config in configs:
        print(f"\n--- {config_name} ---")
        domain_aps = {}
        all_aps = []
        for src in all_domain_data:
            domain_data = all_domain_data[src]
            valid_targets = domain_data['valid_targets']
            delta_norm = domain_data['usc'].domain_shift_norm_
            class_aps = []
            for eval_name, coco_cls, src_cls in valid_targets:
                # 确定γ
                if config['sensor_gamma']:
                    gamma = SENSOR_CONFIG[src]['gamma']
                else:
                    gamma = config['gamma']

                # 确定α
                if config['adaptive_alpha']:
                    # 找这个类别的pred_alpha
                    class_info = next(c for c in all_class_data if c['domain'] == src and c['class'] == coco_cls)
                    alpha = class_info['pred_alpha']
                else:
                    alpha = config['alpha']

                # 确定模板
                if config['sensor_template']:
                    template = SENSOR_CONFIG[src]['template']
                else:
                    template = config['template']

                # 确定g2
                class_info = next(c for c in all_class_data if c['domain'] == src and c['class'] == coco_cls)
                g2 = class_info['g2_sensor'] if config['sensor_template'] else class_info['g2_default']

                # 评估
                ap = evaluate_config(domain_data, src_cls, coco_cls, gamma, alpha, template,
                                     use_soft_gate=config['soft_gate'], delta_norm=delta_norm, g2=g2)
                class_aps.append(ap)
                all_aps.append(ap)

            domain_aps[src] = float(np.mean(class_aps))
            print(f"  {src:<12}: AP={domain_aps[src]:.4f} ({len(class_aps)}类)")

        results_by_config[config_name] = {
            'domain_aps': domain_aps,
            'overall_ap': float(np.mean(all_aps)),
            'config': config,
        }
        print(f"  总体平均: {results_by_config[config_name]['overall_ap']:.4f}")

    # Oracle
    print(f"\n--- Oracle逐类最优(γ=-2.0, α) ---")
    oracle_aps = [c['oracle_ap_gamma2'] for c in all_class_data]
    print(f"  总体平均: {np.mean(oracle_aps):.4f}")

    # 阶段4：汇总对比
    print(f"\n{'='*60}")
    print("阶段4：完整框架 vs 消融 汇总")
    print(f"{'='*60}")
    print(f"{'配置':<20} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'总体':>8} {'vs C1':>8}")
    print("-" * 75)
    c1_ap = results_by_config['C1_原始USC']['overall_ap']
    for config_name, res in results_by_config.items():
        d = res['domain_aps']
        gain = res['overall_ap'] - c1_ap
        print(f"{config_name:<20} {d.get('DIOR',0):>8.4f} {d.get('NWPU',0):>8.4f} "
              f"{d.get('DOTA',0):>8.4f} {d.get('VisDrone',0):>10.4f} "
              f"{res['overall_ap']:>8.4f} {gain:>+8.4f}")
    print(f"{'Oracle(γ=-2)':<20} {'':>8} {'':>8} {'':>8} {'':>10} {np.mean(oracle_aps):>8.4f}")

    # 保存结果
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    final_results = {
        'configs': {k: convert(v) for k, v in results_by_config.items()},
        'oracle_ap': float(np.mean(oracle_aps)),
        'n_classes': n,
        'n_domains': len(source_names),
        'sensor_config': SENSOR_CONFIG,
        'det_stats_correlation': {
            f: {'rho': float(spearmanr([c['det_stats'][f] for c in all_class_data], y_alpha)[0]),
                'p': float(spearmanr([c['det_stats'][f] for c in all_class_data], y_alpha)[1])}
            for f in feature_names
        },
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU显存已释放")


if __name__ == '__main__':
    main()
