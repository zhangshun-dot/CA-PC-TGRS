# -*- coding: utf-8 -*-
"""det_stats自适应(γ, α) LOO验证
对每个类别，用det_stats(det_mean, det_var, n_boxes)线性预测(γ, α)
LOO：留出一个类别，用其余类别训练，预测留出类别
对比：固定(γ,α) / det_stats自适应(γ,α) / Oracle逐类最优(γ,α)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LinearRegression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate,
)
import tge_probe as T

OUT = DATA / 'usc_detstats_gamma_alpha_loo_results.json'
GAMMAS = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 5.0]

# 复用v2的AffineUSC和数据加载
from usc_rank1_affine_gpu_v2 import AffineUSC, load_domain_data, evaluate_with_params


def extract_det_stats(target_data, coco_cls):
    """从目标域检测结果中提取det_stats特征"""
    data = target_data[coco_cls]
    # 从per_img提取所有候选框的检测器分数(gd)
    all_scores = []
    all_boxes = []
    for img in data['per_img']:
        if 'gd' in img and len(img['gd']) > 0:
            scores = img['gd']
            if hasattr(scores, 'tolist'):
                all_scores.extend(scores.tolist())
            else:
                all_scores.extend(list(scores))
        if 'box' in img and len(img['box']) > 0:
            boxes = img['box']
            if hasattr(boxes, 'tolist'):
                all_boxes.extend(boxes.tolist())
            else:
                all_boxes.extend(list(boxes))
    n_boxes = len(all_scores)
    scores_arr = np.array(all_scores) if n_boxes > 0 else np.array([0.0])
    det_mean = float(np.mean(scores_arr))
    det_var = float(np.var(scores_arr))
    return {
        'det_mean': det_mean,
        'det_var': det_var,
        'n_boxes': n_boxes,
        'det_std': float(np.sqrt(det_var)),
        'det_max': float(np.max(scores_arr)),
        'det_median': float(np.median(scores_arr)),
    }


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"\n域对: {source_names}")
    print(f"γ范围: {GAMMAS}")
    print(f"α范围: {ALPHAS}")

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
            print(f"  完成，有效类别: {len(data['valid_targets'])}")

    # 阶段2：对每个类别扫描所有(γ,α)组合，得到Oracle结果和det_stats
    print(f"\n{'='*60}")
    print("阶段2：逐类别(γ,α)网格扫描 + det_stats提取")
    print(f"{'='*60}")

    all_class_data = []  # 每个元素: {domain, class, det_stats, oracle_gamma, oracle_alpha, oracle_ap, sweep_results}

    for src in all_domain_data:
        domain_data = all_domain_data[src]
        valid_targets = domain_data['valid_targets']
        target_data = domain_data['target_data']
        print(f"\n--- {src}→COCO ({len(valid_targets)}类) ---")

        for eval_name, coco_cls, src_cls in valid_targets:
            # 提取det_stats
            det_stats = extract_det_stats(target_data, coco_cls)

            # 扫描所有(γ,α)组合
            best_ap = -1
            best_gamma = 0.0
            best_alpha = 0.0
            sweep = {}
            for gamma in GAMMAS:
                for alpha in ALPHAS:
                    # 单类别评估
                    usc = domain_data['usc']
                    protos_gpu = domain_data['protos_gpu']
                    all_protos = domain_data['all_protos']
                    text_deltas = domain_data['text_deltas']

                    p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, alpha)
                    p_corr = p_corr_gpu.cpu().numpy()
                    neg_raw = build_neg_proto(all_protos, src_cls)
                    if neg_raw is not None:
                        neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
                        neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, gamma, alpha)
                        neg = neg_corr_gpu.cpu().numpy()
                    else:
                        neg = None
                    delta = text_deltas[src_cls]
                    p_hat = p_corr + 1.0 * delta
                    p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
                    ap = evaluate(target_data[coco_cls], p_hat, neg)

                    key = f"{gamma}_{alpha}"
                    sweep[key] = float(ap)
                    if ap > best_ap:
                        best_ap = ap
                        best_gamma = gamma
                        best_alpha = alpha

            class_info = {
                'domain': src,
                'class': coco_cls,
                'src_class': src_cls,
                'det_stats': det_stats,
                'oracle_gamma': best_gamma,
                'oracle_alpha': best_alpha,
                'oracle_ap': float(best_ap),
                'sweep': sweep,
            }
            all_class_data.append(class_info)
            print(f"  {coco_cls:<12}: oracle(γ={best_gamma:>4.1f}, α={best_alpha:>4.1f}) "
                  f"AP={best_ap:.4f}, n_boxes={det_stats['n_boxes']}, "
                  f"det_mean={det_stats['det_mean']:.4f}")

    print(f"\n总类别数: {len(all_class_data)}")

    # 阶段3：det_stats自适应(γ,α) LOO验证
    print(f"\n{'='*60}")
    print("阶段3：det_stats自适应(γ,α) LOO验证")
    print(f"{'='*60}")

    # 特征矩阵
    feature_names = ['det_mean', 'det_var', 'n_boxes']
    X = np.array([[c['det_stats'][f] for f in feature_names] for c in all_class_data])
    # 标准化特征
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    y_gamma = np.array([c['oracle_gamma'] for c in all_class_data])
    y_alpha = np.array([c['oracle_alpha'] for c in all_class_data])

    # LOO预测
    n = len(all_class_data)
    pred_gamma = np.zeros(n)
    pred_alpha = np.zeros(n)

    for i in range(n):
        # 留出第i个类别
        X_train = np.delete(X_norm, i, axis=0)
        y_gamma_train = np.delete(y_gamma, i)
        y_alpha_train = np.delete(y_alpha, i)
        X_test = X_norm[i:i+1]

        # 训练线性回归
        model_gamma = LinearRegression()
        model_gamma.fit(X_train, y_gamma_train)
        pred_gamma[i] = model_gamma.predict(X_test)[0]

        model_alpha = LinearRegression()
        model_alpha.fit(X_train, y_alpha_train)
        pred_alpha[i] = model_alpha.predict(X_test)[0]

    # 将预测值映射到最近的离散(γ,α)组合
    gamma_array = np.array(GAMMAS)
    alpha_array = np.array(ALPHAS)

    pred_gamma_discrete = np.array([gamma_array[np.argmin(np.abs(gamma_array - g))] for g in pred_gamma])
    pred_alpha_discrete = np.array([alpha_array[np.argmin(np.abs(alpha_array - a))] for a in pred_alpha])

    # 评估预测的(γ,α)在每个类别上的AP
    loo_aps = []
    for i, c in enumerate(all_class_data):
        g = pred_gamma_discrete[i]
        a = pred_alpha_discrete[i]
        key = f"{g}_{a}"
        ap = c['sweep'][key]
        loo_aps.append(ap)
        c['pred_gamma'] = float(g)
        c['pred_alpha'] = float(a)
        c['pred_ap'] = float(ap)

    # 固定(γ,α)基线：用全局最佳(γ,α)
    # 先找全局平均AP最高的(γ,α)
    global_best_ap = -1
    global_best_gamma = 0.0
    global_best_alpha = 0.0
    for g in GAMMAS:
        for a in ALPHAS:
            key = f"{g}_{a}"
            avg_ap = np.mean([c['sweep'][key] for c in all_class_data])
            if avg_ap > global_best_ap:
                global_best_ap = avg_ap
                global_best_gamma = g
                global_best_alpha = a

    fixed_aps = [c['sweep'][f"{global_best_gamma}_{global_best_alpha}"] for c in all_class_data]
    oracle_aps = [c['oracle_ap'] for c in all_class_data]

    print(f"\n全局最佳固定(γ,α): ({global_best_gamma}, {global_best_alpha}), 平均AP={global_best_ap:.4f}")
    print(f"det_stats自适应(γ,α) LOO平均AP: {np.mean(loo_aps):.4f}")
    print(f"Oracle逐类最优(γ,α)平均AP: {np.mean(oracle_aps):.4f}")
    print(f"\n固定→自适应增益: {np.mean(loo_aps) - np.mean(fixed_aps):+.4f}")
    print(f"自适应→Oracle差距: {np.mean(oracle_aps) - np.mean(loo_aps):+.4f}")
    print(f"自适应达Oracle比例: {np.mean(loo_aps)/np.mean(oracle_aps)*100:.1f}%")

    # 按域对分组
    print(f"\n{'='*60}")
    print("按域对分组结果")
    print(f"{'='*60}")
    print(f"{'域对':<12} {'固定(γ,α)':>12} {'自适应(γ,α)':>14} {'Oracle':>10} {'自适应增益':>12} {'成功率':>8}")
    print("-" * 75)
    domain_results = {}
    for src in source_names:
        idx = [i for i, c in enumerate(all_class_data) if c['domain'] == src]
        if not idx:
            continue
        fixed_avg = np.mean([fixed_aps[i] for i in idx])
        loo_avg = np.mean([loo_aps[i] for i in idx])
        oracle_avg = np.mean([oracle_aps[i] for i in idx])
        gain = loo_avg - fixed_avg
        # 成功率：自适应AP > 固定AP的类别比例
        success = np.mean([1 if loo_aps[i] > fixed_aps[i] else 0 for i in idx])
        print(f"{src:<12} {fixed_avg:>12.4f} {loo_avg:>14.4f} {oracle_avg:>10.4f} "
              f"{gain:>+12.4f} {success:>7.0%}")
        domain_results[src] = {
            'fixed_ap': float(fixed_avg),
            'loo_ap': float(loo_avg),
            'oracle_ap': float(oracle_avg),
            'gain': float(gain),
            'success_rate': float(success),
            'n_classes': len(idx),
        }

    # 相关性分析
    print(f"\n{'='*60}")
    print("det_stats与最优(γ,α)的相关性")
    print(f"{'='*60}")
    from scipy.stats import spearmanr
    for f in feature_names:
        vals = np.array([c['det_stats'][f] for c in all_class_data])
        rho_g, p_g = spearmanr(vals, y_gamma)
        rho_a, p_a = spearmanr(vals, y_alpha)
        print(f"  {f:<12}: γ ρ={rho_g:+.3f} (p={p_g:.4f}), α ρ={rho_a:+.3f} (p={p_a:.4f})")

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

    results = {
        'summary': {
            'n_classes': n,
            'n_domains': len(source_names),
            'fixed_gamma': global_best_gamma,
            'fixed_alpha': global_best_alpha,
            'fixed_ap': float(np.mean(fixed_aps)),
            'loo_ap': float(np.mean(loo_aps)),
            'oracle_ap': float(np.mean(oracle_aps)),
            'loo_gain_over_fixed': float(np.mean(loo_aps) - np.mean(fixed_aps)),
            'loo_oracle_ratio': float(np.mean(loo_aps) / np.mean(oracle_aps)),
        },
        'domain_results': domain_results,
        'class_results': all_class_data,
        'feature_names': feature_names,
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU显存已释放")


if __name__ == '__main__':
    main()
