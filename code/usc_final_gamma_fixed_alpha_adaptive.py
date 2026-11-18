# -*- coding: utf-8 -*-
"""第二篇最终核心实验：固定γ=-2.0 + det_stats自适应α LOO验证
对比：
1. 原始USC（γ=0, α=固定最佳）
2. 秩1仿射USC固定（γ=-2.0, α=固定最佳）
3. det_stats自适应α（γ=-2.0固定，α由det_stats线性预测）
4. Oracle逐类最优α（γ=-2.0固定）
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
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate,
)
import tge_probe as T
from usc_rank1_affine_gpu_v2 import AffineUSC, load_domain_data

OUT = DATA / 'usc_final_gamma_fixed_alpha_adaptive_results.json'
GAMMA_FIXED = -2.0  # 固定γ=-2.0


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
        'det_max': float(np.max(scores_arr)),
        'det_median': float(np.median(scores_arr)),
    }


def evaluate_single_class(domain_data, src_cls, coco_cls, gamma, alpha):
    """评估单个类别在指定(γ,α)下的AP"""
    usc = domain_data['usc']
    protos_gpu = domain_data['protos_gpu']
    all_protos = domain_data['all_protos']
    text_deltas = domain_data['text_deltas']
    target_data = domain_data['target_data']

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
    return float(ap)


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"\n域对: {source_names}")
    print(f"固定γ: {GAMMA_FIXED}")
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

    # 阶段2：逐类别扫描α（γ固定为-2.0），同时扫描γ=0作为原始USC基线
    print(f"\n{'='*60}")
    print("阶段2：逐类别α扫描（γ=-2.0固定 + γ=0基线）")
    print(f"{'='*60}")

    all_class_data = []

    for src in all_domain_data:
        domain_data = all_domain_data[src]
        valid_targets = domain_data['valid_targets']
        print(f"\n--- {src}→COCO ({len(valid_targets)}类) ---")

        for eval_name, coco_cls, src_cls in valid_targets:
            det_stats = extract_det_stats(domain_data['target_data'], coco_cls)

            # 扫描α（γ=-2.0固定）
            best_ap_affine = -1
            best_alpha_affine = 0.0
            sweep_affine = {}
            for alpha in ALPHAS:
                ap = evaluate_single_class(domain_data, src_cls, coco_cls, GAMMA_FIXED, alpha)
                sweep_affine[alpha] = ap
                if ap > best_ap_affine:
                    best_ap_affine = ap
                    best_alpha_affine = alpha

            # 扫描α（γ=0，原始USC基线）
            best_ap_original = -1
            best_alpha_original = 0.0
            sweep_original = {}
            for alpha in ALPHAS:
                ap = evaluate_single_class(domain_data, src_cls, coco_cls, 0.0, alpha)
                sweep_original[alpha] = ap
                if ap > best_ap_original:
                    best_ap_original = ap
                    best_alpha_original = alpha

            class_info = {
                'domain': src,
                'class': coco_cls,
                'src_class': src_cls,
                'det_stats': det_stats,
                # 秩1仿射USC（γ=-2.0）
                'affine_oracle_alpha': best_alpha_affine,
                'affine_oracle_ap': best_ap_affine,
                'affine_sweep': sweep_affine,
                # 原始USC（γ=0）
                'original_oracle_alpha': best_alpha_original,
                'original_oracle_ap': best_ap_original,
                'original_sweep': sweep_original,
            }
            all_class_data.append(class_info)
            print(f"  {coco_cls:<12}: 仿射(γ=-2) α*={best_alpha_affine:>4.1f} AP={best_ap_affine:.4f} | "
                  f"原始(γ=0) α*={best_alpha_original:>4.1f} AP={best_ap_original:.4f} | "
                  f"n_boxes={det_stats['n_boxes']}, det_mean={det_stats['det_mean']:.4f}")

    n = len(all_class_data)
    print(f"\n总类别数: {n}")

    # 阶段3：det_stats自适应α LOO验证（γ=-2.0固定）
    print(f"\n{'='*60}")
    print("阶段3：det_stats自适应α LOO验证（γ=-2.0固定）")
    print(f"{'='*60}")

    feature_names = ['det_mean', 'det_var', 'n_boxes']
    X = np.array([[c['det_stats'][f] for f in feature_names] for c in all_class_data])
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    y_alpha = np.array([c['affine_oracle_alpha'] for c in all_class_data])

    # LOO预测
    pred_alpha = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X_norm, i, axis=0)
        y_train = np.delete(y_alpha, i)
        X_test = X_norm[i:i+1]
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred_alpha[i] = model.predict(X_test)[0]

    # 映射到最近的离散α
    alpha_array = np.array(ALPHAS)
    pred_alpha_discrete = np.array([alpha_array[np.argmin(np.abs(alpha_array - a))] for a in pred_alpha])

    # 评估预测的α
    loo_aps = []
    for i, c in enumerate(all_class_data):
        a = pred_alpha_discrete[i]
        ap = c['affine_sweep'][a]
        loo_aps.append(ap)
        c['pred_alpha'] = float(a)
        c['pred_ap'] = float(ap)

    # 固定α基线：用全局最佳固定α（γ=-2.0）
    global_best_ap = -1
    global_best_alpha = 0.0
    for a in ALPHAS:
        avg_ap = np.mean([c['affine_sweep'][a] for c in all_class_data])
        if avg_ap > global_best_ap:
            global_best_ap = avg_ap
            global_best_alpha = a

    fixed_aps = [c['affine_sweep'][global_best_alpha] for c in all_class_data]
    oracle_aps = [c['affine_oracle_ap'] for c in all_class_data]

    # 原始USC（γ=0）的固定最佳α和Oracle
    original_global_best_ap = -1
    original_global_best_alpha = 0.0
    for a in ALPHAS:
        avg_ap = np.mean([c['original_sweep'][a] for c in all_class_data])
        if avg_ap > original_global_best_ap:
            original_global_best_ap = avg_ap
            original_global_best_alpha = a
    original_fixed_aps = [c['original_sweep'][original_global_best_alpha] for c in all_class_data]
    original_oracle_aps = [c['original_oracle_ap'] for c in all_class_data]

    print(f"\n=== 总体结果 ===")
    print(f"原始USC (γ=0):  固定α={original_global_best_alpha}, AP={np.mean(original_fixed_aps):.4f} | Oracle AP={np.mean(original_oracle_aps):.4f}")
    print(f"秩1仿射 (γ=-2): 固定α={global_best_alpha}, AP={np.mean(fixed_aps):.4f} | Oracle AP={np.mean(oracle_aps):.4f}")
    print(f"det_stats自适应α (γ=-2): LOO AP={np.mean(loo_aps):.4f}")
    print(f"\n原始USC固定 → 仿射固定: {np.mean(fixed_aps) - np.mean(original_fixed_aps):+.4f}")
    print(f"仿射固定 → det_stats自适应: {np.mean(loo_aps) - np.mean(fixed_aps):+.4f}")
    print(f"det_stats自适应 → Oracle: {np.mean(oracle_aps) - np.mean(loo_aps):+.4f}")
    print(f"det_stats自适应达Oracle比例: {np.mean(loo_aps)/np.mean(oracle_aps)*100:.1f}%")

    # 按域对分组
    print(f"\n{'='*60}")
    print("按域对分组结果")
    print(f"{'='*60}")
    print(f"{'域对':<12} {'原始USC固定':>12} {'仿射固定':>10} {'自适应α':>10} {'Oracle':>10} {'自适应增益':>12} {'成功率':>8}")
    print("-" * 80)
    domain_results = {}
    for src in source_names:
        idx = [i for i, c in enumerate(all_class_data) if c['domain'] == src]
        if not idx:
            continue
        orig_fixed = np.mean([original_fixed_aps[i] for i in idx])
        aff_fixed = np.mean([fixed_aps[i] for i in idx])
        loo_avg = np.mean([loo_aps[i] for i in idx])
        oracle_avg = np.mean([oracle_aps[i] for i in idx])
        gain = loo_avg - aff_fixed
        success = np.mean([1 if loo_aps[i] > fixed_aps[i] else 0 for i in idx])
        print(f"{src:<12} {orig_fixed:>12.4f} {aff_fixed:>10.4f} {loo_avg:>10.4f} {oracle_avg:>10.4f} "
              f"{gain:>+12.4f} {success:>7.0%}")
        domain_results[src] = {
            'original_fixed_ap': float(orig_fixed),
            'affine_fixed_ap': float(aff_fixed),
            'loo_ap': float(loo_avg),
            'oracle_ap': float(oracle_avg),
            'gain_over_fixed': float(gain),
            'success_rate': float(success),
            'n_classes': len(idx),
        }

    # 相关性分析
    print(f"\n{'='*60}")
    print("det_stats与最优α的相关性（γ=-2.0固定）")
    print(f"{'='*60}")
    for f in feature_names:
        vals = np.array([c['det_stats'][f] for c in all_class_data])
        rho, p = spearmanr(vals, y_alpha)
        print(f"  {f:<12}: ρ={rho:+.3f} (p={p:.4f})")

    # 逐类别详细结果
    print(f"\n{'='*60}")
    print("逐类别详细结果")
    print(f"{'='*60}")
    print(f"{'域对':<10} {'类别':<12} {'n_boxes':>8} {'固定αAP':>8} {'预测α':>6} {'自适应AP':>9} {'Oracleα':>7} {'OracleAP':>8}")
    print("-" * 75)
    for c in all_class_data:
        fixed_ap = c['affine_sweep'][global_best_alpha]
        print(f"{c['domain']:<10} {c['class']:<12} {c['det_stats']['n_boxes']:>8} "
              f"{fixed_ap:>8.4f} {c['pred_alpha']:>6.1f} {c['pred_ap']:>9.4f} "
              f"{c['affine_oracle_alpha']:>7.1f} {c['affine_oracle_ap']:>8.4f}")

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
        'config': {
            'gamma_fixed': GAMMA_FIXED,
            'alphas': ALPHAS,
            'feature_names': feature_names,
            'n_classes': n,
            'n_domains': len(source_names),
        },
        'summary': {
            'original_fixed_alpha': original_global_best_alpha,
            'original_fixed_ap': float(np.mean(original_fixed_aps)),
            'original_oracle_ap': float(np.mean(original_oracle_aps)),
            'affine_fixed_alpha': global_best_alpha,
            'affine_fixed_ap': float(np.mean(fixed_aps)),
            'affine_oracle_ap': float(np.mean(oracle_aps)),
            'det_stats_adaptive_loo_ap': float(np.mean(loo_aps)),
            'affine_gain_over_original': float(np.mean(fixed_aps) - np.mean(original_fixed_aps)),
            'adaptive_gain_over_fixed': float(np.mean(loo_aps) - np.mean(fixed_aps)),
            'adaptive_oracle_ratio': float(np.mean(loo_aps) / np.mean(oracle_aps)),
        },
        'domain_results': domain_results,
        'class_results': all_class_data,
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU显存已释放")


if __name__ == '__main__':
    main()
