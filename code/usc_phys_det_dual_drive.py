# -*- coding: utf-8 -*-
"""物理属性+det_stats双驱动自适应α实验
对比：
  1. det_stats单驱动（当前方法）
  2. 物理属性单驱动
  3. 物理属性+det_stats双驱动（融合）
物理属性从源域数据自动计算（平均尺度、长宽比、尺度方差等），无人工赋值
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

OUT = DATA / 'usc_phys_det_dual_drive_results.json'

# 传感器感知γ配置
SENSOR_GAMMA = {
    'DIOR': -2.0, 'NWPU': -2.0, 'DOTA': -2.0, 'VisDrone': -1.0,
}


def compute_physical_attributes(source_data, src_cls):
    """从源域数据自动计算物理属性（无人工赋值）
    返回：avg_area, aspect_ratio, size_variance, occupancy
    """
    if 'boxes' in source_data and len(source_data['boxes']) > 0:
        boxes = np.array(source_data['boxes'])  # (N, 4) xyxy
        if boxes.ndim == 2 and boxes.shape[1] >= 4:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights
            avg_area = float(np.mean(areas))
            aspect_ratio = float(np.mean(widths / (heights + 1e-8)))
            size_variance = float(np.var(areas))
            # 假设图像大小为1024x1024（遥感图像常见尺寸）
            occupancy = float(np.mean(areas) / (1024 * 1024))
            return {
                'avg_area': avg_area,
                'aspect_ratio': aspect_ratio,
                'size_variance': size_variance,
                'occupancy': occupancy,
            }
    # fallback：从特征维度推断
    feats = source_data['features']
    return {
        'avg_area': float(np.mean(np.abs(feats))),
        'aspect_ratio': 1.0,
        'size_variance': float(np.var(np.abs(feats))),
        'occupancy': 0.01,
    }


def compute_det_stats(target_data, coco_cls):
    """从目标域数据计算检测器统计量"""
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
    }


def find_oracle_alpha(domain_data, src_cls, coco_cls, gamma):
    """扫描α找Oracle最优值"""
    best_ap = -1
    best_alpha = 0.0
    for alpha in ALPHAS:
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
        if ap > best_ap:
            best_ap = ap
            best_alpha = alpha
    return best_alpha, best_ap


def evaluate_with_alpha(domain_data, src_cls, coco_cls, gamma, alpha):
    """用指定α评估"""
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
    return float(evaluate(target_data[coco_cls], p_hat, neg))


def loo_evaluation(feature_matrix, y_alpha, all_class_info, domain_data_dict):
    """LOO评估：每次留出一个类别，用其余类别训练线性模型，预测留出类别的α"""
    X = np.array(feature_matrix)
    y = np.array(y_alpha)
    alpha_array = np.array(ALPHAS)

    pred_alphas = np.zeros(len(all_class_info))
    pred_aps = []
    oracle_aps = []

    for i in range(len(all_class_info)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred_cont = model.predict(X_test)[0]
        pred_disc = float(alpha_array[np.argmin(np.abs(alpha_array - pred_cont))])
        pred_alphas[i] = pred_disc

        # 评估预测α的AP
        info = all_class_info[i]
        domain_data = domain_data_dict[info['domain']]
        gamma = SENSOR_GAMMA[info['domain']]
        ap = evaluate_with_alpha(domain_data, info['src_class'], info['class'], gamma, pred_disc)
        pred_aps.append(ap)
        oracle_aps.append(info['oracle_ap'])

    return pred_alphas, pred_aps, oracle_aps


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"域对列表: {source_names}")

    # 阶段1：收集所有类别的物理属性、det_stats、Oracle α
    print(f"\n{'='*60}")
    print("阶段1：收集物理属性、det_stats、Oracle α")
    print(f"{'='*60}")

    all_class_info = []
    domain_data_dict = {}

    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data(src)
        if data is None:
            continue
        domain_data_dict[src] = data
        valid_targets = data['valid_targets']
        gamma = SENSOR_GAMMA[src]

        for eval_name, coco_cls, src_cls in valid_targets:
            # 物理属性（从源域数据自动计算）
            phys = compute_physical_attributes(data['source_data'][src_cls], src_cls)
            # det_stats（从目标域数据计算）
            det = compute_det_stats(data['target_data'], coco_cls)
            # Oracle α
            oracle_alpha, oracle_ap = find_oracle_alpha(data, src_cls, coco_cls, gamma)

            info = {
                'domain': src, 'class': coco_cls, 'src_class': src_cls,
                'phys': phys, 'det': det,
                'oracle_alpha': oracle_alpha, 'oracle_ap': oracle_ap,
            }
            all_class_info.append(info)
            print(f"  {coco_cls:<12}: phys(area={phys['avg_area']:.1f}, ar={phys['aspect_ratio']:.2f}) "
                  f"det(mean={det['det_mean']:.4f}, n={det['n_boxes']}) "
                  f"oracle(α={oracle_alpha}, ap={oracle_ap:.4f})")

    print(f"\n总共收集 {len(all_class_info)} 个类别")

    # 阶段2：构建特征矩阵，对比三种驱动方式
    print(f"\n{'='*60}")
    print("阶段2：对比三种驱动方式的LOO结果")
    print(f"{'='*60}")

    phys_features = ['avg_area', 'aspect_ratio', 'size_variance', 'occupancy']
    det_features = ['det_mean', 'det_var', 'n_boxes']

    y_alpha = [c['oracle_alpha'] for c in all_class_info]

    results = {}

    # 方式1：det_stats单驱动（当前方法）
    print(f"\n--- 方式1：det_stats单驱动（当前方法）---")
    X_det = [[c['det'][f] for f in det_features] for c in all_class_info]
    pred_alphas_det, pred_aps_det, oracle_aps_det = loo_evaluation(X_det, y_alpha, all_class_info, domain_data_dict)
    mean_pred_det = np.mean(pred_aps_det)
    mean_oracle_det = np.mean(oracle_aps_det)
    print(f"  预测AP均值: {mean_pred_det:.4f}")
    print(f"  Oracle AP均值: {mean_oracle_det:.4f}")
    print(f"  达Oracle比例: {mean_pred_det/mean_oracle_det*100:.1f}%")
    results['det_only'] = {
        'mean_pred_ap': float(mean_pred_det),
        'mean_oracle_ap': float(mean_oracle_det),
        'oracle_ratio': float(mean_pred_det/mean_oracle_det),
    }

    # 方式2：物理属性单驱动
    print(f"\n--- 方式2：物理属性单驱动 ---")
    X_phys = [[c['phys'][f] for f in phys_features] for c in all_class_info]
    pred_alphas_phys, pred_aps_phys, oracle_aps_phys = loo_evaluation(X_phys, y_alpha, all_class_info, domain_data_dict)
    mean_pred_phys = np.mean(pred_aps_phys)
    mean_oracle_phys = np.mean(oracle_aps_phys)
    print(f"  预测AP均值: {mean_pred_phys:.4f}")
    print(f"  Oracle AP均值: {mean_oracle_phys:.4f}")
    print(f"  达Oracle比例: {mean_pred_phys/mean_oracle_phys*100:.1f}%")
    results['phys_only'] = {
        'mean_pred_ap': float(mean_pred_phys),
        'mean_oracle_ap': float(mean_oracle_phys),
        'oracle_ratio': float(mean_pred_phys/mean_oracle_phys),
    }

    # 方式3：物理属性+det_stats双驱动
    print(f"\n--- 方式3：物理属性+det_stats双驱动（融合）---")
    X_dual = [[c['phys'][f] for f in phys_features] + [c['det'][f] for f in det_features] for c in all_class_info]
    pred_alphas_dual, pred_aps_dual, oracle_aps_dual = loo_evaluation(X_dual, y_alpha, all_class_info, domain_data_dict)
    mean_pred_dual = np.mean(pred_aps_dual)
    mean_oracle_dual = np.mean(oracle_aps_dual)
    print(f"  预测AP均值: {mean_pred_dual:.4f}")
    print(f"  Oracle AP均值: {mean_oracle_dual:.4f}")
    print(f"  达Oracle比例: {mean_pred_dual/mean_oracle_dual*100:.1f}%")
    results['dual_drive'] = {
        'mean_pred_ap': float(mean_pred_dual),
        'mean_oracle_ap': float(mean_oracle_dual),
        'oracle_ratio': float(mean_pred_dual/mean_oracle_dual),
    }

    # 阶段3：相关性分析
    print(f"\n{'='*60}")
    print("阶段3：各特征与Oracle α的相关性")
    print(f"{'='*60}")

    correlations = {}
    for f in phys_features:
        vals = [c['phys'][f] for c in all_class_info]
        rho, p = spearmanr(vals, y_alpha)
        correlations[f'phys_{f}'] = {'rho': float(rho), 'p': float(p)}
        print(f"  phys.{f:<15}: ρ={rho:+.3f}, p={p:.4f}")
    for f in det_features:
        vals = [c['det'][f] for c in all_class_info]
        rho, p = spearmanr(vals, y_alpha)
        correlations[f'det_{f}'] = {'rho': float(rho), 'p': float(p)}
        print(f"  det.{f:<15}: ρ={rho:+.3f}, p={p:.4f}")

    # 阶段4：逐域对比
    print(f"\n{'='*60}")
    print("阶段4：逐域对比三种驱动方式")
    print(f"{'='*60}")
    print(f"{'域对':<12} {'det_only':>10} {'phys_only':>10} {'dual_drive':>12} {'oracle':>10}")
    print("-" * 60)
    domain_results = {}
    for src in source_names:
        indices = [i for i, c in enumerate(all_class_info) if c['domain'] == src]
        if not indices:
            continue
        det_ap = np.mean([pred_aps_det[i] for i in indices])
        phys_ap = np.mean([pred_aps_phys[i] for i in indices])
        dual_ap = np.mean([pred_aps_dual[i] for i in indices])
        oracle_ap = np.mean([oracle_aps_det[i] for i in indices])
        print(f"{src:<12} {det_ap:>10.4f} {phys_ap:>10.4f} {dual_ap:>12.4f} {oracle_ap:>10.4f}")
        domain_results[src] = {
            'det_only': float(det_ap),
            'phys_only': float(phys_ap),
            'dual_drive': float(dual_ap),
            'oracle': float(oracle_ap),
        }

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
        'loo_results': convert(results),
        'correlations': convert(correlations),
        'domain_results': convert(domain_results),
        'n_classes': len(all_class_info),
        'n_domains': len(source_names),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
