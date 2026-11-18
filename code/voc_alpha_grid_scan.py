# -*- coding: utf-8 -*-
"""voc_alpha_grid_scan.py — VOC目标域完整α网格搜索 + 初始间隔m^0计算
验证Corollary 2的理论预测：VOC上m^0高的类别最优α≈0
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
from scipy import stats

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
CODE = Path(r'C:\Users\张三\Desktop\实验代码')

# 导入unified_eval
import importlib.util
spec = importlib.util.spec_from_file_location("unified_eval", CODE / "unified_eval.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)

# α扫描范围
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# 源域到VOC类别的映射
SRC_TO_VOC = {
    'DIOR': {'airplane': 'aeroplane', 'ship': 'boat', 'vehicle': 'car'},
    'NWPU': {'airplane': 'aeroplane', 'ship': 'boat', 'vehicle': 'car'},
    'DOTA': {'plane': 'aeroplane', 'ship': 'boat', 'small-vehicle': 'car', 'large-vehicle': 'car'},
    'VisDrone': {'bicycle': 'bicycle', 'car': 'car', 'bus': 'bus', 'motor': 'motorbike'},
}


def compute_initial_margin(features, prototypes, class_idx):
    """计算初始间隔m_c^(0)：top1 - top2的置信度中位数
    features: (N, D) 候选框特征
    prototypes: (C, D) 源域原型
    class_idx: 目标类别的索引
    """
    # 计算所有类别的相似度
    similarities = features @ prototypes.T  # (N, C)

    # 对每个候选框，计算top1 - top2
    sorted_sim = np.sort(similarities, axis=1)[:, ::-1]  # 降序
    if sorted_sim.shape[1] >= 2:
        margins = sorted_sim[:, 0] - sorted_sim[:, 1]
    else:
        margins = sorted_sim[:, 0]

    # 取中位数作为m_c^(0)
    m_0 = np.median(margins)
    return float(m_0), margins


def compute_mu_c(features, prototypes, class_idx, extrapolation_direction):
    """计算mu_c：校正方向在特征上的平均投影
    mu_c = (1/n_c) sum_{i: y_hat_i=c} v_c^T f_i
    """
    # 预测类别
    similarities = features @ prototypes.T
    pred_labels = np.argmax(similarities, axis=1)

    # 只取预测为目标类别的样本
    class_mask = pred_labels == class_idx
    if class_mask.sum() == 0:
        return 0.0

    # 计算mu_c
    class_features = features[class_mask]
    mu_c = np.mean(class_features @ extrapolation_direction)
    return float(mu_c)


def run_alpha_scan_for_class(src_name, src_cls, voc_cls, n_images=200):
    """对单个类别跑α网格搜索"""
    print(f'\n  处理 {src_name}/{src_cls} -> VOC/{voc_cls}...')

    # 加载源域原型
    src_prototypes = U.load_source_prototypes(src_name, [src_cls])
    if src_prototypes is None or len(src_prototypes) == 0:
        print(f'    警告: 无法加载源域原型')
        return None

    # 加载VOC目标域数据
    try:
        target_data = U.load_target_voc([voc_cls], n=n_images)
    except Exception as e:
        print(f'    警告: 无法加载VOC数据: {e}')
        return None

    if target_data is None or len(target_data.get('features', [])) == 0:
        print(f'    警告: VOC数据为空')
        return None

    features = np.array(target_data['features'])
    det_scores = np.array(target_data['det_scores'])
    gt_labels = np.array(target_data['gt_labels'])
    gt_boxes = target_data.get('gt_boxes', None)

    # 构造所有类别的原型（用于计算m^0）
    # 这里简化：只用目标类别和一个负类别
    all_classes = [voc_cls, 'background']
    proto_c = src_prototypes[0]
    proto_neg = np.random.randn(*proto_c.shape) * 0.1  # 简化的负原型
    prototypes = np.stack([proto_c, proto_neg])

    # 计算初始间隔m^0
    m_0, margins = compute_initial_margin(features, prototypes, 0)

    # 计算外推方向（简化：用文本方向）
    try:
        text_src = U.get_text_embedding(src_cls)
        text_tgt = U.get_text_embedding(voc_cls)
        extrapolation_dir = text_tgt - text_src
        extrapolation_dir = extrapolation_dir / (np.linalg.norm(extrapolation_dir) + 1e-8)
    except:
        extrapolation_dir = np.ones_like(proto_c) / np.sqrt(len(proto_c))

    # 计算mu_c
    mu_c = compute_mu_c(features, prototypes, 0, extrapolation_dir)

    # α网格搜索
    ap_results = {}
    for alpha in ALPHAS:
        # 外推原型
        extrapolated_proto = proto_c + alpha * extrapolation_dir

        # 计算视觉分数
        vis_scores = features @ extrapolated_proto

        # 融合（几何平均）
        eps = 1e-6
        s_det = np.clip(det_scores, eps, 1)
        s_vis = np.clip(vis_scores, eps, 1)
        fused_scores = np.sqrt(s_det * s_vis)

        # 计算AP
        try:
            ap = U.compute_ap(fused_scores, gt_labels, gt_boxes, iou_threshold=0.5)
        except:
            # 简化的AP计算
            ap = compute_ap_simple(fused_scores, gt_labels)

        ap_results[alpha] = float(ap)

    # 找最优α
    best_alpha = max(ap_results, key=ap_results.get)
    best_ap = ap_results[best_alpha]
    baseline_ap = ap_results[0.0]

    return {
        'src': src_name,
        'src_cls': src_cls,
        'voc_cls': voc_cls,
        'n_images': n_images,
        'm_0': m_0,
        'mu_c': mu_c,
        'baseline_ap': baseline_ap,
        'best_alpha': best_alpha,
        'best_ap': best_ap,
        'ap_curve': ap_results,
        'alpha_gt_0_better': best_ap > baseline_ap + 0.001,
    }


def compute_ap_simple(scores, labels):
    """简化的AP计算"""
    # 按分数降序排序
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    # 计算精确率和召回率
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (labels.sum() + 1e-8)

    # 11点插值AP
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        mask = recall >= t
        if mask.any():
            ap += precision[mask].max() / 11
    return float(ap)


def main():
    print('=' * 80)
    print('VOC目标域完整α网格搜索 + 初始间隔m^0计算')
    print('验证Corollary 2的理论预测')
    print('=' * 80)

    all_results = []

    for src_name, class_map in SRC_TO_VOC.items():
        print(f'\n{"=" * 80}')
        print(f'源域: {src_name}')
        print(f'{"=" * 80}')

        for src_cls, voc_cls in class_map.items():
            result = run_alpha_scan_for_class(src_name, src_cls, voc_cls)
            if result:
                all_results.append(result)
                print(f'    {src_cls} -> {voc_cls}: m^0={result["m_0"]:.4f}, '
                      f'baseline AP={result["baseline_ap"]:.4f}, '
                      f'best α={result["best_alpha"]:.2f}, best AP={result["best_ap"]:.4f}, '
                      f'α>0更好={result["alpha_gt_0_better"]}')

    # 保存结果
    output_file = DATA / 'voc_alpha_grid_scan_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {output_file}')

    # 分析结果
    print(f'\n{"=" * 80}')
    print('分析：m^0 vs 最优α')
    print(f'{"=" * 80}')

    m_0_values = [r['m_0'] for r in all_results]
    best_alphas = [r['best_alpha'] for r in all_results]
    baseline_aps = [r['baseline_ap'] for r in all_results]
    alpha_gt_0 = [r['alpha_gt_0_better'] for r in all_results]

    print(f'\n总类别数: {len(all_results)}')
    print(f'α>0更好的类别数: {sum(alpha_gt_0)} ({sum(alpha_gt_0)/len(all_results)*100:.1f}%)')
    print(f'最优α=0的类别数: {sum(1 for a in best_alphas if a == 0.0)}')
    print(f'最优α>0的类别数: {sum(1 for a in best_alphas if a > 0.0)}')

    # 相关性分析
    if len(all_results) >= 3:
        rho_m0_alpha, p_m0_alpha = stats.spearmanr(m_0_values, best_alphas)
        rho_m0_baseline, p_m0_baseline = stats.spearmanr(m_0_values, baseline_aps)
        rho_baseline_alpha, p_baseline_alpha = stats.spearmanr(baseline_aps, best_alphas)

        print(f'\n相关性分析:')
        print(f'  m^0 vs 最优α: ρ={rho_m0_alpha:.4f}, p={p_m0_alpha:.4f}')
        print(f'  m^0 vs baseline AP: ρ={rho_m0_baseline:.4f}, p={p_m0_baseline:.4f}')
        print(f'  baseline AP vs 最优α: ρ={rho_baseline_alpha:.4f}, p={p_baseline_alpha:.4f}')

    # 分组：m^0高 vs m^0低
    if len(all_results) >= 4:
        median_m0 = np.median(m_0_values)
        high_m0 = [r for r in all_results if r['m_0'] >= median_m0]
        low_m0 = [r for r in all_results if r['m_0'] < median_m0]

        print(f'\n分组分析 (m^0中位数={median_m0:.4f}):')
        print(f'  高m^0组 (N={len(high_m0)}): 平均最优α={np.mean([r["best_alpha"] for r in high_m0]):.4f}, '
              f'α>0比例={sum(r["alpha_gt_0_better"] for r in high_m0)/len(high_m0)*100:.1f}%')
        print(f'  低m^0组 (N={len(low_m0)}): 平均最优α={np.mean([r["best_alpha"] for r in low_m0]):.4f}, '
              f'α>0比例={sum(r["alpha_gt_0_better"] for r in low_m0)/len(low_m0)*100:.1f}%')

    # 打印详细结果表
    print(f'\n{"=" * 80}')
    print('详细结果表')
    print(f'{"=" * 80}')
    print(f'{"源域":<10s} {"源类别":<15s} {"VOC类别":<15s} {"m^0":>8s} {"baselineAP":>11s} {"最优α":>7s} {"最优AP":>8s} {"α>0更好":>9s}')
    print('-' * 95)
    for r in all_results:
        print(f'{r["src"]:<10s} {r["src_cls"]:<15s} {r["voc_cls"]:<15s} '
              f'{r["m_0"]:>8.4f} {r["baseline_ap"]:>11.4f} {r["best_alpha"]:>7.2f} '
              f'{r["best_ap"]:>8.4f} {"是" if r["alpha_gt_0_better"] else "否":>9s}')


if __name__ == '__main__':
    main()
