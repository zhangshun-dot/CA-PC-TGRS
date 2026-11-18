# -*- coding: utf-8 -*-
"""VOC目标域新因子验证实验
扩大样本量：4个源域→VOC，7个类别，预计~22个新样本
合并COCO+VOC后总样本量~44
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from scipy import stats
from pathlib import Path
import pickle

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, PRE_FILTER,
    SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, build_neg_proto, evaluate, USC,
)
import tge_probe as T
from voc_loader import load_target_voc, VOC2COCO

device = 'cpu'
OUT_FILE = DATA / 'new_factor_validation_voc_results.json'

# VOC目标域类别（和遥感对应的）
VOC_TARGET_CLASSES = [
    ('aeroplane', 'airplane'),
    ('bicycle', 'bicycle'),
    ('boat', 'boat'),
    ('bus', 'bus'),
    ('car', 'car'),
    ('motorbike', 'motorcycle'),
    ('train', 'train'),
]


def compute_text_confusability(cls_name):
    """计算文本混淆度：类别在CLIP文本空间中与其他类别的最小距离"""
    all_classes = [c[1] for c in VOC_TARGET_CLASSES] + ['airplane', 'ship', 'vehicle', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat']
    all_classes = list(set(all_classes))
    embs = []
    for c in all_classes:
        emb = T.text_avg([TGT_TMPL], c).cpu().numpy()
        embs.append(emb / (np.linalg.norm(emb) + 1e-8))
    embs = np.array(embs)
    cls_emb = T.text_avg([TGT_TMPL], cls_name).cpu().numpy()
    cls_emb = cls_emb / (np.linalg.norm(cls_emb) + 1e-8)
    sims = embs @ cls_emb
    # 排除自己
    sims = sims[sims < 0.999]
    return float(np.max(sims)) if len(sims) > 0 else 0.5


def compute_detector_stats(target_data):
    """计算检测器统计因子"""
    per_img = target_data.get('per_img', [])
    all_scores = []
    all_boxes = []
    for img_data in per_img:
        scores = img_data.get('gd', None)
        boxes = img_data.get('box', None)
        if scores is not None and len(scores) > 0:
            all_scores.append(scores)
        if boxes is not None and len(boxes) > 0:
            all_boxes.append(boxes)
    if all_scores:
        all_scores = np.concatenate(all_scores)
        conf_variance = float(np.var(all_scores))
        conf_mean = float(np.mean(all_scores))
        n_boxes = int(len(all_scores))
    else:
        conf_variance = 0.0
        conf_mean = 0.0
        n_boxes = 0
    if all_boxes:
        all_boxes = np.vstack(all_boxes)
        widths = all_boxes[:, 2] - all_boxes[:, 0]
        heights = all_boxes[:, 3] - all_boxes[:, 1]
        sizes = np.sqrt(np.maximum(widths * heights, 0))
        size_variance = float(np.var(sizes))
        size_mean = float(np.mean(sizes))
    else:
        size_variance = 0.0
        size_mean = 0.0
    return {
        'conf_variance': conf_variance, 'conf_mean': conf_mean,
        'n_boxes': n_boxes, 'size_variance': size_variance, 'size_mean': size_mean,
    }


def find_optimal_alpha(usc, target_data, all_protos, protos_gpu, text_deltas, src_cls, alphas=ALPHAS):
    """扫描α找到最优AP"""
    best_ap = -1
    best_alpha = None
    all_aps = {}
    for alpha in alphas:
        p_corr_gpu = usc.correct_prototype(protos_gpu[src_cls], alpha)
        p_corr = p_corr_gpu.cpu().numpy() if hasattr(p_corr_gpu, 'cpu') else p_corr_gpu
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            neg_corr_gpu = usc.correct_prototype(neg_gpu, alpha)
            neg = neg_corr_gpu.cpu().numpy() if hasattr(neg_corr_gpu, 'cpu') else neg_corr_gpu
        else:
            neg = None
        delta = text_deltas[src_cls]
        p_hat = p_corr + 1.0 * delta
        p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
        ap = evaluate(target_data, p_hat, neg)
        all_aps[alpha] = ap
        if ap > best_ap:
            best_ap = ap
            best_alpha = alpha
    return best_alpha, best_ap, all_aps


def main():
    print("=" * 60)
    print("VOC目标域新因子验证实验")
    print("=" * 60)

    all_factor_data = []

    for source_name in SOURCE_LOADERS.keys():
        print(f"\n{'='*60}")
        print(f"处理域对: {source_name} → VOC")
        print(f"{'='*60}")

        load_func, _ = SOURCE_LOADERS[source_name]

        # 加载源域
        print(f"  加载源域 {source_name}...")
        source_data = {}
        for voc_cls, coco_cls in VOC_TARGET_CLASSES:
            # 尝试用COCO类别名加载源域（因为源域加载函数用的是COCO类别名）
            src_cls_name = coco_cls
            # 特殊映射
            if coco_cls == 'airplane':
                src_cls_name = 'airplane'
            elif coco_cls == 'motorcycle':
                src_cls_name = 'vehicle'
            elif coco_cls == 'bicycle':
                src_cls_name = 'vehicle'
            elif coco_cls == 'train':
                continue  # 源域没有train类别
            data = load_func(src_cls_name, n=50)
            if data is not None:
                source_data[voc_cls] = data
                print(f"    {voc_cls} (src={src_cls_name}): {data['features'].shape[0]} features")

        if len(source_data) < 2:
            print(f"  ⚠️ 源域类别不足，跳过")
            continue

        # 加载VOC目标域
        print(f"  加载VOC目标域...")
        target_data = {}
        valid_targets = []
        for voc_cls, coco_cls in VOC_TARGET_CLASSES:
            if voc_cls not in source_data:
                continue
            print(f"    加载VOC {voc_cls}...")
            data = load_target_voc(voc_cls, n=200)
            if data is not None:
                target_data[voc_cls] = data
                valid_targets.append(voc_cls)
                print(f"      {data['n_crops']} crops")

        if len(valid_targets) < 2:
            print(f"  ⚠️ 目标域类别不足，跳过")
            continue

        # 构建USC
        all_src = np.vstack([source_data[s]['features'] for s in valid_targets])
        all_tgt = np.vstack([target_data[c]['features'] for c in valid_targets])
        usc = USC(n_components=K_PCA)
        usc.fit(all_src, all_tgt)

        all_protos = {s: source_data[s]['prototype'] for s in valid_targets}
        protos_gpu = {s: torch.tensor(p, dtype=torch.float32, device=device) for s, p in all_protos.items()}

        # 文本偏移
        text_deltas = {}
        for voc_cls in valid_targets:
            coco_cls = dict(VOC_TARGET_CLASSES)[voc_cls]
            s_emb = T.text_avg([SRC_TMPL_OVERHEAD], coco_cls).cpu().numpy()
            d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
            text_deltas[voc_cls] = d_emb - s_emb

        # 对每个类别计算因子和最优α
        for voc_cls in valid_targets:
            coco_cls = dict(VOC_TARGET_CLASSES)[voc_cls]
            print(f"\n  --- 类别: {voc_cls} ({coco_cls}) ---")

            # 1. 文本混淆度
            tc = compute_text_confusability(coco_cls)
            print(f"    文本混淆度: {tc:.4f}")

            # 2. 文本域敏感度
            embs = []
            for tmpl in [TGT_TMPL, 'a photo of a {}', 'an image of a {}']:
                emb = T.text_avg([tmpl], coco_cls).cpu().numpy()
                embs.append(emb / (np.linalg.norm(emb) + 1e-8))
            tds = float(np.var(embs))
            print(f"    文本域敏感度: {tds:.6f}")

            # 3. 检测器统计
            det_stats = compute_detector_stats(target_data[voc_cls])
            print(f"    检测器置信度均值: {det_stats['conf_mean']:.6f}, 方差: {det_stats['conf_variance']:.6f}")
            print(f"    候选框数量: {det_stats['n_boxes']}")
            print(f"    候选框尺寸均值: {det_stats['size_mean']:.2f}, 方差: {det_stats['size_variance']:.4f}")

            # 4. 找到最优α
            best_alpha, best_ap, all_aps = find_optimal_alpha(
                usc, target_data[voc_cls], all_protos, protos_gpu, text_deltas, voc_cls
            )
            print(f"    最优α: {best_alpha}, 最优AP: {best_ap:.4f}")

            # 收集数据
            all_factor_data.append({
                'domain': source_name,
                'target_domain': 'VOC',
                'src_cls': voc_cls,
                'tgt_cls': coco_cls,
                'text_confusability': tc,
                'text_domain_sensitivity': tds,
                'det_conf_variance': det_stats['conf_variance'],
                'det_conf_mean': det_stats['conf_mean'],
                'n_boxes': det_stats['n_boxes'],
                'box_size_variance': det_stats['size_variance'],
                'box_size_mean': det_stats['size_mean'],
                'optimal_alpha': best_alpha,
                'optimal_ap': best_ap,
                'all_aps': {str(k): v for k, v in all_aps.items()},
            })

    # 相关性分析
    print(f"\n{'='*60}")
    print("相关性分析（因子值 vs 最优α）")
    print(f"{'='*60}")
    print(f"VOC样本数: {len(all_factor_data)}")

    factors = [
        ('text_confusability', '文本混淆度'),
        ('text_domain_sensitivity', '文本域敏感度'),
        ('det_conf_variance', '检测器置信度方差'),
        ('det_conf_mean', '检测器置信度均值'),
        ('n_boxes', '候选框数量'),
        ('box_size_variance', '候选框尺寸方差'),
        ('box_size_mean', '候选框尺寸均值'),
    ]

    alphas = np.array([d['optimal_alpha'] for d in all_factor_data])
    correlation_results = {}
    for factor_key, factor_name in factors:
        values = np.array([d[factor_key] for d in all_factor_data])
        if np.std(values) < 1e-10:
            print(f"\n  {factor_name}: 所有样本值相同，跳过")
            correlation_results[factor_key] = {'name': factor_name, 'spearman_rho': None, 'p_value': None, 'significant': False}
            continue
        rho, p_value = stats.spearmanr(values, alphas)
        print(f"\n  {factor_name} ({factor_key}):")
        print(f"    Spearman ρ = {rho:.4f}, p = {p_value:.4f}")
        if abs(rho) > 0.5 and p_value < 0.05:
            print(f"    ✅ 显著相关！有预测力")
        elif abs(rho) > 0.3:
            print(f"    ⚠️ 弱相关，预测力有限")
        else:
            print(f"    ❌ 无显著相关，没有预测力")
        correlation_results[factor_key] = {
            'name': factor_name, 'spearman_rho': float(rho),
            'p_value': float(p_value), 'significant': abs(rho) > 0.5 and p_value < 0.05,
        }

    # 保存
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
        'all_factor_data': convert(all_factor_data),
        'correlation_results': convert(correlation_results),
        'n_samples': len(all_factor_data),
        'target_domain': 'VOC',
        'note': 'VOC目标域新因子验证实验（扩大样本量）',
    }
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT_FILE}")


if __name__ == '__main__':
    main()
