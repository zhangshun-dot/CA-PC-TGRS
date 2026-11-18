# -*- coding: utf-8 -*-
"""新因子验证实验：测试除物理先验外的其他零训练因子是否有预测力
因子列表：
  1. 文本混淆度（Text Confusability）：类别在文本空间中与其他类别的最小距离
  2. 检测器置信度方差（Detector Confidence Variance）：候选框置信度的方差
  3. 文本域敏感度（Text Domain Sensitivity）：不同文本模板下嵌入的方差
  4. 候选框尺寸方差（Box Size Variance）：候选框尺寸的方差
  5. n_boxes（已验证有效，作为对照）

验证流程：
  1. 计算每个类别的因子值（零训练，直接统计）
  2. 计算每个类别的最优校正强度（oracle α，扫描α找到最优AP）
  3. 计算因子值与最优α的Spearman相关系数
  4. 如果相关性高（|ρ| > 0.5, p < 0.05），说明这个因子有预测力
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate,
)
import tge_probe as T

OUT = DATA / 'new_factor_validation_results.json'

# 多个文本模板，用于计算文本域敏感度
TEXT_TEMPLATES = [
    "a photo of a {}.",
    "a {}.",
    "an aerial view of a {}.",
    "a {} in the scene.",
    "this is a {}.",
    "a {} from above.",
    "a {} on the ground.",
    "a {} in the image.",
]


class Rank1AffineUSC:
    """秩1仿射USC"""
    def __init__(self, use_pca=True, n_components=30):
        self.use_pca = use_pca
        self.n_components = n_components
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        self._v_gpu = None

    def fit(self, source_features, target_features):
        if self.use_pca:
            all_feat = np.vstack([source_features, target_features])
            mean_ = all_feat.mean(axis=0)
            X_centered = all_feat - mean_
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            K = min(self.n_components, Vt.shape[0])
            self.components_ = Vt[:K]
            self.mean_ = mean_
            src_proj = (source_features - mean_) @ self.components_.T
            tgt_proj = (target_features - mean_) @ self.components_.T
        else:
            self.components_ = None
            self.mean_ = None
            src_proj = source_features
            tgt_proj = target_features

        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros_like(shift)

        self._v_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        if self.components_ is not None:
            self._components_gpu = torch.tensor(self.components_, dtype=torch.float32, device=device)
            self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        return self

    def correct_prototype_gpu(self, p_gpu, alpha=1.0):
        if self.components_ is not None:
            Z = (p_gpu - self._mean_gpu) @ self._components_gpu.T
        else:
            Z = p_gpu
        v = self._v_gpu
        Z_corr = Z + alpha * v
        if self.components_ is not None:
            p_corr = Z_corr @ self._components_gpu + self._mean_gpu
        else:
            p_corr = Z_corr
        p_corr = p_corr / torch.clamp(torch.norm(p_corr), min=1e-8)
        return p_corr


def compute_text_confusability(category_name, all_categories):
    """计算文本混淆度：该类别与其他类别的最小文本嵌入距离"""
    t_c = T.text_avg([TGT_TMPL], category_name).cpu().numpy()
    min_dist = float('inf')
    for other in all_categories:
        if other == category_name:
            continue
        t_other = T.text_avg([TGT_TMPL], other).cpu().numpy()
        dist = np.linalg.norm(t_c - t_other)
        if dist < min_dist:
            min_dist = dist
    return float(min_dist)


def compute_text_domain_sensitivity(category_name):
    """计算文本域敏感度：不同文本模板下嵌入的方差"""
    embeddings = []
    for template in TEXT_TEMPLATES:
        text = template.format(category_name)
        emb = T.text_avg([text], category_name).cpu().numpy()
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    # 计算嵌入的方差（所有维度的方差之和）
    variance = float(np.sum(np.var(embeddings, axis=0)))
    return variance


def compute_detector_stats(target_data):
    """计算检测器统计因子：置信度方差、候选框数量、候选框尺寸方差
    target_data 格式: {'per_img': [...], 'features': F, 'n_crops': N}
    per_img[i] = {'box': np.array(N,4), 'gd': np.array(N,), 'gt': np.array(M,4), 'crop_idx': np.array(N,)}
    """
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
        n_boxes = int(len(all_scores))
        conf_mean = float(np.mean(all_scores))
    else:
        conf_variance = 0.0
        n_boxes = 0
        conf_mean = 0.0

    if all_boxes:
        all_boxes = np.vstack(all_boxes)
        # boxes 格式: [x1, y1, x2, y2]
        widths = all_boxes[:, 2] - all_boxes[:, 0]
        heights = all_boxes[:, 3] - all_boxes[:, 1]
        sizes = np.sqrt(np.maximum(widths * heights, 0))
        size_variance = float(np.var(sizes))
        size_mean = float(np.mean(sizes))
    else:
        size_variance = 0.0
        size_mean = 0.0

    return {
        'conf_variance': conf_variance,
        'conf_mean': conf_mean,
        'n_boxes': n_boxes,
        'size_variance': size_variance,
        'size_mean': size_mean,
    }


def find_optimal_alpha(domain_data, category_key):
    """扫描α找到最优AP"""
    usc = domain_data['usc']
    target_data = domain_data['target_data'][category_key]
    all_protos = domain_data['all_protos']
    protos_gpu = domain_data['protos_gpu']
    text_deltas = domain_data['text_deltas']
    src_cls = domain_data['valid_targets_dict'][category_key]

    best_ap = -1
    best_alpha = None
    for alpha in ALPHAS:
        p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], alpha)
        p_corr = p_corr_gpu.cpu().numpy()
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, alpha)
            neg = neg_corr_gpu.cpu().numpy()
        else:
            neg = None
        delta = text_deltas[src_cls]
        p_hat = p_corr + 1.0 * delta
        p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
        ap = evaluate(target_data, p_hat, neg)
        if ap > best_ap:
            best_ap = ap
            best_alpha = alpha
    return best_alpha, best_ap


def load_domain_data_batch(source_name):
    """分批加载单个域对的数据（一次只加载一个域对，减少IO压力）"""
    load_func, targets = SOURCE_LOADERS[source_name]
    print(f"  加载源域 {source_name}...")
    source_data = {}
    for eval_name, coco_cls, src_cls in targets:
        data = load_func(src_cls, n=50)
        if data is not None:
            source_data[src_cls] = data
    print(f"  加载目标域...")
    target_data = {}
    valid_targets = []
    for eval_name, coco_cls, src_cls in targets:
        if src_cls not in source_data:
            continue
        data = load_target_coco(coco_cls, n=200)
        if data is not None:
            target_data[coco_cls] = data
            valid_targets.append((eval_name, coco_cls, src_cls))
    if len(valid_targets) < 2:
        return None

    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])

    print(f"  拟合秩1仿射USC...")
    usc = Rank1AffineUSC(use_pca=True, n_components=30)
    usc.fit(all_src, all_tgt)

    all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}
    text_deltas = {}
    for eval_name, coco_cls, src_cls in valid_targets:
        s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
        d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
        text_deltas[src_cls] = d_emb - s_emb
    protos_gpu = {s: torch.tensor(p, dtype=torch.float32, device=device)
                   for s, p in all_protos.items()}

    valid_targets_dict = {c: s for _, c, s in valid_targets}

    return {
        'source_data': source_data,
        'target_data': target_data,
        'usc': usc,
        'all_protos': all_protos,
        'protos_gpu': protos_gpu,
        'valid_targets': valid_targets,
        'valid_targets_dict': valid_targets_dict,
        'text_deltas': text_deltas,
    }


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"域对列表: {source_names}")
    print(f"新因子验证实验（零训练因子 vs 最优α的相关性）")

    # 收集所有类别
    all_categories = set()
    for src in source_names:
        _, targets = SOURCE_LOADERS[src]
        for _, coco_cls, src_cls in targets:
            all_categories.add(coco_cls)
            all_categories.add(src_cls)
    all_categories = sorted(list(all_categories))
    print(f"所有类别: {all_categories}")

    # 逐个域对处理（分批模式，减少IO压力）
    all_factor_data = []  # 每个元素是一个类别的因子数据

    for source_name in source_names:
        print(f"\n{'='*60}")
        print(f"处理域对: {source_name}")
        print(f"{'='*60}")

        try:
            domain_data = load_domain_data_batch(source_name)
            if domain_data is None:
                print(f"  ⚠️ 域对 {source_name} 数据不足，跳过")
                continue

            for eval_name, coco_cls, src_cls in domain_data['valid_targets']:
                print(f"\n  --- 类别: {src_cls} → {coco_cls} ---")

                # 1. 计算文本混淆度
                tc = compute_text_confusability(coco_cls, all_categories)
                print(f"    文本混淆度: {tc:.4f}")

                # 2. 计算文本域敏感度
                tds = compute_text_domain_sensitivity(coco_cls)
                print(f"    文本域敏感度: {tds:.4f}")

                # 3. 计算检测器统计因子
                det_stats = compute_detector_stats(domain_data['target_data'][coco_cls])
                print(f"    检测器置信度均值: {det_stats['conf_mean']:.6f}, 方差: {det_stats['conf_variance']:.6f}")
                print(f"    候选框数量: {det_stats['n_boxes']}")
                print(f"    候选框尺寸均值: {det_stats['size_mean']:.2f}, 方差: {det_stats['size_variance']:.4f}")

                # 4. 找到最优α
                best_alpha, best_ap = find_optimal_alpha(domain_data, coco_cls)
                print(f"    最优α: {best_alpha}, 最优AP: {best_ap:.4f}")

                # 收集数据
                all_factor_data.append({
                    'domain': source_name,
                    'src_cls': src_cls,
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
                })

            # 释放内存
            del domain_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ 域对 {source_name} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 计算相关性
    print(f"\n{'='*60}")
    print("相关性分析（因子值 vs 最优α）")
    print(f"{'='*60}")

    if len(all_factor_data) < 3:
        print(f"⚠️ 样本太少（{len(all_factor_data)}），无法计算可靠的相关性")
    else:
        factors = [
            ('text_confusability', '文本混淆度'),
            ('text_domain_sensitivity', '文本域敏感度'),
            ('det_conf_variance', '检测器置信度方差'),
            ('det_conf_mean', '检测器置信度均值'),
            ('n_boxes', '候选框数量（对照）'),
            ('box_size_variance', '候选框尺寸方差'),
            ('box_size_mean', '候选框尺寸均值'),
        ]

        alphas = np.array([d['optimal_alpha'] for d in all_factor_data])

        correlation_results = {}
        for factor_key, factor_name in factors:
            values = np.array([d[factor_key] for d in all_factor_data])
            # 检查是否有足够的变异（如果全部相同，Spearman会返回NaN）
            if np.std(values) < 1e-10:
                print(f"\n  {factor_name} ({factor_key}):")
                print(f"    ⚠️ 所有样本值相同（{values[0]:.6f}），无法计算相关性")
                correlation_results[factor_key] = {
                    'name': factor_name,
                    'spearman_rho': None,
                    'p_value': None,
                    'significant': False,
                    'note': '所有样本值相同，无法计算相关性',
                }
                continue
            # Spearman相关系数
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
                'name': factor_name,
                'spearman_rho': float(rho),
                'p_value': float(p_value),
                'significant': bool(abs(rho) > 0.5 and p_value < 0.05),
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
        if isinstance(obj, tuple):
            return [convert(v) for v in obj]
        return obj

    final_results = {
        'all_factor_data': convert(all_factor_data),
        'correlation_results': convert(correlation_results) if 'correlation_results' in dir() else {},
        'n_samples': len(all_factor_data),
        'note': '新因子验证实验：零训练因子 vs 最优α的相关性',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
