# -*- coding: utf-8 -*-
"""自适应谱归一化实验（秩1仿射版本）
核心思想：根据域偏移大小自动调整谱归一化的权重
- 域偏移大的场景（VisDrone）用强归一化
- 域偏移小的场景（DIOR/NWPU）用弱归一化甚至不归一化

对比：
  1. 无谱归一化（w=0，基线）
  2. 全局谱归一化（w=1）
  3. 线性自适应：w = clip(a·δ + b, 0, 1)
  4. 归一化自适应：w = δ / δ_max
  5. sigmoid自适应：w = sigmoid(a·δ + b)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate,
)
import tge_probe as T

OUT = DATA / 'usc_adaptive_spectral_norm_results.json'


class Rank1AffineUSC:
    """秩1仿射USC（第二篇论文核心方法，只有1个参数α）"""
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


def apply_spectral_norm(features, src_std, tgt_std, weight):
    """应用谱归一化
    weight=0: 不归一化
    weight=1: 完全归一化
    weight在0-1之间: 部分归一化
    """
    if weight <= 0:
        return features
    scale = tgt_std / (src_std + 1e-8)
    # 插值：原始特征 + weight * (归一化特征 - 原始特征)
    return features + weight * (features * scale - features)


def compute_domain_shift(source_features, target_features):
    """计算域偏移大小（特征均值的欧氏距离）"""
    src_mean = source_features.mean(axis=0)
    tgt_mean = target_features.mean(axis=0)
    return float(np.linalg.norm(tgt_mean - src_mean))


def load_domain_data(source_name, spectral_weight=0.0):
    """加载域对数据，可选谱归一化权重"""
    load_func, targets = SOURCE_LOADERS[source_name]
    print(f"  加载源域...")
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

    # 计算域偏移大小
    domain_shift = compute_domain_shift(all_src, all_tgt)
    print(f"  域偏移大小: {domain_shift:.4f}")

    # 计算谱归一化系数
    src_std = np.std(all_src, axis=0) + 1e-8
    tgt_std = np.std(all_tgt, axis=0) + 1e-8

    # 应用谱归一化
    if spectral_weight > 0:
        print(f"  应用谱归一化（权重={spectral_weight:.4f}）...")
        all_src = apply_spectral_norm(all_src, src_std, tgt_std, spectral_weight)
        all_tgt = apply_spectral_norm(all_tgt, src_std, tgt_std, spectral_weight)
        for s in source_data:
            source_data[s]['features'] = apply_spectral_norm(source_data[s]['features'], src_std, tgt_std, spectral_weight)
            source_data[s]['prototype'] = apply_spectral_norm(source_data[s]['prototype'], src_std, tgt_std, spectral_weight)
        for c in target_data:
            target_data[c]['features'] = apply_spectral_norm(target_data[c]['features'], src_std, tgt_std, spectral_weight)

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

    return {
        'source_data': source_data,
        'target_data': target_data,
        'usc': usc,
        'all_protos': all_protos,
        'protos_gpu': protos_gpu,
        'valid_targets': valid_targets,
        'text_deltas': text_deltas,
        'domain_shift': domain_shift,
    }


def evaluate_with_params(domain_data, alpha):
    usc = domain_data['usc']
    target_data = domain_data['target_data']
    all_protos = domain_data['all_protos']
    protos_gpu = domain_data['protos_gpu']
    text_deltas = domain_data['text_deltas']
    valid_targets = domain_data['valid_targets']
    aps = []
    for eval_name, coco_cls, src_cls in valid_targets:
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
        ap = evaluate(target_data[coco_cls], p_hat, neg)
        aps.append(ap)
    return float(np.mean(aps))


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"域对列表: {source_names}")
    print(f"自适应谱归一化实验（秩1仿射）")

    # 第一步：计算每个域对的域偏移大小
    print(f"\n{'='*60}")
    print("第一步：计算每个域对的域偏移大小")
    print(f"{'='*60}")
    domain_shifts = {}
    for src in source_names:
        data = load_domain_data(src, spectral_weight=0.0)
        if data:
            domain_shifts[src] = data['domain_shift']
            print(f"  {src}: 域偏移={data['domain_shift']:.4f}")
            del data
    delta_max = max(domain_shifts.values())
    print(f"  最大域偏移: {delta_max:.4f}")

    # 第二步：测试不同的谱归一化权重
    print(f"\n{'='*60}")
    print("第二步：测试不同的谱归一化权重（全局）")
    print(f"{'='*60}")
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    global_results = {}
    for w in weights:
        print(f"\n--- 谱归一化权重 w={w} ---")
        w_results = {}
        for src in source_names:
            data = load_domain_data(src, spectral_weight=w)
            if data is None:
                continue
            best_ap = -1
            best_alpha = None
            for alpha in ALPHAS:
                ap = evaluate_with_params(data, alpha)
                if ap > best_ap:
                    best_ap = ap
                    best_alpha = alpha
            print(f"  {src}: AP={best_ap:.4f}, α={best_alpha}")
            w_results[src] = {'ap': best_ap, 'alpha': best_alpha}
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        global_results[w] = w_results

    # 第三步：自适应谱归一化
    print(f"\n{'='*60}")
    print("第三步：自适应谱归一化")
    print(f"{'='*60}")

    # 方法1：归一化自适应 w = δ / δ_max
    print(f"\n--- 方法1：归一化自适应 w = δ / δ_max ---")
    adaptive1_results = {}
    for src in source_names:
        w = domain_shifts[src] / delta_max
        print(f"  {src}: δ={domain_shifts[src]:.4f}, w={w:.4f}")
        data = load_domain_data(src, spectral_weight=w)
        if data is None:
            continue
        best_ap = -1
        best_alpha = None
        for alpha in ALPHAS:
            ap = evaluate_with_params(data, alpha)
            if ap > best_ap:
                best_ap = ap
                best_alpha = alpha
        print(f"    AP={best_ap:.4f}, α={best_alpha}")
        adaptive1_results[src] = {'ap': best_ap, 'alpha': best_alpha, 'w': w}
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'='*60}")
    print("汇总：全局权重 vs 自适应权重")
    print(f"{'='*60}")
    print(f"{'方法':<30} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8}")
    print("-" * 80)

    # 无归一化
    aps = [global_results[0.0].get(s, {}).get('ap', 0) for s in source_names]
    print(f"{'无谱归一化 (w=0)':<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f}")

    # 全局权重
    for w in [0.25, 0.5, 0.75, 1.0]:
        aps = [global_results[w].get(s, {}).get('ap', 0) for s in source_names]
        print(f"{'全局谱归一化 (w=' + str(w) + ')':<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f}")

    # 自适应
    aps = [adaptive1_results.get(s, {}).get('ap', 0) for s in source_names]
    print(f"{'自适应 (w=δ/δ_max)':<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f}")

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
        'domain_shifts': convert(domain_shifts),
        'delta_max': float(delta_max),
        'global_results': convert(global_results),
        'adaptive1_results': convert(adaptive1_results),
        'note': '自适应谱归一化实验（秩1仿射）',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
