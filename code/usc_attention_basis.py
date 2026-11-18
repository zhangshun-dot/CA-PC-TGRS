# -*- coding: utf-8 -*-
"""注意力驱动的共享基底USC实验
核心思想：借鉴Transformer自注意力机制，每个基底向量作为token生成Q/K/V，
通过自注意力动态学习基底间的关系，增强共享基底的表达能力。
最简单实现：Q=K=V=基底向量本身（固定，零训练，不需要任何参数）
对比：
  1. 原始USC（单域PCA基底）
  2. 简单共享基底USC（多域联合PCA基底，无注意力）
  3. 注意力增强共享基底USC（多域联合PCA基底 + 自注意力增强）
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
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate, evaluate_baseline,
)
import tge_probe as T

OUT = DATA / 'usc_attention_basis_results.json'

# 传感器感知γ配置
SENSOR_GAMMA = {
    'DIOR': -2.0, 'NWPU': -2.0, 'DOTA': -2.0, 'VisDrone': -1.0,
}


class AttentionBasisUSC:
    """注意力驱动的共享基底USC
    基底B从多个源域联合学习（PCA闭式解）
    自注意力增强：Q=K=V=基底向量本身（固定，零训练）
    """
    def __init__(self, n_components=30, use_attention=True, temperature=None):
        self.n_components = n_components
        self.use_attention = use_attention
        self.mean_ = None
        self.components_ = None  # 原始共享基底B
        self.components_attn_ = None  # 注意力增强后的基底B'
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        self._components_gpu = None
        self._mean_gpu = None
        self._v_domain_gpu = None
        # 注意力温度，如果为None则用√d（d=K）
        self.temperature = temperature if temperature is not None else np.sqrt(n_components)

    def fit_shared_basis(self, all_source_features_list):
        """从多个源域联合学习共享基底B（PCA闭式解）"""
        all_features = np.vstack(all_source_features_list)
        self.mean_ = all_features.mean(axis=0)
        X_centered = all_features - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        K = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:K]  # (K, D)
        print(f"    共享基底学习完成: {len(all_source_features_list)}个源域, "
              f"{all_features.shape[0]}个样本, K={K}")
        explained_var = np.sum(S[:K]**2) / np.sum(S**2)
        print(f"    前{K}主成分解释方差比例: {explained_var*100:.1f}%")

        # 自注意力增强
        if self.use_attention:
            self._apply_attention()
        else:
            self.components_attn_ = self.components_
        return self

    def _apply_attention(self):
        """自注意力增强：Q=K=V=基底向量本身
        α_ij = softmax(b_i·b_j / √d)
        b_i' = Σ_j α_ij · b_j
        """
        B = self.components_  # (K, D)
        K, D = B.shape
        # 计算注意力分数：Q·K^T = B·B^T (K, K)
        scores = B @ B.T / self.temperature  # (K, K)
        # softmax（每行）
        scores = scores - scores.max(axis=1, keepdims=True)  # 数值稳定
        attn_weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)  # (K, K)
        # 注意力加权：B' = attn_weights · B (K, D)
        self.components_attn_ = attn_weights @ B  # (K, D)

        # 打印注意力统计
        print(f"    自注意力增强完成")
        print(f"    注意力权重均值: {attn_weights.mean():.4f}, 最大: {attn_weights.max():.4f}")
        # 熵（衡量注意力集中度）
        entropy = -np.sum(attn_weights * np.log(attn_weights + 1e-10), axis=1).mean()
        print(f"    平均注意力熵: {entropy:.4f} (最大可能熵={np.log(K):.4f})")
        # 基底变化量
        basis_change = np.linalg.norm(self.components_attn_ - self.components_) / np.linalg.norm(self.components_)
        print(f"    基底相对变化量: {basis_change*100:.2f}%")

    def fit_domain_shift(self, source_features, target_features):
        """计算域偏移方向（在基底空间）"""
        # 用注意力增强后的基底投影
        src_proj = (source_features - self.mean_) @ self.components_attn_.T
        tgt_proj = (target_features - self.mean_) @ self.components_attn_.T
        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros(self.n_components)
        self._components_gpu = torch.tensor(self.components_attn_, dtype=torch.float32, device=device)
        self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        self._v_domain_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0):
        """GPU校正单个原型（在注意力增强的基底空间）"""
        Z = (p_gpu - self._mean_gpu) @ self._components_gpu.T  # (K,)
        vZ = torch.dot(Z, self._v_domain_gpu)
        Z_affine = Z + gamma * vZ * self._v_domain_gpu  # 缩放
        Z_corr = Z_affine + alpha * self._v_domain_gpu  # 平移
        X_corr = Z_corr @ self._components_gpu + self._mean_gpu
        X_corr = X_corr / torch.clamp(torch.norm(X_corr), min=1e-8)
        return X_corr


def load_source_features(source_name):
    """加载单个源域的所有特征"""
    load_func, targets = SOURCE_LOADERS[source_name]
    all_features = []
    for eval_name, coco_cls, src_cls in targets:
        data = load_func(src_cls, n=50)
        if data is not None and 'features' in data:
            all_features.append(data['features'])
    if all_features:
        return np.vstack(all_features)
    return None


def load_domain_data(source_name, usc):
    """加载单个域对数据，使用给定的USC"""
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
    print(f"  计算域偏移...")
    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
    usc.fit_domain_shift(all_src, all_tgt)
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
    }


def evaluate_with_params(domain_data, gamma, alpha):
    """用指定γ,α评估"""
    usc = domain_data['usc']
    target_data = domain_data['target_data']
    all_protos = domain_data['all_protos']
    protos_gpu = domain_data['protos_gpu']
    text_deltas = domain_data['text_deltas']
    valid_targets = domain_data['valid_targets']
    aps = []
    for eval_name, coco_cls, src_cls in valid_targets:
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
        aps.append(ap)
    return float(np.mean(aps))


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"域对列表: {source_names}")

    # 阶段1：从多个源域联合学习共享基底
    print(f"\n{'='*60}")
    print("阶段1：从多个源域联合学习共享基底")
    print(f"{'='*60}")
    all_source_features = []
    for src in source_names:
        feats = load_source_features(src)
        if feats is not None:
            all_source_features.append(feats)
            print(f"  {src}: {feats.shape}")

    # 原始USC的最佳结果（从之前的实验）
    original_usc_best = {
        'DIOR': 0.3836, 'NWPU': 0.3845, 'DOTA': 0.3864, 'VisDrone': 0.3406,
    }

    results = {}

    # 阶段2：简单共享基底（无注意力）
    print(f"\n{'='*60}")
    print("阶段2：简单共享基底USC（无注意力）")
    print(f"{'='*60}")
    usc_simple = AttentionBasisUSC(n_components=K_PCA, use_attention=False).fit_shared_basis(all_source_features)
    simple_results = {}
    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data(src, usc_simple)
        if data is None:
            continue
        best_ap = -1
        best_params = None
        for gamma in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]:
            for alpha in ALPHAS:
                ap = evaluate_with_params(data, gamma, alpha)
                if ap > best_ap:
                    best_ap = ap
                    best_params = (gamma, alpha)
        print(f"  简单共享基底最佳: γ={best_params[0]}, α={best_params[1]}, AP={best_ap:.4f}")
        print(f"  原始USC最佳: AP={original_usc_best.get(src, 0):.4f}")
        gain = best_ap - original_usc_best.get(src, 0)
        print(f"  提升: {gain:+.4f}")
        simple_results[src] = {'ap': best_ap, 'params': best_params, 'gain': gain}
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results['simple_shared'] = simple_results

    # 阶段3：注意力增强共享基底
    print(f"\n{'='*60}")
    print("阶段3：注意力增强共享基底USC")
    print(f"{'='*60}")
    usc_attn = AttentionBasisUSC(n_components=K_PCA, use_attention=True).fit_shared_basis(all_source_features)
    attn_results = {}
    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data(src, usc_attn)
        if data is None:
            continue
        best_ap = -1
        best_params = None
        for gamma in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]:
            for alpha in ALPHAS:
                ap = evaluate_with_params(data, gamma, alpha)
                if ap > best_ap:
                    best_ap = ap
                    best_params = (gamma, alpha)
        print(f"  注意力共享基底最佳: γ={best_params[0]}, α={best_params[1]}, AP={best_ap:.4f}")
        print(f"  原始USC最佳: AP={original_usc_best.get(src, 0):.4f}")
        gain = best_ap - original_usc_best.get(src, 0)
        print(f"  提升: {gain:+.4f}")
        attn_results[src] = {'ap': best_ap, 'params': best_params, 'gain': gain}
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results['attention_shared'] = attn_results

    # 阶段4：汇总对比
    print(f"\n{'='*60}")
    print("阶段4：三种基底方式汇总对比")
    print(f"{'='*60}")
    print(f"{'域对':<12} {'原始USC':>10} {'简单共享':>10} {'注意力共享':>12} {'简单提升':>10} {'注意力提升':>12}")
    print("-" * 70)
    for src in source_names:
        orig = original_usc_best.get(src, 0)
        simp = simple_results.get(src, {}).get('ap', 0)
        attn = attn_results.get(src, {}).get('ap', 0)
        print(f"{src:<12} {orig:>10.4f} {simp:>10.4f} {attn:>12.4f} "
              f"{simp-orig:>+10.4f} {attn-orig:>+12.4f}")
    orig_mean = np.mean([original_usc_best.get(s, 0) for s in source_names])
    simp_mean = np.mean([simple_results.get(s, {}).get('ap', 0) for s in source_names])
    attn_mean = np.mean([attn_results.get(s, {}).get('ap', 0) for s in source_names])
    print(f"{'平均':<12} {orig_mean:>10.4f} {simp_mean:>10.4f} {attn_mean:>12.4f} "
          f"{simp_mean-orig_mean:>+10.4f} {attn_mean-orig_mean:>+12.4f}")

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
        'results': convert(results),
        'original_baseline': convert(original_usc_best),
        'n_domains': len(source_names),
        'n_components': K_PCA,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
