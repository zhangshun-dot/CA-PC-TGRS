# -*- coding: utf-8 -*-
"""秩1仿射USC实验（PyTorch GPU版）
Z_corr = (I + γ·v_domain·v_domain^T)·Z + α·v_domain
只有2个参数：γ（各向异性缩放系数）和α（平移系数）
扫描γ和α的组合，对比原始USC（γ=0）
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate, evaluate_baseline,
)
import tge_probe as T

OUT = DATA / 'usc_rank1_affine_results.json'

# γ扫描范围
GAMMAS = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 5.0]


class AffineUSC:
    """秩1仿射USC：Z_corr = (I + γ·v·v^T)·Z + α·v"""
    def __init__(self, n_components=30):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        # GPU张量
        self._components_gpu = None
        self._mean_gpu = None
        self._v_domain_gpu = None
        self._vvt_gpu = None

    def fit(self, source_features, target_features):
        all_features = np.vstack([source_features, target_features])
        self.mean_ = all_features.mean(axis=0)
        X_centered = all_features - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        K = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:K]
        src_proj = self._project(source_features)
        tgt_proj = self._project(target_features)
        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros(K)

        # 移到GPU
        self._components_gpu = torch.tensor(self.components_, dtype=torch.float32, device=device)
        self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        self._v_domain_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        # v·v^T (K×K)
        self._vvt_gpu = torch.outer(self._v_domain_gpu, self._v_domain_gpu)
        return self

    def _project(self, X):
        return (X - self.mean_) @ self.components_.T

    def _reconstruct(self, Z):
        return Z @ self.components_ + self.mean_

    def correct_gpu(self, X_gpu, gamma=0.0, alpha=1.0):
        """GPU版校正：Z_corr = (I + γ·v·v^T)·Z + α·v"""
        # 投影到低秩空间
        Z = (X_gpu - self._mean_gpu) @ self._components_gpu.T  # (N, K)
        # 秩1仿射修正：(I + γ·v·v^T)·Z = Z + γ·v·(v^T·Z)
        vZ = Z @ self._v_domain_gpu  # (N,)
        Z_affine = Z + gamma * torch.outer(vZ, self._v_domain_gpu)  # (N, K)
        # 平移校正
        Z_corr = Z_affine + alpha * self._v_domain_gpu  # (N, K)
        # 回投影
        X_corr = Z_corr @ self._components_gpu + self._mean_gpu  # (N, D)
        # L2归一化
        norms = torch.norm(X_corr, dim=1, keepdim=True)
        X_corr = X_corr / torch.clamp(norms, min=1e-8)
        return X_corr

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0):
        return self.correct_gpu(p_gpu.unsqueeze(0), gamma, alpha)[0]


def run_domain_experiment(source_name, gamma, alpha):
    """跑单个域对，指定γ和α"""
    load_func, targets = SOURCE_LOADERS[source_name]

    # 加载源域
    source_data = {}
    for eval_name, coco_cls, src_cls in targets:
        data = load_func(src_cls, n=50)
        if data is not None:
            source_data[src_cls] = data

    # 加载目标域
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
        return None, None

    # 拟合仿射USC
    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
    usc = AffineUSC(n_components=K_PCA).fit(all_src, all_tgt)
    all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}

    # 原型移到GPU
    protos_gpu = {s: torch.tensor(p, dtype=torch.float32, device=device)
                   for s, p in all_protos.items()}

    per_class_ap = {}
    for eval_name, coco_cls, src_cls in valid_targets:
        # 仿射校正正原型
        p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, alpha)
        p_corr = p_corr_gpu.cpu().numpy()

        # 负原型（也做同样校正）
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, gamma, alpha)
            neg = neg_corr_gpu.cpu().numpy()
        else:
            neg = None

        # 文本外推（用原始模板）
        s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
        d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
        delta = d_emb - s_emb
        p_hat = p_corr + 1.0 * delta
        p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)

        ap = evaluate(target_data[coco_cls], p_hat, neg)
        per_class_ap[coco_cls] = float(ap)

    avg_ap = float(np.mean(list(per_class_ap.values())))
    return per_class_ap, avg_ap


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"\n域对: {source_names}")
    print(f"γ范围: {GAMMAS}")
    print(f"α范围: {ALPHAS}")
    print(f"总组合数: {len(GAMMAS)} × {len(ALPHAS)} = {len(GAMMAS)*len(ALPHAS)}")

    all_results = {}

    # 对每个域对，扫描γ×α
    for src in source_names:
        print(f"\n{'='*60}")
        print(f"域对: {src}→COCO")
        print(f"{'='*60}")

        domain_results = {}
        best_ap = -1
        best_gamma = 0.0
        best_alpha = 0.0

        for gamma in GAMMAS:
            for alpha in ALPHAS:
                per_class, avg = run_domain_experiment(src, gamma, alpha)
                if per_class is None:
                    continue
                key = f"γ={gamma}_α={alpha}"
                domain_results[key] = {
                    'gamma': gamma, 'alpha': alpha,
                    'avg_ap': avg, 'per_class': per_class,
                }
                if avg > best_ap:
                    best_ap = avg
                    best_gamma = gamma
                    best_alpha = alpha
                print(f"  γ={gamma:>5.1f}, α={alpha:>4.1f}: AP50={avg:.4f}"
                      f"{'  ← best' if avg == best_ap else ''}")

        # 原始USC（γ=0）的最佳α
        baseline_best_ap = max(
            domain_results[f"γ=0.0_α={a}"]['avg_ap'] for a in ALPHAS
            if f"γ=0.0_α={a}" in domain_results
        )

        print(f"\n  原始USC最佳(γ=0): AP50={baseline_best_ap:.4f}")
        print(f"  秩1仿射最佳(γ={best_gamma}, α={best_alpha}): AP50={best_ap:.4f}")
        print(f"  增益: {best_ap - baseline_best_ap:+.4f}")

        all_results[src] = {
            'sweep_results': domain_results,
            'best_gamma': best_gamma,
            'best_alpha': best_alpha,
            'best_ap': best_ap,
            'baseline_best_ap': baseline_best_ap,
            'gain': best_ap - baseline_best_ap,
        }

    # 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    print(f"{'域对':<12} {'原始USC最佳':>12} {'秩1仿射最佳':>12} {'增益':>10} {'最佳(γ,α)':>15}")
    print("-" * 65)
    for src, r in all_results.items():
        print(f"{src:<12} {r['baseline_best_ap']:>12.4f} {r['best_ap']:>12.4f} "
              f"{r['gain']:>+10.4f} ({r['best_gamma']:>4.1f},{r['best_alpha']:>4.1f})")

    avg_baseline = np.mean([r['baseline_best_ap'] for r in all_results.values()])
    avg_affine = np.mean([r['best_ap'] for r in all_results.values()])
    print("-" * 65)
    print(f"{'平均':<12} {avg_baseline:>12.4f} {avg_affine:>12.4f} {avg_affine-avg_baseline:>+10.4f}")

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

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    # 释放GPU显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU显存已释放")


if __name__ == '__main__':
    main()
