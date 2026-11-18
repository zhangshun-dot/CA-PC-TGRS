# -*- coding: utf-8 -*-
"""秩1仿射USC实验（优化版：数据只加载一次，γ/α扫描只做校正）
Z_corr = (I + γ·v_domain·v_domain^T)·Z + α·v_domain
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate, evaluate_baseline,
)
import tge_probe as T

OUT = DATA / 'usc_rank1_affine_results.json'
GAMMAS = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 5.0]


class AffineUSC:
    def __init__(self, n_components=30):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        self._components_gpu = None
        self._mean_gpu = None
        self._v_domain_gpu = None

    def fit(self, source_features, target_features):
        all_features = np.vstack([source_features, target_features])
        self.mean_ = all_features.mean(axis=0)
        X_centered = all_features - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        K = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:K]
        src_proj = (source_features - self.mean_) @ self.components_.T
        tgt_proj = (target_features - self.mean_) @ self.components_.T
        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros(K)
        self._components_gpu = torch.tensor(self.components_, dtype=torch.float32, device=device)
        self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        self._v_domain_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0):
        """GPU校正单个原型：Z_corr = (I + γ·v·v^T)·Z + α·v"""
        Z = (p_gpu - self._mean_gpu) @ self._components_gpu.T  # (K,)
        vZ = torch.dot(Z, self._v_domain_gpu)  # scalar
        Z_affine = Z + gamma * vZ * self._v_domain_gpu  # (K,)
        Z_corr = Z_affine + alpha * self._v_domain_gpu  # (K,)
        X_corr = Z_corr @ self._components_gpu + self._mean_gpu  # (D,)
        X_corr = X_corr / torch.clamp(torch.norm(X_corr), min=1e-8)
        return X_corr


def load_domain_data(source_name):
    """加载单个域对的全部数据并缓存"""
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
    print(f"  拟合USC...")
    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
    usc = AffineUSC(n_components=K_PCA).fit(all_src, all_tgt)
    all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}
    # 预计算文本外推方向（每个类别）
    text_deltas = {}
    for eval_name, coco_cls, src_cls in valid_targets:
        s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
        d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
        text_deltas[src_cls] = d_emb - s_emb
    # 原型移到GPU
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
    """用指定γ,α评估（数据已缓存，只做校正和评估）"""
    usc = domain_data['usc']
    target_data = domain_data['target_data']
    all_protos = domain_data['all_protos']
    protos_gpu = domain_data['protos_gpu']
    valid_targets = domain_data['valid_targets']
    text_deltas = domain_data['text_deltas']

    per_class_ap = {}
    for eval_name, coco_cls, src_cls in valid_targets:
        # GPU校正正原型
        p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, alpha)
        p_corr = p_corr_gpu.cpu().numpy()
        # 负原型校正
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, gamma, alpha)
            neg = neg_corr_gpu.cpu().numpy()
        else:
            neg = None
        # 文本外推
        delta = text_deltas[src_cls]
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

    # 阶段1：加载所有域对数据（只做一次）
    print(f"\n{'='*60}")
    print("阶段1：加载所有域对数据（只做一次）")
    print(f"{'='*60}")
    all_domain_data = {}
    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data(src)
        if data is not None:
            all_domain_data[src] = data
            print(f"  完成，有效类别: {len(data['valid_targets'])}, delta_norm={data['usc'].domain_shift_norm_:.4f}")

    # 阶段2：γ×α扫描
    print(f"\n{'='*60}")
    print("阶段2：γ×α网格扫描")
    print(f"{'='*60}")

    all_results = {}
    for src in all_domain_data:
        print(f"\n--- {src}→COCO ---")
        domain_data = all_domain_data[src]
        domain_results = {}
        best_ap = -1
        best_gamma = 0.0
        best_alpha = 0.0

        for gamma in GAMMAS:
            for alpha in ALPHAS:
                per_class, avg = evaluate_with_params(domain_data, gamma, alpha)
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

        baseline_best_ap = max(
            domain_results[f"γ=0.0_α={a}"]['avg_ap'] for a in ALPHAS
            if f"γ=0.0_α={a}" in domain_results
        )
        print(f"\n  原始USC最佳(γ=0): {baseline_best_ap:.4f}")
        print(f"  秩1仿射最佳(γ={best_gamma}, α={best_alpha}): {best_ap:.4f}")
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU显存已释放")


if __name__ == '__main__':
    main()
