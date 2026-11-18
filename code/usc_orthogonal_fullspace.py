# -*- coding: utf-8 -*-
"""变体B：直接在完整正交空间做秩2仿射校正（不做PCA降维）
核心思想：不在正交空间里做PCA，直接在完整512维空间做秩2仿射校正。
理论预测：结果应该和原始空间完全一样（因为秩2仿射在正交变换下协变）。
如果不一样，就是重要发现！
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
from scipy.linalg import hadamard

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate,
)
import tge_probe as T

OUT = DATA / 'usc_orthogonal_fullspace_results.json'


def generate_hadamard_matrix(n):
    H = hadamard(n)
    H = H / np.sqrt(n)
    return H.astype(np.float32)


def generate_dct_matrix(n):
    from scipy.fft import dct
    DCT = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            if k == 0:
                DCT[k, i] = np.sqrt(1.0/n)
            else:
                DCT[k, i] = np.sqrt(2.0/n) * np.cos(np.pi * (2*i+1) * k / (2*n))
    return DCT.astype(np.float32)


def generate_random_orthogonal(n, seed=42):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q.astype(np.float32)


class FullSpaceUSC:
    """直接在完整空间做秩2仿射校正（不做PCA降维）"""
    def __init__(self, Q_matrix=None, space_name='original'):
        self.Q = Q_matrix  # 正交变换矩阵，如果为None则用原始空间
        self.space_name = space_name
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        self.w_orth_ = None
        self._Q_gpu = None
        self._v_gpu = None
        self._w_gpu = None

    def fit(self, source_features, target_features):
        # 变换到正交空间（如果Q不为None）
        if self.Q is not None:
            src = source_features @ self.Q.T
            tgt = target_features @ self.Q.T
        else:
            src = source_features
            tgt = target_features

        # 计算域偏移方向（完整空间，不降维）
        shift = tgt.mean(axis=0) - src.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros_like(shift)

        # v的主正交方向w（完整空间）
        v = self.domain_shift_
        # 用源域特征的协方差矩阵，找与v正交的最大方差方向
        X_centered = src - src.mean(axis=0)
        cov = X_centered.T @ X_centered
        P_perp = np.eye(len(v)) - np.outer(v, v)
        cov_perp = P_perp @ cov @ P_perp
        # 用幂迭代找最大特征值对应的特征向量（避免完整eigh计算512x512）
        w = np.random.randn(len(v))
        w = w - np.dot(w, v) * v
        w = w / (np.linalg.norm(w) + 1e-8)
        for _ in range(50):
            w_new = cov_perp @ w
            w_new = w_new - np.dot(w_new, v) * v
            w_new = w_new / (np.linalg.norm(w_new) + 1e-8)
            if np.linalg.norm(w_new - w) < 1e-6:
                break
            w = w_new
        self.w_orth_ = w

        # 移到GPU
        if self.Q is not None:
            self._Q_gpu = torch.tensor(self.Q, dtype=torch.float32, device=device)
        self._v_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        self._w_gpu = torch.tensor(self.w_orth_, dtype=torch.float32, device=device)

        print(f"    域偏移范数: {self.domain_shift_norm_:.4f}, "
              f"v·w正交性: {np.dot(self.domain_shift_, self.w_orth_):.8f}")
        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0, theta_deg=0.0):
        # 变换到正交空间
        if self._Q_gpu is not None:
            p_orth = p_gpu @ self._Q_gpu.T
        else:
            p_orth = p_gpu

        v = self._v_gpu
        w = self._w_gpu
        theta = theta_deg * np.pi / 180.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # 在(v,w)平面内旋转
        vZ = torch.dot(p_orth, v)
        wZ = torch.dot(p_orth, w)
        vZ_rot = cos_t * vZ - sin_t * wZ
        wZ_rot = sin_t * vZ + cos_t * wZ
        Z_rot = p_orth - vZ * v - wZ * w + vZ_rot * v + wZ_rot * w

        # 沿v方向缩放（秩1仿射）
        vZ_new = torch.dot(Z_rot, v)
        Z_scaled = Z_rot + gamma * vZ_new * v

        # 沿v方向平移
        Z_corr = Z_scaled + alpha * v

        # 从正交空间变换回来
        if self._Q_gpu is not None:
            X_corr = Z_corr @ self._Q_gpu
        else:
            X_corr = Z_corr

        X_corr = X_corr / torch.clamp(torch.norm(X_corr), min=1e-8)
        return X_corr


def load_domain_data(source_name, usc):
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
    print(f"  拟合USC（{usc.space_name}空间，完整维度，无PCA）...")
    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
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
    }


def evaluate_with_params(domain_data, gamma, alpha, theta_deg=0.0):
    usc = domain_data['usc']
    target_data = domain_data['target_data']
    all_protos = domain_data['all_protos']
    protos_gpu = domain_data['protos_gpu']
    text_deltas = domain_data['text_deltas']
    valid_targets = domain_data['valid_targets']
    aps = []
    for eval_name, coco_cls, src_cls in valid_targets:
        p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, alpha, theta_deg)
        p_corr = p_corr_gpu.cpu().numpy()
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, gamma, alpha, theta_deg)
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
    D = 512

    print(f"\n生成正交矩阵（维度={D}）...")
    Q_hadamard = generate_hadamard_matrix(D)
    Q_dct = generate_dct_matrix(D)
    Q_random = generate_random_orthogonal(D, seed=42)

    spaces = [
        ('original', None),
        ('hadamard', Q_hadamard),
        ('dct', Q_dct),
        ('random_ortho', Q_random),
    ]

    # 变体A的结果（PCA降维版），用于对比
    variant_a_results = {
        'DIOR': 0.3884, 'NWPU': 0.3910, 'DOTA': 0.3903, 'VisDrone': 0.3443,
    }

    all_results = {}

    for space_name, Q in spaces:
        print(f"\n{'='*60}")
        print(f"空间: {space_name}（完整维度，无PCA降维）")
        print(f"{'='*60}")
        space_results = {}
        for src in source_names:
            print(f"\n--- {src}→COCO ---")
            usc = FullSpaceUSC(Q, space_name=space_name)
            data = load_domain_data(src, usc)
            if data is None:
                continue
            best_ap = -1
            best_params = None
            gammas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
            thetas = [-30, -15, 0, 15, 30]
            for gamma in gammas:
                for alpha in ALPHAS:
                    for theta in thetas:
                        ap = evaluate_with_params(data, gamma, alpha, theta)
                        if ap > best_ap:
                            best_ap = ap
                            best_params = (gamma, alpha, theta)
            print(f"  最佳: γ={best_params[0]}, α={best_params[1]}, θ={best_params[2]}°, AP={best_ap:.4f}")
            print(f"  变体A(PCA版)最佳: AP={variant_a_results.get(src, 0):.4f}")
            diff = best_ap - variant_a_results.get(src, 0)
            print(f"  与变体A差异: {diff:+.4f}")
            space_results[src] = {
                'ap': best_ap,
                'params': {'gamma': best_params[0], 'alpha': best_params[1], 'theta': best_params[2]},
                'diff_vs_variant_a': diff,
            }
            del data, usc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results[space_name] = space_results

    # 汇总对比
    print(f"\n{'='*60}")
    print("汇总：变体B（完整空间，无PCA） vs 变体A（PCA降维）")
    print(f"{'='*60}")
    print(f"{'域对':<12} {'变体A(PCA)':>12} {'原始空间B':>12} {'Hadamard B':>12} {'DCT B':>10} {'随机正交B':>12}")
    print("-" * 75)
    for src in source_names:
        va = variant_a_results.get(src, 0)
        orig = all_results['original'].get(src, {}).get('ap', 0)
        had = all_results['hadamard'].get(src, {}).get('ap', 0)
        dct = all_results['dct'].get(src, {}).get('ap', 0)
        rnd = all_results['random_ortho'].get(src, {}).get('ap', 0)
        print(f"{src:<12} {va:>12.4f} {orig:>12.4f} {had:>12.4f} {dct:>10.4f} {rnd:>12.4f}")

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
        'all_results': convert(all_results),
        'variant_a_results': convert(variant_a_results),
        'n_domains': len(source_names),
        'feature_dim': D,
        'note': '变体B：直接在完整512维正交空间做秩2仿射校正，不做PCA降维',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
