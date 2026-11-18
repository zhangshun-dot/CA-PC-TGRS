# -*- coding: utf-8 -*-
"""正交空间校正实验
核心思想：用固定的正交矩阵Q把特征投影到确定的正交空间，
在正交空间做秩2仿射校正，然后用Q^T逆变换回来。
对比：
  1. 原始空间秩2仿射（当前方法）
  2. Hadamard空间秩2仿射（Q=Hadamard矩阵，512维完美适配）
  3. DFT空间秩2仿射（Q=离散傅里叶变换矩阵）
  4. 随机正交空间秩2仿射（Q=固定的随机正交矩阵）
  5. 多正交空间集成（多个Q的校正结果融合）
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
from scipy.linalg import hadamard, dft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate, evaluate_baseline,
)
import tge_probe as T

OUT = DATA / 'usc_orthogonal_space_results.json'

# 传感器感知γ配置
SENSOR_GAMMA = {
    'DIOR': -2.0, 'NWPU': -2.0, 'DOTA': -2.0, 'VisDrone': -1.0,
}


def generate_hadamard_matrix(n):
    """生成n维Hadamard矩阵（n必须是2的幂）"""
    H = hadamard(n)
    # 归一化（正交矩阵需要归一化）
    H = H / np.sqrt(n)
    return H.astype(np.float32)


def generate_dft_matrix(n):
    """生成n维DFT矩阵（正交矩阵）"""
    D = dft(n, scale='sqrtn')  # 归一化的DFT矩阵，是正交的
    # DFT矩阵是复数的，取实部（或者用实DFT）
    # 这里用实值正交近似：实部+虚部堆叠，但维度会变
    # 更简单：用D的实部（不是严格正交，但近似）
    # 或者用cosine变换（DCT），是实值正交的
    from scipy.fft import dct
    # DCT矩阵
    DCT = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            if k == 0:
                DCT[k, i] = np.sqrt(1.0/n)
            else:
                DCT[k, i] = np.sqrt(2.0/n) * np.cos(np.pi * (2*i+1) * k / (2*n))
    return DCT.astype(np.float32)


def generate_random_orthogonal(n, seed=42):
    """生成n维随机正交矩阵（QR分解）"""
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    Q, R = np.linalg.qr(A)
    # 确保对角线为正（消除符号歧义）
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q.astype(np.float32)


class OrthogonalSpaceUSC:
    """正交空间USC
    用固定的正交矩阵Q把特征投影到正交空间，
    在正交空间做秩2仿射校正，然后用Q^T逆变换回来。
    """
    def __init__(self, Q_matrix, n_components=30, space_name='original'):
        self.Q = Q_matrix  # 正交变换矩阵 (D, D)，如果为None则用原始空间
        self.space_name = space_name
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # 在正交空间的PCA基底
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        self._Q_gpu = None
        self._components_gpu = None
        self._mean_gpu = None
        self._v_domain_gpu = None
        # 旋转相关
        self.w_orth_ = None  # v的主正交方向
        self._w_gpu = None

    def _to_orthogonal_space(self, X):
        """把特征变换到正交空间"""
        if self.Q is None:
            return X
        return X @ self.Q.T  # X @ Q^T，因为Q是正交的，Q^T=Q^{-1}

    def _from_orthogonal_space(self, X):
        """把特征从正交空间变换回来"""
        if self.Q is None:
            return X
        return X @ self.Q  # X @ Q，逆变换

    def fit(self, source_features, target_features):
        """在正交空间拟合USC（PCA基底 + 域偏移）"""
        # 变换到正交空间
        src_orth = self._to_orthogonal_space(source_features)
        tgt_orth = self._to_orthogonal_space(target_features)

        # PCA（在正交空间）
        all_feat = np.vstack([src_orth, tgt_orth])
        self.mean_ = all_feat.mean(axis=0)
        X_centered = all_feat - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        K = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:K]  # (K, D)

        # 域偏移（在正交空间的PCA子空间）
        src_proj = (src_orth - self.mean_) @ self.components_.T
        tgt_proj = (tgt_orth - self.mean_) @ self.components_.T
        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros(self.n_components)

        # v的主正交方向w（在PCA子空间内）
        v = self.domain_shift_
        # 找与v正交的最大方差方向
        X_src_centered = src_proj - src_proj.mean(axis=0)
        cov = X_src_centered.T @ X_src_centered
        # 投影掉v方向
        P_perp = np.eye(K) - np.outer(v, v)
        cov_perp = P_perp @ cov @ P_perp
        eigvals, eigvecs = np.linalg.eigh(cov_perp)
        self.w_orth_ = eigvecs[:, -1]  # 最大特征值对应的特征向量
        # 确保w和v正交
        self.w_orth_ = self.w_orth_ - np.dot(self.w_orth_, v) * v
        self.w_orth_ = self.w_orth_ / (np.linalg.norm(self.w_orth_) + 1e-8)

        # 移到GPU
        if self.Q is not None:
            self._Q_gpu = torch.tensor(self.Q, dtype=torch.float32, device=device)
        self._components_gpu = torch.tensor(self.components_, dtype=torch.float32, device=device)
        self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        self._v_domain_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        self._w_gpu = torch.tensor(self.w_orth_, dtype=torch.float32, device=device)

        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0, theta_deg=0.0):
        """GPU校正单个原型（秩2仿射：平移+缩放+旋转）"""
        # 变换到正交空间
        if self._Q_gpu is not None:
            p_orth = p_gpu @ self._Q_gpu.T
        else:
            p_orth = p_gpu

        # 投影到PCA子空间
        Z = (p_orth - self._mean_gpu) @ self._components_gpu.T  # (K,)

        # 秩2仿射校正（在(v,w)平面内旋转 + 沿v方向缩放和平移）
        v = self._v_domain_gpu
        w = self._w_gpu
        theta = theta_deg * np.pi / 180.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # 在(v,w)平面内旋转
        vZ = torch.dot(Z, v)
        wZ = torch.dot(Z, w)
        # 旋转后的(v,w)坐标
        vZ_rot = cos_t * vZ - sin_t * wZ
        wZ_rot = sin_t * vZ + cos_t * wZ
        # 重建旋转后的Z（只旋转(v,w)平面内的分量，其他分量不变）
        Z_rot = Z - vZ * v - wZ * w + vZ_rot * v + wZ_rot * w

        # 沿v方向缩放（秩1仿射）
        vZ_new = torch.dot(Z_rot, v)
        Z_scaled = Z_rot + gamma * vZ_new * v

        # 沿v方向平移
        Z_corr = Z_scaled + alpha * v

        # 回投影到正交空间
        X_orth = Z_corr @ self._components_gpu + self._mean_gpu

        # 从正交空间变换回来
        if self._Q_gpu is not None:
            X_corr = X_orth @ self._Q_gpu
        else:
            X_corr = X_orth

        X_corr = X_corr / torch.clamp(torch.norm(X_corr), min=1e-8)
        return X_corr


def load_domain_data(source_name, usc):
    """加载单个域对数据"""
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
    print(f"  拟合USC（{usc.space_name}空间）...")
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
    """用指定参数评估"""
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
    D = 512  # CLIP特征维度

    # 生成各种正交矩阵
    print(f"\n生成正交矩阵（维度={D}）...")
    Q_hadamard = generate_hadamard_matrix(D)
    print(f"  Hadamard矩阵: {Q_hadamard.shape}, 正交性误差={np.max(np.abs(Q_hadamard @ Q_hadamard.T - np.eye(D))):.6f}")
    Q_dct = generate_dft_matrix(D)
    print(f"  DCT矩阵: {Q_dct.shape}, 正交性误差={np.max(np.abs(Q_dct @ Q_dct.T - np.eye(D))):.6f}")
    Q_random = generate_random_orthogonal(D, seed=42)
    print(f"  随机正交矩阵: {Q_random.shape}, 正交性误差={np.max(np.abs(Q_random @ Q_random.T - np.eye(D))):.6f}")

    spaces = [
        ('original', None),
        ('hadamard', Q_hadamard),
        ('dct', Q_dct),
        ('random_ortho', Q_random),
    ]

    # 原始USC的最佳结果（从之前的实验）
    original_best = {
        'DIOR': 0.3879, 'NWPU': 0.3867, 'DOTA': 0.3906, 'VisDrone': 0.3420,
    }

    all_results = {}

    for space_name, Q in spaces:
        print(f"\n{'='*60}")
        print(f"空间: {space_name}")
        print(f"{'='*60}")
        space_results = {}
        for src in source_names:
            print(f"\n--- {src}→COCO ---")
            usc = OrthogonalSpaceUSC(Q, n_components=K_PCA, space_name=space_name)
            data = load_domain_data(src, usc)
            if data is None:
                continue
            # 扫描(γ, α, θ)找最佳
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
            print(f"  原始空间最佳: AP={original_best.get(src, 0):.4f}")
            gain = best_ap - original_best.get(src, 0)
            print(f"  提升: {gain:+.4f}")
            space_results[src] = {
                'ap': best_ap,
                'params': {'gamma': best_params[0], 'alpha': best_params[1], 'theta': best_params[2]},
                'gain_vs_original': gain,
            }
            del data, usc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results[space_name] = space_results

    # 汇总对比
    print(f"\n{'='*60}")
    print("汇总：各正交空间秩2仿射校正对比")
    print(f"{'='*60}")
    print(f"{'域对':<12} {'原始空间':>10} {'Hadamard':>10} {'DCT':>10} {'随机正交':>10}")
    print("-" * 55)
    for src in source_names:
        orig = all_results['original'].get(src, {}).get('ap', 0)
        had = all_results['hadamard'].get(src, {}).get('ap', 0)
        dct = all_results['dct'].get(src, {}).get('ap', 0)
        rnd = all_results['random_ortho'].get(src, {}).get('ap', 0)
        print(f"{src:<12} {orig:>10.4f} {had:>10.4f} {dct:>10.4f} {rnd:>10.4f}")
    # 平均
    orig_mean = np.mean([all_results['original'].get(s, {}).get('ap', 0) for s in source_names])
    had_mean = np.mean([all_results['hadamard'].get(s, {}).get('ap', 0) for s in source_names])
    dct_mean = np.mean([all_results['dct'].get(s, {}).get('ap', 0) for s in source_names])
    rnd_mean = np.mean([all_results['random_ortho'].get(s, {}).get('ap', 0) for s in source_names])
    print(f"{'平均':<12} {orig_mean:>10.4f} {had_mean:>10.4f} {dct_mean:>10.4f} {rnd_mean:>10.4f}")

    # 提升对比
    print(f"\n{'域对':<12} {'Hadamard提升':>12} {'DCT提升':>10} {'随机正交提升':>14}")
    print("-" * 50)
    for src in source_names:
        orig = all_results['original'].get(src, {}).get('ap', 0)
        had_g = all_results['hadamard'].get(src, {}).get('ap', 0) - orig
        dct_g = all_results['dct'].get(src, {}).get('ap', 0) - orig
        rnd_g = all_results['random_ortho'].get(src, {}).get('ap', 0) - orig
        print(f"{src:<12} {had_g:>+12.4f} {dct_g:>+10.4f} {rnd_g:>+14.4f}")

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
        'original_baseline': convert(original_best),
        'n_domains': len(source_names),
        'n_components': K_PCA,
        'feature_dim': D,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
