# -*- coding: utf-8 -*-
"""频域滤波校正实验
核心思想：在DCT频域空间里，用不同的校正操作（而不是秩2仿射）：
  - 低频维度（整体结构、类别语义）：强校正
  - 高频维度（细节、纹理、噪声）：弱校正或不校正
  - 校正强度随频率衰减：w_k = exp(-k/K) 或 w_k = 1/(1 + k/K)
对比：
  1. 原始空间秩2仿射（baseline）
  2. DCT空间秩2仿射（协变，应该和原始空间一样）
  3. DCT空间频域滤波校正（低频强校正、高频弱校正）
  4. DCT空间只校正低频（高频保持不变）
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
from scipy.fft import dct, idct

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate,
)
import tge_probe as T

OUT = DATA / 'usc_dct_filtering_results.json'


def dct_transform(x):
    """DCT变换（沿最后一个维度）"""
    return dct(x, type=2, norm='ortho', axis=-1)


def idct_transform(x):
    """逆DCT变换（沿最后一个维度）"""
    return idct(x, type=2, norm='ortho', axis=-1)


def get_frequency_weights(n, mode='exp', decay=0.02):
    """获取频率衰减权重
    mode:
      - 'exp': w_k = exp(-decay * k)
      - 'inv': w_k = 1 / (1 + decay * k)
      - 'step': w_k = 1 if k < n*cutoff else 0
    """
    k = np.arange(n)
    if mode == 'exp':
        w = np.exp(-decay * k)
    elif mode == 'inv':
        w = 1.0 / (1.0 + decay * k)
    elif mode == 'step':
        cutoff = 0.3  # 只校正前30%的低频
        w = (k < n * cutoff).astype(np.float32)
    else:
        w = np.ones(n)
    return w.astype(np.float32)


class DCTFilteringUSC:
    """DCT频域滤波校正USC"""
    def __init__(self, filter_mode='exp', decay=0.02, use_pca=True, n_components=30):
        self.filter_mode = filter_mode
        self.decay = decay
        self.use_pca = use_pca
        self.n_components = n_components
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        self.w_orth_ = None
        self.freq_weights_ = None
        self._v_gpu = None
        self._w_gpu = None
        self._freq_weights_gpu = None

    def fit(self, source_features, target_features):
        # 先做DCT变换
        src_dct = dct_transform(source_features)
        tgt_dct = dct_transform(target_features)

        # 频率衰减权重
        D = src_dct.shape[1]
        self.freq_weights_ = get_frequency_weights(D, self.filter_mode, self.decay)
        print(f"    频率权重: mode={self.filter_mode}, decay={self.decay}, "
              f"前5个={self.freq_weights_[:5]}, 后5个={self.freq_weights_[-5:]}")

        if self.use_pca:
            # 在DCT空间做PCA降维
            all_feat = np.vstack([src_dct, tgt_dct])
            mean_ = all_feat.mean(axis=0)
            X_centered = all_feat - mean_
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            K = min(self.n_components, Vt.shape[0])
            self.components_ = Vt[:K]
            self.mean_ = mean_

            # 投影到PCA子空间
            src_proj = (src_dct - mean_) @ self.components_.T
            tgt_proj = (tgt_dct - mean_) @ self.components_.T
        else:
            # 不做PCA，直接用完整DCT空间
            self.components_ = None
            self.mean_ = None
            src_proj = src_dct
            tgt_proj = tgt_dct

        # 计算域偏移方向
        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros_like(shift)

        # v的主正交方向w
        v = self.domain_shift_
        X_centered = src_proj - src_proj.mean(axis=0)
        cov = X_centered.T @ X_centered
        P_perp = np.eye(len(v)) - np.outer(v, v)
        cov_perp = P_perp @ cov @ P_perp
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
        self._v_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        self._w_gpu = torch.tensor(self.w_orth_, dtype=torch.float32, device=device)
        self._freq_weights_gpu = torch.tensor(self.freq_weights_, dtype=torch.float32, device=device)
        if self.components_ is not None:
            self._components_gpu = torch.tensor(self.components_, dtype=torch.float32, device=device)
            self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)

        print(f"    域偏移范数: {self.domain_shift_norm_:.4f}, "
              f"v·w正交性: {np.dot(self.domain_shift_, self.w_orth_):.8f}")
        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0, theta_deg=0.0):
        # DCT变换
        p_dct = torch.tensor(dct_transform(p_gpu.cpu().numpy()), dtype=torch.float32, device=device)

        if self.components_ is not None:
            # 投影到PCA子空间
            Z = (p_dct - self._mean_gpu) @ self._components_gpu.T
        else:
            Z = p_dct

        v = self._v_gpu
        w = self._w_gpu
        theta = theta_deg * np.pi / 180.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # 在(v,w)平面内旋转
        vZ = torch.dot(Z, v)
        wZ = torch.dot(Z, w)
        vZ_rot = cos_t * vZ - sin_t * wZ
        wZ_rot = sin_t * vZ + cos_t * wZ
        Z_rot = Z - vZ * v - wZ * w + vZ_rot * v + wZ_rot * w

        # 沿v方向缩放（秩1仿射）
        vZ_new = torch.dot(Z_rot, v)
        Z_scaled = Z_rot + gamma * vZ_new * v

        # 沿v方向平移
        Z_corr = Z_scaled + alpha * v

        # 回投影到DCT空间
        if self.components_ is not None:
            p_dct_corr = Z_corr @ self._components_gpu + self._mean_gpu
        else:
            p_dct_corr = Z_corr

        # 频域滤波：用频率权重混合原始和校正后的DCT系数
        # 低频：强校正（权重高，用校正后的）
        # 高频：弱校正（权重低，用原始的）
        p_dct_final = self._freq_weights_gpu * p_dct_corr + (1.0 - self._freq_weights_gpu) * p_dct

        # 逆DCT变换
        p_final = torch.tensor(idct_transform(p_dct_final.cpu().numpy()), dtype=torch.float32, device=device)
        p_final = p_final / torch.clamp(torch.norm(p_final), min=1e-8)
        return p_final


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
    print(f"  拟合USC（DCT频域滤波, mode={usc.filter_mode}）...")
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

    # 原始空间秩2仿射的最佳结果（从之前的实验）
    original_best = {
        'DIOR': 0.3884, 'NWPU': 0.3910, 'DOTA': 0.3903, 'VisDrone': 0.3443,
    }

    # 测试不同的频域滤波配置
    filter_configs = [
        ('exp_0.01', 'exp', 0.01, True, 30),
        ('exp_0.02', 'exp', 0.02, True, 30),
        ('exp_0.05', 'exp', 0.05, True, 30),
        ('inv_0.01', 'inv', 0.01, True, 30),
        ('inv_0.05', 'inv', 0.05, True, 30),
        ('step_0.3', 'step', 0.3, True, 30),
        ('no_filter', 'none', 0.0, True, 30),  # 无滤波，应该和原始空间一样
    ]

    all_results = {}

    for config_name, filter_mode, decay, use_pca, n_components in filter_configs:
        print(f"\n{'='*60}")
        print(f"配置: {config_name} (mode={filter_mode}, decay={decay})")
        print(f"{'='*60}")
        config_results = {}
        for src in source_names:
            print(f"\n--- {src}→COCO ---")
            usc = DCTFilteringUSC(filter_mode=filter_mode, decay=decay,
                                   use_pca=use_pca, n_components=n_components)
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
            print(f"  原始空间最佳: AP={original_best.get(src, 0):.4f}")
            gain = best_ap - original_best.get(src, 0)
            print(f"  提升: {gain:+.4f}")
            config_results[src] = {
                'ap': best_ap,
                'params': {'gamma': best_params[0], 'alpha': best_params[1], 'theta': best_params[2]},
                'gain_vs_original': gain,
            }
            del data, usc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results[config_name] = config_results

    # 汇总对比
    print(f"\n{'='*60}")
    print("汇总：各频域滤波配置对比")
    print(f"{'='*60}")
    print(f"{'配置':<15} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8} {'平均提升':>10}")
    print("-" * 75)
    for config_name, _, _, _, _ in filter_configs:
        results = all_results[config_name]
        aps = [results.get(s, {}).get('ap', 0) for s in source_names]
        gains = [results.get(s, {}).get('gain_vs_original', 0) for s in source_names]
        mean_ap = np.mean(aps)
        mean_gain = np.mean(gains)
        print(f"{config_name:<15} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {mean_ap:>8.4f} {mean_gain:>+10.4f}")

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
        'note': 'DCT频域滤波校正：低频强校正、高频弱校正',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
