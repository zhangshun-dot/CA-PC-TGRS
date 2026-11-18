# -*- coding: utf-8 -*-
"""傅里叶变换频域校正实验
核心思想：用傅里叶变换（DFT）代替DCT，利用傅里叶变换的幅值+相位信息
方向1：傅里叶幅值空间校正（只在幅值空间做秩2仿射，相位保持不变）
方向2：傅里叶幅值+相位分别校正
方向3：傅里叶空间多尺度校正
对比：
  1. 原始空间秩2仿射（baseline）
  2. DCT空间秩2仿射（协变，应该和原始空间一样）
  3. 傅里叶幅值空间秩2仿射
  4. 傅里叶幅值+相位分别校正
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

OUT = DATA / 'usc_fourier_results.json'


def dft_transform(x):
    """离散傅里叶变换，返回复数"""
    return np.fft.fft(x, axis=-1)


def idft_transform(x):
    """逆离散傅里叶变换"""
    return np.fft.ifft(x, axis=-1).real


def get_frequency_weights(n, mode='exp', decay=0.02):
    """获取频率衰减权重（傅里叶频率是对称的）"""
    k = np.arange(n)
    # 傅里叶频率：0是直流，然后递增到n/2，再递减到0
    # 用距离直流的距离作为频率
    freq = np.minimum(k, n - k)
    if mode == 'exp':
        w = np.exp(-decay * freq)
    elif mode == 'inv':
        w = 1.0 / (1.0 + decay * freq)
    elif mode == 'step':
        cutoff = n * 0.3
        w = (freq < cutoff).astype(np.float32)
    else:
        w = np.ones(n)
    return w.astype(np.float32)


class FourierMagnitudeUSC:
    """傅里叶幅值空间校正USC
    只在幅值空间做秩2仿射校正，相位保持不变
    """
    def __init__(self, use_pca=True, n_components=30):
        self.use_pca = use_pca
        self.n_components = n_components
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        self.w_orth_ = None
        self._v_gpu = None
        self._w_gpu = None

    def fit(self, source_features, target_features):
        # 傅里叶变换，取幅值
        src_fft = dft_transform(source_features)
        tgt_fft = dft_transform(target_features)
        src_mag = np.abs(src_fft)
        tgt_mag = np.abs(tgt_fft)

        if self.use_pca:
            all_feat = np.vstack([src_mag, tgt_mag])
            mean_ = all_feat.mean(axis=0)
            X_centered = all_feat - mean_
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            K = min(self.n_components, Vt.shape[0])
            self.components_ = Vt[:K]
            self.mean_ = mean_
            src_proj = (src_mag - mean_) @ self.components_.T
            tgt_proj = (tgt_mag - mean_) @ self.components_.T
        else:
            self.components_ = None
            self.mean_ = None
            src_proj = src_mag
            tgt_proj = tgt_mag

        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros_like(shift)

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

        self._v_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        self._w_gpu = torch.tensor(self.w_orth_, dtype=torch.float32, device=device)
        if self.components_ is not None:
            self._components_gpu = torch.tensor(self.components_, dtype=torch.float32, device=device)
            self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)

        print(f"    域偏移范数: {self.domain_shift_norm_:.4f}, "
              f"v·w正交性: {np.dot(self.domain_shift_, self.w_orth_):.8f}")
        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0, theta_deg=0.0):
        # 傅里叶变换
        p_np = p_gpu.cpu().numpy()
        p_fft = np.fft.fft(p_np)
        p_mag = np.abs(p_fft)
        p_phase = np.angle(p_fft)

        p_mag_gpu = torch.tensor(p_mag, dtype=torch.float32, device=device)

        if self.components_ is not None:
            Z = (p_mag_gpu - self._mean_gpu) @ self._components_gpu.T
        else:
            Z = p_mag_gpu

        v = self._v_gpu
        w = self._w_gpu
        theta = theta_deg * np.pi / 180.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        vZ = torch.dot(Z, v)
        wZ = torch.dot(Z, w)
        vZ_rot = cos_t * vZ - sin_t * wZ
        wZ_rot = sin_t * vZ + cos_t * wZ
        Z_rot = Z - vZ * v - wZ * w + vZ_rot * v + wZ_rot * w

        vZ_new = torch.dot(Z_rot, v)
        Z_scaled = Z_rot + gamma * vZ_new * v

        Z_corr = Z_scaled + alpha * v

        if self.components_ is not None:
            p_mag_corr = Z_corr @ self._components_gpu + self._mean_gpu
        else:
            p_mag_corr = Z_corr

        # 保持相位不变，用校正后的幅值重建
        p_mag_corr_np = p_mag_corr.cpu().numpy()
        p_mag_corr_np = np.maximum(p_mag_corr_np, 0)  # 幅值非负
        p_fft_corr = p_mag_corr_np * np.exp(1j * p_phase)
        p_final_np = np.fft.ifft(p_fft_corr).real

        p_final = torch.tensor(p_final_np, dtype=torch.float32, device=device)
        p_final = p_final / torch.clamp(torch.norm(p_final), min=1e-8)
        return p_final


class FourierMagPhaseUSC:
    """傅里叶幅值+相位分别校正USC
    幅值空间做秩2仿射校正，相位空间做简单的平移校正
    """
    def __init__(self, use_pca=True, n_components=20):
        self.use_pca = use_pca
        self.n_components = n_components
        self.mag_usc = None
        self.phase_shift_ = None

    def fit(self, source_features, target_features):
        # 幅值空间USC
        self.mag_usc = FourierMagnitudeUSC(use_pca=self.use_pca, n_components=self.n_components)
        self.mag_usc.fit(source_features, target_features)

        # 相位偏移（目标域平均相位 - 源域平均相位）
        src_fft = dft_transform(source_features)
        tgt_fft = dft_transform(target_features)
        src_phase = np.angle(src_fft)
        tgt_phase = np.angle(tgt_fft)
        # 相位差需要处理环绕（-π到π）
        phase_diff = tgt_phase.mean(axis=0) - src_phase.mean(axis=0)
        # 把相位差映射到[-π, π]
        phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
        self.phase_shift_ = phase_diff
        print(f"    相位偏移范数: {np.linalg.norm(self.phase_shift_):.4f}")
        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0, theta_deg=0.0, phase_alpha=0.0):
        # 先做幅值校正
        p_mag_corr = self.mag_usc.correct_prototype_gpu(p_gpu, gamma, alpha, theta_deg)

        # 相位校正（在原始空间做，因为相位校正等价于时域卷积）
        p_np = p_mag_corr.cpu().numpy()
        p_fft = np.fft.fft(p_np)
        p_mag = np.abs(p_fft)
        p_phase = np.angle(p_fft)

        # 相位平移
        p_phase_corr = p_phase + phase_alpha * self.phase_shift_
        p_fft_corr = p_mag * np.exp(1j * p_phase_corr)
        p_final_np = np.fft.ifft(p_fft_corr).real

        p_final = torch.tensor(p_final_np, dtype=torch.float32, device=device)
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
    print(f"  拟合USC（傅里叶变换）...")
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


def evaluate_with_params(domain_data, gamma, alpha, theta_deg=0.0, phase_alpha=0.0):
    usc = domain_data['usc']
    target_data = domain_data['target_data']
    all_protos = domain_data['all_protos']
    protos_gpu = domain_data['protos_gpu']
    text_deltas = domain_data['text_deltas']
    valid_targets = domain_data['valid_targets']
    aps = []
    for eval_name, coco_cls, src_cls in valid_targets:
        if isinstance(usc, FourierMagPhaseUSC):
            p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, alpha, theta_deg, phase_alpha)
        else:
            p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, alpha, theta_deg)
        p_corr = p_corr_gpu.cpu().numpy()
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            if isinstance(usc, FourierMagPhaseUSC):
                neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, gamma, alpha, theta_deg, phase_alpha)
            else:
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

    original_best = {
        'DIOR': 0.3884, 'NWPU': 0.3910, 'DOTA': 0.3903, 'VisDrone': 0.3443,
    }

    configs = [
        ('fourier_magnitude', FourierMagnitudeUSC(use_pca=True, n_components=30), False),
        ('fourier_mag_phase', FourierMagPhaseUSC(use_pca=True, n_components=20), True),
    ]

    all_results = {}

    for config_name, usc, use_phase in configs:
        print(f"\n{'='*60}")
        print(f"配置: {config_name}")
        print(f"{'='*60}")
        config_results = {}
        for src in source_names:
            print(f"\n--- {src}→COCO ---")
            data = load_domain_data(src, usc)
            if data is None:
                continue

            best_ap = -1
            best_params = None
            gammas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
            thetas = [-30, -15, 0, 15, 30]

            if use_phase:
                phase_alphas = [0.0, 0.1, 0.3, 0.5, 1.0]
                for gamma in gammas:
                    for alpha in ALPHAS:
                        for theta in thetas:
                            for phase_alpha in phase_alphas:
                                ap = evaluate_with_params(data, gamma, alpha, theta, phase_alpha)
                                if ap > best_ap:
                                    best_ap = ap
                                    best_params = (gamma, alpha, theta, phase_alpha)
            else:
                for gamma in gammas:
                    for alpha in ALPHAS:
                        for theta in thetas:
                            ap = evaluate_with_params(data, gamma, alpha, theta)
                            if ap > best_ap:
                                best_ap = ap
                                best_params = (gamma, alpha, theta)

            print(f"  最佳AP: {best_ap:.4f}, 参数: {best_params}")
            print(f"  原始空间最佳: AP={original_best.get(src, 0):.4f}")
            gain = best_ap - original_best.get(src, 0)
            print(f"  提升: {gain:+.4f}")
            config_results[src] = {
                'ap': best_ap,
                'params': str(best_params),
                'gain_vs_original': gain,
            }
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results[config_name] = config_results
        del usc

    # 汇总
    print(f"\n{'='*60}")
    print("汇总：傅里叶变换 vs 原始空间")
    print(f"{'='*60}")
    print(f"{'配置':<25} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8} {'平均提升':>10}")
    print("-" * 85)
    print(f"{'原始空间(基线)':<25} {original_best['DIOR']:>8.4f} {original_best['NWPU']:>8.4f} {original_best['DOTA']:>8.4f} {original_best['VisDrone']:>10.4f} {np.mean(list(original_best.values())):>8.4f} {'+0.0000':>10}")
    for config_name, _, _ in configs:
        results = all_results[config_name]
        aps = [results.get(s, {}).get('ap', 0) for s in source_names]
        gains = [results.get(s, {}).get('gain_vs_original', 0) for s in source_names]
        mean_ap = np.mean(aps)
        mean_gain = np.mean(gains)
        print(f"{config_name:<25} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {mean_ap:>8.4f} {mean_gain:>+10.4f}")

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
        'note': '傅里叶变换频域校正：幅值空间秩2仿射 + 相位保持/校正',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
