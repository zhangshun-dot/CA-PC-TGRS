# -*- coding: utf-8 -*-
"""频域多尺度校正实验
核心思想：在DCT频域空间里，把频率维度分成多个尺度（低/中/高频），
每个尺度独立做秩2仿射校正（有自己的校正参数），然后融合。
因为不同频率尺度的域偏移特性不同：
  - 低频（整体结构、类别语义）：域偏移大，需要强校正
  - 中频（部件、纹理）：域偏移中等
  - 高频（细节、噪声）：域偏移小，需要弱校正
对比：
  1. 原始空间秩2仿射（baseline）
  2. 2尺度校正（低频/高频）
  3. 3尺度校正（低频/中频/高频）
  4. 4尺度校正（低/中低/中高/高）
  5. 参数共享 vs 独立的对比
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

OUT = DATA / 'usc_dct_multiscale_results.json'


def dct_transform(x):
    return dct(x, type=2, norm='ortho', axis=-1)


def idct_transform(x):
    return idct(x, type=2, norm='ortho', axis=-1)


class MultiScaleUSC:
    """频域多尺度校正USC"""
    def __init__(self, n_scales=3, param_mode='shared', use_pca=True, n_components=30):
        """
        n_scales: 尺度数量（2/3/4）
        param_mode: 参数模式
          - 'shared': 所有尺度共享相同的γ, α, θ（退化为普通频域校正）
          - 'independent': 每个尺度有独立的γ, α, θ（参数多，表达能力强）
          - 'semi': γ共享，α和θ独立（平衡）
        use_pca: 是否在每个尺度内做PCA降维
        n_components: PCA降维维度
        """
        self.n_scales = n_scales
        self.param_mode = param_mode
        self.use_pca = use_pca
        self.n_components = n_components
        self.scale_boundaries = None  # 每个尺度的边界
        self.scales = []  # 每个尺度的USC对象

    def _compute_scale_boundaries(self, D):
        """计算每个尺度的边界"""
        boundaries = []
        step = D // self.n_scales
        for i in range(self.n_scales):
            start = i * step
            end = (i + 1) * step if i < self.n_scales - 1 else D
            boundaries.append((start, end))
        return boundaries

    def fit(self, source_features, target_features):
        # DCT变换
        src_dct = dct_transform(source_features)
        tgt_dct = dct_transform(target_features)
        D = src_dct.shape[1]

        # 计算尺度边界
        self.scale_boundaries = self._compute_scale_boundaries(D)
        print(f"    尺度划分: {self.n_scales}尺度, 边界={self.scale_boundaries}")

        # 每个尺度独立拟合
        self.scales = []
        for i, (start, end) in enumerate(self.scale_boundaries):
            scale_src = src_dct[:, start:end]
            scale_tgt = tgt_dct[:, start:end]
            scale_name = ['低频', '中低频', '中高频', '高频'][i] if i < 4 else f'尺度{i}'

            if self.use_pca:
                # 在该尺度内做PCA降维
                all_feat = np.vstack([scale_src, scale_tgt])
                mean_ = all_feat.mean(axis=0)
                X_centered = all_feat - mean_
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                K = min(self.n_components, Vt.shape[0], end - start)
                components_ = Vt[:K]

                # 投影到PCA子空间
                src_proj = (scale_src - mean_) @ components_.T
                tgt_proj = (scale_tgt - mean_) @ components_.T
            else:
                components_ = None
                mean_ = None
                src_proj = scale_src
                tgt_proj = scale_tgt

            # 计算域偏移方向
            shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
            shift_norm = float(np.linalg.norm(shift))
            if shift_norm > 1e-8:
                v = shift / shift_norm
            else:
                v = np.zeros_like(shift)

            # v的主正交方向w
            X_centered = src_proj - src_proj.mean(axis=0)
            cov = X_centered.T @ X_centered
            P_perp = np.eye(len(v)) - np.outer(v, v)
            cov_perp = P_perp @ cov @ P_perp
            w = np.random.randn(len(v))
            w = w - np.dot(w, v) * v
            w = w / (np.linalg.norm(w) + 1e-8)
            for _ in range(30):
                w_new = cov_perp @ w
                w_new = w_new - np.dot(w_new, v) * v
                w_new = w_new / (np.linalg.norm(w_new) + 1e-8)
                if np.linalg.norm(w_new - w) < 1e-6:
                    break
                w = w_new

            # 移到GPU
            scale_data = {
                'start': start,
                'end': end,
                'name': scale_name,
                'components': components_,
                'mean': mean_,
                'v': v,
                'w': w,
                'shift_norm': shift_norm,
                'v_gpu': torch.tensor(v, dtype=torch.float32, device=device),
                'w_gpu': torch.tensor(w, dtype=torch.float32, device=device),
            }
            if components_ is not None:
                scale_data['components_gpu'] = torch.tensor(components_, dtype=torch.float32, device=device)
                scale_data['mean_gpu'] = torch.tensor(mean_, dtype=torch.float32, device=device)

            self.scales.append(scale_data)
            print(f"    {scale_name} [{start}:{end}]: 域偏移范数={shift_norm:.4f}, "
                  f"v·w正交性={np.dot(v, w):.8f}")

        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0, theta_deg=0.0,
                                gamma_per_scale=None, alpha_per_scale=None, theta_per_scale=None):
        """
        多尺度校正
        如果param_mode='shared'，用全局的gamma, alpha, theta
        如果param_mode='independent'，用gamma_per_scale, alpha_per_scale, theta_per_scale（列表）
        如果param_mode='semi'，gamma共享，alpha和theta独立
        """
        # DCT变换
        p_dct = torch.tensor(dct_transform(p_gpu.cpu().numpy()), dtype=torch.float32, device=device)
        p_dct_corr = p_dct.clone()

        theta = theta_deg * np.pi / 180.0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        for i, scale in enumerate(self.scales):
            start, end = scale['start'], scale['end']

            # 确定该尺度的参数
            if self.param_mode == 'shared':
                g, a, t = gamma, alpha, theta_deg
            elif self.param_mode == 'independent':
                g = gamma_per_scale[i] if gamma_per_scale is not None else gamma
                a = alpha_per_scale[i] if alpha_per_scale is not None else alpha
                t = theta_per_scale[i] if theta_per_scale is not None else theta_deg
            elif self.param_mode == 'semi':
                g = gamma  # γ共享
                a = alpha_per_scale[i] if alpha_per_scale is not None else alpha
                t = theta_per_scale[i] if theta_per_scale is not None else theta_deg
            else:
                g, a, t = gamma, alpha, theta_deg

            t_rad = t * np.pi / 180.0
            cos_t_i = np.cos(t_rad)
            sin_t_i = np.sin(t_rad)

            # 提取该尺度的DCT系数
            p_scale = p_dct[start:end]

            # 投影到PCA子空间（如果有）
            if scale['components'] is not None:
                Z = (p_scale - scale['mean_gpu']) @ scale['components_gpu'].T
            else:
                Z = p_scale

            v = scale['v_gpu']
            w = scale['w_gpu']

            # 在(v,w)平面内旋转
            vZ = torch.dot(Z, v)
            wZ = torch.dot(Z, w)
            vZ_rot = cos_t_i * vZ - sin_t_i * wZ
            wZ_rot = sin_t_i * vZ + cos_t_i * wZ
            Z_rot = Z - vZ * v - wZ * w + vZ_rot * v + wZ_rot * w

            # 沿v方向缩放
            vZ_new = torch.dot(Z_rot, v)
            Z_scaled = Z_rot + g * vZ_new * v

            # 沿v方向平移
            Z_corr = Z_scaled + a * v

            # 回投影到该尺度的DCT空间
            if scale['components'] is not None:
                p_scale_corr = Z_corr @ scale['components_gpu'] + scale['mean_gpu']
            else:
                p_scale_corr = Z_corr

            # 放回完整DCT系数
            p_dct_corr[start:end] = p_scale_corr

        # 逆DCT变换
        p_final = torch.tensor(idct_transform(p_dct_corr.cpu().numpy()), dtype=torch.float32, device=device)
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
    print(f"  拟合USC（多尺度校正, {usc.n_scales}尺度, mode={usc.param_mode}）...")
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


def evaluate_with_params(domain_data, gamma, alpha, theta_deg=0.0,
                          gamma_per_scale=None, alpha_per_scale=None, theta_per_scale=None):
    usc = domain_data['usc']
    target_data = domain_data['target_data']
    all_protos = domain_data['all_protos']
    protos_gpu = domain_data['protos_gpu']
    text_deltas = domain_data['text_deltas']
    valid_targets = domain_data['valid_targets']
    aps = []
    for eval_name, coco_cls, src_cls in valid_targets:
        p_corr_gpu = usc.correct_prototype_gpu(
            protos_gpu[src_cls], gamma, alpha, theta_deg,
            gamma_per_scale, alpha_per_scale, theta_per_scale
        )
        p_corr = p_corr_gpu.cpu().numpy()
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            neg_corr_gpu = usc.correct_prototype_gpu(
                neg_gpu, gamma, alpha, theta_deg,
                gamma_per_scale, alpha_per_scale, theta_per_scale
            )
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

    # 原始空间秩2仿射的最佳结果
    original_best = {
        'DIOR': 0.3884, 'NWPU': 0.3910, 'DOTA': 0.3903, 'VisDrone': 0.3443,
    }

    # 测试配置
    configs = [
        # (名称, 尺度数, 参数模式)
        ('2scale_shared', 2, 'shared'),
        ('2scale_semi', 2, 'semi'),
        ('2scale_independent', 2, 'independent'),
        ('3scale_shared', 3, 'shared'),
        ('3scale_semi', 3, 'semi'),
        ('3scale_independent', 3, 'independent'),
    ]

    all_results = {}

    for config_name, n_scales, param_mode in configs:
        print(f"\n{'='*60}")
        print(f"配置: {config_name} ({n_scales}尺度, mode={param_mode})")
        print(f"{'='*60}")
        config_results = {}
        for src in source_names:
            print(f"\n--- {src}→COCO ---")
            usc = MultiScaleUSC(n_scales=n_scales, param_mode=param_mode,
                                 use_pca=True, n_components=20)
            data = load_domain_data(src, usc)
            if data is None:
                continue

            best_ap = -1
            best_params = None

            if param_mode == 'shared':
                # 共享参数：搜索全局γ, α, θ
                gammas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
                thetas = [-30, -15, 0, 15, 30]
                for gamma in gammas:
                    for alpha in ALPHAS:
                        for theta in thetas:
                            ap = evaluate_with_params(data, gamma, alpha, theta)
                            if ap > best_ap:
                                best_ap = ap
                                best_params = (gamma, alpha, theta)
            elif param_mode == 'semi':
                # semi：γ共享，α和θ独立
                gammas = [-2.0, -1.0, 0.0, 1.0]
                alpha_options = [0.0, 0.3, 1.0]
                theta_options = [-15, 0, 15]
                if n_scales == 2:
                    for gamma in gammas:
                        for a0 in alpha_options:
                            for a1 in alpha_options:
                                for t0 in theta_options:
                                    for t1 in theta_options:
                                        ap = evaluate_with_params(
                                            data, gamma, 0.0, 0,
                                            None, [a0, a1], [t0, t1]
                                        )
                                        if ap > best_ap:
                                            best_ap = ap
                                            best_params = (gamma, [a0, a1], [t0, t1])
                elif n_scales == 3:
                    for gamma in gammas:
                        for a0 in alpha_options:
                            for a1 in alpha_options:
                                for a2 in alpha_options:
                                    for t0 in theta_options:
                                        for t1 in theta_options:
                                            for t2 in theta_options:
                                                ap = evaluate_with_params(
                                                    data, gamma, 0.0, 0,
                                                    None, [a0, a1, a2], [t0, t1, t2]
                                                )
                                                if ap > best_ap:
                                                    best_ap = ap
                                                    best_params = (gamma, [a0, a1, a2], [t0, t1, t2])
            elif param_mode == 'independent':
                # independent：每个尺度独立的γ, α, θ
                # 为了控制搜索空间，用较少的选项
                gamma_options = [-1.0, 0.0, 1.0]
                alpha_options = [0.0, 0.5, 1.0]
                theta_options = [-15, 0, 15]
                if n_scales == 2:
                    for g0 in gamma_options:
                        for g1 in gamma_options:
                            for a0 in alpha_options:
                                for a1 in alpha_options:
                                    for t0 in theta_options:
                                        for t1 in theta_options:
                                            ap = evaluate_with_params(
                                                data, 0.0, 0.0, 0,
                                                [g0, g1], [a0, a1], [t0, t1]
                                            )
                                            if ap > best_ap:
                                                best_ap = ap
                                                best_params = ([g0, g1], [a0, a1], [t0, t1])
                elif n_scales == 3:
                    # 3尺度搜索空间太大，减少选项
                    gamma_options_3 = [0.0, 1.0]
                    alpha_options_3 = [0.0, 1.0]
                    theta_options_3 = [0, 15]
                    for g0 in gamma_options_3:
                        for g1 in gamma_options_3:
                            for g2 in gamma_options_3:
                                for a0 in alpha_options_3:
                                    for a1 in alpha_options_3:
                                        for a2 in alpha_options_3:
                                            for t0 in theta_options_3:
                                                for t1 in theta_options_3:
                                                    for t2 in theta_options_3:
                                                        ap = evaluate_with_params(
                                                            data, 0.0, 0.0, 0,
                                                            [g0, g1, g2], [a0, a1, a2], [t0, t1, t2]
                                                        )
                                                        if ap > best_ap:
                                                            best_ap = ap
                                                            best_params = ([g0, g1, g2], [a0, a1, a2], [t0, t1, t2])

            print(f"  最佳AP: {best_ap:.4f}, 参数: {best_params}")
            print(f"  原始空间最佳: AP={original_best.get(src, 0):.4f}")
            gain = best_ap - original_best.get(src, 0)
            print(f"  提升: {gain:+.4f}")
            config_results[src] = {
                'ap': best_ap,
                'params': str(best_params),
                'gain_vs_original': gain,
            }
            del data, usc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results[config_name] = config_results

    # 汇总对比
    print(f"\n{'='*60}")
    print("汇总：各多尺度校正配置对比")
    print(f"{'='*60}")
    print(f"{'配置':<25} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8} {'平均提升':>10}")
    print("-" * 85)
    for config_name, _, _ in configs:
        results = all_results[config_name]
        aps = [results.get(s, {}).get('ap', 0) for s in source_names]
        gains = [results.get(s, {}).get('gain_vs_original', 0) for s in source_names]
        mean_ap = np.mean(aps)
        mean_gain = np.mean(gains)
        print(f"{config_name:<25} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {mean_ap:>8.4f} {mean_gain:>+10.4f}")

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
        'note': '频域多尺度校正：每个频率尺度独立做秩2仿射校正',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
