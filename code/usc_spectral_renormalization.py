# -*- coding: utf-8 -*-
"""方向1+方向2：重新定义频率 + 谱归一化
方向1：按特征方差重新排序维度（方差大的=低频，方差小的=高频）
方向2：谱归一化/谱白化（把源域谱分布归一化到目标域）
对比：
  1. 原始空间秩2仿射（基线）
  2. 按方差重新排序后的秩2仿射
  3. 谱归一化 + 秩2仿射
  4. 按方差重新排序 + 谱归一化 + 秩2仿射
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

OUT = DATA / 'usc_spectral_renormalization_results.json'


class Rank2AffineUSC:
    """秩2仿射USC（和原始空间一样，但可以应用在重新排序/归一化后的空间）"""
    def __init__(self, use_pca=True, n_components=30):
        self.use_pca = use_pca
        self.n_components = n_components
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        self.w_orth_ = None
        self._v_gpu = None
        self._w_gpu = None

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
        if self.components_ is not None:
            Z = (p_gpu - self._mean_gpu) @ self._components_gpu.T
        else:
            Z = p_gpu

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
            p_corr = Z_corr @ self._components_gpu + self._mean_gpu
        else:
            p_corr = Z_corr

        p_corr = p_corr / torch.clamp(torch.norm(p_corr), min=1e-8)
        return p_corr


def reorder_by_variance(source_features, target_features):
    """按特征方差重新排序维度（方差大的在前=低频，方差小的在后=高频）"""
    all_feat = np.vstack([source_features, target_features])
    variances = np.var(all_feat, axis=0)
    # 按方差从大到小排序
    sorted_indices = np.argsort(-variances)
    print(f"    方差排序: 前5个维度方差={variances[sorted_indices[:5]]}, "
          f"后5个维度方差={variances[sorted_indices[-5:]]}")
    return sorted_indices


def spectral_normalization(source_features, target_features):
    """谱归一化：把源域的谱分布（每个维度的标准差）归一化到目标域"""
    src_std = np.std(source_features, axis=0) + 1e-8
    tgt_std = np.std(target_features, axis=0) + 1e-8
    # 归一化系数：目标域标准差 / 源域标准差
    scale = tgt_std / src_std
    print(f"    谱归一化系数: 均值={np.mean(scale):.4f}, "
          f"范围=[{np.min(scale):.4f}, {np.max(scale):.4f}]")
    return scale


def load_domain_data(source_name, transform_type='none'):
    """加载域对数据，可选变换：none/variance_reorder/spectral_norm/both"""
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

    # 应用变换
    sorted_indices = None
    scale = None

    if transform_type in ['variance_reorder', 'both']:
        print(f"  按方差重新排序维度...")
        sorted_indices = reorder_by_variance(all_src, all_tgt)
        all_src = all_src[:, sorted_indices]
        all_tgt = all_tgt[:, sorted_indices]
        for s in source_data:
            source_data[s]['features'] = source_data[s]['features'][:, sorted_indices]
            source_data[s]['prototype'] = source_data[s]['prototype'][sorted_indices]
        for c in target_data:
            target_data[c]['features'] = target_data[c]['features'][:, sorted_indices]

    if transform_type in ['spectral_norm', 'both']:
        print(f"  谱归一化...")
        scale = spectral_normalization(all_src, all_tgt)
        all_src = all_src * scale
        all_tgt = all_tgt * scale
        for s in source_data:
            source_data[s]['features'] = source_data[s]['features'] * scale
            source_data[s]['prototype'] = source_data[s]['prototype'] * scale
        for c in target_data:
            target_data[c]['features'] = target_data[c]['features'] * scale

    print(f"  拟合USC...")
    usc = Rank2AffineUSC(use_pca=True, n_components=30)
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

    original_best = {
        'DIOR': 0.3884, 'NWPU': 0.3910, 'DOTA': 0.3903, 'VisDrone': 0.3443,
    }

    transform_types = ['none', 'variance_reorder', 'spectral_norm', 'both']
    transform_names = {
        'none': '原始空间（基线）',
        'variance_reorder': '按方差重新排序',
        'spectral_norm': '谱归一化',
        'both': '方差重排+谱归一化',
    }

    all_results = {}

    for transform_type in transform_types:
        print(f"\n{'='*60}")
        print(f"变换类型: {transform_names[transform_type]}")
        print(f"{'='*60}")
        config_results = {}
        for src in source_names:
            print(f"\n--- {src}→COCO ---")
            data = load_domain_data(src, transform_type)
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

            print(f"  最佳AP: {best_ap:.4f}, 参数: {best_params}")
            print(f"  原始空间最佳: AP={original_best.get(src, 0):.4f}")
            gain = best_ap - original_best.get(src, 0)
            print(f"  提升: {gain:+.4f}")
            config_results[src] = {
                'ap': best_ap,
                'params': best_params,
                'gain_vs_original': gain,
            }
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results[transform_type] = config_results

    # 汇总
    print(f"\n{'='*60}")
    print("汇总：各变换类型对比")
    print(f"{'='*60}")
    print(f"{'变换类型':<25} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8} {'平均提升':>10}")
    print("-" * 85)
    for transform_type in transform_types:
        results = all_results[transform_type]
        aps = [results.get(s, {}).get('ap', 0) for s in source_names]
        gains = [results.get(s, {}).get('gain_vs_original', 0) for s in source_names]
        mean_ap = np.mean(aps)
        mean_gain = np.mean(gains)
        print(f"{transform_names[transform_type]:<25} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {mean_ap:>8.4f} {mean_gain:>+10.4f}")

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
        'all_results': convert(all_results),
        'original_baseline': convert(original_best),
        'transform_names': transform_names,
        'n_domains': len(source_names),
        'note': '方向1+方向2：重新定义频率 + 谱归一化',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
