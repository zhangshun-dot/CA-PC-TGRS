# -*- coding: utf-8 -*-
"""旋转感知秩2仿射USC实验
Z_corr = R(θ) · (I + γ·v·v^T) · Z + α·v
其中R(θ)是在(v,w)平面内的2D旋转，w是v正交补空间中方差最大的方向
参数：γ(缩放), α(平移), θ(旋转) —— 比秩1仿射只多1个参数
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

OUT = DATA / 'usc_rank2_rotation_results.json'
GAMMAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
# 旋转角度（度），转换成弧度
THETAS_DEG = [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]


class RotationAffineUSC:
    """旋转感知秩2仿射USC
    在域偏移方向v和其主正交方向w张成的2D平面内加旋转
    """
    def __init__(self, n_components=30):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.domain_shift_ = None  # v
        self.ortho_direction_ = None  # w（v正交补空间中方差最大的方向）
        self.domain_shift_norm_ = None
        self._components_gpu = None
        self._mean_gpu = None
        self._v_domain_gpu = None
        self._w_ortho_gpu = None

    def fit(self, source_features, target_features):
        all_features = np.vstack([source_features, target_features])
        self.mean_ = all_features.mean(axis=0)
        X_centered = all_features - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        K = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:K]

        # 投影到低秩空间
        src_proj = (source_features - self.mean_) @ self.components_.T
        tgt_proj = (target_features - self.mean_) @ self.components_.T

        # 域偏移方向v
        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros(K)

        # 找v的正交补空间中方差最大的方向w
        # 方法：把所有特征投影到v的正交补空间，然后做PCA找第一主成分
        v = self.domain_shift_
        # 投影到正交补空间：Z_perp = Z - <Z,v>·v
        all_proj = np.vstack([src_proj, tgt_proj])
        all_centered = all_proj - all_proj.mean(axis=0)
        # 去掉v方向的分量
        v_components = all_centered @ v  # (N,)
        all_perp = all_centered - np.outer(v_components, v)
        # 在正交补空间中做PCA
        if np.linalg.norm(all_perp) > 1e-8:
            U_perp, S_perp, Vt_perp = np.linalg.svd(all_perp, full_matrices=False)
            self.ortho_direction_ = Vt_perp[0]  # 第一主成分
            # 确保w与v正交（数值误差）
            self.ortho_direction_ = self.ortho_direction_ - np.dot(self.ortho_direction_, v) * v
            self.ortho_direction_ = self.ortho_direction_ / (np.linalg.norm(self.ortho_direction_) + 1e-8)
        else:
            # 如果正交补空间退化，随便找一个正交方向
            w = np.random.randn(K)
            w = w - np.dot(w, v) * v
            self.ortho_direction_ = w / (np.linalg.norm(w) + 1e-8)

        # 移到GPU
        self._components_gpu = torch.tensor(self.components_, dtype=torch.float32, device=device)
        self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        self._v_domain_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        self._w_ortho_gpu = torch.tensor(self.ortho_direction_, dtype=torch.float32, device=device)

        print(f"    域偏移范数: {self.domain_shift_norm_:.4f}")
        print(f"    v·w（正交性）: {np.dot(self.domain_shift_, self.ortho_direction_):.6f}")
        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0, theta_deg=0.0):
        """GPU校正单个原型：Z_corr = R(θ)·(I + γ·v·v^T)·Z + α·v
        θ是旋转角度（度），在(v,w)平面内旋转
        """
        # 投影到低秩空间
        Z = (p_gpu - self._mean_gpu) @ self._components_gpu.T  # (K,)

        # 秩1仿射：缩放 + 平移
        vZ = torch.dot(Z, self._v_domain_gpu)  # scalar
        Z_affine = Z + gamma * vZ * self._v_domain_gpu  # 缩放
        Z_affine = Z_affine + alpha * self._v_domain_gpu  # 平移

        # 2D旋转：在(v,w)平面内旋转θ
        if abs(theta_deg) > 1e-8:
            theta = theta_deg * np.pi / 180.0
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            v = self._v_domain_gpu
            w = self._w_ortho_gpu
            Zv = torch.dot(Z_affine, v)  # 在v方向的分量
            Zw = torch.dot(Z_affine, w)  # 在w方向的分量
            # 旋转后的(v,w)平面内分量
            # Z_rot = Z + [(cos-1)·Zv + sin·Zw]·v + [(cos-1)·Zw - sin·Zv]·w
            Z_rot = Z_affine + ((cos_t - 1.0) * Zv + sin_t * Zw) * v + ((cos_t - 1.0) * Zw - sin_t * Zv) * w
        else:
            Z_rot = Z_affine

        # 回投影到原始空间
        X_corr = Z_rot @ self._components_gpu + self._mean_gpu  # (D,)
        X_corr = X_corr / torch.clamp(torch.norm(X_corr), min=1e-8)
        return X_corr


def load_domain_data(source_name):
    """加载单个域对的全部数据并缓存（用RotationAffineUSC）"""
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
    print(f"  拟合旋转感知USC...")
    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
    usc = RotationAffineUSC(n_components=K_PCA).fit(all_src, all_tgt)
    all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}
    # 预计算文本外推方向
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


def evaluate_with_params(domain_data, gamma, alpha, theta_deg):
    """用指定γ,α,θ评估"""
    usc = domain_data['usc']
    target_data = domain_data['target_data']
    all_protos = domain_data['all_protos']
    protos_gpu = domain_data['protos_gpu']
    text_deltas = domain_data['text_deltas']
    valid_targets = domain_data['valid_targets']

    aps = []
    for eval_name, coco_cls, src_cls in valid_targets:
        # 校正正原型
        p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], gamma, alpha, theta_deg)
        p_corr = p_corr_gpu.cpu().numpy()
        # 校正负原型
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, gamma, alpha, theta_deg)
            neg = neg_corr_gpu.cpu().numpy()
        else:
            neg = None
        # 文本外推
        delta = text_deltas[src_cls]
        p_hat = p_corr + 1.0 * delta
        p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
        # 评估
        ap = evaluate(target_data[coco_cls], p_hat, neg)
        aps.append(ap)
    return float(np.mean(aps))


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"域对列表: {source_names}")

    # 阶段1：先做θ扫描（固定γ=-2.0, α=0.0，秩1仿射最佳配置）
    print(f"\n{'='*60}")
    print("阶段1：θ扫描（固定γ=-2.0, α=0.0）")
    print(f"{'='*60}")

    theta_scan_results = {}
    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data(src)
        if data is None:
            continue
        domain_aps = {}
        for theta in THETAS_DEG:
            ap = evaluate_with_params(data, gamma=-2.0, alpha=0.0, theta_deg=theta)
            domain_aps[theta] = ap
            print(f"    θ={theta:>4}°: AP={ap:.4f}")
        best_theta = max(domain_aps, key=domain_aps.get)
        print(f"    最佳θ={best_theta}°, AP={domain_aps[best_theta]:.4f}")
        theta_scan_results[src] = domain_aps
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 阶段2：联合扫描(γ, α, θ)——用阶段1找到的最佳θ
    print(f"\n{'='*60}")
    print("阶段2：联合扫描(γ, α, θ)")
    print(f"{'='*60}")

    full_results = {}
    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data(src)
        if data is None:
            continue
        best_theta = max(theta_scan_results[src], key=theta_scan_results[src].get)
        print(f"  使用最佳θ={best_theta}°")

        best_ap = -1
        best_params = None
        grid_results = {}
        for gamma in GAMMAS:
            for alpha in ALPHAS:
                ap = evaluate_with_params(data, gamma=gamma, alpha=alpha, theta_deg=best_theta)
                grid_results[f"{gamma}_{alpha}"] = ap
                if ap > best_ap:
                    best_ap = ap
                    best_params = (gamma, alpha, best_theta)
        print(f"  最佳: γ={best_params[0]}, α={best_params[1]}, θ={best_params[2]}°, AP={best_ap:.4f}")
        full_results[src] = {
            'best_ap': best_ap,
            'best_params': {'gamma': best_params[0], 'alpha': best_params[1], 'theta': best_params[2]},
            'grid_results': grid_results,
        }
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 阶段3：对比秩1仿射 vs 旋转感知秩2仿射
    print(f"\n{'='*60}")
    print("阶段3：对比秩1仿射 vs 旋转感知秩2仿射")
    print(f"{'='*60}")

    # 秩1仿射的最佳结果（从之前的实验）
    rank1_best = {
        'DIOR': {'gamma': -2.0, 'alpha': 0.0, 'ap': 0.3836},
        'NWPU': {'gamma': -2.0, 'alpha': 0.0, 'ap': 0.3845},
        'DOTA': {'gamma': -2.0, 'alpha': 0.7, 'ap': 0.3864},
        'VisDrone': {'gamma': -1.0, 'alpha': 1.0, 'ap': 0.3406},
    }

    print(f"{'域对':<12} {'秩1仿射AP':>10} {'秩2旋转AP':>10} {'提升':>10} {'最佳θ':>8}")
    print("-" * 55)
    for src in source_names:
        if src in full_results and src in rank1_best:
            r1_ap = rank1_best[src]['ap']
            r2_ap = full_results[src]['best_ap']
            gain = r2_ap - r1_ap
            best_theta = full_results[src]['best_params']['theta']
            print(f"{src:<12} {r1_ap:>10.4f} {r2_ap:>10.4f} {gain:>+10.4f} {best_theta:>6}°")

    # 保存结果
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    final_results = {
        'theta_scan': convert(theta_scan_results),
        'full_grid': convert(full_results),
        'rank1_baseline': convert(rank1_best),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")


if __name__ == '__main__':
    main()
