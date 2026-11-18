# -*- coding: utf-8 -*-
"""共享基底空间学习USC实验
核心思想：从多个源域联合学习共享基底B（PCA闭式解，零训练）
校正在基底空间进行：c_s' = c_s + α·v_base，然后回投影
对比：
  1. 原始USC（单域PCA基底）
  2. 共享基底USC（多域联合PCA基底）
  3. 共享基底 + det_stats自适应α
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate, evaluate_baseline,
)
import tge_probe as T

OUT = DATA / 'usc_shared_basis_results.json'

# 传感器感知γ配置
SENSOR_GAMMA = {
    'DIOR': -2.0, 'NWPU': -2.0, 'DOTA': -2.0, 'VisDrone': -1.0,
}


class SharedBasisUSC:
    """共享基底空间USC
    基底B从多个源域联合学习（PCA闭式解）
    校正在基底空间进行
    """
    def __init__(self, n_components=30):
        self.n_components = n_components
        self.mean_ = None  # 全局均值（多域联合）
        self.components_ = None  # 共享基底B
        self.domain_shift_ = None  # 域偏移方向（在基底空间）
        self.domain_shift_norm_ = None
        self._components_gpu = None
        self._mean_gpu = None
        self._v_domain_gpu = None

    def fit_shared_basis(self, all_source_features_list):
        """从多个源域联合学习共享基底B（PCA闭式解）
        all_source_features_list: [src_features_1, src_features_2, ...]
        """
        all_features = np.vstack(all_source_features_list)
        self.mean_ = all_features.mean(axis=0)
        X_centered = all_features - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        K = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:K]
        print(f"    共享基底学习完成: {len(all_source_features_list)}个源域, "
              f"{all_features.shape[0]}个样本, K={K}")
        # 解释方差比例
        explained_var = np.sum(S[:K]**2) / np.sum(S**2)
        print(f"    前{K}主成分解释方差比例: {explained_var*100:.1f}%")
        return self

    def fit_domain_shift(self, source_features, target_features):
        """计算域偏移方向（在基底空间）"""
        src_proj = (source_features - self.mean_) @ self.components_.T
        tgt_proj = (target_features - self.mean_) @ self.components_.T
        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros(self.n_components)
        self._components_gpu = torch.tensor(self.components_, dtype=torch.float32, device=device)
        self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        self._v_domain_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        return self

    def correct_prototype_gpu(self, p_gpu, gamma=0.0, alpha=1.0):
        """GPU校正单个原型（在共享基底空间）
        Z_corr = (I + γ·v·v^T)·Z + α·v  （秩1仿射，在基底空间）
        """
        # 投影到共享基底空间
        Z = (p_gpu - self._mean_gpu) @ self._components_gpu.T  # (K,)
        # 秩1仿射校正（在基底空间）
        vZ = torch.dot(Z, self._v_domain_gpu)
        Z_affine = Z + gamma * vZ * self._v_domain_gpu  # 缩放
        Z_corr = Z_affine + alpha * self._v_domain_gpu  # 平移
        # 回投影到原始空间
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


def load_domain_data_shared(source_name, shared_usc):
    """加载单个域对数据，使用共享基底USC"""
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

    # 用共享基底USC计算域偏移
    print(f"  计算域偏移（共享基底空间）...")
    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
    shared_usc.fit_domain_shift(all_src, all_tgt)

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
        'usc': shared_usc,
        'all_protos': all_protos,
        'protos_gpu': protos_gpu,
        'valid_targets': valid_targets,
        'text_deltas': text_deltas,
    }


def evaluate_with_params(domain_data, gamma, alpha):
    """用指定γ,α评估（共享基底USC）"""
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

    # 阶段1：从多个源域联合学习共享基底B
    print(f"\n{'='*60}")
    print("阶段1：从多个源域联合学习共享基底B")
    print(f"{'='*60}")

    all_source_features = []
    for src in source_names:
        feats = load_source_features(src)
        if feats is not None:
            all_source_features.append(feats)
            print(f"  {src}: {feats.shape}")

    shared_usc = SharedBasisUSC(n_components=K_PCA).fit_shared_basis(all_source_features)

    # 阶段2：共享基底USC vs 原始USC（单域基底）
    print(f"\n{'='*60}")
    print("阶段2：共享基底USC vs 原始USC（单域基底）")
    print(f"{'='*60}")

    # 原始USC的最佳结果（从之前的实验）
    original_usc_best = {
        'DIOR': 0.3836, 'NWPU': 0.3845, 'DOTA': 0.3864, 'VisDrone': 0.3406,
    }

    shared_results = {}
    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        data = load_domain_data_shared(src, shared_usc)
        if data is None:
            continue
        # 扫描(γ, α)找最佳
        best_ap = -1
        best_params = None
        for gamma in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]:
            for alpha in ALPHAS:
                ap = evaluate_with_params(data, gamma, alpha)
                if ap > best_ap:
                    best_ap = ap
                    best_params = (gamma, alpha)
        print(f"  共享基底最佳: γ={best_params[0]}, α={best_params[1]}, AP={best_ap:.4f}")
        print(f"  原始USC最佳: AP={original_usc_best.get(src, 0):.4f}")
        gain = best_ap - original_usc_best.get(src, 0)
        print(f"  提升: {gain:+.4f}")
        shared_results[src] = {
            'shared_best_ap': best_ap,
            'shared_best_params': {'gamma': best_params[0], 'alpha': best_params[1]},
            'original_best_ap': original_usc_best.get(src, 0),
            'gain': gain,
        }
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 阶段3：汇总对比
    print(f"\n{'='*60}")
    print("阶段3：共享基底USC vs 原始USC 汇总")
    print(f"{'='*60}")
    print(f"{'域对':<12} {'原始USC':>10} {'共享基底':>10} {'提升':>10}")
    print("-" * 45)
    for src in source_names:
        if src in shared_results:
            orig = shared_results[src]['original_best_ap']
            shared = shared_results[src]['shared_best_ap']
            gain = shared_results[src]['gain']
            print(f"{src:<12} {orig:>10.4f} {shared:>10.4f} {gain:>+10.4f}")
    orig_mean = np.mean([shared_results[s]['original_best_ap'] for s in shared_results])
    shared_mean = np.mean([shared_results[s]['shared_best_ap'] for s in shared_results])
    print(f"{'平均':<12} {orig_mean:>10.4f} {shared_mean:>10.4f} {shared_mean-orig_mean:>+10.4f}")

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
        'shared_results': convert(shared_results),
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
