# -*- coding: utf-8 -*-
"""自适应谱归一化实验（IO优化版：每个域对只加载一次，在内存中做不同权重的谱归一化）"""
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

OUT = DATA / 'usc_adaptive_spectral_norm_results.json'


class Rank1AffineUSC:
    def __init__(self, use_pca=True, n_components=30):
        self.use_pca = use_pca
        self.n_components = n_components
        self.domain_shift_ = None
        self.domain_shift_norm_ = None
        self._v_gpu = None

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

        self._v_gpu = torch.tensor(self.domain_shift_, dtype=torch.float32, device=device)
        if self.components_ is not None:
            self._components_gpu = torch.tensor(self.components_, dtype=torch.float32, device=device)
            self._mean_gpu = torch.tensor(self.mean_, dtype=torch.float32, device=device)
        return self

    def correct_prototype_gpu(self, p_gpu, alpha=1.0):
        if self.components_ is not None:
            Z = (p_gpu - self._mean_gpu) @ self._components_gpu.T
        else:
            Z = p_gpu
        v = self._v_gpu
        Z_corr = Z + alpha * v
        if self.components_ is not None:
            p_corr = Z_corr @ self._components_gpu + self._mean_gpu
        else:
            p_corr = Z_corr
        p_corr = p_corr / torch.clamp(torch.norm(p_corr), min=1e-8)
        return p_corr


def apply_spectral_norm(features, src_std, tgt_std, weight):
    if weight <= 0:
        return features
    scale = tgt_std / (src_std + 1e-8)
    return features + weight * (features * scale - features)


def load_domain_raw(source_name):
    """加载域对原始数据（不做谱归一化），只加载一次"""
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

    domain_shift = float(np.linalg.norm(all_tgt.mean(axis=0) - all_src.mean(axis=0)))
    src_std = np.std(all_src, axis=0) + 1e-8
    tgt_std = np.std(all_tgt, axis=0) + 1e-8

    text_deltas = {}
    for eval_name, coco_cls, src_cls in valid_targets:
        s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
        d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
        text_deltas[src_cls] = d_emb - s_emb

    return {
        'source_data': source_data,
        'target_data': target_data,
        'valid_targets': valid_targets,
        'domain_shift': domain_shift,
        'src_std': src_std,
        'tgt_std': tgt_std,
        'text_deltas': text_deltas,
    }


def evaluate_with_spectral_weight(raw_data, spectral_weight):
    """在内存中对原始数据应用谱归一化，然后评估"""
    source_data = {}
    for s, data in raw_data['source_data'].items():
        source_data[s] = {
            'features': apply_spectral_norm(data['features'], raw_data['src_std'], raw_data['tgt_std'], spectral_weight),
            'prototype': apply_spectral_norm(data['prototype'], raw_data['src_std'], raw_data['tgt_std'], spectral_weight),
        }
    target_data = {}
    for c, data in raw_data['target_data'].items():
        target_data[c] = {
            'features': apply_spectral_norm(data['features'], raw_data['src_std'], raw_data['tgt_std'], spectral_weight),
            'boxes': data.get('boxes', None),
            'scores': data.get('scores', None),
        }
        # 保留其他字段
        for k, v in data.items():
            if k not in target_data[c]:
                target_data[c][k] = v

    all_src = np.vstack([source_data[s]['features'] for _, _, s in raw_data['valid_targets']])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in raw_data['valid_targets']])

    usc = Rank1AffineUSC(use_pca=True, n_components=30)
    usc.fit(all_src, all_tgt)

    all_protos = {s: source_data[s]['prototype'] for _, _, s in raw_data['valid_targets']}
    protos_gpu = {s: torch.tensor(p, dtype=torch.float32, device=device) for s, p in all_protos.items()}

    best_ap = -1
    best_alpha = None
    for alpha in ALPHAS:
        aps = []
        for eval_name, coco_cls, src_cls in raw_data['valid_targets']:
            p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], alpha)
            p_corr = p_corr_gpu.cpu().numpy()
            neg_raw = build_neg_proto(all_protos, src_cls)
            if neg_raw is not None:
                neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
                neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, alpha)
                neg = neg_corr_gpu.cpu().numpy()
            else:
                neg = None
            delta = raw_data['text_deltas'][src_cls]
            p_hat = p_corr + 1.0 * delta
            p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
            ap = evaluate(target_data[coco_cls], p_hat, neg)
            aps.append(ap)
        mean_ap = float(np.mean(aps))
        if mean_ap > best_ap:
            best_ap = mean_ap
            best_alpha = alpha

    return best_ap, best_alpha


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"域对列表: {source_names}")
    print(f"自适应谱归一化实验（IO优化版）")

    # 第一步：一次性加载所有域对的原始数据
    print(f"\n{'='*60}")
    print("第一步：加载所有域对原始数据（每个域对只加载一次）")
    print(f"{'='*60}")
    raw_data_all = {}
    domain_shifts = {}
    for src in source_names:
        print(f"\n--- {src} ---")
        raw = load_domain_raw(src)
        if raw:
            raw_data_all[src] = raw
            domain_shifts[src] = raw['domain_shift']
            print(f"  域偏移: {raw['domain_shift']:.4f}")
        else:
            print(f"  ⚠️ 数据不足，跳过")
    delta_max = max(domain_shifts.values()) if domain_shifts else 1.0
    print(f"\n最大域偏移: {delta_max:.4f}")

    # 第二步：测试不同的谱归一化权重（在内存中做，不重新加载数据）
    print(f"\n{'='*60}")
    print("第二步：测试不同的谱归一化权重（全局）")
    print(f"{'='*60}")
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    global_results = {}
    for w in weights:
        print(f"\n--- 全局权重 w={w} ---")
        w_results = {}
        for src in source_names:
            if src not in raw_data_all:
                continue
            best_ap, best_alpha = evaluate_with_spectral_weight(raw_data_all[src], w)
            print(f"  {src}: AP={best_ap:.4f}, α={best_alpha}")
            w_results[src] = {'ap': best_ap, 'alpha': best_alpha}
        global_results[w] = w_results

    # 第三步：自适应谱归一化（w = δ / δ_max）
    print(f"\n{'='*60}")
    print("第三步：自适应谱归一化（w = δ / δ_max）")
    print(f"{'='*60}")
    adaptive_results = {}
    for src in source_names:
        if src not in raw_data_all:
            continue
        w = domain_shifts[src] / delta_max
        print(f"\n  {src}: δ={domain_shifts[src]:.4f}, w={w:.4f}")
        best_ap, best_alpha = evaluate_with_spectral_weight(raw_data_all[src], w)
        print(f"    AP={best_ap:.4f}, α={best_alpha}")
        adaptive_results[src] = {'ap': best_ap, 'alpha': best_alpha, 'w': w}

    # 汇总
    print(f"\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    print(f"{'方法':<30} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8}")
    print("-" * 80)
    for w in weights:
        aps = [global_results.get(w, {}).get(s, {}).get('ap', 0) for s in source_names]
        print(f"{'全局w=' + str(w):<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f}")
    aps = [adaptive_results.get(s, {}).get('ap', 0) for s in source_names]
    print(f"{'自适应w=δ/δ_max':<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f}")

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
        'domain_shifts': convert(domain_shifts),
        'delta_max': float(delta_max),
        'global_results': convert(global_results),
        'adaptive_results': convert(adaptive_results),
        'note': '自适应谱归一化实验（IO优化版，每个域对只加载一次）',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
