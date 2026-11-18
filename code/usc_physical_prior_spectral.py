# -*- coding: utf-8 -*-
"""物理先验 + 频域校正实验（秩1仿射版本）
核心思想：用物理先验（类别尺度大小）指导谱归一化的权重分配
- 大目标（飞机、船）→ 整体结构重要 → 谱归一化权重大
- 中等目标（车辆、巴士）→ 谱归一化权重中等
- 小目标（摩托车、自行车）→ 细节重要 → 谱归一化权重小

对比：
  1. 无谱归一化（基线）
  2. 全局谱归一化（w=1，所有类别相同权重）
  3. 物理先验指导的谱归一化（每个类别不同权重，基于尺度大小）
  4. 物理先验 + 自适应（尺度权重 × 域偏移自适应）
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

OUT = DATA / 'usc_physical_prior_spectral_results.json'

# 物理先验：类别尺度大小（0-1，越大表示目标越大）
# 基于遥感地物的典型物理尺寸
CATEGORY_SCALE_PRIOR = {
    'airplane': 0.9,      # 大目标（翼展数十米）
    'ship': 0.85,          # 大目标（船长数十米）
    'vehicle': 0.5,        # 中等目标（车长4-5米）
    'car': 0.5,            # 中等目标
    'truck': 0.6,          # 中等偏大目标
    'bus': 0.65,           # 中等偏大目标（车长10米）
    'motorcycle': 0.2,     # 小目标（车长2米）
    'bicycle': 0.15,       # 小目标（车长1.5米）
    'boat': 0.4,           # 中等偏小目标
    'tennis-court': 0.95,  # 大目标
    'basketball-court': 0.9,  # 大目标
    'ground-track-field': 1.0,  # 最大目标
    'storage-tank': 0.7,   # 中等偏大目标
    'swimming-pool': 0.75, # 中等偏大目标
}


class Rank1AffineUSC:
    """秩1仿射USC（第二篇论文核心方法，只有1个参数α）"""
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


def apply_spectral_norm_per_class(features, src_std, tgt_std, weight):
    """应用谱归一化（每个类别可以有不同权重）"""
    if weight <= 0:
        return features
    scale = tgt_std / (src_std + 1e-8)
    return features + weight * (features * scale - features)


def compute_domain_shift(source_features, target_features):
    """计算域偏移大小"""
    src_mean = source_features.mean(axis=0)
    tgt_mean = target_features.mean(axis=0)
    return float(np.linalg.norm(tgt_mean - src_mean))


def get_category_scale(category_name):
    """获取类别的尺度先验"""
    name_lower = category_name.lower()
    for key, value in CATEGORY_SCALE_PRIOR.items():
        if key in name_lower:
            return value
    return 0.5  默认中等尺度


def load_domain_data(source_name, spectral_mode='none', domain_delta_max=1.0):
    """加载域对数据
    spectral_mode:
      - 'none': 无谱归一化
      - 'global': 全局谱归一化（w=1）
      - 'physical': 物理先验指导的谱归一化（每个类别不同权重）
      - 'physical_adaptive': 物理先验 + 域偏移自适应
    """
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

    domain_shift = compute_domain_shift(all_src, all_tgt)
    print(f"  域偏移大小: {domain_shift:.4f}")

    src_std = np.std(all_src, axis=0) + 1e-8
    tgt_std = np.std(all_tgt, axis=0) + 1e-8

    # 应用谱归一化
    if spectral_mode == 'global':
        print(f"  全局谱归一化（w=1.0）...")
        all_src = apply_spectral_norm_per_class(all_src, src_std, tgt_std, 1.0)
        all_tgt = apply_spectral_norm_per_class(all_tgt, src_std, tgt_std, 1.0)
        for s in source_data:
            source_data[s]['features'] = apply_spectral_norm_per_class(source_data[s]['features'], src_std, tgt_std, 1.0)
            source_data[s]['prototype'] = apply_spectral_norm_per_class(source_data[s]['prototype'], src_std, tgt_std, 1.0)
        for c in target_data:
            target_data[c]['features'] = apply_spectral_norm_per_class(target_data[c]['features'], src_std, tgt_std, 1.0)

    elif spectral_mode == 'physical':
        print(f"  物理先验指导的谱归一化...")
        # 每个类别用不同权重
        for eval_name, coco_cls, src_cls in valid_targets:
            scale = get_category_scale(src_cls)
            # 尺度越大，谱归一化权重越大
            # 大目标（scale=0.9）→ w=0.9
            # 小目标（scale=0.15）→ w=0.15
            w = scale
            print(f"    {src_cls}: 尺度先验={scale:.2f}, 谱归一化权重={w:.2f}")
            source_data[src_cls]['features'] = apply_spectral_norm_per_class(source_data[src_cls]['features'], src_std, tgt_std, w)
            source_data[src_cls]['prototype'] = apply_spectral_norm_per_class(source_data[src_cls]['prototype'], src_std, tgt_std, w)
            target_data[coco_cls]['features'] = apply_spectral_norm_per_class(target_data[coco_cls]['features'], src_std, tgt_std, w)
        # 重新计算all_src和all_tgt
        all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
        all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])

    elif spectral_mode == 'physical_adaptive':
        print(f"  物理先验 + 域偏移自适应...")
        # 域偏移自适应因子：δ / δ_max
        adaptive_factor = min(domain_shift / domain_delta_max, 1.0)
        print(f"    域偏移自适应因子: {adaptive_factor:.4f}")
        for eval_name, coco_cls, src_cls in valid_targets:
            scale = get_category_scale(src_cls)
            # 物理先验权重 × 域偏移自适应因子
            w = scale * adaptive_factor
            print(f"    {src_cls}: 尺度先验={scale:.2f}, 自适应因子={adaptive_factor:.4f}, 最终权重={w:.4f}")
            source_data[src_cls]['features'] = apply_spectral_norm_per_class(source_data[src_cls]['features'], src_std, tgt_std, w)
            source_data[src_cls]['prototype'] = apply_spectral_norm_per_class(source_data[src_cls]['prototype'], src_std, tgt_std, w)
            target_data[coco_cls]['features'] = apply_spectral_norm_per_class(target_data[coco_cls]['features'], src_std, tgt_std, w)
        all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
        all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])

    print(f"  拟合秩1仿射USC...")
    usc = Rank1AffineUSC(use_pca=True, n_components=30)
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
        'domain_shift': domain_shift,
    }


def evaluate_with_params(domain_data, alpha):
    usc = domain_data['usc']
    target_data = domain_data['target_data']
    all_protos = domain_data['all_protos']
    protos_gpu = domain_data['protos_gpu']
    text_deltas = domain_data['text_deltas']
    valid_targets = domain_data['valid_targets']
    aps = []
    for eval_name, coco_cls, src_cls in valid_targets:
        p_corr_gpu = usc.correct_prototype_gpu(protos_gpu[src_cls], alpha)
        p_corr = p_corr_gpu.cpu().numpy()
        neg_raw = build_neg_proto(all_protos, src_cls)
        if neg_raw is not None:
            neg_gpu = torch.tensor(neg_raw, dtype=torch.float32, device=device)
            neg_corr_gpu = usc.correct_prototype_gpu(neg_gpu, alpha)
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
    print(f"物理先验 + 频域校正实验（秩1仿射）")

    # 第一步：计算每个域对的域偏移大小
    print(f"\n{'='*60}")
    print("第一步：计算每个域对的域偏移大小")
    print(f"{'='*60}")
    domain_shifts = {}
    for src in source_names:
        data = load_domain_data(src, spectral_mode='none')
        if data:
            domain_shifts[src] = data['domain_shift']
            print(f"  {src}: 域偏移={data['domain_shift']:.4f}")
            del data
    delta_max = max(domain_shifts.values())
    print(f"  最大域偏移: {delta_max:.4f}")

    # 第二步：测试不同的谱归一化模式
    print(f"\n{'='*60}")
    print("第二步：测试不同的谱归一化模式")
    print(f"{'='*60}")

    modes = ['none', 'global', 'physical', 'physical_adaptive']
    mode_names = {
        'none': '无谱归一化（基线）',
        'global': '全局谱归一化（w=1）',
        'physical': '物理先验指导（按尺度）',
        'physical_adaptive': '物理先验 + 域偏移自适应',
    }

    all_results = {}
    for mode in modes:
        print(f"\n--- {mode_names[mode]} ---")
        mode_results = {}
        for src in source_names:
            data = load_domain_data(src, spectral_mode=mode, domain_delta_max=delta_max)
            if data is None:
                continue
            best_ap = -1
            best_alpha = None
            for alpha in ALPHAS:
                ap = evaluate_with_params(data, alpha)
                if ap > best_ap:
                    best_ap = ap
                    best_alpha = alpha
            print(f"  {src}: AP={best_ap:.4f}, α={best_alpha}")
            mode_results[src] = {'ap': best_ap, 'alpha': best_alpha}
            del data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_results[mode] = mode_results

    # 汇总
    print(f"\n{'='*60}")
    print("汇总：物理先验 + 频域校正")
    print(f"{'='*60}")
    print(f"{'模式':<35} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8} {'vs基线':>10}")
    print("-" * 95)

    baseline_aps = [all_results['none'].get(s, {}).get('ap', 0) for s in source_names]
    baseline_mean = np.mean(baseline_aps)

    for mode in modes:
        aps = [all_results[mode].get(s, {}).get('ap', 0) for s in source_names]
        mean_ap = np.mean(aps)
        gain = mean_ap - baseline_mean
        print(f"{mode_names[mode]:<35} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {mean_ap:>8.4f} {gain:>+10.4f}")

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
        'domain_shifts': convert(domain_shifts),
        'delta_max': float(delta_max),
        'category_scale_prior': CATEGORY_SCALE_PRIOR,
        'all_results': convert(all_results),
        'mode_names': mode_names,
        'note': '物理先验 + 频域校正实验（秩1仿射）',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
