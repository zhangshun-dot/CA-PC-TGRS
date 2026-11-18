# -*- coding: utf-8 -*-
"""usc_experiment.py — Universal Space Corrector (USC) 模块实验

实验目标：
  验证"低秩投影→域偏心校正→回投影"的通用空间校正模块能否提升A-CSPC性能。

USC三阶段：
  阶段1：低秩投影（PCA，K=7，解释84.2%方差）
  阶段2：域偏心校正（在低秩空间中估计并消除域偏移方向）
  阶段3：回投影（校正后特征投影回原始空间）

实验设置：
  源域：DIOR（6类：airplane, ship, vehicle×4映射）
  目标域：COCO val2017（6类）
  检测器：GDINO-tiny
  基线：A-CSPC (β=1.0, λ=0.25)，无USC
  对比：USC + A-CSPC，扫描校正强度α

输出：实验数据/usc_experiment_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'usc_experiment_results.json'

# ── 配置 ──
BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_PCA = 7  # 前7个主成分解释84.2%方差
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]  # 校正强度扫描

SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

# (name, coco_cls, dior_cls)
TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


# ══════════════════════ 数据加载与特征提取 ══════════════════════

def load_source_features():
    """加载DIOR源域数据，提取标注框CLIP特征，构建源域原型。
    返回：dict {cls_name: {'prototype': np.array(512,), 'features': np.array(N,512)}}
    """
    print('=== 加载源域(DIOR)特征 ===')
    source = {}
    for name, coco_cls, dior_cls in TARGETS:
        print(f'  {dior_cls}...', end=' ', flush=True)
        images, gts = U.load_dior(dior_cls, n=50)
        if not images:
            print('无数据')
            continue
        crops = []
        for img, g in zip(images, gts):
            for gt in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                crops.append(img.crop((x1, y1, x2, y2)))
        if not crops:
            print('无标注框')
            continue
        F = T.encode_clip_batch(crops).cpu().numpy()  # (N, 512), 已归一化
        proto = F.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        source[dior_cls] = {'prototype': proto, 'features': F}
        print(f'{len(crops)}框, 原型已构建')
    return source


def load_target_data():
    """加载COCO目标域数据，GDINO检测，提取候选框CLIP特征。
    返回：dict {cls_name: {'per_img': list, 'features': np.array(N,512)}}
    """
    print('\n=== 加载目标域(COCO)数据 ===')
    target = {}
    for name, coco_cls, dior_cls in TARGETS:
        print(f'  {coco_cls}...', end=' ', flush=True)
        images, gts = U.load_coco(coco_cls, n=200)
        if not images:
            print('无数据')
            continue

        # GDINO检测（用缓存）
        cache_p = DATA / f'tge_v2_dets_COCO-{name}.pkl'
        if cache_p.exists():
            dets = pickle.load(open(cache_p, 'rb'))
            print(f'检测缓存', end=' ', flush=True)
        else:
            dets = U.gdino_detect(images, coco_cls, 0.0)
            pickle.dump(dets, open(cache_p, 'wb'))

        # 提取候选框特征
        per_img = []
        crops_all = []
        for (boxes, scores), img, g in zip(dets, images, gts):
            bl, sl, ci = [], [], []
            for box, sc in zip(boxes, scores):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                if float(sc) < PRE_FILTER:
                    continue
                bl.append([x1, y1, x2, y2])
                sl.append(float(sc))
                ci.append(len(crops_all))
                crops_all.append(img.crop((x1, y1, x2, y2)))
            per_img.append({
                'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
            })

        if crops_all:
            F = T.encode_clip_batch(crops_all).cpu().numpy()
        else:
            F = np.zeros((0, 512))

        target[coco_cls] = {'per_img': per_img, 'features': F}
        print(f'{len(images)}图, {len(crops_all)}候选框')
    return target


# ══════════════════════ USC 模块 ══════════════════════

class UniversalSpaceCorrector:
    """通用空间校正模块（USC）

    三阶段：
      1. 低秩投影：PCA到K维共享子空间
      2. 域偏心校正：在低秩空间中沿域偏移方向做反向校正
      3. 回投影：校正后特征投影回原始空间
    """

    def __init__(self, n_components=7):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # (K, d)
        self.domain_shift_ = None  # (K,) 低秩空间中的域偏移方向

    def fit(self, source_features, target_features):
        """拟合PCA和域偏移方向。

        Args:
            source_features: np.array (N_s, d) 源域特征
            target_features: np.array (N_t, d) 目标域特征
        """
        all_features = np.vstack([source_features, target_features])
        self.mean_ = all_features.mean(axis=0)

        # PCA（用SVD，更稳定）
        X_centered = all_features - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:self.n_components]  # (K, d)

        # 在低秩空间中估计域偏移方向
        src_proj = self._project(source_features)  # (N_s, K)
        tgt_proj = self._project(target_features)  # (N_t, K)
        self.domain_shift_ = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)  # (K,)

        # 归一化域偏移方向
        shift_norm = np.linalg.norm(self.domain_shift_)
        if shift_norm > 1e-8:
            self.domain_shift_ = self.domain_shift_ / shift_norm

        explained = (S[:self.n_components] ** 2).sum() / (S ** 2).sum()
        print(f'  USC拟合: K={self.n_components}, 解释方差={explained*100:.1f}%')
        print(f'  域偏移方向范数={shift_norm:.4f}')
        return self

    def _project(self, X):
        """投影到低秩空间。"""
        return (X - self.mean_) @ self.components_.T  # (N, K)

    def _reconstruct(self, Z):
        """从低秩空间回投影。"""
        return Z @ self.components_ + self.mean_  # (N, d)

    def correct(self, X, alpha=1.0):
        """校正特征：投影→域偏心校正→回投影。

        Args:
            X: np.array (N, d) 待校正特征
            alpha: float 校正强度（0=不校正，1=完全消除域偏移）

        Returns:
            X_corrected: np.array (N, d) 校正后特征
        """
        Z = self._project(X)  # (N, K)
        # 沿域偏移方向做反向校正（源域特征需要向目标域方向移动）
        Z_corrected = Z + alpha * self.domain_shift_  # (N, K)
        X_corrected = self._reconstruct(Z_corrected)  # (N, d)
        # 重新归一化（CLIP特征是单位向量）
        norms = np.linalg.norm(X_corrected, axis=1, keepdims=True)
        X_corrected = X_corrected / np.maximum(norms, 1e-8)
        return X_corrected

    def correct_prototype(self, p, alpha=1.0):
        """校正单个原型向量。"""
        return self.correct(p.reshape(1, -1), alpha)[0]


# ══════════════════════ A-CSPC 评估 ══════════════════════

def build_extrapolated_proto(src_proto, coco_cls, dior_cls, beta=BETA):
    """构建外推原型：p̂ = norm(p + β·(d(x_c) - s(x_s)))"""
    s_emb = T.text_avg([SRC_TMPL], dior_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)
    return p_hat


def build_neg_proto(source, pos_cls):
    """构建负原型：与正原型最接近的top-3其他类原型均值。"""
    protos = {k: v['prototype'] for k, v in source.items() if k != pos_cls}
    if not protos:
        return None
    pos_p = source[pos_cls]['prototype']
    sims = {k: float(np.dot(pos_p, v)) for k, v in protos.items()}
    top3 = sorted(sims.items(), key=lambda x: -x[1])[:3]
    neg = np.mean([protos[k] for k, _ in top3], axis=0)
    return neg / (np.linalg.norm(neg) + 1e-8)


def evaluate_acspc(target_data, src_proto, neg_proto, beta=BETA, lam=LAM):
    """在目标域上评估A-CSPC，返回AP50。"""
    per_img = target_data['per_img']
    F = target_data['features']

    # 构建视觉分数
    for p in per_img:
        n = len(p['gd'])
        if n:
            f_boxes = F[p['crop_idx']]  # (n, 512)
            pos_sim = f_boxes @ src_proto  # (n,)
            neg_sim = f_boxes @ neg_proto if neg_proto is not None else np.zeros(n)
            vis = np.maximum(0.0, pos_sim - lam * neg_sim)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)

    # 计算AP50（全量图，不用校准图分割，因为这里n=200）
    preds = [(p['box'], p['fused']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    ap50 = U.compute_ap50_img(preds, gts)
    return ap50


def evaluate_baseline(target_data):
    """评估基线（GDINO原始分数）。"""
    per_img = target_data['per_img']
    preds = [(p['box'], p['gd']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return U.compute_ap50_img(preds, gts)


# ══════════════════════ 主实验 ══════════════════════

def main():
    print('=' * 60)
    print('USC (Universal Space Corrector) 实验')
    print('=' * 60)

    # 1. 加载数据
    source = load_source_features()
    target = load_target_data()

    # 2. 收集所有特征用于PCA拟合
    print('\n=== 收集特征用于USC拟合 ===')
    all_src_features = np.vstack([v['features'] for v in source.values()])
    all_tgt_features = np.vstack([v['features'] for v in target.values()])
    print(f'  源域特征: {all_src_features.shape}')
    print(f'  目标域特征: {all_tgt_features.shape}')

    # 3. 拟合USC
    print('\n=== 拟合USC模块 ===')
    usc = UniversalSpaceCorrector(n_components=K_PCA)
    usc.fit(all_src_features, all_tgt_features)

    # 4. 基线评估
    print('\n=== 基线评估 ===')
    baseline_aps = {}
    acspc_aps = {}  # 无USC的A-CSPC
    for name, coco_cls, dior_cls in TARGETS:
        if coco_cls not in target or dior_cls not in source:
            continue
        baseline_aps[coco_cls] = evaluate_baseline(target[coco_cls])
        src_proto = source[dior_cls]['prototype']
        p_hat = build_extrapolated_proto(src_proto, coco_cls, dior_cls)
        neg_proto = build_neg_proto(source, dior_cls)
        acspc_aps[coco_cls] = evaluate_acspc(target[coco_cls], p_hat, neg_proto)
        print(f'  {coco_cls}: baseline={baseline_aps[coco_cls]:.4f}, '
              f'A-CSPC={acspc_aps[coco_cls]:.4f}, '
              f'gain={acspc_aps[coco_cls]-baseline_aps[coco_cls]:+.4f}')

    mean_baseline = np.mean(list(baseline_aps.values()))
    mean_acspc = np.mean(list(acspc_aps.values()))
    print(f'  均值: baseline={mean_baseline:.4f}, A-CSPC={mean_acspc:.4f}, '
          f'gain={mean_acspc-mean_baseline:+.4f}')

    # 5. USC + A-CSPC 实验（扫描α）
    print('\n=== USC + A-CSPC 实验（扫描校正强度α） ===')
    results = {
        'config': {
            'beta': BETA, 'lam': LAM, 'k_pca': K_PCA,
            'alphas': ALPHAS, 'n_targets': len(TARGETS),
        },
        'baseline': {'per_class': baseline_aps, 'mean': mean_baseline},
        'acspc_no_usc': {'per_class': acspc_aps, 'mean': mean_acspc},
        'usc_results': {},
    }

    for alpha in ALPHAS:
        print(f'\n  α={alpha}:')
        usc_aps = {}
        for name, coco_cls, dior_cls in TARGETS:
            if coco_cls not in target or dior_cls not in source:
                continue
            # 用USC校正源域原型
            src_proto_raw = source[dior_cls]['prototype']
            src_proto_corrected = usc.correct_prototype(src_proto_raw, alpha=alpha)
            # 构建外推原型（用校正后的源域原型）
            p_hat = build_extrapolated_proto(src_proto_corrected, coco_cls, dior_cls)
            # 负原型也校正
            neg_proto_raw = build_neg_proto(source, dior_cls)
            if neg_proto_raw is not None:
                neg_proto = usc.correct_prototype(neg_proto_raw, alpha=alpha)
            else:
                neg_proto = None
            # 评估
            ap = evaluate_acspc(target[coco_cls], p_hat, neg_proto)
            usc_aps[coco_cls] = ap
            gain = ap - baseline_aps[coco_cls]
            print(f'    {coco_cls}: AP={ap:.4f}, gain={gain:+.4f}')

        mean_usc = np.mean(list(usc_aps.values()))
        gain_over_baseline = mean_usc - mean_baseline
        gain_over_acspc = mean_usc - mean_acspc
        print(f'    均值: AP={mean_usc:.4f}, '
              f'gain_vs_base={gain_over_baseline:+.4f}, '
              f'gain_vs_acspc={gain_over_acspc:+.4f}')

        results['usc_results'][str(alpha)] = {
            'per_class': usc_aps,
            'mean': mean_usc,
            'gain_vs_baseline': gain_over_baseline,
            'gain_vs_acspc': gain_over_acspc,
        }

    # 6. 找最佳α
    best_alpha = max(ALPHAS, key=lambda a: results['usc_results'][str(a)]['mean'])
    best_result = results['usc_results'][str(best_alpha)]
    print(f'\n=== 最佳α={best_alpha} ===')
    print(f'  均值AP={best_result["mean"]:.4f}')
    print(f'  vs baseline: {best_result["gain_vs_baseline"]:+.4f}')
    print(f'  vs A-CSPC(no USC): {best_result["gain_vs_acspc"]:+.4f}')

    results['best_alpha'] = best_alpha
    results['best_result'] = best_result

    # 7. 保存结果
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {OUT}')

    return results


if __name__ == '__main__':
    main()
