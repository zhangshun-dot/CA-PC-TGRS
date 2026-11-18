# -*- coding: utf-8 -*-
"""usc_experiment_v2.py — USC模块改进实验

改进点：
  1. PCA维度扫描：K ∈ {7, 15, 30, 50, 100, 512}，找到最优低秩维度
  2. 三模式消融：USC-only / TGE-only / USC+TGE，验证空间校正的独立贡献
  3. 逐类最优α分析：找到每个类别的最优校正强度，分析与物理先验的相关性

输出：实验数据/usc_experiment_v2_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'usc_experiment_v2_results.json'

# ── 配置 ──
BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_VALUES = [7, 15, 30, 50, 100, 512]  # PCA维度扫描
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


# ══════════════════════ 数据加载（复用v1的缓存逻辑） ══════════════════════

def load_source_features():
    print('=== 加载源域(DIOR)特征 ===')
    source = {}
    for name, coco_cls, dior_cls in TARGETS:
        print(f'  {dior_cls}...', end=' ', flush=True)
        images, gts = U.load_dior(dior_cls, n=50)
        if not images:
            print('无数据'); continue
        crops = []
        for img, g in zip(images, gts):
            for gt in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10: continue
                crops.append(img.crop((x1, y1, x2, y2)))
        if not crops:
            print('无标注框'); continue
        F = T.encode_clip_batch(crops).cpu().numpy()
        proto = F.mean(axis=0); proto = proto / (np.linalg.norm(proto) + 1e-8)
        source[dior_cls] = {'prototype': proto, 'features': F}
        print(f'{len(crops)}框')
    return source


def load_target_data():
    print('\n=== 加载目标域(COCO)数据 ===')
    target = {}
    for name, coco_cls, dior_cls in TARGETS:
        print(f'  {coco_cls}...', end=' ', flush=True)
        images, gts = U.load_coco(coco_cls, n=200)
        if not images:
            print('无数据'); continue
        cache_p = DATA / f'tge_v2_dets_COCO-{name}.pkl'
        if cache_p.exists():
            dets = pickle.load(open(cache_p, 'rb'))
        else:
            dets = U.gdino_detect(images, coco_cls, 0.0)
            pickle.dump(dets, open(cache_p, 'wb'))
        per_img = []; crops_all = []
        for (boxes, scores), img, g in zip(dets, images, gts):
            bl, sl, ci = [], [], []
            for box, sc in zip(boxes, scores):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10: continue
                if float(sc) < PRE_FILTER: continue
                bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
                crops_all.append(img.crop((x1, y1, x2, y2)))
            per_img.append({
                'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
            })
        F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
        target[coco_cls] = {'per_img': per_img, 'features': F}
        print(f'{len(images)}图, {len(crops_all)}候选框')
    return target


# ══════════════════════ USC 模块（支持任意K） ══════════════════════

class USC:
    def __init__(self, n_components=7):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.domain_shift_ = None
        self.explained_variance_ = None

    def fit(self, source_features, target_features):
        all_features = np.vstack([source_features, target_features])
        self.mean_ = all_features.mean(axis=0)
        X_centered = all_features - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        K = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:K]
        self.explained_variance_ = (S[:K] ** 2).sum() / (S ** 2).sum()
        src_proj = self._project(source_features)
        tgt_proj = self._project(target_features)
        self.domain_shift_ = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        shift_norm = np.linalg.norm(self.domain_shift_)
        if shift_norm > 1e-8:
            self.domain_shift_ = self.domain_shift_ / shift_norm
        return self

    def _project(self, X):
        return (X - self.mean_) @ self.components_.T

    def _reconstruct(self, Z):
        return Z @ self.components_ + self.mean_

    def correct(self, X, alpha=1.0):
        Z = self._project(X)
        Z_corrected = Z + alpha * self.domain_shift_
        X_corrected = self._reconstruct(Z_corrected)
        norms = np.linalg.norm(X_corrected, axis=1, keepdims=True)
        return X_corrected / np.maximum(norms, 1e-8)

    def correct_prototype(self, p, alpha=1.0):
        return self.correct(p.reshape(1, -1), alpha)[0]


# ══════════════════════ 评估函数 ══════════════════════

def build_proto(src_proto, coco_cls, dior_cls, beta=BETA, use_tge=True):
    """构建原型：use_tge=True时做文本外推，False时只用源域原型。"""
    if not use_tge:
        return src_proto.copy()
    s_emb = T.text_avg([SRC_TMPL], dior_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def build_neg_proto(source, pos_cls):
    protos = {k: v['prototype'] for k, v in source.items() if k != pos_cls}
    if not protos: return None
    pos_p = source[pos_cls]['prototype']
    sims = {k: float(np.dot(pos_p, v)) for k, v in protos.items()}
    top3 = sorted(sims.items(), key=lambda x: -x[1])[:3]
    neg = np.mean([protos[k] for k, _ in top3], axis=0)
    return neg / (np.linalg.norm(neg) + 1e-8)


def evaluate(target_data, proto, neg_proto, lam=LAM):
    per_img = target_data['per_img']
    F = target_data['features']
    for p in per_img:
        n = len(p['gd'])
        if n:
            f_boxes = F[p['crop_idx']]
            pos_sim = f_boxes @ proto
            neg_sim = f_boxes @ neg_proto if neg_proto is not None else np.zeros(n)
            vis = np.maximum(0.0, pos_sim - lam * neg_sim)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)
    preds = [(p['box'], p['fused']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return U.compute_ap50_img(preds, gts)


def evaluate_baseline(target_data):
    per_img = target_data['per_img']
    preds = [(p['box'], p['gd']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return U.compute_ap50_img(preds, gts)


# ══════════════════════ 主实验 ══════════════════════

def main():
    print('=' * 60)
    print('USC v2 改进实验')
    print('=' * 60)

    source = load_source_features()
    target = load_target_data()

    all_src = np.vstack([v['features'] for v in source.values()])
    all_tgt = np.vstack([v['features'] for v in target.values()])
    print(f'\n源域特征: {all_src.shape}, 目标域特征: {all_tgt.shape}')

    # 基线
    print('\n=== 基线评估 ===')
    baseline_aps = {}
    for name, coco_cls, dior_cls in TARGETS:
        if coco_cls in target:
            baseline_aps[coco_cls] = evaluate_baseline(target[coco_cls])
    mean_baseline = np.mean(list(baseline_aps.values()))
    print(f'  均值基线AP: {mean_baseline:.4f}')

    results = {
        'config': {'beta': BETA, 'lam': LAM, 'k_values': K_VALUES, 'alphas': ALPHAS},
        'baseline': {'per_class': baseline_aps, 'mean': mean_baseline},
        'pca_scan': {},
        'ablation': {},
        'per_class_best_alpha': {},
    }

    # ═══ 实验1：PCA维度扫描 ═══
    print('\n' + '=' * 60)
    print('实验1：PCA维度扫描（K值对USC性能的影响）')
    print('=' * 60)

    for K in K_VALUES:
        print(f'\n--- K={K} ---')
        usc = USC(n_components=K).fit(all_src, all_tgt)
        print(f'  解释方差: {usc.explained_variance_*100:.1f}%')

        k_results = {}
        for alpha in ALPHAS:
            aps = {}
            for name, coco_cls, dior_cls in TARGETS:
                if coco_cls not in target or dior_cls not in source: continue
                src_corr = usc.correct_prototype(source[dior_cls]['prototype'], alpha)
                p_hat = build_proto(src_corr, coco_cls, dior_cls, use_tge=True)
                neg_raw = build_neg_proto(source, dior_cls)
                neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
                aps[coco_cls] = evaluate(target[coco_cls], p_hat, neg)
            mean_ap = np.mean(list(aps.values()))
            k_results[str(alpha)] = {'per_class': aps, 'mean': mean_ap}

        best_alpha = max(ALPHAS, key=lambda a: k_results[str(a)]['mean'])
        best_mean = k_results[str(best_alpha)]['mean']
        print(f'  最佳α={best_alpha}, 均值AP={best_mean:.4f}, '
              f'gain={best_mean-mean_baseline:+.4f}')
        results['pca_scan'][str(K)] = {
            'explained_variance': usc.explained_variance_,
            'best_alpha': best_alpha,
            'best_mean': best_mean,
            'best_gain': best_mean - mean_baseline,
            'all_alphas': k_results,
        }

    # ═══ 实验2：三模式消融（用最佳K） ═══
    print('\n' + '=' * 60)
    print('实验2：三模式消融（USC-only / TGE-only / USC+TGE）')
    print('=' * 60)

    best_K = max(K_VALUES, key=lambda k: results['pca_scan'][str(k)]['best_mean'])
    print(f'使用最佳K={best_K}')
    usc = USC(n_components=best_K).fit(all_src, all_tgt)

    ablation_modes = {
        'baseline': {'use_usc': False, 'use_tge': False, 'alpha': 0.0},
        'tge_only': {'use_usc': False, 'use_tge': True, 'alpha': 0.0},
        'usc_only': {'use_usc': True, 'use_tge': False, 'alpha': None},  # 扫描α
        'usc_plus_tge': {'use_usc': True, 'use_tge': True, 'alpha': None},  # 扫描α
    }

    for mode_name, mode_cfg in ablation_modes.items():
        print(f'\n--- {mode_name} ---')
        if mode_cfg['alpha'] is not None:
            # 固定α
            alpha = mode_cfg['alpha']
            aps = {}
            for name, coco_cls, dior_cls in TARGETS:
                if coco_cls not in target or dior_cls not in source: continue
                src_p = source[dior_cls]['prototype']
                if mode_cfg['use_usc']:
                    src_p = usc.correct_prototype(src_p, alpha)
                p_hat = build_proto(src_p, coco_cls, dior_cls, use_tge=mode_cfg['use_tge'])
                neg_raw = build_neg_proto(source, dior_cls)
                if mode_cfg['use_usc'] and neg_raw is not None:
                    neg = usc.correct_prototype(neg_raw, alpha)
                else:
                    neg = neg_raw
                aps[coco_cls] = evaluate(target[coco_cls], p_hat, neg)
            mean_ap = np.mean(list(aps.values()))
            print(f'  均值AP={mean_ap:.4f}, gain={mean_ap-mean_baseline:+.4f}')
            results['ablation'][mode_name] = {
                'alpha': alpha, 'per_class': aps, 'mean': mean_ap,
                'gain': mean_ap - mean_baseline,
            }
        else:
            # 扫描α
            mode_results = {}
            for alpha in ALPHAS:
                aps = {}
                for name, coco_cls, dior_cls in TARGETS:
                    if coco_cls not in target or dior_cls not in source: continue
                    src_p = usc.correct_prototype(source[dior_cls]['prototype'], alpha)
                    p_hat = build_proto(src_p, coco_cls, dior_cls, use_tge=mode_cfg['use_tge'])
                    neg_raw = build_neg_proto(source, dior_cls)
                    neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
                    aps[coco_cls] = evaluate(target[coco_cls], p_hat, neg)
                mean_ap = np.mean(list(aps.values()))
                mode_results[str(alpha)] = {'per_class': aps, 'mean': mean_ap}
            best_alpha = max(ALPHAS, key=lambda a: mode_results[str(a)]['mean'])
            best_mean = mode_results[str(best_alpha)]['mean']
            print(f'  最佳α={best_alpha}, 均值AP={best_mean:.4f}, '
                  f'gain={best_mean-mean_baseline:+.4f}')
            results['ablation'][mode_name] = {
                'best_alpha': best_alpha, 'best_mean': best_mean,
                'best_gain': best_mean - mean_baseline,
                'all_alphas': mode_results,
            }

    # ═══ 实验3：逐类最优α分析 ═══
    print('\n' + '=' * 60)
    print('实验3：逐类最优α分析')
    print('=' * 60)

    per_class_best = {}
    for name, coco_cls, dior_cls in TARGETS:
        if coco_cls not in target or dior_cls not in source: continue
        best_alpha = 0.0
        best_ap = baseline_aps[coco_cls]
        alpha_aps = {}
        for alpha in ALPHAS:
            src_corr = usc.correct_prototype(source[dior_cls]['prototype'], alpha)
            p_hat = build_proto(src_corr, coco_cls, dior_cls, use_tge=True)
            neg_raw = build_neg_proto(source, dior_cls)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            ap = evaluate(target[coco_cls], p_hat, neg)
            alpha_aps[str(alpha)] = ap
            if ap > best_ap:
                best_ap = ap
                best_alpha = alpha
        per_class_best[coco_cls] = {
            'best_alpha': best_alpha,
            'best_ap': best_ap,
            'baseline_ap': baseline_aps[coco_cls],
            'gain': best_ap - baseline_aps[coco_cls],
            'all_alphas': alpha_aps,
        }
        print(f'  {coco_cls}: 最佳α={best_alpha}, AP={best_ap:.4f}, '
              f'gain={best_ap-baseline_aps[coco_cls]:+.4f}')

    # 逐类最优α的"oracle"组合（每个类别用自己的最优α）
    oracle_mean = np.mean([v['best_ap'] for v in per_class_best.values()])
    print(f'\n  逐类最优α组合(oracle): 均值AP={oracle_mean:.4f}, '
          f'gain={oracle_mean-mean_baseline:+.4f}')
    results['per_class_best_alpha'] = per_class_best
    results['per_class_oracle_mean'] = oracle_mean
    results['per_class_oracle_gain'] = oracle_mean - mean_baseline

    # 保存
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {OUT}')
    return results


if __name__ == '__main__':
    main()
