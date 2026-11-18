# -*- coding: utf-8 -*-
"""usc_experiment_v3.py — 物理先验驱动的逐类自适应α实验

核心问题：能否用物理先验R(c)（G/S/I/C四因子）预测每个类别的最优校正强度α(c)？

实验设计：
  1. 为6个类别定义G/S/I/C物理先验值
  2. 计算风险评分R(c) = w_G·G + w_S·S + w_I·I + w_C·C
  3. 自适应α：α(c) = α_min + (α_max - α_min) · sigmoid(k·(R(c) - R_0))
  4. 扫描超参数（权重、α_min、α_max、k、R_0），找到最佳配置
  5. 和固定α、逐类oracle对比
  6. 分析物理先验和最佳α的相关性

输出：实验数据/usc_experiment_v3_results.json
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
OUT = DATA / 'usc_experiment_v3_results.json'

# ── 配置 ──
BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_PCA = 30  # v2实验确定的最佳维度

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

# ══════════════════════ 物理先验定义 ══════════════════════
# G: 粒度失配（源域→目标域的类别粒度差异，0=完全匹配，1=粗到细严重失配）
# S: 尺度分辨率（目标域中该类别实例的像素尺度变异，0=尺度统一，1=尺度变化极大）
# I: 类内变异（该类别内部形态变异程度，0=形态统一，1=形态多样）
# C: 类间混淆度（该类别与其他类别的视觉混淆程度，0=独特，1=易混淆）
# 所有值在[0,1]之间

PHYSICAL_PRIORS = {
    'airplane': {
        'G': 0.1,  # airplane→airplane，粒度完全匹配
        'S': 0.4,  # 飞机在COCO中尺度中等，有一定变异
        'I': 0.3,  # 飞机形态相对统一（客机、战斗机，但整体轮廓相似）
        'C': 0.1,  # 飞机和其他类别差异大，不易混淆
    },
    'boat': {
        'G': 0.5,  # ship→boat，源域粗粒度，目标域细粒度，中等失配
        'S': 0.7,  # 船在COCO中尺度变化极大（小艇到货轮）
        'I': 0.8,  # 船的形态变异极大（帆船、货轮、游艇、独木舟）
        'C': 0.4,  # 船和其他水上物体有一定混淆
    },
    'car': {
        'G': 0.9,  # vehicle→car，源域粗粒度，目标域细粒度，严重失配
        'S': 0.8,  # 汽车在COCO中尺度变化大（远近、大小）
        'I': 0.7,  # 汽车形态变异大（轿车、SUV、跑车、面包车）
        'C': 0.7,  # 汽车和truck/bus容易混淆
    },
    'truck': {
        'G': 0.9,  # vehicle→truck，粗到细，严重失配
        'S': 0.5,  # 卡车在COCO中尺度较大，变异中等
        'I': 0.5,  # 卡车形态变异中等（货车、皮卡、工程车）
        'C': 0.6,  # 卡车和car/bus有一定混淆
    },
    'bus': {
        'G': 0.9,  # vehicle→bus，粗到细，严重失配
        'S': 0.4,  # 公交车在COCO中尺度较大，相对统一
        'I': 0.3,  # 公交车形态相对统一（长方形车身）
        'C': 0.5,  # 公交车和truck有一定混淆
    },
    'motorcycle': {
        'G': 0.9,  # vehicle→motorcycle，粗到细，严重失配
        'S': 0.6,  # 摩托车在COCO中尺度较小，有一定变异
        'I': 0.5,  # 摩托车形态变异中等（街车、跑车、巡航、越野）
        'C': 0.5,  # 摩托车和bicycle容易混淆
    },
}


# ══════════════════════ 数据加载 ══════════════════════

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


# ══════════════════════ USC 模块 ══════════════════════

class USC:
    def __init__(self, n_components=30):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.domain_shift_ = None

    def fit(self, source_features, target_features):
        all_features = np.vstack([source_features, target_features])
        self.mean_ = all_features.mean(axis=0)
        X_centered = all_features - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        K = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:K]
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


# ══════════════════════ 物理先验驱动的自适应α ══════════════════════

def compute_risk_scores(weights):
    """计算每个类别的物理先验风险评分R(c)。"""
    R = {}
    for cls, priors in PHYSICAL_PRIORS.items():
        R[cls] = (weights['G'] * priors['G'] +
                   weights['S'] * priors['S'] +
                   weights['I'] * priors['I'] +
                   weights['C'] * priors['C'])
    return R


def adaptive_alpha(R_c, alpha_min, alpha_max, k, R_0):
    """sigmoid映射：R(c) → α(c)。"""
    return alpha_min + (alpha_max - alpha_min) / (1 + np.exp(-k * (R_c - R_0)))


# ══════════════════════ 主实验 ══════════════════════

def main():
    print('=' * 60)
    print('USC v3：物理先验驱动的逐类自适应α实验')
    print('=' * 60)

    source = load_source_features()
    target = load_target_data()

    all_src = np.vstack([v['features'] for v in source.values()])
    all_tgt = np.vstack([v['features'] for v in target.values()])

    # 拟合USC
    usc = USC(n_components=K_PCA).fit(all_src, all_tgt)

    # 基线
    baseline_aps = {}
    for name, coco_cls, dior_cls in TARGETS:
        if coco_cls in target:
            baseline_aps[coco_cls] = evaluate_baseline(target[coco_cls])
    mean_baseline = np.mean(list(baseline_aps.values()))
    print(f'\n基线均值AP: {mean_baseline:.4f}')

    # ═══ 实验1：逐类α扫描（得到oracle最佳α） ═══
    print('\n' + '=' * 60)
    print('实验1：逐类α扫描（oracle最佳α）')
    print('=' * 60)

    ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    oracle_best = {}
    for name, coco_cls, dior_cls in TARGETS:
        if coco_cls not in target or dior_cls not in source: continue
        best_alpha = 0.0
        best_ap = baseline_aps[coco_cls]
        for alpha in ALPHAS:
            src_corr = usc.correct_prototype(source[dior_cls]['prototype'], alpha)
            p_hat = build_proto(src_corr, coco_cls, dior_cls, use_tge=True)
            neg_raw = build_neg_proto(source, dior_cls)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            ap = evaluate(target[coco_cls], p_hat, neg)
            if ap > best_ap:
                best_ap = ap
                best_alpha = alpha
        oracle_best[coco_cls] = {'best_alpha': best_alpha, 'best_ap': best_ap}
        print(f'  {coco_cls}: oracle最佳α={best_alpha}, AP={best_ap:.4f}, '
              f'gain={best_ap-baseline_aps[coco_cls]:+.4f}')

    oracle_mean = np.mean([v['best_ap'] for v in oracle_best.values()])
    print(f'  Oracle均值AP: {oracle_mean:.4f}, gain={oracle_mean-mean_baseline:+.4f}')

    # ═══ 实验2：物理先验和最佳α的相关性分析 ═══
    print('\n' + '=' * 60)
    print('实验2：物理先验和最佳α的相关性分析')
    print('=' * 60)

    classes = list(PHYSICAL_PRIORS.keys())
    oracle_alphas = np.array([oracle_best[c]['best_alpha'] for c in classes])

    # 各单因子和最佳α的Spearman相关
    print('\n各物理先验因子与oracle最佳α的Spearman相关：')
    for factor in ['G', 'S', 'I', 'C']:
        values = np.array([PHYSICAL_PRIORS[c][factor] for c in classes])
        from scipy.stats import spearmanr
        rho, p = spearmanr(values, oracle_alphas)
        print(f'  {factor}: ρ={rho:.3f}, p={p:.3f}')

    # 等权R和最佳α的相关
    R_equal = compute_risk_scores({'G': 0.25, 'S': 0.25, 'I': 0.25, 'C': 0.25})
    R_values = np.array([R_equal[c] for c in classes])
    rho, p = spearmanr(R_values, oracle_alphas)
    print(f'\n等权R(c)与oracle最佳α: ρ={rho:.3f}, p={p:.3f}')
    print(f'  R值: {dict(zip(classes, [f"{R_equal[c]:.3f}" for c in classes]))}')
    print(f'  oracle α: {dict(zip(classes, oracle_alphas))}')

    # ═══ 实验3：物理先验驱动的自适应α（扫描超参数） ═══
    print('\n' + '=' * 60)
    print('实验3：物理先验驱动的自适应α（扫描超参数）')
    print('=' * 60)

    # 超参数搜索空间
    weight_configs = [
        {'G': 0.25, 'S': 0.25, 'I': 0.25, 'C': 0.25},  # 等权
        {'G': 0.4, 'S': 0.2, 'I': 0.2, 'C': 0.2},  # G偏重
        {'G': 0.2, 'S': 0.4, 'I': 0.2, 'C': 0.2},  # S偏重
        {'G': 0.2, 'S': 0.2, 'I': 0.4, 'C': 0.2},  # I偏重
        {'G': 0.2, 'S': 0.2, 'I': 0.2, 'C': 0.4},  # C偏重
        {'G': 0.3, 'S': 0.3, 'I': 0.2, 'C': 0.2},  # G+S偏重
        {'G': 0.1, 'S': 0.3, 'I': 0.3, 'C': 0.3},  # S+I+C偏重
    ]

    alpha_mins = [0.0, 0.1, 0.2]
    alpha_maxs = [1.0, 1.5, 2.0]
    ks = [2.0, 5.0, 10.0]  # sigmoid斜率
    R_0s = [0.3, 0.4, 0.5, 0.6]  # sigmoid中点

    best_config = None
    best_mean_ap = 0
    all_results = []

    total = len(weight_configs) * len(alpha_mins) * len(alpha_maxs) * len(ks) * len(R_0s)
    print(f'超参数组合总数: {total}')
    count = 0

    for weights in weight_configs:
        R = compute_risk_scores(weights)
        for alpha_min in alpha_mins:
            for alpha_max in alpha_maxs:
                if alpha_max <= alpha_min: continue
                for k in ks:
                    for R_0 in R_0s:
                        count += 1
                        if count % 50 == 0:
                            print(f'  进度: {count}/{total}')

                        # 计算每个类别的自适应α
                        alpha_per_class = {}
                        for c in classes:
                            alpha_per_class[c] = adaptive_alpha(R[c], alpha_min, alpha_max, k, R_0)

                        # 评估
                        aps = {}
                        for name, coco_cls, dior_cls in TARGETS:
                            if coco_cls not in target or dior_cls not in source: continue
                            alpha = alpha_per_class[coco_cls]
                            src_corr = usc.correct_prototype(source[dior_cls]['prototype'], alpha)
                            p_hat = build_proto(src_corr, coco_cls, dior_cls, use_tge=True)
                            neg_raw = build_neg_proto(source, dior_cls)
                            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
                            aps[coco_cls] = evaluate(target[coco_cls], p_hat, neg)

                        mean_ap = np.mean(list(aps.values()))
                        all_results.append({
                            'weights': weights, 'alpha_min': alpha_min,
                            'alpha_max': alpha_max, 'k': k, 'R_0': R_0,
                            'mean_ap': mean_ap, 'gain': mean_ap - mean_baseline,
                            'alpha_per_class': alpha_per_class, 'per_class': aps,
                        })

                        if mean_ap > best_mean_ap:
                            best_mean_ap = mean_ap
                            best_config = all_results[-1]

    print(f'\n最佳配置:')
    print(f'  权重: {best_config["weights"]}')
    print(f'  α_min={best_config["alpha_min"]}, α_max={best_config["alpha_max"]}')
    print(f'  k={best_config["k"]}, R_0={best_config["R_0"]}')
    print(f'  均值AP={best_config["mean_ap"]:.4f}, gain={best_config["gain"]:+.4f}')
    alpha_dict = best_config["alpha_per_class"]
    ap_dict = best_config["per_class"]
    print(f'  逐类α: {dict(zip(classes, [f"{alpha_dict[c]:.2f}" for c in classes]))}')
    print(f'  逐类AP: {dict(zip(classes, [f"{ap_dict[c]:.4f}" for c in classes]))}')

    # ═══ 实验4：和固定α、oracle对比 ═══
    print('\n' + '=' * 60)
    print('实验4：综合对比')
    print('=' * 60)

    # 固定α=1.5（v2最佳）
    fixed_aps = {}
    for name, coco_cls, dior_cls in TARGETS:
        if coco_cls not in target or dior_cls not in source: continue
        src_corr = usc.correct_prototype(source[dior_cls]['prototype'], 1.5)
        p_hat = build_proto(src_corr, coco_cls, dior_cls, use_tge=True)
        neg_raw = build_neg_proto(source, dior_cls)
        neg = usc.correct_prototype(neg_raw, 1.5) if neg_raw is not None else None
        fixed_aps[coco_cls] = evaluate(target[coco_cls], p_hat, neg)
    fixed_mean = np.mean(list(fixed_aps.values()))

    # TGE-only（无USC）
    tge_aps = {}
    for name, coco_cls, dior_cls in TARGETS:
        if coco_cls not in target or dior_cls not in source: continue
        p_hat = build_proto(source[dior_cls]['prototype'], coco_cls, dior_cls, use_tge=True)
        neg = build_neg_proto(source, dior_cls)
        tge_aps[coco_cls] = evaluate(target[coco_cls], p_hat, neg)
    tge_mean = np.mean(list(tge_aps.values()))

    print(f'\n{"配置":<30} {"均值AP":<10} {"vs基线":<10} {"vs TGE":<10} {"vs固定α":<10}')
    print('-' * 70)
    print(f'{"基线(GDINO)":<30} {mean_baseline:<10.4f} {"—":<10} {"—":<10} {"—":<10}')
    print(f'{"TGE-only (A-CSPC)":<30} {tge_mean:<10.4f} {tge_mean-mean_baseline:<+10.4f} {"—":<10} {"—":<10}')
    print(f'{"USC+TGE (固定α=1.5)":<30} {fixed_mean:<10.4f} {fixed_mean-mean_baseline:<+10.4f} {fixed_mean-tge_mean:<+10.4f} {"—":<10}')
    print(f'{"USC+TGE (物理先验自适应α)":<30} {best_config["mean_ap"]:<10.4f} {best_config["gain"]:<+10.4f} {best_config["mean_ap"]-tge_mean:<+10.4f} {best_config["mean_ap"]-fixed_mean:<+10.4f}')
    print(f'{"USC+TGE (逐类oracle α)":<30} {oracle_mean:<10.4f} {oracle_mean-mean_baseline:<+10.4f} {oracle_mean-tge_mean:<+10.4f} {oracle_mean-fixed_mean:<+10.4f}')

    # 自适应α达到oracle的比例
    oracle_gap = oracle_mean - mean_baseline
    adaptive_gap = best_config['mean_ap'] - mean_baseline
    oracle_ratio = adaptive_gap / oracle_gap if oracle_gap > 0 else 0
    print(f'\n物理先验自适应α达到oracle增益的: {oracle_ratio*100:.1f}%')

    # 保存结果
    results = {
        'config': {'beta': BETA, 'lam': LAM, 'k_pca': K_PCA, 'n_classes': len(classes)},
        'physical_priors': PHYSICAL_PRIORS,
        'baseline_mean': float(mean_baseline),
        'tge_only_mean': float(tge_mean),
        'fixed_alpha_mean': float(fixed_mean),
        'fixed_alpha': 1.5,
        'oracle_best': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in oracle_best.items()},
        'oracle_mean': float(oracle_mean),
        'best_adaptive_config': {
            'weights': best_config['weights'],
            'alpha_min': float(best_config['alpha_min']),
            'alpha_max': float(best_config['alpha_max']),
            'k': float(best_config['k']),
            'R_0': float(best_config['R_0']),
            'mean_ap': float(best_config['mean_ap']),
            'gain': float(best_config['gain']),
            'alpha_per_class': {k: float(v) for k, v in best_config['alpha_per_class'].items()},
            'per_class_ap': {k: float(v) for k, v in best_config['per_class'].items()},
        },
        'oracle_ratio': float(oracle_ratio),
        'top10_configs': sorted(all_results, key=lambda x: -x['mean_ap'])[:10],
    }

    # 转换numpy类型
    def convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {OUT}')

    return results


if __name__ == '__main__':
    main()
