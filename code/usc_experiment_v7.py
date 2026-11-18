# -*- coding: utf-8 -*-
"""usc_experiment_v7.py — 反向迁移验证g2门控（COCO→DIOR）

在反向迁移中验证域方向门g2的价值：
  - 正向（DIOR→COCO）：外推方向overhead→photo，g2全部>0，门控无效果
  - 反向（COCO→DIOR）：外推方向photo→overhead，g2可能<0，门控应能关闭有害类别

四种配置对比：
  1. 基线 GDINO
  2. TGE-only (α=0，无USC)
  3. USC+TGE (最佳固定α)
  4. USC+TGE+g2门控 (g2>0启用USC，g2<=0关闭USC)

输出：实验数据/usc_experiment_v7_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'usc_experiment_v7_results.json'

BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_PCA = 30
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# 反向：源域COCO(photo)，目标域DIOR(overhead)
SRC_TMPL = 'a photo of {}'
TGT_TMPL = 'an overhead image of {}'

# 类别映射：(评估名, DIOR目标类, COCO源类)
TARGETS = [
    ('airplane', 'airplane', 'airplane'),
    ('boat', 'ship', 'boat'),
    ('car', 'vehicle', 'car'),
    ('truck', 'vehicle', 'truck'),
    ('bus', 'vehicle', 'bus'),
    ('motorcycle', 'vehicle', 'motorcycle'),
]


# ══════════════════════ 源域加载（COCO作为源域，用GT框提取特征） ══════════════════════

def load_source_coco(coco_cls, n=50):
    """加载COCO作为源域：用GT标注框提取CLIP特征。"""
    images, gts = U.load_coco(coco_cls, n=n)
    if not images:
        return None
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    if not crops:
        return None
    F = T.encode_clip_batch(crops).cpu().numpy()
    proto = F.mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    return {'prototype': proto, 'features': F, 'n_crops': len(crops), 'images': images, 'gts': gts}


# ══════════════════════ 目标域加载（DIOR作为目标域，GDINO检测） ══════════════════════

def load_target_dior(dior_cls, n=200):
    """加载DIOR作为目标域：GDINO检测，提取候选框CLIP特征。"""
    images, gts = U.load_dior(dior_cls, n=n)
    if not images:
        return None
    cache_p = DATA / f'tge_v2_dets_DIOR-{dior_cls}.pkl'
    if cache_p.exists():
        dets = pickle.load(open(cache_p, 'rb'))
        print(f'    命中缓存: {cache_p.name}')
    else:
        print(f'    运行GDINO检测（{len(images)}张图）...')
        dets = U.gdino_detect(images, dior_cls, 0.0)
        pickle.dump(dets, open(cache_p, 'wb'))
        print(f'    缓存已保存: {cache_p.name}')
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
    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return {'per_img': per_img, 'features': F, 'n_crops': len(crops_all)}


# ══════════════════════ USC 模块 ══════════════════════

class USC:
    def __init__(self, n_components=30):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.domain_shift_ = None
        self.domain_shift_norm_ = None

    def fit(self, source_features, target_features):
        all_features = np.vstack([source_features, target_features])
        self.mean_ = all_features.mean(axis=0)
        X_centered = all_features - self.mean_
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        K = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:K]
        src_proj = self._project(source_features)
        tgt_proj = self._project(target_features)
        shift = tgt_proj.mean(axis=0) - src_proj.mean(axis=0)
        self.domain_shift_norm_ = np.linalg.norm(shift)
        if self.domain_shift_norm_ > 1e-8:
            self.domain_shift_ = shift / self.domain_shift_norm_
        else:
            self.domain_shift_ = np.zeros(K)
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

def build_proto(src_proto, src_cls, tgt_cls, beta=BETA, use_tge=True):
    """反向外推：photo→overhead。"""
    if not use_tge:
        return src_proto.copy()
    s_emb = T.text_avg([SRC_TMPL], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], tgt_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def build_neg_proto(protos, pos_cls):
    other = {k: v for k, v in protos.items() if k != pos_cls}
    if not other:
        return None
    pos_p = protos[pos_cls]
    sims = {k: float(np.dot(pos_p, v)) for k, v in other.items()}
    top3 = sorted(sims.items(), key=lambda x: -x[1])[:3]
    neg = np.mean([other[k] for k, _ in top3], axis=0)
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


# ══════════════════════ 主函数 ══════════════════════

def main():
    print('=' * 60)
    print('USC v7：反向迁移验证g2门控（COCO→DIOR）')
    print('=' * 60)

    # 加载源域（COCO）
    print('\n加载COCO源域数据（GT框提取特征）...')
    source_data = {}
    for eval_name, dior_cls, coco_cls in TARGETS:
        print(f'  {coco_cls}...', end=' ', flush=True)
        data = load_source_coco(coco_cls, n=50)
        if data is not None:
            source_data[coco_cls] = data
            print(f'{data["n_crops"]}框')
        else:
            print('无数据')

    # 加载目标域（DIOR）
    print('\n加载DIOR目标域数据（GDINO检测）...')
    target_data = {}
    valid_targets = []
    for eval_name, dior_cls, coco_cls in TARGETS:
        if coco_cls not in source_data:
            continue
        print(f'  {dior_cls}...', end=' ', flush=True)
        data = load_target_dior(dior_cls, n=200)
        if data is not None:
            target_data[dior_cls] = data
            valid_targets.append((eval_name, dior_cls, coco_cls))
            print(f'{data["n_crops"]}候选框')
        else:
            print('无数据')

    if len(valid_targets) < 2:
        print('有效类别太少，退出')
        return

    print(f'\n有效类别数: {len(valid_targets)}')

    # 拟合USC（COCO→DIOR）
    all_src = np.vstack([source_data[c]['features'] for _, _, c in valid_targets])
    all_tgt = np.vstack([target_data[d]['features'] for _, d, _ in valid_targets])
    usc = USC(n_components=K_PCA).fit(all_src, all_tgt)
    print(f'\nUSC域偏移范数: {usc.domain_shift_norm_:.4f}')

    all_protos = {c: source_data[c]['prototype'] for _, _, c in valid_targets}

    # 计算g2（反向：外推方向photo→overhead与域偏移方向的一致性）
    print('\n计算域方向门g2（反向：photo→overhead）...')
    g2_values = {}
    for eval_name, dior_cls, coco_cls in valid_targets:
        s_emb = T.text_avg([SRC_TMPL], coco_cls).cpu().numpy()
        d_emb = T.text_avg([TGT_TMPL], dior_cls).cpu().numpy()
        extrap_dir = d_emb - s_emb
        extrap_dir = extrap_dir / (np.linalg.norm(extrap_dir) + 1e-8)
        extrap_proj = usc._project(extrap_dir.reshape(1, -1))[0]
        extrap_proj = extrap_proj / (np.linalg.norm(extrap_proj) + 1e-8)
        g2 = float(np.dot(extrap_proj, usc.domain_shift_))
        g2_values[dior_cls] = g2
        print(f'  {dior_cls} (源{coco_cls}): g2={g2:.4f} {"✅启用USC" if g2 > 0 else "❌关闭USC"}')

    # 四种配置对比
    print('\n=== 四种配置对比（反向COCO→DIOR）===')

    # 1. 基线
    baseline_aps = {d: evaluate_baseline(target_data[d]) for _, d, _ in valid_targets}
    mean_baseline = np.mean(list(baseline_aps.values()))
    print(f'  1. 基线: {mean_baseline:.4f}')

    # 2. TGE-only (α=0)
    tge_aps = {}
    for eval_name, dior_cls, coco_cls in valid_targets:
        p_hat = build_proto(source_data[coco_cls]['prototype'], coco_cls, dior_cls, use_tge=True)
        neg = build_neg_proto(all_protos, coco_cls)
        tge_aps[dior_cls] = evaluate(target_data[dior_cls], p_hat, neg)
    mean_tge = np.mean(list(tge_aps.values()))
    print(f'  2. TGE-only (α=0): {mean_tge:.4f} (gain={mean_tge-mean_baseline:+.4f})')

    # 3. USC+TGE（扫描最佳α）
    best_alpha = 0.0
    best_mean = mean_tge
    best_aps = tge_aps.copy()
    for alpha in ALPHAS:
        aps = {}
        for eval_name, dior_cls, coco_cls in valid_targets:
            src_corr = usc.correct_prototype(source_data[coco_cls]['prototype'], alpha)
            p_hat = build_proto(src_corr, coco_cls, dior_cls, use_tge=True)
            neg_raw = build_neg_proto(all_protos, coco_cls)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            aps[dior_cls] = evaluate(target_data[dior_cls], p_hat, neg)
        mean_ap = np.mean(list(aps.values()))
        if mean_ap > best_mean:
            best_mean = mean_ap
            best_alpha = alpha
            best_aps = aps.copy()
    print(f'  3. USC+TGE (α={best_alpha}): {best_mean:.4f} (gain={best_mean-mean_baseline:+.4f})')

    # 4. USC+TGE+g2门控
    g2_threshold = 0.0
    gated_aps = {}
    enabled_classes = []
    disabled_classes = []
    for eval_name, dior_cls, coco_cls in valid_targets:
        if g2_values[dior_cls] > g2_threshold:
            alpha = best_alpha
            enabled_classes.append(dior_cls)
        else:
            alpha = 0.0
            disabled_classes.append(dior_cls)
        src_corr = usc.correct_prototype(source_data[coco_cls]['prototype'], alpha)
        p_hat = build_proto(src_corr, coco_cls, dior_cls, use_tge=True)
        neg_raw = build_neg_proto(all_protos, coco_cls)
        neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
        gated_aps[dior_cls] = evaluate(target_data[dior_cls], p_hat, neg)
    gated_mean = np.mean(list(gated_aps.values()))
    print(f'  4. USC+TGE+g2门控: {gated_mean:.4f} (gain={gated_mean-mean_baseline:+.4f})')
    print(f'     g2>0启用USC: {enabled_classes}')
    print(f'     g2<=0关闭USC: {disabled_classes}')

    # 逐类对比
    print('\n=== 逐类对比 ===')
    print(f'{"类别":<12} {"基线":<8} {"TGE":<8} {"USC+TGE":<10} {"+g2门控":<10} {"g2值":<8} {"门控决策"}')
    print('-' * 80)
    for eval_name, dior_cls, coco_cls in valid_targets:
        decision = '启用' if g2_values[dior_cls] > 0 else '关闭'
        print(f'{dior_cls:<12} {baseline_aps[dior_cls]:<8.4f} {tge_aps[dior_cls]:<8.4f} '
              f'{best_aps[dior_cls]:<10.4f} {gated_aps[dior_cls]:<10.4f} '
              f'{g2_values[dior_cls]:<8.4f} {decision}')

    # 正向 vs 反向对比
    print('\n=== 正向 vs 反向 g2对比 ===')
    print(f'  正向(DIOR→COCO): 全部g2>0，门控无效果（v6结果）')
    print(f'  反向(COCO→DIOR): {len([v for v in g2_values.values() if v > 0])}/{len(g2_values)}类g2>0，'
          f'{len([v for v in g2_values.values() if v <= 0])}/{len(g2_values)}类g2<=0')
    print(f'  g2门控增益(over USC+TGE): {gated_mean - best_mean:+.4f}')

    # 保存结果
    results = {
        'direction': 'COCO→DIOR (reverse)',
        'n_classes': len(valid_targets),
        'domain_shift_norm': float(usc.domain_shift_norm_),
        'g2_values': {k: float(v) for k, v in g2_values.items()},
        'g2_enabled': enabled_classes,
        'g2_disabled': disabled_classes,
        'configs': {
            'baseline': {'mean': float(mean_baseline), 'per_class': {k: float(v) for k, v in baseline_aps.items()}},
            'tge_only': {'mean': float(mean_tge), 'gain': float(mean_tge - mean_baseline), 'per_class': {k: float(v) for k, v in tge_aps.items()}},
            'usc_tge': {'mean': float(best_mean), 'best_alpha': float(best_alpha), 'gain': float(best_mean - mean_baseline), 'per_class': {k: float(v) for k, v in best_aps.items()}},
            'usc_tge_g2': {'mean': float(gated_mean), 'gain': float(gated_mean - mean_baseline), 'g2_gain_over_usc': float(gated_mean - best_mean), 'per_class': {k: float(v) for k, v in gated_aps.items()}},
        },
    }

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
