# -*- coding: utf-8 -*-
"""usc_experiment_v9.py — G因子跨域验证 + 基于G的自适应α映射"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import unified_eval as U
import tge_probe as T
from scipy.stats import spearmanr

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'usc_experiment_v9_results.json'

BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_PCA = 30
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

SRC_TMPL_OVERHEAD = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

NWPU_DIR = Path(r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10')
DOTA_IMG_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
DOTA_LBL_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'

NWPU_CLASSES = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
                 'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']
NWPU_CLS_ID = {c: i + 1 for i, c in enumerate(NWPU_CLASSES)}

REMOTE_TARGETS = [
    ('airplane', 'airplane', 'airplane'), ('boat', 'boat', 'ship'),
    ('car', 'car', 'vehicle'), ('truck', 'truck', 'vehicle'),
    ('bus', 'bus', 'vehicle'), ('motorcycle', 'motorcycle', 'vehicle'),
]

DOTA_TARGETS = [
    ('airplane', 'airplane', 'plane'), ('boat', 'boat', 'ship'),
    ('car', 'car', 'small-vehicle'), ('truck', 'truck', 'large-vehicle'),
    ('bus', 'bus', 'large-vehicle'), ('motorcycle', 'motorcycle', 'small-vehicle'),
]


def _extract_features(images, gts):
    if not images: return None
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10: continue
            crops.append(img.crop((x1, y1, x2, y2)))
    if not crops: return None
    F = T.encode_clip_batch(crops).cpu().numpy()
    proto = F.mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    return {'prototype': proto, 'features': F, 'n_crops': len(crops)}


def load_dior_src(cls_name, n=50):
    images, gts = U.load_dior(cls_name, n=n)
    return _extract_features(images, gts)


def load_nwpu_src(cls_name, n=50):
    ann_dir = NWPU_DIR / 'ground_truth'
    img_dir = NWPU_DIR / 'positive_image_set'
    cls_id = NWPU_CLS_ID.get(cls_name)
    if cls_id is None: return None
    images, gts = [], []
    for txt_path in sorted(ann_dir.glob('*.txt')):
        boxes = []
        for line in txt_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            nums = [int(v) for v in re.findall(r'-?\d+', line.strip())]
            if len(nums) < 5: continue
            x1, y1, x2, y2, cat = nums[-5], nums[-4], nums[-3], nums[-2], nums[-1]
            if cat != cls_id: continue
            boxes.append([x1, y1, x2, y2])
        if not boxes: continue
        img_path = None
        for ext in ('.jpg', '.JPG', '.jpeg', '.png', '.tif'):
            for cand in (txt_path.stem, txt_path.stem.lstrip('0') or '0', txt_path.stem.zfill(4)):
                p = img_dir / (cand + ext)
                if p.exists(): img_path = p; break
            if img_path: break
        if img_path is None: continue
        try:
            images.append(Image.open(img_path).convert('RGB'))
            gts.append(boxes)
        except Exception: continue
        if len(images) >= n: break
    return _extract_features(images, gts)


def load_dota_src(cls_name, n=50):
    quads_by_img = defaultdict(list)
    for fn in Path(DOTA_LBL_DIR).glob('*.txt'):
        img_name = fn.stem + '.png'
        img_path = Path(DOTA_IMG_DIR) / img_name
        if not img_path.exists(): continue
        with open(fn) as f: lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue
            try:
                q = list(map(float, parts[:8]))
                if parts[8] == cls_name:
                    xs, ys = q[0::2], q[1::2]
                    quads_by_img[img_name].append([min(xs), min(ys), max(xs), max(ys)])
            except Exception: continue
    images, gts = [], []
    for img_name, boxes in quads_by_img.items():
        try: im = Image.open(Path(DOTA_IMG_DIR) / img_name).convert('RGB')
        except Exception: continue
        images.append(im); gts.append(boxes)
        if len(images) >= n: break
    return _extract_features(images, gts)


SOURCE_LOADERS = {
    'DIOR': (load_dior_src, REMOTE_TARGETS),
    'NWPU': (load_nwpu_src, REMOTE_TARGETS),
    'DOTA': (load_dota_src, DOTA_TARGETS),
}


def load_target_coco(coco_cls, n=200):
    images, gts = U.load_coco(coco_cls, n=n)
    if not images: return None
    cache_p = DATA / f'tge_v2_dets_COCO-{coco_cls}.pkl'
    if cache_p.exists(): dets = pickle.load(open(cache_p, 'rb'))
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
    return {'per_img': per_img, 'features': F, 'n_crops': len(crops_all)}


class USC:
    def __init__(self, n_components=30):
        self.n_components = n_components; self.mean_ = None
        self.components_ = None; self.domain_shift_ = None; self.domain_shift_norm_ = None
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
        self.domain_shift_norm_ = float(np.linalg.norm(shift))
        if self.domain_shift_norm_ > 1e-8: self.domain_shift_ = shift / self.domain_shift_norm_
        else: self.domain_shift_ = np.zeros(K)
        return self
    def _project(self, X): return (X - self.mean_) @ self.components_.T
    def _reconstruct(self, Z): return Z @ self.components_ + self.mean_
    def correct(self, X, alpha=1.0):
        Z = self._project(X)
        Z_corrected = Z + alpha * self.domain_shift_
        X_corrected = self._reconstruct(Z_corrected)
        norms = np.linalg.norm(X_corrected, axis=1, keepdims=True)
        return X_corrected / np.maximum(norms, 1e-8)
    def correct_prototype(self, p, alpha=1.0): return self.correct(p.reshape(1, -1), alpha)[0]


def compute_G(src_cls, tgt_cls, src_tmpl=SRC_TMPL_OVERHEAD, tgt_tmpl=TGT_TMPL):
    """粒度失配因子：1 - 源域文本嵌入与目标域文本嵌入的余弦相似度"""
    s_emb = T.text_avg([src_tmpl], src_cls).cpu().numpy()
    d_emb = T.text_avg([tgt_tmpl], tgt_cls).cpu().numpy()
    cos_sim = float(np.dot(s_emb, d_emb) / (np.linalg.norm(s_emb) * np.linalg.norm(d_emb) + 1e-8))
    return 1.0 - cos_sim


def build_proto(src_proto, src_cls, tgt_cls, src_tmpl, beta=BETA, use_tge=True):
    if not use_tge: return src_proto.copy()
    s_emb = T.text_avg([src_tmpl], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], tgt_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def build_neg_proto(protos, pos_cls):
    other = {k: v for k, v in protos.items() if k != pos_cls}
    if not other: return None
    pos_p = protos[pos_cls]
    sims = {k: float(np.dot(pos_p, v)) for k, v in other.items()}
    top3 = sorted(sims.items(), key=lambda x: -x[1])[:3]
    neg = np.mean([other[k] for k, _ in top3], axis=0)
    return neg / (np.linalg.norm(neg) + 1e-8)


def evaluate(target_data, proto, neg_proto, lam=LAM):
    per_img = target_data['per_img']; F = target_data['features']
    for p in per_img:
        n = len(p['gd'])
        if n:
            f_boxes = F[p['crop_idx']]
            pos_sim = f_boxes @ proto
            neg_sim = f_boxes @ neg_proto if neg_proto is not None else np.zeros(n)
            vis = np.maximum(0.0, pos_sim - lam * neg_sim)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else: p['fused'] = np.zeros(0)
    preds = [(p['box'], p['fused']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return U.compute_ap50_img(preds, gts)


def main():
    results = {}

    for source_name, (load_func, targets) in SOURCE_LOADERS.items():
        print(f'\n=== {source_name}→COCO ===')
        source_data = {}
        for eval_name, coco_cls, src_cls in targets:
            data = load_func(src_cls, n=50)
            if data is not None: source_data[src_cls] = data

        target_data = {}; valid_targets = []
        for eval_name, coco_cls, src_cls in targets:
            if src_cls not in source_data: continue
            data = load_target_coco(coco_cls, n=200)
            if data is not None:
                target_data[coco_cls] = data
                valid_targets.append((eval_name, coco_cls, src_cls))
        if len(valid_targets) < 2: continue

        all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
        all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
        usc = USC(n_components=K_PCA).fit(all_src, all_tgt)
        all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}

        # 逐类：计算G因子 + 扫描α找最佳α
        per_class = {}
        for eval_name, coco_cls, src_cls in valid_targets:
            G = compute_G(src_cls, coco_cls)
            best_alpha = 0.0; best_ap = -1
            alpha_aps = {}
            for alpha in ALPHAS:
                if alpha == 0.0:
                    src_corr = source_data[src_cls]['prototype'].copy()
                    neg = build_neg_proto(all_protos, src_cls)
                else:
                    src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], alpha)
                    neg_raw = build_neg_proto(all_protos, src_cls)
                    neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
                p_hat = build_proto(src_corr, src_cls, coco_cls, SRC_TMPL_OVERHEAD, use_tge=True)
                ap = evaluate(target_data[coco_cls], p_hat, neg)
                alpha_aps[alpha] = ap
                if ap > best_ap: best_ap = ap; best_alpha = alpha
            per_class[coco_cls] = {
                'G': float(G), 'best_alpha': float(best_alpha), 'best_ap': float(best_ap),
                'alpha_aps': {str(k): float(v) for k, v in alpha_aps.items()},
            }
            print(f'  {coco_cls}: G={G:.4f}, best_alpha={best_alpha}, best_ap={best_ap:.4f}')

        # G因子与最佳α的Spearman相关性
        G_vals = [v['G'] for v in per_class.values()]
        alpha_vals = [v['best_alpha'] for v in per_class.values()]
        rho, p = spearmanr(G_vals, alpha_vals)
        print(f'  G vs best_alpha: ρ={rho:+.4f}, p={p:.4f}')

        # 基于G的自适应α映射（线性）
        # α(c) = α_max - (α_max - α_min) * (G(c) - G_min) / (G_max - G_min)
        G_min, G_max = min(G_vals), max(G_vals)
        alpha_min, alpha_max = 0.0, 2.0
        adaptive_aps = {}
        for eval_name, coco_cls, src_cls in valid_targets:
            G = per_class[coco_cls]['G']
            if G_max > G_min:
                alpha_adapt = alpha_max - (alpha_max - alpha_min) * (G - G_min) / (G_max - G_min)
            else:
                alpha_adapt = (alpha_min + alpha_max) / 2
            # 取最近的ALPHAS中的值
            alpha_adapt = min(ALPHAS, key=lambda x: abs(x - alpha_adapt))
            if alpha_adapt == 0.0:
                src_corr = source_data[src_cls]['prototype'].copy()
                neg = build_neg_proto(all_protos, src_cls)
            else:
                src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], alpha_adapt)
                neg_raw = build_neg_proto(all_protos, src_cls)
                neg = usc.correct_prototype(neg_raw, alpha_adapt) if neg_raw is not None else None
            p_hat = build_proto(src_corr, src_cls, coco_cls, SRC_TMPL_OVERHEAD, use_tge=True)
            ap = evaluate(target_data[coco_cls], p_hat, neg)
            adaptive_aps[coco_cls] = {'alpha': float(alpha_adapt), 'ap': float(ap)}

        mean_adaptive = float(np.mean([v['ap'] for v in adaptive_aps.values()]))
        mean_oracle = float(np.mean([v['best_ap'] for v in per_class.values()]))
        mean_fixed_best = max(ALPHAS, key=lambda a: np.mean([
            per_class[c]['alpha_aps'][str(a)] for c in per_class
        ]))
        mean_fixed = float(np.mean([per_class[c]['alpha_aps'][str(mean_fixed_best)] for c in per_class]))

        print(f'  固定α={mean_fixed_best}: {mean_fixed:.4f}')
        print(f'  自适应α(G线性): {mean_adaptive:.4f}')
        print(f'  Oracle逐类: {mean_oracle:.4f}')
        print(f'  自适应达到oracle: {mean_adaptive/mean_oracle*100:.1f}%')

        results[source_name] = {
            'n_classes': len(valid_targets),
            'domain_shift_norm': float(usc.domain_shift_norm_),
            'per_class': per_class,
            'G_alpha_correlation': {'rho': float(rho), 'p': float(p)},
            'adaptive_linear': {
                'mean_ap': mean_adaptive,
                'per_class': adaptive_aps,
                'G_range': [float(G_min), float(G_max)],
                'alpha_range': [float(alpha_min), float(alpha_max)],
            },
            'fixed_best': {'alpha': float(mean_fixed_best), 'mean_ap': mean_fixed},
            'oracle': {'mean_ap': mean_oracle},
        }

    # 汇总三个源域
    print('\n' + '=' * 60)
    print('汇总：三个跨域源域')
    print('=' * 60)
    all_G = []; all_alpha = []
    for source_name, res in results.items():
        for c, v in res['per_class'].items():
            all_G.append(v['G']); all_alpha.append(v['best_alpha'])
    rho_all, p_all = spearmanr(all_G, all_alpha)
    print(f'合并G vs best_alpha (n={len(all_G)}): ρ={rho_all:+.4f}, p={p_all:.4f}')

    mean_fixed = np.mean([r['fixed_best']['mean_ap'] for r in results.values()])
    mean_adaptive = np.mean([r['adaptive_linear']['mean_ap'] for r in results.values()])
    mean_oracle = np.mean([r['oracle']['mean_ap'] for r in results.values()])
    print(f'固定α平均: {mean_fixed:.4f}')
    print(f'自适应α平均: {mean_adaptive:.4f}')
    print(f'Oracle平均: {mean_oracle:.4f}')
    print(f'自适应达到oracle: {mean_adaptive/mean_oracle*100:.1f}%')

    results['summary'] = {
        'merged_correlation': {'rho': float(rho_all), 'p': float(p_all), 'n': len(all_G)},
        'mean_fixed': float(mean_fixed),
        'mean_adaptive': float(mean_adaptive),
        'mean_oracle': float(mean_oracle),
        'adaptive_oracle_ratio': float(mean_adaptive / mean_oracle),
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {OUT}')


if __name__ == '__main__':
    main()
