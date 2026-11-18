# -*- coding: utf-8 -*-
"""usc_experiment_v11.py — 新预测因子探索 + LOO验证"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import unified_eval as U
import tge_probe as T
from scipy.stats import spearmanr

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'usc_experiment_v11_results.json'

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


def evaluate_baseline(target_data):
    per_img = target_data['per_img']
    preds = [(p['box'], p['gd']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return U.compute_ap50_img(preds, gts)


def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


def fit_linear(train_factors, train_alphas):
    X = np.column_stack([train_factors, np.ones(len(train_factors))])
    y = np.array(train_alphas)
    sol, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return sol[:-1], sol[-1]


def main():
    # 第一阶段：收集所有域对的逐类因子（包括新因子）和最佳α
    all_data = {}
    for source_name, (load_func, targets) in SOURCE_LOADERS.items():
        print(f'\n=== 加载 {source_name}→COCO ===')
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

        per_class = {}
        for eval_name, coco_cls, src_cls in valid_targets:
            # 基础因子
            G = compute_G(src_cls, coco_cls)
            baseline_ap = evaluate_baseline(target_data[coco_cls])

            # 新因子1：原型置信度（源域原型与源域特征的平均余弦相似度）
            src_feats = source_data[src_cls]['features']
            proto_conf = float(np.mean(src_feats @ source_data[src_cls]['prototype']))

            # 新因子2：类内方差（源域特征的平均方差）
            intra_var = float(np.mean(np.var(src_feats, axis=0)))

            # 新因子3：检测器分数均值
            all_gd = np.concatenate([p['gd'] for p in target_data[coco_cls]['per_img']])
            det_mean = float(np.mean(all_gd)) if len(all_gd) > 0 else 0.0

            # 新因子4：检测器分数方差
            det_var = float(np.var(all_gd)) if len(all_gd) > 0 else 0.0

            # 新因子5：候选框数量
            n_boxes = int(len(all_gd))

            # 新因子6：源域裁剪数量
            n_src_crops = int(source_data[src_cls]['n_crops'])

            # 新因子7：正负原型相似度
            neg_proto = build_neg_proto(all_protos, src_cls)
            pos_neg_sim = float(np.dot(source_data[src_cls]['prototype'], neg_proto)) if neg_proto is not None else 0.0

            # 扫描α找最佳
            best_alpha = 0.0; best_ap = -1
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
                if ap > best_ap: best_ap = ap; best_alpha = alpha

            per_class[coco_cls] = {
                'G': float(G), 'baseline_ap': float(baseline_ap),
                'proto_conf': proto_conf, 'intra_var': intra_var,
                'det_mean': det_mean, 'det_var': det_var,
                'n_boxes': n_boxes, 'n_src_crops': n_src_crops,
                'pos_neg_sim': pos_neg_sim,
                'best_alpha': float(best_alpha), 'best_ap': float(best_ap),
                'source': source_name,
            }
            print(f'  {coco_cls}: G={G:.4f}, conf={proto_conf:.4f}, intra_var={intra_var:.6f}, '
                  f'det_mean={det_mean:.4f}, n_boxes={n_boxes}, bestα={best_alpha}, bestAP={best_ap:.4f}')

        all_data[source_name] = {
            'per_class': per_class, 'usc': usc, 'source_data': source_data,
            'target_data': target_data, 'all_protos': all_protos, 'valid_targets': valid_targets,
        }

    # 第二阶段：所有因子与最佳α的相关性
    print('\n' + '=' * 60)
    print('所有因子相关性分析（合并所有类别，n=18）')
    print('=' * 60)
    all_classes = []
    for source_name, data in all_data.items():
        for coco_cls, v in data['per_class'].items():
            all_classes.append((source_name, coco_cls, v))

    all_factors = ['G', 'baseline_ap', 'proto_conf', 'intra_var', 'det_mean', 'det_var',
                    'n_boxes', 'n_src_crops', 'pos_neg_sim']
    factor_correlations = {}
    for factor_name in all_factors:
        vals = [v[factor_name] for _, _, v in all_classes]
        alphas = [v['best_alpha'] for _, _, v in all_classes]
        rho, p = spearmanr(vals, alphas)
        factor_correlations[factor_name] = {'rho': float(rho), 'p': float(p)}
        sig = '✅显著' if p < 0.05 else '❌'
        print(f'  {factor_name}: ρ={rho:+.4f}, p={p:.4f} {sig}')

    # 第三阶段：LOO验证，尝试不同因子组合
    print('\n' + '=' * 60)
    print('LOO验证：不同因子组合对比')
    print('=' * 60)

    source_names = list(all_data.keys())

    # 因子组合定义
    factor_combos = {
        'G_only': ['G'],
        'G+baseline': ['G', 'baseline_ap'],
        'proto_conf_only': ['proto_conf'],
        'det_stats': ['det_mean', 'det_var', 'n_boxes'],
        'all_new': ['proto_conf', 'intra_var', 'det_mean', 'det_var', 'n_boxes', 'pos_neg_sim'],
        'G+all_new': ['G', 'proto_conf', 'intra_var', 'det_mean', 'det_var', 'n_boxes', 'pos_neg_sim'],
        'top3_by_corr': [],  # 动态选择相关性最高的3个因子
    }

    # 选择相关性最高的3个因子（按|ρ|排序）
    sorted_factors = sorted(all_factors, key=lambda f: abs(factor_correlations[f]['rho']), reverse=True)
    factor_combos['top3_by_corr'] = sorted_factors[:3]
    print(f'  相关性最高的3个因子: {factor_combos["top3_by_corr"]}')

    loo_results = {combo_name: {} for combo_name in factor_combos}
    fixed_results = {}
    oracle_results = {}

    for test_source in source_names:
        train_sources = [s for s in source_names if s != test_source]
        print(f'\n--- 留出 {test_source} ---')

        # 找测试域的固定最佳α（需要重新评估所有α）
        test_data = all_data[test_source]
        usc = test_data['usc']
        source_data = test_data['source_data']
        target_data = test_data['target_data']
        all_protos = test_data['all_protos']
        valid_targets = test_data['valid_targets']

        alpha_aps_per_class = {}
        for eval_name, coco_cls, src_cls in valid_targets:
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
            alpha_aps_per_class[coco_cls] = alpha_aps

        fixed_mean_aps = {a: np.mean([alpha_aps_per_class[c][a] for c in alpha_aps_per_class]) for a in ALPHAS}
        best_fixed_alpha = max(fixed_mean_aps, key=fixed_mean_aps.get)
        fixed_results[test_source] = float(fixed_mean_aps[best_fixed_alpha])
        oracle_results[test_source] = float(np.mean([test_data['per_class'][c]['best_ap'] for c in test_data['per_class']]))
        print(f'  固定最佳α={best_fixed_alpha}: {fixed_mean_aps[best_fixed_alpha]:.4f}')
        print(f'  Oracle: {oracle_results[test_source]:.4f}')

        # 对每种因子组合做LOO
        for combo_name, factor_list in factor_combos.items():
            # 收集训练数据
            train_factors = []
            train_alphas = []
            for s in train_sources:
                for coco_cls, v in all_data[s]['per_class'].items():
                    train_factors.append([v[f] for f in factor_list])
                    train_alphas.append(v['best_alpha'])
            train_factors = np.array(train_factors)
            train_alphas = np.array(train_alphas)

            # 标准化
            factor_mean = train_factors.mean(axis=0)
            factor_std = train_factors.std(axis=0) + 1e-8
            train_factors_norm = (train_factors - factor_mean) / factor_std

            # 拟合线性映射
            w, b = fit_linear(train_factors_norm, train_alphas)

            # 在测试域上评估
            test_aps = []
            for coco_cls in alpha_aps_per_class:
                v = test_data['per_class'][coco_cls]
                factors = np.array([v[f] for f in factor_list])
                factors_norm = (factors - factor_mean) / factor_std
                alpha_pred = float(np.clip(np.dot(factors_norm, w) + b, 0.0, 2.0))
                alpha_pred = min(ALPHAS, key=lambda x: abs(x - alpha_pred))
                test_aps.append(alpha_aps_per_class[coco_cls][alpha_pred])

            mean_ap = float(np.mean(test_aps))
            loo_results[combo_name][test_source] = mean_ap
            print(f'  {combo_name}: {mean_ap:.4f} (vs固定: {mean_ap-fixed_results[test_source]:+.4f})')

    # 汇总
    print('\n' + '=' * 60)
    print('LOO汇总（三个域对的平均）')
    print('=' * 60)
    summary = {}
    mean_fixed = float(np.mean(list(fixed_results.values())))
    mean_oracle = float(np.mean(list(oracle_results.values())))
    print(f'  固定最佳α: {mean_fixed:.4f}')
    print(f'  Oracle逐类: {mean_oracle:.4f}')
    summary['fixed_best'] = mean_fixed
    summary['oracle'] = mean_oracle

    for combo_name in factor_combos:
        mean_ap = float(np.mean(list(loo_results[combo_name].values())))
        summary[combo_name] = mean_ap
        print(f'  {combo_name}: {mean_ap:.4f} (vs固定: {mean_ap-mean_fixed:+.4f}, 达oracle: {mean_ap/mean_oracle*100:.1f}%)')

    results = {
        'factor_correlations': factor_correlations,
        'factor_combos_used': {k: v for k, v in factor_combos.items()},
        'loo_results': loo_results,
        'fixed_results': fixed_results,
        'oracle_results': oracle_results,
        'summary': summary,
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {OUT}')


if __name__ == '__main__':
    main()
