# -*- coding: utf-8 -*-
"""usc_experiment_v6.py — 综合实验：域偏移投影因子 + 域偏移门控 + USC+g2结合

三个任务：
  1. 验证"域偏移投影"作为物理先验因子的预测能力（vs G/S/I/C）
  2. 设计域偏移门控，在跨域(DIOR→COCO)和同域(VOC→COCO)上验证
  3. 验证USC + 域方向门g2结合的效果

输出：实验数据/usc_experiment_v6_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'usc_experiment_v6_results.json'

BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_PCA = 30
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

SRC_TMPL_OVERHEAD = 'an overhead image of {}'
SRC_TMPL_PHOTO = 'a photo of {}'
TGT_TMPL = 'a photo of {}'

VOC_DIR = Path(r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007')

# ══════════════════════ 数据加载 ══════════════════════

def load_source_dior(cls_name, n=50):
    images, gts = U.load_dior(cls_name, n=n)
    return _extract_source_features(images, gts)


def load_source_voc(cls_name, n=50):
    ann_dir = VOC_DIR / 'Annotations'
    img_dir = VOC_DIR / 'JPEGImages'
    images, gts = [], []
    for xml_path in sorted(ann_dir.glob('*.xml')):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            if obj.find('name').text != cls_name:
                continue
            b = obj.find('bndbox')
            boxes.append([float(b.find('xmin').text), float(b.find('ymin').text),
                          float(b.find('xmax').text), float(b.find('ymax').text)])
        if not boxes:
            continue
        img_path = img_dir / (root.find('filename').text)
        if not img_path.exists():
            continue
        from PIL import Image
        images.append(Image.open(str(img_path)).convert('RGB'))
        gts.append(boxes)
        if len(images) >= n:
            break
    return _extract_source_features(images, gts)


def _extract_source_features(images, gts):
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
    return {'prototype': proto, 'features': F, 'n_crops': len(crops)}


def load_target_coco(coco_cls, n=200):
    images, gts = U.load_coco(coco_cls, n=n)
    if not images:
        return None
    cache_p = DATA / f'tge_v2_dets_COCO-{coco_cls}.pkl'
    if cache_p.exists():
        dets = pickle.load(open(cache_p, 'rb'))
    else:
        dets = U.gdino_detect(images, coco_cls, 0.0)
        pickle.dump(dets, open(cache_p, 'wb'))
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

    def prototype_shift_projection(self, p):
        """计算原型在域偏移方向上的投影长度。"""
        p_proj = self._project(p.reshape(1, -1))[0]
        return float(np.dot(p_proj, self.domain_shift_))


# ══════════════════════ 评估函数 ══════════════════════

def build_proto(src_proto, src_cls, coco_cls, src_tmpl, beta=BETA, use_tge=True):
    if not use_tge:
        return src_proto.copy()
    s_emb = T.text_avg([src_tmpl], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
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


# ══════════════════════ 任务1：域偏移投影因子 ══════════════════════

def task1_domain_shift_projection():
    """验证域偏移投影作为物理先验因子的预测能力。"""
    print('\n' + '=' * 60)
    print('任务1：域偏移投影因子 vs G/S/I/C')
    print('=' * 60)

    # DIOR→COCO 6类
    TARGETS = [
        ('airplane', 'airplane', 'airplane'),
        ('boat', 'boat', 'ship'),
        ('car', 'car', 'vehicle'),
        ('truck', 'truck', 'vehicle'),
        ('bus', 'bus', 'vehicle'),
        ('motorcycle', 'motorcycle', 'vehicle'),
    ]

    print('\n加载DIOR源域数据...')
    source_data = {}
    for eval_name, coco_cls, src_cls in TARGETS:
        data = load_source_dior(src_cls, n=50)
        if data is not None:
            source_data[src_cls] = data
            print(f'  {src_cls}: {data["n_crops"]}框')

    print('\n加载COCO目标域数据...')
    target_data = {}
    for eval_name, coco_cls, src_cls in TARGETS:
        data = load_target_coco(coco_cls, n=200)
        if data is not None:
            target_data[coco_cls] = data
            print(f'  {coco_cls}: {data["n_crops"]}候选框')

    # 拟合USC
    all_src = np.vstack([source_data[s]['features'] for _, _, s in TARGETS])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in TARGETS])
    usc = USC(n_components=K_PCA).fit(all_src, all_tgt)

    all_protos = {s: source_data[s]['prototype'] for _, _, s in TARGETS}

    # 逐类oracle最佳α
    print('\n扫描逐类oracle最佳α...')
    oracle_best = {}
    for eval_name, coco_cls, src_cls in TARGETS:
        best_alpha = 0.0
        best_ap = evaluate_baseline(target_data[coco_cls])
        for alpha in ALPHAS:
            src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], alpha)
            p_hat = build_proto(src_corr, src_cls, coco_cls, SRC_TMPL_OVERHEAD, use_tge=True)
            neg_raw = build_neg_proto(all_protos, src_cls)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            ap = evaluate(target_data[coco_cls], p_hat, neg)
            if ap > best_ap:
                best_ap = ap
                best_alpha = alpha
        oracle_best[coco_cls] = {'best_alpha': best_alpha, 'best_ap': best_ap}
        print(f'  {coco_cls}: oracle α={best_alpha}, AP={best_ap:.4f}')

    # 计算各因子
    print('\n计算各物理先验因子...')
    factors = {}
    for eval_name, coco_cls, src_cls in TARGETS:
        proto = source_data[src_cls]['prototype']
        F_src = source_data[src_cls]['features']
        F_tgt = target_data[coco_cls]['features']

        # G：文本嵌入余弦距离
        s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
        d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
        G = 1.0 - float(np.dot(s_emb, d_emb))

        # S：目标域候选框尺度变异系数
        areas = []
        for p in target_data[coco_cls]['per_img']:
            if len(p['box']) > 0:
                w = p['box'][:, 2] - p['box'][:, 0]
                h = p['box'][:, 3] - p['box'][:, 1]
                areas.extend((w * h).tolist())
        if len(areas) > 1:
            areas = np.array(areas)
            S = min(1.0, np.std(areas) / (np.mean(areas) + 1e-8) / 2.0)
        else:
            S = 0.5

        # I：源域类内变异
        if len(F_src) > 1:
            n_sample = min(100, len(F_src))
            idx = np.random.choice(len(F_src), n_sample, replace=False)
            F_sample = F_src[idx]
            sim_matrix = F_sample @ F_sample.T
            upper = sim_matrix[np.triu_indices(n_sample, k=1)]
            I = 1.0 - np.mean(upper)
        else:
            I = 0.5

        # C：类间混淆
        other_sims = []
        for other_name, other_proto in all_protos.items():
            if other_name != src_cls:
                other_sims.append(float(np.dot(proto, other_proto)))
        C = np.mean(other_sims) if other_sims else 0.5

        # D：域偏移投影（新因子）
        D = usc.prototype_shift_projection(proto)

        factors[coco_cls] = {'G': G, 'S': S, 'I': I, 'C': C, 'D': D}
        print(f'  {coco_cls}: G={G:.3f}, S={S:.3f}, I={I:.3f}, C={C:.3f}, D={D:.4f}')

    # 相关性分析
    print('\n=== 各因子 vs oracle最佳α 相关性 ===')
    oracle_alphas = np.array([oracle_best[c]['best_alpha'] for _, c, _ in TARGETS])
    correlations = {}
    for factor_name in ['G', 'S', 'I', 'C', 'D']:
        values = np.array([factors[c][factor_name] for _, c, _ in TARGETS])
        rho, p = spearmanr(values, oracle_alphas)
        correlations[factor_name] = {'rho': float(rho), 'p': float(p)}
        print(f'  {factor_name}: ρ={rho:.3f}, p={p:.3f}')

    # 等权组合
    R_gsic = np.array([np.mean([factors[c][f] for f in ['G', 'S', 'I', 'C']]) for _, c, _ in TARGETS])
    rho_gsic, p_gsic = spearmanr(R_gsic, oracle_alphas)
    print(f'  等权G/S/I/C: ρ={rho_gsic:.3f}, p={p_gsic:.3f}')

    R_d = np.array([factors[c]['D'] for _, c, _ in TARGETS])
    rho_d, p_d = spearmanr(R_d, oracle_alphas)
    print(f'  域偏移投影D: ρ={rho_d:.3f}, p={p_d:.3f}')

    return {
        'factors': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in factors.items()},
        'oracle_best': {k: {'best_alpha': float(v['best_alpha']), 'best_ap': float(v['best_ap'])} for k, v in oracle_best.items()},
        'correlations': {k: {'rho': float(v['rho']), 'p': float(v['p'])} for k, v in correlations.items()},
        'correlation_gsic': {'rho': float(rho_gsic), 'p': float(p_gsic)},
        'correlation_d': {'rho': float(rho_d), 'p': float(p_d)},
        'domain_shift_norm': float(usc.domain_shift_norm_),
    }


# ══════════════════════ 任务2：域偏移门控 ══════════════════════

def task2_domain_gate():
    """设计域偏移门控，在跨域和同域上验证。"""
    print('\n' + '=' * 60)
    print('任务2：域偏移门控（跨域DIOR→COCO vs 同域VOC→COCO）')
    print('=' * 60)

    # 跨域：DIOR→COCO 6类
    DIOR_TARGETS = [
        ('airplane', 'airplane', 'airplane'),
        ('boat', 'boat', 'ship'),
        ('car', 'car', 'vehicle'),
        ('truck', 'truck', 'vehicle'),
        ('bus', 'bus', 'vehicle'),
        ('motorcycle', 'motorcycle', 'vehicle'),
    ]

    # 同域：VOC→COCO 6类（选和DIOR对应的类别）
    VOC_TARGETS = [
        ('aeroplane', 'airplane', 'aeroplane'),
        ('boat', 'boat', 'boat'),
        ('car', 'car', 'car'),
        ('bus', 'bus', 'bus'),
        ('motorbike', 'motorcycle', 'motorbike'),
        ('bicycle', 'bicycle', 'bicycle'),  # 额外加一个
    ]

    results = {}

    for domain_name, targets, src_tmpl, load_func in [
        ('DIOR→COCO (跨域)', DIOR_TARGETS, SRC_TMPL_OVERHEAD, load_source_dior),
        ('VOC→COCO (同域)', VOC_TARGETS, SRC_TMPL_PHOTO, load_source_voc),
    ]:
        print(f'\n--- {domain_name} ---')

        # 加载源域
        source_data = {}
        for eval_name, coco_cls, src_cls in targets:
            data = load_func(src_cls, n=50)
            if data is not None:
                source_data[src_cls] = data

        # 加载目标域
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
            continue

        # 拟合USC
        all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
        all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
        usc = USC(n_components=K_PCA).fit(all_src, all_tgt)

        all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}

        # 基线
        baseline_aps = {c: evaluate_baseline(target_data[c]) for _, c, _ in valid_targets}
        mean_baseline = np.mean(list(baseline_aps.values()))

        # TGE-only（α=0，无USC）
        tge_aps = {}
        for eval_name, coco_cls, src_cls in valid_targets:
            p_hat = build_proto(source_data[src_cls]['prototype'], src_cls, coco_cls, src_tmpl, use_tge=True)
            neg = build_neg_proto(all_protos, src_cls)
            tge_aps[coco_cls] = evaluate(target_data[coco_cls], p_hat, neg)
        mean_tge = np.mean(list(tge_aps.values()))

        # USC最佳固定α
        best_alpha = 0.0
        best_mean = mean_tge
        for alpha in ALPHAS:
            aps = {}
            for eval_name, coco_cls, src_cls in valid_targets:
                src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], alpha)
                p_hat = build_proto(src_corr, src_cls, coco_cls, src_tmpl, use_tge=True)
                neg_raw = build_neg_proto(all_protos, src_cls)
                neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
                aps[coco_cls] = evaluate(target_data[coco_cls], p_hat, neg)
            mean_ap = np.mean(list(aps.values()))
            if mean_ap > best_mean:
                best_mean = mean_ap
                best_alpha = alpha

        print(f'  域偏移范数: {usc.domain_shift_norm_:.4f}')
        print(f'  基线: {mean_baseline:.4f}')
        print(f'  TGE-only: {mean_tge:.4f} (gain={mean_tge-mean_baseline:+.4f})')
        print(f'  USC最佳α={best_alpha}: {best_mean:.4f} (gain={best_mean-mean_baseline:+.4f})')
        print(f'  USC增益(over TGE): {best_mean-mean_tge:+.4f}')

        results[domain_name] = {
            'n_classes': len(valid_targets),
            'domain_shift_norm': float(usc.domain_shift_norm_),
            'baseline_mean': float(mean_baseline),
            'tge_only_mean': float(mean_tge),
            'tge_only_gain': float(mean_tge - mean_baseline),
            'usc_best_alpha': float(best_alpha),
            'usc_best_mean': float(best_mean),
            'usc_best_gain': float(best_mean - mean_baseline),
            'usc_gain_over_tge': float(best_mean - mean_tge),
        }

    # 门控阈值分析
    print('\n=== 域偏移门控阈值分析 ===')
    dior_shift = results['DIOR→COCO (跨域)']['domain_shift_norm']
    voc_shift = results['VOC→COCO (同域)']['domain_shift_norm']
    print(f'  跨域(DIOR→COCO)域偏移范数: {dior_shift:.4f}, USC有效: +{results["DIOR→COCO (跨域)"]["usc_gain_over_tge"]:.4f}')
    print(f'  同域(VOC→COCO)域偏移范数: {voc_shift:.4f}, USC有效: {results["VOC→COCO (同域)"]["usc_gain_over_tge"]:+.4f}')
    print(f'  建议门控阈值: {(dior_shift + voc_shift) / 2:.4f}')

    return results


# ══════════════════════ 任务3：USC + g2结合 ══════════════════════

def task3_usc_g2():
    """验证USC + 域方向门g2结合的效果。"""
    print('\n' + '=' * 60)
    print('任务3：USC + 域方向门g2结合')
    print('=' * 60)

    # DIOR→COCO 6类
    TARGETS = [
        ('airplane', 'airplane', 'airplane'),
        ('boat', 'boat', 'ship'),
        ('car', 'car', 'vehicle'),
        ('truck', 'truck', 'vehicle'),
        ('bus', 'bus', 'vehicle'),
        ('motorcycle', 'motorcycle', 'vehicle'),
    ]

    print('\n加载DIOR源域数据...')
    source_data = {}
    for eval_name, coco_cls, src_cls in TARGETS:
        data = load_source_dior(src_cls, n=50)
        if data is not None:
            source_data[src_cls] = data

    print('\n加载COCO目标域数据...')
    target_data = {}
    for eval_name, coco_cls, src_cls in TARGETS:
        data = load_target_coco(coco_cls, n=200)
        if data is not None:
            target_data[coco_cls] = data

    # 拟合USC
    all_src = np.vstack([source_data[s]['features'] for _, _, s in TARGETS])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in TARGETS])
    usc = USC(n_components=K_PCA).fit(all_src, all_tgt)
    all_protos = {s: source_data[s]['prototype'] for _, _, s in TARGETS}

    # 计算g2（域方向门）
    # g2 = 外推方向与域偏移方向的一致性
    # 外推方向：d(x_c) - s(x_s(c))
    # 域偏移方向：usc.domain_shift_（在低秩空间）
    print('\n计算域方向门g2...')
    g2_values = {}
    for eval_name, coco_cls, src_cls in TARGETS:
        s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
        d_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
        extrap_dir = d_emb - s_emb
        extrap_dir = extrap_dir / (np.linalg.norm(extrap_dir) + 1e-8)

        # 将外推方向投影到低秩空间
        extrap_proj = usc._project(extrap_dir.reshape(1, -1))[0]
        extrap_proj = extrap_proj / (np.linalg.norm(extrap_proj) + 1e-8)

        # g2 = 外推方向与域偏移方向的余弦相似度
        g2 = float(np.dot(extrap_proj, usc.domain_shift_))
        g2_values[coco_cls] = g2
        print(f'  {coco_cls}: g2={g2:.4f}')

    # 四种配置对比
    print('\n=== 四种配置对比 ===')
    configs = {
        'baseline': {},
        'TGE-only (α=0)': {'alpha': 0.0, 'use_tge': True, 'use_g2': False},
        'USC+TGE (最佳α)': {'alpha': None, 'use_tge': True, 'use_g2': False},  # None表示扫描
        'USC+TGE+g2门控': {'alpha': None, 'use_tge': True, 'use_g2': True},
    }

    results = {}

    # 基线
    baseline_aps = {c: evaluate_baseline(target_data[c]) for _, c, _ in TARGETS}
    results['baseline'] = {'mean': float(np.mean(list(baseline_aps.values()))), 'per_class': {k: float(v) for k, v in baseline_aps.items()}}
    print(f'  基线: {results["baseline"]["mean"]:.4f}')

    # TGE-only
    tge_aps = {}
    for eval_name, coco_cls, src_cls in TARGETS:
        p_hat = build_proto(source_data[src_cls]['prototype'], src_cls, coco_cls, SRC_TMPL_OVERHEAD, use_tge=True)
        neg = build_neg_proto(all_protos, src_cls)
        tge_aps[coco_cls] = evaluate(target_data[coco_cls], p_hat, neg)
    results['TGE-only (α=0)'] = {'mean': float(np.mean(list(tge_aps.values()))), 'per_class': {k: float(v) for k, v in tge_aps.items()}}
    print(f'  TGE-only: {results["TGE-only (α=0)"]["mean"]:.4f}')

    # USC+TGE（扫描最佳α）
    best_alpha = 0.0
    best_mean = results['TGE-only (α=0)']['mean']
    best_aps = tge_aps.copy()
    for alpha in ALPHAS:
        aps = {}
        for eval_name, coco_cls, src_cls in TARGETS:
            src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], alpha)
            p_hat = build_proto(src_corr, src_cls, coco_cls, SRC_TMPL_OVERHEAD, use_tge=True)
            neg_raw = build_neg_proto(all_protos, src_cls)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            aps[coco_cls] = evaluate(target_data[coco_cls], p_hat, neg)
        mean_ap = np.mean(list(aps.values()))
        if mean_ap > best_mean:
            best_mean = mean_ap
            best_alpha = alpha
            best_aps = aps.copy()
    results['USC+TGE (最佳α)'] = {'mean': float(best_mean), 'best_alpha': float(best_alpha), 'per_class': {k: float(v) for k, v in best_aps.items()}}
    print(f'  USC+TGE (α={best_alpha}): {best_mean:.4f}')

    # USC+TGE+g2门控
    # g2>0时启用USC，g2<=0时只用TGE（α=0）
    g2_threshold = 0.0
    gated_aps = {}
    for eval_name, coco_cls, src_cls in TARGETS:
        if g2_values[coco_cls] > g2_threshold:
            # 启用USC，用最佳α
            alpha = best_alpha
        else:
            # 关闭USC，用α=0
            alpha = 0.0
        src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], alpha)
        p_hat = build_proto(src_corr, src_cls, coco_cls, SRC_TMPL_OVERHEAD, use_tge=True)
        neg_raw = build_neg_proto(all_protos, src_cls)
        neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
        gated_aps[coco_cls] = evaluate(target_data[coco_cls], p_hat, neg)
    gated_mean = np.mean(list(gated_aps.values()))
    results['USC+TGE+g2门控'] = {
        'mean': float(gated_mean),
        'g2_threshold': g2_threshold,
        'g2_values': {k: float(v) for k, v in g2_values.items()},
        'per_class': {k: float(v) for k, v in gated_aps.items()},
    }
    print(f'  USC+TGE+g2门控: {gated_mean:.4f}')
    print(f'    g2>0启用USC的类别: {[c for _, c, _ in TARGETS if g2_values[c] > g2_threshold]}')
    print(f'    g2<=0关闭USC的类别: {[c for _, c, _ in TARGETS if g2_values[c] <= g2_threshold]}')

    return results


# ══════════════════════ 主函数 ══════════════════════

def main():
    print('=' * 60)
    print('USC v6：域偏移投影因子 + 域偏移门控 + USC+g2结合')
    print('=' * 60)

    # 任务1
    task1_results = task1_domain_shift_projection()

    # 任务2
    task2_results = task2_domain_gate()

    # 任务3
    task3_results = task3_usc_g2()

    # 汇总
    all_results = {
        'task1_domain_shift_projection': task1_results,
        'task2_domain_gate': task2_results,
        'task3_usc_g2': task3_results,
    }

    def convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存: {OUT}')

    return all_results


if __name__ == '__main__':
    main()
