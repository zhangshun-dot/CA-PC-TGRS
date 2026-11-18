# -*- coding: utf-8 -*-
"""usc_experiment_v8.py — 综合实验：物理先验20类验证 + 完整门控模块测试

任务B：物理先验因子20类验证（VOC→COCO）
  - 计算G/S/I/C/D（域偏移投影）各因子
  - 扫描逐类oracle最佳α
  - 计算各因子与oracleα的Spearman相关性

任务C：完整门控模块测试（4个域对）
  - 整合：域偏移门控 + g2方向门控
  - 门控逻辑：域偏移小 OR g2<0 → 关闭USC；否则启用USC
  - 测试域对：DIOR→COCO, NWPU→COCO, DOTA→COCO, VOC→COCO
  - 对比：基线、TGE-only、USC+TGE、完整门控

输出：实验数据/usc_experiment_v8_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from PIL import Image
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'usc_experiment_v8_results.json'

BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_PCA = 30
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

SRC_TMPL_OVERHEAD = 'an overhead image of {}'
SRC_TMPL_PHOTO = 'a photo of {}'
TGT_TMPL = 'a photo of {}'

VOC_DIR = Path(r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007')
NWPU_DIR = Path(r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10')
DOTA_IMG_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
DOTA_LBL_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'

NWPU_CLASSES = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
                 'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']
NWPU_CLS_ID = {c: i + 1 for i, c in enumerate(NWPU_CLASSES)}

# VOC→COCO 20类
VOC20_TARGETS = [
    ('aeroplane', 'airplane', 'aeroplane'),
    ('bicycle', 'bicycle', 'bicycle'),
    ('bird', 'bird', 'bird'),
    ('boat', 'boat', 'boat'),
    ('bottle', 'bottle', 'bottle'),
    ('bus', 'bus', 'bus'),
    ('car', 'car', 'car'),
    ('cat', 'cat', 'cat'),
    ('chair', 'chair', 'chair'),
    ('cow', 'cow', 'cow'),
    ('diningtable', 'dining table', 'diningtable'),
    ('dog', 'dog', 'dog'),
    ('horse', 'horse', 'horse'),
    ('motorbike', 'motorcycle', 'motorbike'),
    ('person', 'person', 'person'),
    ('pottedplant', 'potted plant', 'pottedplant'),
    ('sheep', 'sheep', 'sheep'),
    ('sofa', 'couch', 'sofa'),
    ('train', 'train', 'train'),
    ('tvmonitor', 'tv', 'tvmonitor'),
]

# 6类遥感→COCO（用于多源域门控测试）
REMOTE_TARGETS = [
    ('airplane', 'airplane', 'airplane'),
    ('boat', 'boat', 'ship'),
    ('car', 'car', 'vehicle'),
    ('truck', 'truck', 'vehicle'),
    ('bus', 'bus', 'vehicle'),
    ('motorcycle', 'motorcycle', 'vehicle'),
]


# ══════════════════════ 源域加载函数 ══════════════════════

def _extract_features(images, gts):
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


def load_dior_src(cls_name, n=50):
    images, gts = U.load_dior(cls_name, n=n)
    return _extract_features(images, gts)


def load_voc_src(cls_name, n=50):
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
        images.append(Image.open(str(img_path)).convert('RGB'))
        gts.append(boxes)
        if len(images) >= n:
            break
    return _extract_features(images, gts)


def load_nwpu_src(cls_name, n=50):
    ann_dir = NWPU_DIR / 'ground_truth'
    img_dir = NWPU_DIR / 'positive_image_set'
    cls_id = NWPU_CLS_ID.get(cls_name)
    if cls_id is None:
        return None
    images, gts = [], []
    for txt_path in sorted(ann_dir.glob('*.txt')):
        boxes = []
        for line in txt_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            nums = [int(v) for v in re.findall(r'-?\d+', line.strip())]
            if len(nums) < 5:
                continue
            x1, y1, x2, y2, cat = nums[-5], nums[-4], nums[-3], nums[-2], nums[-1]
            if cat != cls_id:
                continue
            boxes.append([x1, y1, x2, y2])
        if not boxes:
            continue
        img_path = None
        for ext in ('.jpg', '.JPG', '.jpeg', '.png', '.tif'):
            for cand in (txt_path.stem, txt_path.stem.lstrip('0') or '0', txt_path.stem.zfill(4)):
                p = img_dir / (cand + ext)
                if p.exists():
                    img_path = p
                    break
            if img_path:
                break
        if img_path is None:
            continue
        try:
            images.append(Image.open(img_path).convert('RGB'))
            gts.append(boxes)
        except Exception:
            continue
        if len(images) >= n:
            break
    return _extract_features(images, gts)


def load_dota_src(cls_name, n=50):
    quads_by_img = defaultdict(list)
    for fn in Path(DOTA_LBL_DIR).glob('*.txt'):
        img_name = fn.stem + '.png'
        img_path = Path(DOTA_IMG_DIR) / img_name
        if not img_path.exists():
            continue
        with open(fn) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                q = list(map(float, parts[:8]))
                if parts[8] == cls_name:
                    xs, ys = q[0::2], q[1::2]
                    quads_by_img[img_name].append([min(xs), min(ys), max(xs), max(ys)])
            except Exception:
                continue
    images, gts = [], []
    for img_name, boxes in quads_by_img.items():
        try:
            im = Image.open(Path(DOTA_IMG_DIR) / img_name).convert('RGB')
        except Exception:
            continue
        images.append(im)
        gts.append(boxes)
        if len(images) >= n:
            break
    return _extract_features(images, gts)


# DOTA类别映射
DOTA_TARGETS = [
    ('airplane', 'airplane', 'plane'),
    ('boat', 'boat', 'ship'),
    ('car', 'car', 'small-vehicle'),
    ('truck', 'truck', 'large-vehicle'),
    ('bus', 'bus', 'large-vehicle'),
    ('motorcycle', 'motorcycle', 'small-vehicle'),
]

SOURCE_LOADERS = {
    'DIOR': (load_dior_src, REMOTE_TARGETS, SRC_TMPL_OVERHEAD),
    'NWPU': (load_nwpu_src, REMOTE_TARGETS, SRC_TMPL_OVERHEAD),
    'DOTA': (load_dota_src, DOTA_TARGETS, SRC_TMPL_OVERHEAD),
    'VOC': (load_voc_src, VOC20_TARGETS, SRC_TMPL_PHOTO),
}


# ══════════════════════ 目标域加载（COCO） ══════════════════════

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
        p_proj = self._project(p.reshape(1, -1))[0]
        return float(np.dot(p_proj, self.domain_shift_))

    def compute_g2(self, src_cls, tgt_cls, src_tmpl, tgt_tmpl=TGT_TMPL):
        """计算域方向门g2：外推方向与域偏移方向的一致性。"""
        s_emb = T.text_avg([src_tmpl], src_cls).cpu().numpy()
        d_emb = T.text_avg([tgt_tmpl], tgt_cls).cpu().numpy()
        extrap_dir = d_emb - s_emb
        extrap_dir = extrap_dir / (np.linalg.norm(extrap_dir) + 1e-8)
        extrap_proj = self._project(extrap_dir.reshape(1, -1))[0]
        extrap_proj = extrap_proj / (np.linalg.norm(extrap_proj) + 1e-8)
        return float(np.dot(extrap_proj, self.domain_shift_))


# ══════════════════════ 评估函数 ══════════════════════

def build_proto(src_proto, src_cls, tgt_cls, src_tmpl, beta=BETA, use_tge=True):
    if not use_tge:
        return src_proto.copy()
    s_emb = T.text_avg([src_tmpl], src_cls).cpu().numpy()
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


# ══════════════════════ 任务B：物理先验20类验证 ══════════════════════

def task_b_physical_priors_20class():
    """VOC→COCO 20类物理先验因子验证。"""
    print('\n' + '=' * 60)
    print('任务B：物理先验因子20类验证（VOC→COCO）')
    print('=' * 60)

    load_func = load_voc_src
    src_tmpl = SRC_TMPL_PHOTO
    targets = VOC20_TARGETS

    # 加载源域
    print('\n加载VOC源域数据...')
    source_data = {}
    for eval_name, coco_cls, voc_cls in targets:
        data = load_func(voc_cls, n=50)
        if data is not None:
            source_data[voc_cls] = data
            print(f'  {voc_cls}: {data["n_crops"]}框')

    # 加载目标域
    print('\n加载COCO目标域数据...')
    target_data = {}
    valid_targets = []
    for eval_name, coco_cls, voc_cls in targets:
        if voc_cls not in source_data:
            continue
        data = load_target_coco(coco_cls, n=200)
        if data is not None:
            target_data[coco_cls] = data
            valid_targets.append((eval_name, coco_cls, voc_cls))
            print(f'  {coco_cls}: {data["n_crops"]}候选框')

    print(f'\n有效类别数: {len(valid_targets)}')

    # 拟合USC
    all_src = np.vstack([source_data[v]['features'] for _, _, v in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
    usc = USC(n_components=K_PCA).fit(all_src, all_tgt)
    all_protos = {v: source_data[v]['prototype'] for _, _, v in valid_targets}

    # 逐类oracle最佳α
    print('\n扫描逐类oracle最佳α...')
    oracle_best = {}
    for eval_name, coco_cls, voc_cls in valid_targets:
        best_alpha = 0.0
        best_ap = evaluate_baseline(target_data[coco_cls])
        for alpha in ALPHAS:
            src_corr = usc.correct_prototype(source_data[voc_cls]['prototype'], alpha)
            p_hat = build_proto(src_corr, voc_cls, coco_cls, src_tmpl, use_tge=True)
            neg_raw = build_neg_proto(all_protos, voc_cls)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            ap = evaluate(target_data[coco_cls], p_hat, neg)
            if ap > best_ap:
                best_ap = ap
                best_alpha = alpha
        oracle_best[coco_cls] = {'best_alpha': best_alpha, 'best_ap': best_ap}

    # 计算各因子
    print('\n计算各物理先验因子...')
    factors = {}
    for eval_name, coco_cls, voc_cls in valid_targets:
        proto = source_data[voc_cls]['prototype']
        F_src = source_data[voc_cls]['features']

        # G：文本嵌入余弦距离
        s_emb = T.text_avg([src_tmpl], voc_cls).cpu().numpy()
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
            if other_name != voc_cls:
                other_sims.append(float(np.dot(proto, other_proto)))
        C = np.mean(other_sims) if other_sims else 0.5

        # D：域偏移投影（新因子）
        D = usc.prototype_shift_projection(proto)

        factors[coco_cls] = {'G': G, 'S': S, 'I': I, 'C': C, 'D': D}

    # 相关性分析
    print('\n=== 各因子 vs oracle最佳α 相关性（20类）===')
    oracle_alphas = np.array([oracle_best[c]['best_alpha'] for _, c, _ in valid_targets])
    correlations = {}
    for factor_name in ['G', 'S', 'I', 'C', 'D']:
        values = np.array([factors[c][factor_name] for _, c, _ in valid_targets])
        rho, p = spearmanr(values, oracle_alphas)
        correlations[factor_name] = {'rho': float(rho), 'p': float(p)}
        sig = '✅' if p < 0.05 else '❌'
        print(f'  {factor_name}: ρ={rho:+.3f}, p={p:.3f} {sig}')

    # 等权组合
    R_gsic = np.array([np.mean([factors[c][f] for f in ['G', 'S', 'I', 'C']]) for _, c, _ in valid_targets])
    rho_gsic, p_gsic = spearmanr(R_gsic, oracle_alphas)
    print(f'  等权G/S/I/C: ρ={rho_gsic:+.3f}, p={p_gsic:.3f}')

    R_d = np.array([factors[c]['D'] for _, c, _ in valid_targets])
    rho_d, p_d = spearmanr(R_d, oracle_alphas)
    print(f'  域偏移投影D: ρ={rho_d:+.3f}, p={p_d:.3f}')

    return {
        'n_classes': len(valid_targets),
        'domain_shift_norm': float(usc.domain_shift_norm_),
        'factors': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in factors.items()},
        'oracle_best': {k: {'best_alpha': float(v['best_alpha']), 'best_ap': float(v['best_ap'])} for k, v in oracle_best.items()},
        'correlations': {k: {'rho': float(v['rho']), 'p': float(v['p'])} for k, v in correlations.items()},
        'correlation_gsic': {'rho': float(rho_gsic), 'p': float(p_gsic)},
        'correlation_d': {'rho': float(rho_d), 'p': float(p_d)},
    }


# ══════════════════════ 任务C：完整门控模块测试 ══════════════════════

def task_c_full_gating():
    """完整门控模块（域偏移门控+g2方向门控）在4个域对上测试。"""
    print('\n' + '=' * 60)
    print('任务C：完整门控模块测试（4个域对）')
    print('=' * 60)

    # 域偏移门控阈值（从v6结果：跨域0.3736，同域0.1552，中点0.2644）
    DOMAIN_SHIFT_THRESHOLD = 0.2644
    G2_THRESHOLD = 0.0

    results = {}

    for source_name, (load_func, targets, src_tmpl) in SOURCE_LOADERS.items():
        print(f'\n--- {source_name}→COCO ---')

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

        # 计算g2
        g2_values = {}
        for eval_name, coco_cls, src_cls in valid_targets:
            g2_values[coco_cls] = usc.compute_g2(src_cls, coco_cls, src_tmpl)

        # 域偏移门控决策
        domain_shift_enabled = usc.domain_shift_norm_ > DOMAIN_SHIFT_THRESHOLD

        # 1. 基线
        baseline_aps = {c: evaluate_baseline(target_data[c]) for _, c, _ in valid_targets}
        mean_baseline = np.mean(list(baseline_aps.values()))

        # 2. TGE-only (α=0)
        tge_aps = {}
        for eval_name, coco_cls, src_cls in valid_targets:
            p_hat = build_proto(source_data[src_cls]['prototype'], src_cls, coco_cls, src_tmpl, use_tge=True)
            neg = build_neg_proto(all_protos, src_cls)
            tge_aps[coco_cls] = evaluate(target_data[coco_cls], p_hat, neg)
        mean_tge = np.mean(list(tge_aps.values()))

        # 3. USC+TGE（最佳固定α）
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

        # 4. 完整门控模块
        # 门控逻辑：域偏移小 OR g2<0 → 关闭USC(α=0)；否则启用USC(最佳α)
        gated_aps = {}
        enabled_classes = []
        disabled_classes = []
        for eval_name, coco_cls, src_cls in valid_targets:
            g2_enabled = g2_values[coco_cls] > G2_THRESHOLD
            usc_enabled = domain_shift_enabled and g2_enabled
            if usc_enabled:
                alpha = best_alpha
                enabled_classes.append(coco_cls)
            else:
                alpha = 0.0
                disabled_classes.append(coco_cls)
            src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], alpha)
            p_hat = build_proto(src_corr, src_cls, coco_cls, src_tmpl, use_tge=True)
            neg_raw = build_neg_proto(all_protos, src_cls)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            gated_aps[coco_cls] = evaluate(target_data[coco_cls], p_hat, neg)
        gated_mean = np.mean(list(gated_aps.values()))

        print(f'  域偏移范数: {usc.domain_shift_norm_:.4f} (阈值{DOMAIN_SHIFT_THRESHOLD:.4f}, '
              f'{"启用" if domain_shift_enabled else "关闭"})')
        print(f'  g2范围: [{min(g2_values.values()):.4f}, {max(g2_values.values()):.4f}]')
        print(f'  基线: {mean_baseline:.4f}')
        print(f'  TGE-only: {mean_tge:.4f} (gain={mean_tge-mean_baseline:+.4f})')
        print(f'  USC+TGE (α={best_alpha}): {best_mean:.4f} (gain={best_mean-mean_baseline:+.4f})')
        print(f'  完整门控: {gated_mean:.4f} (gain={gated_mean-mean_baseline:+.4f})')
        print(f'    启用USC: {enabled_classes}')
        print(f'    关闭USC: {disabled_classes}')
        print(f'    门控增益(over USC+TGE): {gated_mean-best_mean:+.4f}')

        results[source_name] = {
            'n_classes': len(valid_targets),
            'domain_shift_norm': float(usc.domain_shift_norm_),
            'domain_shift_enabled': domain_shift_enabled,
            'g2_values': {k: float(v) for k, v in g2_values.items()},
            'g2_enabled_count': len([v for v in g2_values.values() if v > G2_THRESHOLD]),
            'g2_disabled_count': len([v for v in g2_values.values() if v <= G2_THRESHOLD]),
            'enabled_classes': enabled_classes,
            'disabled_classes': disabled_classes,
            'configs': {
                'baseline': {'mean': float(mean_baseline)},
                'tge_only': {'mean': float(mean_tge), 'gain': float(mean_tge - mean_baseline)},
                'usc_tge': {'mean': float(best_mean), 'best_alpha': float(best_alpha), 'gain': float(best_mean - mean_baseline)},
                'full_gating': {'mean': float(gated_mean), 'gain': float(gated_mean - mean_baseline), 'gain_over_usc': float(gated_mean - best_mean)},
            },
        }

    return results


# ══════════════════════ 主函数 ══════════════════════

def main():
    print('=' * 60)
    print('USC v8：物理先验20类验证 + 完整门控模块测试')
    print('=' * 60)

    # 任务B
    task_b_results = task_b_physical_priors_20class()

    # 任务C
    task_c_results = task_c_full_gating()

    # 汇总
    all_results = {
        'task_b_physical_priors_20class': task_b_results,
        'task_c_full_gating': task_c_results,
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
