# -*- coding: utf-8 -*-
"""usc_experiment_v4.py — 综合实验：数据驱动物理先验 + VOC→COCO 20类 + 低α诊断

三个任务：
  1. 数据驱动的物理先验定义（G/S/I/C从数据中计算，而非主观赋值）
  2. 扩展到VOC→COCO 20类验证泛化性
  3. 深入诊断truck/motorcycle低α原因

输出：实验数据/usc_experiment_v4_results.json
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
OUT = DATA / 'usc_experiment_v4_results.json'
VOC_DIR = Path(r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007')

BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_PCA = 30

SRC_TMPL = 'a photo of {}'  # VOC源域用自然图像模板
TGT_TMPL = 'a photo of {}'  # COCO目标域也用自然图像模板

# VOC→COCO 20类完整映射
VOC20_TARGETS = [
    ('aeroplane', 'airplane'),
    ('bicycle', 'bicycle'),
    ('bird', 'bird'),
    ('boat', 'boat'),
    ('bottle', 'bottle'),
    ('bus', 'bus'),
    ('car', 'car'),
    ('cat', 'cat'),
    ('chair', 'chair'),
    ('cow', 'cow'),
    ('diningtable', 'dining table'),
    ('dog', 'dog'),
    ('horse', 'horse'),
    ('motorbike', 'motorcycle'),
    ('person', 'person'),
    ('pottedplant', 'potted plant'),
    ('sheep', 'sheep'),
    ('sofa', 'couch'),
    ('train', 'train'),
    ('tvmonitor', 'tv'),
]


# ══════════════════════ 数据加载 ══════════════════════

def load_voc(cls_name, n=100):
    """VOC2007 trainval加载。"""
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
    return images, gts


def load_source_voc(voc_cls, n=50):
    """加载VOC源域数据，提取标注框CLIP特征和原型。"""
    images, gts = load_voc(voc_cls, n=n)
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
    """加载COCO目标域数据，GDINO检测，提取候选框CLIP特征。"""
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


# ══════════════════════ 数据驱动的物理先验 ══════════════════════

def compute_data_driven_priors(source_data, target_data, all_source_protos, cls_name):
    """从数据中计算G/S/I/C物理先验。

    G（粒度失配）：源域类别文本嵌入和目标域类别文本嵌入的余弦距离（1-相似度）
    S（尺度变异）：目标域候选框尺度的变异系数（std/mean）
    I（类内变异）：源域标注框特征的类内方差（1-平均两两余弦相似度）
    C（类间混淆）：该类别原型和其他类别原型的平均余弦相似度
    """
    priors = {}

    # G：文本嵌入余弦距离
    s_emb = T.text_avg([SRC_TMPL], cls_name).cpu().numpy()
    # 目标域类别名（COCO名）
    coco_name = cls_name  # 对于VOC→COCO，类别名基本一致
    d_emb = T.text_avg([TGT_TMPL], coco_name).cpu().numpy()
    G = 1.0 - float(np.dot(s_emb, d_emb))
    priors['G'] = max(0.0, min(1.0, G))

    # S：目标域候选框尺度变异系数
    if target_data is not None and len(target_data['features']) > 0:
        # 用候选框的面积作为尺度代理
        areas = []
        for p in target_data['per_img']:
            if len(p['box']) > 0:
                w = p['box'][:, 2] - p['box'][:, 0]
                h = p['box'][:, 3] - p['box'][:, 1]
                areas.extend((w * h).tolist())
        if len(areas) > 1:
            areas = np.array(areas)
            cv = np.std(areas) / (np.mean(areas) + 1e-8)
            # 归一化到[0,1]，假设CV在[0,2]范围内
            S = min(1.0, cv / 2.0)
        else:
            S = 0.5
    else:
        S = 0.5
    priors['S'] = S

    # I：源域类内变异（1-平均两两余弦相似度）
    if source_data is not None and len(source_data['features']) > 1:
        F = source_data['features']
        # 采样最多100个特征计算平均相似度
        if len(F) > 100:
            idx = np.random.choice(len(F), 100, replace=False)
            F_sample = F[idx]
        else:
            F_sample = F
        sim_matrix = F_sample @ F_sample.T
        # 取上三角（不含对角线）
        n = len(F_sample)
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        avg_sim = np.mean(upper_tri)
        I = 1.0 - avg_sim
    else:
        I = 0.5
    priors['I'] = max(0.0, min(1.0, I))

    # C：类间混淆（和其他类别原型的平均余弦相似度）
    if all_source_protos is not None and len(all_source_protos) > 1:
        proto = source_data['prototype']
        other_sims = []
        for other_name, other_proto in all_source_protos.items():
            if other_name != cls_name:
                other_sims.append(float(np.dot(proto, other_proto)))
        C = np.mean(other_sims) if other_sims else 0.5
    else:
        C = 0.5
    priors['C'] = max(0.0, min(1.0, C))

    return priors


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

def build_proto(src_proto, cls_name, beta=BETA, use_tge=True):
    if not use_tge:
        return src_proto.copy()
    s_emb = T.text_avg([SRC_TMPL], cls_name).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], cls_name).cpu().numpy()
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


# ══════════════════════ 主实验 ══════════════════════

def main():
    print('=' * 60)
    print('USC v4：数据驱动物理先验 + VOC→COCO 20类 + 低α诊断')
    print('=' * 60)

    # ═══ 任务1+2：VOC→COCO 20类实验 ═══
    print('\n' + '=' * 60)
    print('任务1+2：VOC→COCO 20类（数据驱动物理先验）')
    print('=' * 60)

    # 加载所有20类的源域和目标域数据
    print('\n加载VOC源域数据...')
    source_data = {}
    for voc_cls, coco_cls in VOC20_TARGETS:
        print(f'  {voc_cls}...', end=' ', flush=True)
        data = load_source_voc(voc_cls, n=50)
        if data is not None:
            source_data[voc_cls] = data
            print(f'{data["n_crops"]}框')
        else:
            print('无数据')

    print('\n加载COCO目标域数据...')
    target_data = {}
    for voc_cls, coco_cls in VOC20_TARGETS:
        if voc_cls not in source_data:
            continue
        print(f'  {coco_cls}...', end=' ', flush=True)
        data = load_target_coco(coco_cls, n=200)
        if data is not None:
            target_data[voc_cls] = data
            print(f'{data["n_crops"]}候选框')
        else:
            print('无数据')

    # 只保留都有数据的类别
    valid_classes = [c for c in source_data if c in target_data]
    print(f'\n有效类别数: {len(valid_classes)}/20')
    print(f'有效类别: {valid_classes}')

    # 收集所有特征用于USC拟合
    all_src_features = np.vstack([source_data[c]['features'] for c in valid_classes])
    all_tgt_features = np.vstack([target_data[c]['features'] for c in valid_classes])
    print(f'源域特征: {all_src_features.shape}, 目标域特征: {all_tgt_features.shape}')

    # 拟合USC
    usc = USC(n_components=K_PCA).fit(all_src_features, all_tgt_features)

    # 所有源域原型（用于负原型和类间混淆计算）
    all_protos = {c: source_data[c]['prototype'] for c in valid_classes}

    # 基线评估
    print('\n=== 基线评估 ===')
    baseline_aps = {}
    for c in valid_classes:
        baseline_aps[c] = evaluate_baseline(target_data[c])
    mean_baseline = np.mean(list(baseline_aps.values()))
    print(f'  20类均值基线AP: {mean_baseline:.4f}')

    # TGE-only（无USC）
    print('\n=== TGE-only（无USC）===')
    tge_aps = {}
    for c in valid_classes:
        p_hat = build_proto(source_data[c]['prototype'], c, use_tge=True)
        neg = build_neg_proto(all_protos, c)
        tge_aps[c] = evaluate(target_data[c], p_hat, neg)
    mean_tge = np.mean(list(tge_aps.values()))
    print(f'  20类均值AP: {mean_tge:.4f}, gain={mean_tge-mean_baseline:+.4f}')

    # 数据驱动的物理先验
    print('\n=== 数据驱动的物理先验 ===')
    data_priors = {}
    for c in valid_classes:
        priors = compute_data_driven_priors(source_data[c], target_data[c], all_protos, c)
        data_priors[c] = priors
        print(f'  {c}: G={priors["G"]:.3f}, S={priors["S"]:.3f}, I={priors["I"]:.3f}, C={priors["C"]:.3f}')

    # 逐类oracle最佳α
    print('\n=== 逐类oracle最佳α扫描 ===')
    ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    oracle_best = {}
    for c in valid_classes:
        best_alpha = 0.0
        best_ap = baseline_aps[c]
        for alpha in ALPHAS:
            src_corr = usc.correct_prototype(source_data[c]['prototype'], alpha)
            p_hat = build_proto(src_corr, c, use_tge=True)
            neg_raw = build_neg_proto(all_protos, c)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            ap = evaluate(target_data[c], p_hat, neg)
            if ap > best_ap:
                best_ap = ap
                best_alpha = alpha
        oracle_best[c] = {'best_alpha': best_alpha, 'best_ap': best_ap}
    oracle_mean = np.mean([v['best_ap'] for v in oracle_best.values()])
    print(f'  Oracle均值AP: {oracle_mean:.4f}, gain={oracle_mean-mean_baseline:+.4f}')

    # 物理先验和oracleα的相关性
    print('\n=== 数据驱动物理先验 vs oracleα 相关性 ===')
    oracle_alphas = np.array([oracle_best[c]['best_alpha'] for c in valid_classes])
    for factor in ['G', 'S', 'I', 'C']:
        values = np.array([data_priors[c][factor] for c in valid_classes])
        rho, p = spearmanr(values, oracle_alphas)
        print(f'  {factor}: ρ={rho:.3f}, p={p:.3f}')

    # 等权R
    R_values = np.array([np.mean([data_priors[c][f] for f in ['G', 'S', 'I', 'C']]) for c in valid_classes])
    rho, p = spearmanr(R_values, oracle_alphas)
    print(f'  等权R: ρ={rho:.3f}, p={p:.3f}')

    # 固定α=1.0（VOC→COCO是同域，可能需要不同α）
    print('\n=== USC+TGE（固定α扫描）===')
    fixed_results = {}
    for alpha in ALPHAS:
        aps = {}
        for c in valid_classes:
            src_corr = usc.correct_prototype(source_data[c]['prototype'], alpha)
            p_hat = build_proto(src_corr, c, use_tge=True)
            neg_raw = build_neg_proto(all_protos, c)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            aps[c] = evaluate(target_data[c], p_hat, neg)
        mean_ap = np.mean(list(aps.values()))
        fixed_results[alpha] = {'mean': mean_ap, 'gain': mean_ap - mean_baseline, 'per_class': aps}
        print(f'  α={alpha}: 均值AP={mean_ap:.4f}, gain={mean_ap-mean_baseline:+.4f}')

    best_fixed_alpha = max(ALPHAS, key=lambda a: fixed_results[a]['mean'])
    print(f'  最佳固定α={best_fixed_alpha}, 均值AP={fixed_results[best_fixed_alpha]["mean"]:.4f}')

    # ═══ 任务3：低α诊断（truck/motorcycle） ═══
    print('\n' + '=' * 60)
    print('任务3：低α类别诊断')
    print('=' * 60)

    # 找出oracleα最低的3个类别
    low_alpha_classes = sorted(valid_classes, key=lambda c: oracle_best[c]['best_alpha'])[:5]
    print(f'\nOracleα最低的5个类别:')
    for c in low_alpha_classes:
        print(f'  {c}: oracleα={oracle_best[c]["best_alpha"]}, '
              f'baseline={baseline_aps[c]:.4f}, '
              f'oracleAP={oracle_best[c]["best_ap"]:.4f}, '
              f'gain={oracle_best[c]["best_ap"]-baseline_aps[c]:+.4f}')

    # 深入诊断：对比低α类别和高α类别的特征统计
    print('\n=== 低α vs 高α 特征统计对比 ===')
    high_alpha_classes = sorted(valid_classes, key=lambda c: -oracle_best[c]['best_alpha'])[:5]

    diagnosis = {}
    for group_name, group_classes in [('low_alpha', low_alpha_classes), ('high_alpha', high_alpha_classes)]:
        print(f'\n  {group_name} ({group_classes}):')
        stats = {
            'classes': group_classes,
            'avg_oracle_alpha': float(np.mean([oracle_best[c]['best_alpha'] for c in group_classes])),
            'avg_baseline_ap': float(np.mean([baseline_aps[c] for c in group_classes])),
            'avg_G': float(np.mean([data_priors[c]['G'] for c in group_classes])),
            'avg_S': float(np.mean([data_priors[c]['S'] for c in group_classes])),
            'avg_I': float(np.mean([data_priors[c]['I'] for c in group_classes])),
            'avg_C': float(np.mean([data_priors[c]['C'] for c in group_classes])),
        }
        # 域偏移方向投影长度（该类别原型在域偏移方向上的投影）
        proj_lengths = []
        for c in group_classes:
            proto = source_data[c]['prototype']
            proto_proj = usc._project(proto.reshape(1, -1))[0]
            proj_len = float(np.dot(proto_proj, usc.domain_shift_))
            proj_lengths.append(proj_len)
        stats['avg_domain_shift_proj'] = float(np.mean(proj_lengths))

        for k, v in stats.items():
            if k != 'classes':
                print(f'    {k}: {v:.4f}' if isinstance(v, float) else f'    {k}: {v}')
        diagnosis[group_name] = stats

    # 保存结果
    results = {
        'config': {'beta': BETA, 'lam': LAM, 'k_pca': K_PCA, 'n_classes': len(valid_classes)},
        'valid_classes': valid_classes,
        'baseline_mean': float(mean_baseline),
        'tge_only_mean': float(mean_tge),
        'tge_only_gain': float(mean_tge - mean_baseline),
        'oracle_mean': float(oracle_mean),
        'oracle_gain': float(oracle_mean - mean_baseline),
        'best_fixed_alpha': float(best_fixed_alpha),
        'best_fixed_mean': float(fixed_results[best_fixed_alpha]['mean']),
        'best_fixed_gain': float(fixed_results[best_fixed_alpha]['gain']),
        'data_priors': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in data_priors.items()},
        'oracle_best': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in oracle_best.items()},
        'fixed_alpha_results': {str(k): {'mean': float(v['mean']), 'gain': float(v['gain'])} for k, v in fixed_results.items()},
        'low_alpha_diagnosis': diagnosis,
        'per_class_results': {
            c: {
                'baseline': float(baseline_aps[c]),
                'tge_only': float(tge_aps[c]),
                'oracle_alpha': float(oracle_best[c]['best_alpha']),
                'oracle_ap': float(oracle_best[c]['best_ap']),
                'best_fixed_ap': float(fixed_results[best_fixed_alpha]['per_class'][c]),
            } for c in valid_classes
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
