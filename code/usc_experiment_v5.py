# -*- coding: utf-8 -*-
"""usc_experiment_v5.py — 多源域USC普适性验证

验证USC模块在四个遥感源域上的普适性：
  - DIOR → COCO（6类）
  - NWPU → COCO（6类）
  - DOTA → COCO（6类，plane/ship/large-vehicle/small-vehicle映射）
  - VisDrone → COCO（4类，无airplane/boat）

每个源域上：
  1. 拟合USC（K=30）
  2. 扫描α ∈ [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
  3. 评估：基线、TGE-only、USC+TGE(最佳α)、逐类oracle
  4. 对比四个源域

输出：实验数据/usc_experiment_v5_results.json
"""
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

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'usc_experiment_v5_results.json'

BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_PCA = 30
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

SRC_TMPL_OVERHEAD = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

# ══════════════════════ 源域配置 ══════════════════════

# 每个源域的类别映射：(评估名, COCO目标类, 源域类, 源域文本模板)
SOURCE_CONFIGS = {
    'DIOR': {
        'load_func': 'dior',
        'targets': [
            ('airplane', 'airplane', 'airplane'),
            ('boat', 'boat', 'ship'),
            ('car', 'car', 'vehicle'),
            ('truck', 'truck', 'vehicle'),
            ('bus', 'bus', 'vehicle'),
            ('motorcycle', 'motorcycle', 'vehicle'),
        ],
        'src_tmpl': SRC_TMPL_OVERHEAD,
    },
    'NWPU': {
        'load_func': 'nwpu',
        'targets': [
            ('airplane', 'airplane', 'airplane'),
            ('boat', 'boat', 'ship'),
            ('car', 'car', 'vehicle'),
            ('truck', 'truck', 'vehicle'),
            ('bus', 'bus', 'vehicle'),
            ('motorcycle', 'motorcycle', 'vehicle'),
        ],
        'src_tmpl': SRC_TMPL_OVERHEAD,
    },
    'DOTA': {
        'load_func': 'dota',
        'targets': [
            ('airplane', 'airplane', 'plane'),
            ('boat', 'boat', 'ship'),
            ('car', 'car', 'small-vehicle'),
            ('truck', 'truck', 'large-vehicle'),
            ('bus', 'bus', 'large-vehicle'),
            ('motorcycle', 'motorcycle', 'small-vehicle'),
        ],
        'src_tmpl': SRC_TMPL_OVERHEAD,
    },
    'VisDrone': {
        'load_func': 'visdrone',
        'targets': [
            ('car', 'car', 'car'),
            ('truck', 'truck', 'truck'),
            ('bus', 'bus', 'bus'),
            ('motorcycle', 'motorcycle', 'motor'),
        ],
        'src_tmpl': SRC_TMPL_OVERHEAD,
    },
}

# ══════════════════════ 源域加载函数 ══════════════════════

NWPU_DIR = Path(r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10')
NWPU_CLASSES = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
                 'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']
NWPU_CLS_ID = {c: i + 1 for i, c in enumerate(NWPU_CLASSES)}

DOTA_IMG_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
DOTA_LBL_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'

VISDRONE_DIR = Path(r'C:\Users\张三\Downloads\VisDrone2019-DET-val\VisDrone2019-DET-val')
VISDRONE_CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                    'tricycle', 'awning-tricycle', 'bus', 'motor']
VISDRONE_CLS_ID = {c: i + 1 for i, c in enumerate(VISDRONE_CLASSES)}


def load_source_dior(cls_name, n=50):
    images, gts = U.load_dior(cls_name, n=n)
    return _extract_source_features(images, gts)


def load_source_nwpu(cls_name, n=50):
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
    return _extract_source_features(images, gts)


def load_source_dota(cls_name, n=50):
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
    return _extract_source_features(images, gts)


def load_source_visdrone(cls_name, n=50):
    cls_id = VISDRONE_CLS_ID.get(cls_name)
    if cls_id is None:
        return None
    ann_dir = VISDRONE_DIR / 'annotations'
    img_dir = VISDRONE_DIR / 'images'
    images, gts = [], []
    for txt in sorted(ann_dir.glob('*.txt')):
        boxes = []
        for line in txt.read_text(encoding='utf-8', errors='ignore').splitlines():
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            try:
                l, t, w, h, score, cat = [float(parts[i]) for i in range(5)] + [int(parts[5])]
            except Exception:
                continue
            if int(cat) != cls_id or score < 1:
                continue
            boxes.append([int(l), int(t), int(l + w), int(t + h)])
        if not boxes:
            continue
        img_path = img_dir / (txt.stem + '.jpg')
        if not img_path.exists():
            continue
        try:
            images.append(Image.open(img_path).convert('RGB'))
            gts.append(boxes)
        except Exception:
            continue
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


LOAD_FUNCS = {
    'dior': load_source_dior,
    'nwpu': load_source_nwpu,
    'dota': load_source_dota,
    'visdrone': load_source_visdrone,
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


# ══════════════════════ 单源域实验 ══════════════════════

def run_source(source_name, config):
    print(f'\n{"="*60}')
    print(f'源域: {source_name}')
    print(f'{"="*60}')

    load_func = LOAD_FUNCS[config['load_func']]
    src_tmpl = config['src_tmpl']
    targets = config['targets']

    # 加载源域
    print('\n加载源域数据...')
    source_data = {}
    for eval_name, coco_cls, src_cls in targets:
        print(f'  {src_cls}...', end=' ', flush=True)
        data = load_func(src_cls, n=50)
        if data is not None:
            source_data[src_cls] = data
            print(f'{data["n_crops"]}框')
        else:
            print('无数据')

    # 加载目标域
    print('\n加载目标域(COCO)数据...')
    target_data = {}
    valid_targets = []
    for eval_name, coco_cls, src_cls in targets:
        if src_cls not in source_data:
            continue
        print(f'  {coco_cls}...', end=' ', flush=True)
        data = load_target_coco(coco_cls, n=200)
        if data is not None:
            target_data[coco_cls] = data
            valid_targets.append((eval_name, coco_cls, src_cls))
            print(f'{data["n_crops"]}候选框')
        else:
            print('无数据')

    if len(valid_targets) < 2:
        print('有效类别太少，跳过')
        return None

    print(f'\n有效类别数: {len(valid_targets)}')

    # 收集特征
    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])

    # 拟合USC
    usc = USC(n_components=K_PCA).fit(all_src, all_tgt)

    # 所有源域原型
    all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}

    # 基线
    baseline_aps = {}
    for eval_name, coco_cls, src_cls in valid_targets:
        baseline_aps[coco_cls] = evaluate_baseline(target_data[coco_cls])
    mean_baseline = np.mean(list(baseline_aps.values()))

    # TGE-only
    tge_aps = {}
    for eval_name, coco_cls, src_cls in valid_targets:
        p_hat = build_proto(source_data[src_cls]['prototype'], src_cls, coco_cls, src_tmpl, use_tge=True)
        neg = build_neg_proto(all_protos, src_cls)
        tge_aps[coco_cls] = evaluate(target_data[coco_cls], p_hat, neg)
    mean_tge = np.mean(list(tge_aps.values()))

    # 固定α扫描
    fixed_results = {}
    for alpha in ALPHAS:
        aps = {}
        for eval_name, coco_cls, src_cls in valid_targets:
            src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], alpha)
            p_hat = build_proto(src_corr, src_cls, coco_cls, src_tmpl, use_tge=True)
            neg_raw = build_neg_proto(all_protos, src_cls)
            neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None
            aps[coco_cls] = evaluate(target_data[coco_cls], p_hat, neg)
        mean_ap = np.mean(list(aps.values()))
        fixed_results[alpha] = {'mean': mean_ap, 'gain': mean_ap - mean_baseline, 'per_class': aps}

    best_fixed_alpha = max(ALPHAS, key=lambda a: fixed_results[a]['mean'])
    best_fixed_mean = fixed_results[best_fixed_alpha]['mean']

    # 逐类oracle
    oracle_best = {}
    for eval_name, coco_cls, src_cls in valid_targets:
        best_alpha = 0.0
        best_ap = baseline_aps[coco_cls]
        for alpha in ALPHAS:
            ap = fixed_results[alpha]['per_class'][coco_cls]
            if ap > best_ap:
                best_ap = ap
                best_alpha = alpha
        oracle_best[coco_cls] = {'best_alpha': best_alpha, 'best_ap': best_ap}
    oracle_mean = np.mean([v['best_ap'] for v in oracle_best.values()])

    # 输出结果
    print(f'\n--- {source_name} 结果 ---')
    print(f'  基线: {mean_baseline:.4f}')
    print(f'  TGE-only: {mean_tge:.4f} (gain={mean_tge-mean_baseline:+.4f})')
    print(f'  最佳固定α={best_fixed_alpha}: {best_fixed_mean:.4f} (gain={best_fixed_mean-mean_baseline:+.4f})')
    print(f'  Oracle: {oracle_mean:.4f} (gain={oracle_mean-mean_baseline:+.4f})')
    print(f'  USC增益(over TGE): {best_fixed_mean-mean_tge:+.4f}')
    print(f'  逐类oracle α: {dict(zip([c for _,c,_ in valid_targets], [oracle_best[c]["best_alpha"] for _,c,_ in valid_targets]))}')

    return {
        'n_classes': len(valid_targets),
        'baseline_mean': float(mean_baseline),
        'tge_only_mean': float(mean_tge),
        'tge_only_gain': float(mean_tge - mean_baseline),
        'best_fixed_alpha': float(best_fixed_alpha),
        'best_fixed_mean': float(best_fixed_mean),
        'best_fixed_gain': float(best_fixed_mean - mean_baseline),
        'usc_gain_over_tge': float(best_fixed_mean - mean_tge),
        'oracle_mean': float(oracle_mean),
        'oracle_gain': float(oracle_mean - mean_baseline),
        'fixed_alpha_results': {str(k): {'mean': float(v['mean']), 'gain': float(v['gain'])} for k, v in fixed_results.items()},
        'oracle_best': {k: {'best_alpha': float(v['best_alpha']), 'best_ap': float(v['best_ap'])} for k, v in oracle_best.items()},
        'per_class_baseline': {k: float(v) for k, v in baseline_aps.items()},
        'per_class_tge': {k: float(v) for k, v in tge_aps.items()},
    }


# ══════════════════════ 主函数 ══════════════════════

def main():
    print('=' * 60)
    print('USC v5：多源域普适性验证（DIOR/NWPU/DOTA/VisDrone → COCO）')
    print('=' * 60)

    all_results = {}
    for source_name, config in SOURCE_CONFIGS.items():
        result = run_source(source_name, config)
        if result is not None:
            all_results[source_name] = result

    # 汇总对比
    print(f'\n{"="*60}')
    print('四源域汇总对比')
    print(f'{"="*60}')
    print(f'{"源域":<12} {"类别数":<6} {"基线":<8} {"TGE-only":<10} {"USC最佳α":<10} {"USC增益":<10} {"Oracle":<10}')
    print('-' * 70)
    for source_name, r in all_results.items():
        print(f'{source_name:<12} {r["n_classes"]:<6} {r["baseline_mean"]:<8.4f} '
              f'{r["tge_only_mean"]:<10.4f} {r["best_fixed_mean"]:<10.4f} '
              f'{r["usc_gain_over_tge"]:<+10.4f} {r["oracle_mean"]:<10.4f}')

    # 保存
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
