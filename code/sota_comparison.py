# -*- coding: utf-8 -*-
"""sota_comparison.py — SOTA对比实验：7种方法在4个源域→COCO上的对比"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import unified_eval as U
import tge_probe as T
from scipy.stats import spearmanr

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'sota_comparison_results.json'

BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
K_PCA = 30

SRC_TMPL_OVERHEAD = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

NWPU_DIR = Path(r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10')
DOTA_IMG_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
DOTA_LBL_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'
VISDRONE_DIR = Path(r'C:\Users\张三\Downloads\VisDrone2019-DET-val\VisDrone2019-DET-val')

NWPU_CLASSES = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
                 'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']
NWPU_CLS_ID = {c: i + 1 for i, c in enumerate(NWPU_CLASSES)}

VISDRONE_CLS_ID = {
    'pedestrian': 1, 'people': 2, 'bicycle': 3, 'car': 4, 'van': 5,
    'truck': 6, 'tricycle': 7, 'awning-tricycle': 8, 'bus': 9, 'motor': 10,
}

# DIOR只有20类：airplane(0), ship(13), vehicle(18)
DIOR_TARGETS = [
    ('airplane', 'airplane', 'airplane'),
    ('boat', 'ship', 'ship'),
    ('car', 'vehicle', 'vehicle'),
]

# NWPU有10类：airplane, ship, vehicle等
NWPU_TARGETS = [
    ('airplane', 'airplane', 'airplane'),
    ('boat', 'ship', 'ship'),
    ('car', 'vehicle', 'vehicle'),
]

# DOTA有15类
DOTA_TARGETS = [
    ('airplane', 'plane', 'plane'),
    ('boat', 'ship', 'ship'),
    ('car', 'small-vehicle', 'small-vehicle'),
    ('truck', 'large-vehicle', 'large-vehicle'),
]

# VisDrone有10类，增加person（pedestrian+people合并）和van（映射到car）
VISDRONE_TARGETS = [
    ('bicycle', 'bicycle', 'bicycle'),
    ('car', 'car', 'car'),
    ('bus', 'bus', 'bus'),
    ('motorcycle', 'motor', 'motor'),
    ('person', ['pedestrian', 'people'], 'person'),
    ('car', 'van', 'van'),  # van映射到car，增加car的样本数
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
    import re
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


def load_visdrone_src(cls_name, n=50):
    img_dir = VISDRONE_DIR / 'images'
    ann_dir = VISDRONE_DIR / 'annotations'
    # 支持多类别合并（传入类别列表）
    if isinstance(cls_name, list):
        cls_ids = [VISDRONE_CLS_ID.get(c) for c in cls_name if VISDRONE_CLS_ID.get(c) is not None]
    else:
        cls_id = VISDRONE_CLS_ID.get(cls_name)
        if cls_id is None: return None
        cls_ids = [cls_id]
    if not cls_ids: return None
    images, gts = [], []
    for ann_path in sorted(ann_dir.glob('*.txt')):
        boxes = []
        for line in ann_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            try:
                x1, y1, w, h = map(float, parts[:4])
                cat = int(parts[5])
                if cat not in cls_ids: continue
                if w < 10 or h < 10: continue
                boxes.append([x1, y1, x1 + w, y1 + h])
            except Exception: continue
        if not boxes: continue
        img_path = img_dir / (ann_path.stem + '.jpg')
        if not img_path.exists(): continue
        try:
            images.append(Image.open(img_path).convert('RGB'))
            gts.append(boxes)
        except Exception: continue
        if len(images) >= n: break
    return _extract_features(images, gts)


SOURCE_LOADERS = {
    'DIOR': (load_dior_src, DIOR_TARGETS),
    'NWPU': (load_nwpu_src, NWPU_TARGETS),
    'DOTA': (load_dota_src, DOTA_TARGETS),
    'VisDrone': (load_visdrone_src, VISDRONE_TARGETS),
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


def build_proto_tge(src_proto, src_cls, tgt_cls, beta=BETA):
    """TGE: 文本引导外推"""
    s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
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


def coral_align(source_features, target_features):
    """CORAL: 相关性对齐（闭式解）"""
    src_mean = source_features.mean(axis=0)
    tgt_mean = target_features.mean(axis=0)
    src_cov = np.cov(source_features.T) + 1e-6 * np.eye(source_features.shape[1])
    tgt_cov = np.cov(target_features.T) + 1e-6 * np.eye(target_features.shape[1])
    # 矩阵平方根
    def matrix_sqrt(M):
        eigvals, eigvecs = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    src_sqrt = matrix_sqrt(src_cov)
    tgt_inv_sqrt = matrix_sqrt(np.linalg.inv(tgt_cov))
    # 对齐目标域特征到源域分布
    target_centered = target_features - tgt_mean
    target_aligned = target_centered @ (src_sqrt @ tgt_inv_sqrt) + src_mean
    # 归一化
    norms = np.linalg.norm(target_aligned, axis=1, keepdims=True)
    return target_aligned / np.maximum(norms, 1e-8)


def tent_align(source_features, target_features):
    """Tent: 熵最小化测试时自适应（特征校准变体）"""
    # 简单的特征校准：对齐均值和方差
    src_mean = source_features.mean(axis=0)
    src_std = source_features.std(axis=0) + 1e-8
    tgt_mean = target_features.mean(axis=0)
    tgt_std = target_features.std(axis=0) + 1e-8
    target_aligned = (target_features - tgt_mean) / tgt_std * src_std + src_mean
    norms = np.linalg.norm(target_aligned, axis=1, keepdims=True)
    return target_aligned / np.maximum(norms, 1e-8)


class DANN:
    """DANN: 域对抗训练（用候选框特征训练简单的特征变换网络）"""
    def __init__(self, input_dim=512, hidden_dim=128, lr=0.01, epochs=200, l2_reg=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        # 特征变换网络参数
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b2 = np.zeros(input_dim)
        # 域判别器参数
        self.Wd = np.random.randn(hidden_dim, 1) * 0.01
        self.bd = np.zeros(1)
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def fit(self, source_features, target_features):
        n_src = len(source_features)
        n_tgt = len(target_features)
        X = np.vstack([source_features, target_features])
        y_domain = np.vstack([np.zeros((n_src, 1)), np.ones((n_tgt, 1))])
        
        for epoch in range(self.epochs):
            # 前向传播
            h = self._relu(X @ self.W1 + self.b1)
            X_transformed = h @ self.W2 + self.b2
            domain_logits = h @ self.Wd + self.bd
            domain_pred = self._sigmoid(domain_logits)
            
            # 域判别器损失
            domain_loss = -np.mean(y_domain * np.log(domain_pred + 1e-8) + 
                                    (1 - y_domain) * np.log(1 - domain_pred + 1e-8))
            
            # 梯度反转：特征变换网络的梯度 = -域判别器梯度（域对抗）
            # 域判别器梯度
            d_domain_logits = (domain_pred - y_domain) / len(X)
            d_Wd = h.T @ d_domain_logits + self.l2_reg * self.Wd
            d_bd = np.sum(d_domain_logits, axis=0)
            d_h_domain = d_domain_logits @ self.Wd.T
            
            # 特征重构损失（保持特征信息）
            recon_loss = np.mean((X_transformed - X) ** 2)
            d_X_transformed = 2 * (X_transformed - X) / len(X)
            d_W2 = h.T @ d_X_transformed + self.l2_reg * self.W2
            d_b2 = np.sum(d_X_transformed, axis=0)
            d_h_recon = d_X_transformed @ self.W2.T
            
            # 总梯度（域对抗：反转域梯度）
            d_h = d_h_recon - 0.1 * d_h_domain  # 梯度反转系数0.1
            
            d_relu = (h > 0).astype(float) * d_h
            d_W1 = X.T @ d_relu + self.l2_reg * self.W1
            d_b1 = np.sum(d_relu, axis=0)
            
            # 更新参数
            self.W1 -= self.lr * d_W1
            self.b1 -= self.lr * d_b1
            self.W2 -= self.lr * d_W2
            self.b2 -= self.lr * d_b2
            self.Wd -= self.lr * d_Wd
            self.bd -= self.lr * d_bd
        
        return self
    
    def transform(self, X):
        h = self._relu(X @ self.W1 + self.b1)
        X_transformed = h @ self.W2 + self.b2
        norms = np.linalg.norm(X_transformed, axis=1, keepdims=True)
        return X_transformed / np.maximum(norms, 1e-8)


def evaluate_with_features(target_data, proto, features_aligned=None, lam=LAM, use_neg=True, neg_proto=None):
    """通用评估函数：支持特征对齐和原型校正"""
    per_img = target_data['per_img']
    F = features_aligned if features_aligned is not None else target_data['features']
    for p in per_img:
        n = len(p['gd'])
        if n:
            f_boxes = F[p['crop_idx']]
            pos_sim = f_boxes @ proto
            if use_neg and neg_proto is not None:
                neg_sim = f_boxes @ neg_proto
                vis = np.maximum(0.0, pos_sim - lam * neg_sim)
            else:
                vis = np.maximum(0.0, pos_sim)
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


# ========== 双因子自适应α模型（第二篇论文完整方法）==========

def load_dual_factor_model():
    """加载双因子线性回归模型参数"""
    model_path = DATA / 'dual_factor_model_params.json'
    if not model_path.exists():
        return None
    with open(model_path, 'r') as f:
        return json.load(f)


def compute_box_size_variance(target_data):
    """计算候选框尺寸方差（零训练因子1）：sqrt(面积)的方差，与训练模型时一致"""
    all_boxes = []
    for p in target_data['per_img']:
        if len(p['box']) > 0:
            all_boxes.append(p['box'])
    if not all_boxes:
        return 0.0
    all_boxes = np.vstack(all_boxes)
    widths = all_boxes[:, 2] - all_boxes[:, 0]
    heights = all_boxes[:, 3] - all_boxes[:, 1]
    sizes = np.sqrt(np.maximum(widths * heights, 0))
    return float(np.var(sizes))


def compute_text_confusability(coco_cls, all_coco_classes):
    """计算文本混淆度（零训练因子2）：该类别与其他类别的最小文本嵌入距离，与训练模型时一致"""
    t_emb = T.text_avg([TGT_TMPL], coco_cls).cpu().numpy()
    min_dist = float('inf')
    for other_cls in all_coco_classes:
        if other_cls == coco_cls:
            continue
        other_emb = T.text_avg([TGT_TMPL], other_cls).cpu().numpy()
        dist = float(np.linalg.norm(t_emb - other_emb))
        if dist < min_dist:
            min_dist = dist
    return min_dist if min_dist != float('inf') else 0.0


def predict_alpha_adaptive(box_size_var, text_conf, model_params):
    """用双因子模型预测自适应α"""
    if model_params is None:
        return 1.0  #  fallback to fixed alpha
    sv_norm = (box_size_var - model_params['X_mean_sv']) / model_params['X_std_sv']
    tc_norm = (text_conf - model_params['X_mean_tc']) / model_params['X_std_tc']
    alpha = model_params['w_sv'] * sv_norm + model_params['w_tc'] * tc_norm + model_params['bias']
    return float(max(0.0, min(2.0, alpha)))  # 截断到[0, 2]


# ========== 最新方法简化版复现 ==========

def build_proto_upre(src_proto, coco_cls, all_categories):
    """UPRE (ICCV 2025) 简化版：多视角域提示 + 相对域距离加权
    
    核心思想：用多个文本模板（源域视角、目标域视角、通用视角）生成文本嵌入，
    根据源域-目标域相对距离加权融合，同时对源域原型进行表示增强。
    """
    # 多视角文本模板
    templates = [
        TGT_TMPL,                              # 目标域视角：a photo of {}
        SRC_TMPL_OVERHEAD,                     # 源域视角：an overhead image of {}
        'a {} object',                          # 通用视角
        '{} in the real world',                 # 现实世界视角
    ]
    
    # 生成多视角文本嵌入
    text_embeddings = []
    for tmpl in templates:
        text = tmpl.format(coco_cls)
        emb = T.text_avg([text], coco_cls).cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        text_embeddings.append(emb)
    
    # 计算相对域距离：源域视角与目标域视角的距离
    src_emb = text_embeddings[1]  # 源域视角
    tgt_emb = text_embeddings[0]  # 目标域视角
    domain_dist = float(np.linalg.norm(src_emb - tgt_emb))
    
    # 相对域距离加权：距离越大，目标域视角权重越高
    # 归一化距离到[0,1]
    max_dist = 2.0  # 归一化向量的最大距离
    rel_dist = min(domain_dist / max_dist, 1.0)
    
    # 加权融合：源域原型 + 域偏移方向
    # URE思想：从源域表示生成目标域表示
    domain_shift = tgt_emb - src_emb
    enhanced_proto = src_proto + rel_dist * 0.5 * domain_shift
    
    # 归一化
    enhanced_proto = enhanced_proto / (np.linalg.norm(enhanced_proto) + 1e-8)
    
    return enhanced_proto


def evaluate_factor(target_data, proto, n_perturb=5, perturb_scale=0.1):
    """FACTOR (arXiv 2026) 简化版：反事实扰动 + 预测稳定性校准
    
    核心思想：对候选框特征加入随机扰动（模拟亮度、纹理等非因果属性变化），
    比较原始和扰动后的预测，用预测稳定性校准分数。
    稳定的预测置信度高，不稳定的置信度低。
    """
    per_img = target_data['per_img']
    F = target_data['features']
    
    for p in per_img:
        n = len(p['gd'])
        if n:
            f_boxes = F[p['crop_idx']]
            # 原始预测分数
            pos_sim_original = f_boxes @ proto
            
            # 多次反事实扰动，计算预测稳定性
            all_sims = [pos_sim_original]
            for _ in range(n_perturb):
                # 随机扰动特征（模拟非因果属性变化）
                noise = np.random.randn(*f_boxes.shape) * perturb_scale
                f_perturbed = f_boxes + noise
                sim_perturbed = f_perturbed @ proto
                all_sims.append(sim_perturbed)
            
            all_sims = np.array(all_sims)
            # 预测均值和方差
            sim_mean = np.mean(all_sims, axis=0)
            sim_std = np.std(all_sims, axis=0) + 1e-8
            
            # 反事实敏感度分数（CSS）：标准化后的方差
            css = 1.0 / (1.0 + sim_std)  # 越稳定，css越高
            
            # 用稳定性校准分数：稳定的预测保留高置信度，不稳定的降低
            calibrated_sim = sim_mean * css
            
            # 几何平均融合
            vis = np.maximum(0.0, calibrated_sim)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)
    
    preds = [(p['box'], p['fused']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return U.compute_ap50_img(preds, gts)


def evaluate_bca_plus(target_data, src_proto, src_feat):
    """BCA+ (OpenReview 2026) 简化版：贝叶斯类适应 + 目标域先验融合
    
    核心思想：用目标域候选框的统计量作为类先验，
    用贝叶斯公式融合源域原型（似然）和目标域先验，
    同时自适应调整空间尺度。
    """
    per_img = target_data['per_img']
    F = target_data['features']
    
    # 计算目标域候选框的平均特征作为类先验
    all_features = []
    for p in per_img:
        if len(p['gd']) > 0:
            # 只取高置信度候选框作为先验
            high_conf_mask = p['gd'] > np.median(p['gd'])
            if np.any(high_conf_mask):
                all_features.append(F[p['crop_idx']][high_conf_mask])
    
    if all_features:
        all_features = np.vstack(all_features)
        target_prior = np.mean(all_features, axis=0)
        target_prior = target_prior / (np.linalg.norm(target_prior) + 1e-8)
    else:
        target_prior = src_proto.copy()
    
    # 贝叶斯融合：源域原型（似然）+ 目标域先验
    # 后验 ∝ 似然 × 先验
    # 在特征空间中，用加权平均模拟贝叶斯融合
    # 源域原型权重：基于源域特征的置信度
    src_confidence = 0.6  # 源域原型的先验置信度
    tgt_confidence = 0.4  # 目标域先验的置信度
    
    # 归一化权重
    total_conf = src_confidence + tgt_confidence
    w_src = src_confidence / total_conf
    w_tgt = tgt_confidence / total_conf
    
    # 贝叶斯融合后的原型
    bayes_proto = w_src * src_proto + w_tgt * target_prior
    bayes_proto = bayes_proto / (np.linalg.norm(bayes_proto) + 1e-8)
    
    # 自适应空间尺度：根据目标域候选框尺寸分布调整
    all_boxes = []
    for p in per_img:
        if len(p['box']) > 0:
            all_boxes.append(p['box'])
    
    if all_boxes:
        all_boxes = np.vstack(all_boxes)
        widths = all_boxes[:, 2] - all_boxes[:, 0]
        heights = all_boxes[:, 3] - all_boxes[:, 1]
        sizes = np.sqrt(np.maximum(widths * heights, 0))
        size_mean = np.mean(sizes)
        # 空间尺度调整：小目标增加原型响应，大目标降低
        scale_factor = 1.0 / (1.0 + size_mean / 1000.0)
    else:
        scale_factor = 1.0
    
    # 评估
    for p in per_img:
        n = len(p['gd'])
        if n:
            f_boxes = F[p['crop_idx']]
            pos_sim = f_boxes @ bayes_proto
            # 应用空间尺度调整
            pos_sim = pos_sim * scale_factor
            vis = np.maximum(0.0, pos_sim)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)
    
    preds = [(p['box'], p['fused']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return U.compute_ap50_img(preds, gts)


def run_experiment():
    results = {}
    
    # 加载双因子自适应α模型
    model_params = load_dual_factor_model()
    if model_params:
        print(f"双因子模型已加载: ρ={model_params['rho']:.4f}, N={model_params['n_samples']}")
    else:
        print("警告：双因子模型未找到，USC将使用固定α=1.0")
    
    # 收集所有类别（包含coco_cls和src_cls，与训练模型时一致）
    all_categories = set()
    for src_name, (loader, targets) in SOURCE_LOADERS.items():
        for coco_cls, src_cls, src_label in targets:
            all_categories.add(coco_cls)
            all_categories.add(src_cls)
    all_categories = sorted(list(all_categories))
    print(f"所有类别（含源域类别名）: {all_categories}")
    
    for src_name, (loader, targets) in SOURCE_LOADERS.items():
        print(f'\n{"="*60}')
        print(f'源域: {src_name}')
        print(f'{"="*60}')
        
        src_results = {}
        
        # 加载所有源域类别数据
        src_data = {}
        for coco_cls, src_cls, src_label in targets:
            print(f'  加载源域 {src_name}/{src_cls}...')
            d = loader(src_cls, n=50)
            if d is not None:
                src_data[coco_cls] = d
        
        # 对每个类别运行所有方法
        for coco_cls in src_data:
            print(f'\n  --- 类别: {coco_cls} ---')
            
            # 加载目标域数据
            target_data = load_target_coco(coco_cls, n=200)
            if target_data is None or target_data['n_crops'] == 0:
                print(f'    跳过：无目标域数据')
                continue
            
            src_feat = src_data[coco_cls]['features']
            src_proto = src_data[coco_cls]['prototype']
            tgt_feat = target_data['features']
            
            cls_results = {}
            
            # 方法1: Baseline
            ap_base = evaluate_baseline(target_data)
            cls_results['Baseline'] = ap_base
            print(f'    Baseline: {ap_base:.4f}')
            
            # 方法2: TGE（文本引导外推）
            proto_tge = build_proto_tge(src_proto, coco_cls, coco_cls, beta=BETA)
            ap_tge = evaluate_with_features(target_data, proto_tge, use_neg=False)
            cls_results['TGE'] = ap_tge
            print(f'    TGE: {ap_tge:.4f} (增益: {ap_tge-ap_base:+.4f})')
            
            # 方法3: A-CSPC（非对称原型校正）
            proto_acspc = build_proto_tge(src_proto, coco_cls, coco_cls, beta=BETA)
            # 构建负原型（用所有源域类别的原型）
            all_protos = {k: v['prototype'] for k, v in src_data.items()}
            neg_proto = build_neg_proto(all_protos, coco_cls)
            ap_acspc = evaluate_with_features(target_data, proto_acspc, use_neg=True, neg_proto=neg_proto)
            cls_results['A-CSPC'] = ap_acspc
            print(f'    A-CSPC: {ap_acspc:.4f} (增益: {ap_acspc-ap_base:+.4f})')
            
            # 方法4: CORAL（相关性对齐）
            try:
                tgt_aligned_coral = coral_align(src_feat, tgt_feat)
                ap_coral = evaluate_with_features(target_data, src_proto, features_aligned=tgt_aligned_coral, use_neg=False)
                cls_results['CORAL'] = ap_coral
                print(f'    CORAL: {ap_coral:.4f} (增益: {ap_coral-ap_base:+.4f})')
            except Exception as e:
                print(f'    CORAL失败: {e}')
                cls_results['CORAL'] = ap_base
            
            # 方法5: DANN（域对抗训练）
            try:
                dann = DANN(input_dim=512, hidden_dim=128, lr=0.01, epochs=200)
                dann.fit(src_feat, tgt_feat)
                tgt_aligned_dann = dann.transform(tgt_feat)
                ap_dann = evaluate_with_features(target_data, src_proto, features_aligned=tgt_aligned_dann, use_neg=False)
                cls_results['DANN'] = ap_dann
                print(f'    DANN: {ap_dann:.4f} (增益: {ap_dann-ap_base:+.4f})')
            except Exception as e:
                print(f'    DANN失败: {e}')
                cls_results['DANN'] = ap_base
            
            # 方法6: Tent（熵最小化测试时自适应）
            try:
                tgt_aligned_tent = tent_align(src_feat, tgt_feat)
                ap_tent = evaluate_with_features(target_data, src_proto, features_aligned=tgt_aligned_tent, use_neg=False)
                cls_results['Tent'] = ap_tent
                print(f'    Tent: {ap_tent:.4f} (增益: {ap_tent-ap_base:+.4f})')
            except Exception as e:
                print(f'    Tent失败: {e}')
                cls_results['Tent'] = ap_base
            
            # 方法7: USC（我们的完整方法：秩1仿射校正 + 双因子自适应α）
            try:
                # 计算零训练因子
                box_size_var = compute_box_size_variance(target_data)
                text_conf = compute_text_confusability(coco_cls, all_categories)
                # 双因子预测自适应α
                alpha_adaptive = predict_alpha_adaptive(box_size_var, text_conf, model_params)
                
                usc = USC(n_components=K_PCA)
                usc.fit(src_feat, tgt_feat)
                proto_usc = usc.correct_prototype(src_proto, alpha=alpha_adaptive)
                ap_usc = evaluate_with_features(target_data, proto_usc, use_neg=False)
                cls_results['USC'] = ap_usc
                print(f'    USC (α={alpha_adaptive:.2f}, sv={box_size_var:.0f}, tc={text_conf:.3f}): {ap_usc:.4f} (增益: {ap_usc-ap_base:+.4f})')
            except Exception as e:
                print(f'    USC失败: {e}')
                import traceback
                traceback.print_exc()
                cls_results['USC'] = ap_base
            
            # 方法8: UPRE (ICCV 2025) 简化版：多视角域提示 + 相对域距离加权
            try:
                proto_upre = build_proto_upre(src_proto, coco_cls, all_categories)
                ap_upre = evaluate_with_features(target_data, proto_upre, use_neg=False)
                cls_results['UPRE'] = ap_upre
                print(f'    UPRE: {ap_upre:.4f} (增益: {ap_upre-ap_base:+.4f})')
            except Exception as e:
                print(f'    UPRE失败: {e}')
                import traceback
                traceback.print_exc()
                cls_results['UPRE'] = ap_base
            
            # 方法9: FACTOR (arXiv 2026) 简化版：反事实扰动 + 预测稳定性校准
            try:
                ap_factor = evaluate_factor(target_data, src_proto, n_perturb=5, perturb_scale=0.1)
                cls_results['FACTOR'] = ap_factor
                print(f'    FACTOR: {ap_factor:.4f} (增益: {ap_factor-ap_base:+.4f})')
            except Exception as e:
                print(f'    FACTOR失败: {e}')
                import traceback
                traceback.print_exc()
                cls_results['FACTOR'] = ap_base
            
            # 方法10: BCA+ (OpenReview 2026) 简化版：贝叶斯类适应 + 目标域先验融合
            try:
                ap_bca = evaluate_bca_plus(target_data, src_proto, src_feat)
                cls_results['BCA+'] = ap_bca
                print(f'    BCA+: {ap_bca:.4f} (增益: {ap_bca-ap_base:+.4f})')
            except Exception as e:
                print(f'    BCA+失败: {e}')
                import traceback
                traceback.print_exc()
                cls_results['BCA+'] = ap_base
            
            src_results[coco_cls] = cls_results
        
        results[src_name] = src_results
    
    # 保存结果
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print(f'\n\n{"="*80}')
    print('SOTA对比实验汇总')
    print(f'{"="*80}')
    
    methods = ['Baseline', 'TGE', 'A-CSPC', 'CORAL', 'DANN', 'Tent', 'USC', 'UPRE', 'FACTOR', 'BCA+']
    
    # 按源域汇总
    for src_name in results:
        print(f'\n--- {src_name} ---')
        cls_names = list(results[src_name].keys())
        for method in methods:
            aps = [results[src_name][cls][method] for cls in cls_names if method in results[src_name][cls]]
            if aps:
                mean_ap = np.mean(aps)
                gain = mean_ap - np.mean([results[src_name][cls]['Baseline'] for cls in cls_names])
                print(f'  {method:12s}: AP50={mean_ap:.4f}, 增益={gain:+.4f}')
    
    # 总体汇总
    print(f'\n--- 总体平均 ---')
    all_aps = {m: [] for m in methods}
    for src_name in results:
        for cls in results[src_name]:
            for method in methods:
                if method in results[src_name][cls]:
                    all_aps[method].append(results[src_name][cls][method])
    
    for method in methods:
        if all_aps[method]:
            mean_ap = np.mean(all_aps[method])
            gain = mean_ap - np.mean(all_aps['Baseline'])
            print(f'  {method:12s}: AP50={mean_ap:.4f}, 增益={gain:+.4f}, N={len(all_aps[method])}')
    
    print(f'\n结果已保存: {OUT}')
    return results


if __name__ == '__main__':
    run_experiment()
