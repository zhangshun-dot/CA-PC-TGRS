# -*- coding: utf-8 -*-
r"""tge_v2_auto_tune_sim_weighted.py — 基于域相似度的源域映射加权实验

方案A：用源域原型与目标域(VOC)原型的余弦相似度作为权重，
      softmax归一化后加权融合多个源域的映射函数预测。

实验设计：
  开发集：DIOR→COCO 6类 + NWPU→COCO 5类 = 11类
  测试集：VOC→COCO 19类

  对比策略：
    1. dior_only:        仅用DIOR映射（上一实验最佳基线）
    2. nwpu_only:        仅用NWPU映射
    3. merged:           合并11类训练一个映射
    4. avg:              简单平均
    5. sim_weighted:     域相似度加权（扫描τ ∈ {0.5,1,2,5,10,20}）
    6. sim_weighted_cls: 类别级相似度加权（更精细）

  如果sim_weighted > dior_only (+0.0459)，则考虑方案B(MoE门控)。

运行: python tge_v2_auto_tune_sim_weighted.py
输出: 实验数据/tge_v2_auto_tune_sim_weighted_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import json
import pickle
import time
import os
import glob
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

from acspc.auto_tuner import GridSearchTuner, LinearMapper, PolynomialMapper, MLPMapper, ClassOptimal
from acspc.coco_categories import get_offline_prior
from acspc.appg import PhysicsPrior

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

K_CALIB = 10
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
PRE_FILTER = 0.03

VOC_ROOT = r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007'
VOC_IMG_DIR = os.path.join(VOC_ROOT, 'JPEGImages')
VOC_ANN_DIR = os.path.join(VOC_ROOT, 'Annotations')

NWPU_ROOT = r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10'
NWPU_IMG_DIR = os.path.join(NWPU_ROOT, 'positive_image_set')
NWPU_ANN_DIR = os.path.join(NWPU_ROOT, 'ground_truth')

NWPU_ID2NAME = {
    1: 'airplane', 2: 'ship', 3: 'storage tank', 4: 'baseball diamond',
    5: 'tennis court', 6: 'basketball court', 7: 'ground track field',
    8: 'harbor', 9: 'bridge', 10: 'vehicle',
}

VOC_TO_COCO = {
    'person': 'person', 'bicycle': 'bicycle', 'car': 'car',
    'motorbike': 'motorcycle', 'aeroplane': 'airplane', 'bus': 'bus',
    'train': 'train', 'truck': 'truck', 'boat': 'boat', 'bird': 'bird',
    'cat': 'cat', 'dog': 'dog', 'horse': 'horse', 'sheep': 'sheep',
    'cow': 'cow', 'bottle': 'bottle', 'chair': 'chair', 'sofa': 'couch',
    'pottedplant': 'potted plant', 'tvmonitor': 'tv',
}

# VOC类别 → DIOR/NWPU中对应的源域类别（用于类别级相似度）
VOC_TO_SRC_CLS = {
    'aeroplane': 'airplane',
    'boat': 'ship',
    'car': 'vehicle',
    'truck': 'vehicle',
    'bus': 'vehicle',
    'motorbike': 'vehicle',
    # 非交通工具类没有直接对应，用None标记
    'person': None, 'bicycle': None, 'train': None, 'bird': None,
    'cat': None, 'dog': None, 'horse': None, 'sheep': None,
    'cow': None, 'bottle': None, 'chair': None, 'sofa': None,
    'pottedplant': None, 'tvmonitor': None,
}

DEV_TARGETS_DIOR = [
    ('DIOR-COCO-airplane',   'airplane',   'airplane'),
    ('DIOR-COCO-boat',       'boat',       'ship'),
    ('DIOR-COCO-car',        'car',        'vehicle'),
    ('DIOR-COCO-truck',      'truck',      'vehicle'),
    ('DIOR-COCO-bus',        'bus',        'vehicle'),
    ('DIOR-COCO-motorcycle', 'motorcycle', 'vehicle'),
]

DEV_TARGETS_NWPU = [
    ('NWPU-COCO-airplane', 'airplane', 'airplane'),
    ('NWPU-COCO-boat',     'boat',     'ship'),
    ('NWPU-COCO-car',      'car',      'vehicle'),
    ('NWPU-COCO-truck',    'truck',    'vehicle'),
    ('NWPU-COCO-bus',      'bus',      'vehicle'),
]

TEST_TARGETS = [
    ('COCO-person',      'person',      'person'),
    ('COCO-bicycle',     'bicycle',     'bicycle'),
    ('COCO-car',         'car',         'car'),
    ('COCO-motorcycle',  'motorcycle',  'motorbike'),
    ('COCO-airplane',    'airplane',    'aeroplane'),
    ('COCO-bus',         'bus',         'bus'),
    ('COCO-train',       'train',       'train'),
    ('COCO-truck',       'truck',       'truck'),
    ('COCO-boat',        'boat',        'boat'),
    ('COCO-bird',        'bird',        'bird'),
    ('COCO-cat',         'cat',         'cat'),
    ('COCO-dog',         'dog',         'dog'),
    ('COCO-horse',       'horse',       'horse'),
    ('COCO-sheep',       'sheep',       'sheep'),
    ('COCO-cow',         'cow',         'cow'),
    ('COCO-bottle',      'bottle',      'bottle'),
    ('COCO-chair',       'chair',       'chair'),
    ('COCO-couch',       'couch',       'sofa'),
    ('COCO-potted plant','potted plant','pottedplant'),
    ('COCO-tv',          'tv',          'tvmonitor'),
]


# ── 基础工具 ──
def load_coco_all(cls_name):
    return U.load_coco(cls_name, n=100000)

def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()

def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops

def v2_cached_dets(images, prompt, name):
    p = DATA / f'tge_v2_dets_COCO-{name}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets

def collect_features(images, gts, dets):
    per_img, crops_all = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                        'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64)})
    if crops_all:
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    return per_img, F

def fused_arrays(per_img, F, W):
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, F[p['crop_idx']] @ W)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)
    return per_img

def dataset_ap(per_img, key):
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    return U.compute_ap50_img(preds, gts)


class ClassData:
    def __init__(self, name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50, source='dior'):
        self.name = name; self.coco_cls = coco_cls; self.src_cls = src_cls
        self.per_img = per_img; self.F = F; self.P_src = P_src
        self.P_neg = P_neg; self.Delta = Delta; self.base50 = base50; self.source = source

    def evaluate(self, beta, lam):
        P_hat = self.P_src + beta * self.Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - lam * self.P_neg.cpu().numpy())
        fused_arrays(self.per_img, self.F, W)
        return dataset_ap(self.per_img, 'fused') - self.base50


def load_nwpu_class(nwpu_class, n_max=50):
    class_id = None
    for cid, cname in NWPU_ID2NAME.items():
        if cname == nwpu_class:
            class_id = cid; break
    if class_id is None:
        return [], []
    images, gts = [], []
    for ann_file in sorted(glob.glob(os.path.join(NWPU_ANN_DIR, '*.txt'))):
        if len(images) >= n_max:
            break
        try:
            boxes = []
            with open(ann_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    m = re.match(r'\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\),\s*(\d+)', line)
                    if m:
                        x1, y1, x2, y2, cid = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
                        if cid == class_id:
                            boxes.append([x1, y1, x2, y2])
            if boxes:
                img_name = os.path.splitext(os.path.basename(ann_file))[0] + '.jpg'
                img_file = os.path.join(NWPU_IMG_DIR, img_name)
                if os.path.exists(img_file):
                    img = Image.open(img_file).convert('RGB')
                    images.append(img); gts.append(boxes)
        except Exception:
            continue
    return images, gts

def load_voc_class(voc_class, n_max=50):
    images, gts = [], []
    for ann_file in sorted(glob.glob(os.path.join(VOC_ANN_DIR, '*.xml'))):
        if len(images) >= n_max: break
        try:
            tree = ET.parse(ann_file); root = tree.getroot()
            boxes = []
            for obj in root.findall('object'):
                if obj.find('name').text == voc_class:
                    b = obj.find('bndbox')
                    boxes.append([float(b.find('xmin').text), float(b.find('ymin').text),
                                  float(b.find('xmax').text), float(b.find('ymax').text)])
            if boxes:
                img_file = os.path.join(VOC_IMG_DIR, root.find('filename').text)
                if os.path.exists(img_file):
                    images.append(Image.open(img_file).convert('RGB')); gts.append(boxes)
        except Exception:
            continue
    return images, gts

def build_protos(source):
    protos = {}
    if source == 'dior':
        for cls in U.DIOR_ID2NAME.values():
            images, gts = U.load_dior(cls)
            p = clip_proto(gt_crops(images, gts))
            if p is not None: protos[cls] = p
    elif source == 'nwpu':
        for cls in ['airplane', 'ship', 'vehicle']:
            images, gts = load_nwpu_class(cls, n_max=50)
            p = clip_proto(gt_crops(images, gts))
            if p is not None: protos[cls] = p
    elif source == 'voc':
        for cls in VOC_TO_COCO.keys():
            images, gts = load_voc_class(cls, n_max=50)
            p = clip_proto(gt_crops(images, gts))
            if p is not None: protos[cls] = p
    return protos

def load_class_data(name, coco_cls, src_cls, protos, source='dior', neg_protos=None):
    images, gts = load_coco_all(coco_cls)
    if len(images) < K_CALIB + 10:
        return None
    dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
    per_img, F = collect_features(images, gts, dets)
    if src_cls not in protos:
        return None
    P_src = protos[src_cls].to(U.DEVICE)
    if neg_protos is None:
        neg_protos = protos
    P_neg = T.get_neg_proto(neg_protos, src_cls)
    if source in ('dior', 'nwpu'):
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
    else:
        s_xs = T.text_emb(['a photo of {}'.format(src_cls)]).cpu()[0]
    d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
    Delta = d_y - s_xs
    base50 = dataset_ap(per_img, 'gd')
    return ClassData(name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50, source)

def get_risk(coco_cls):
    v, m, s, i = get_offline_prior(coco_cls)
    prior = PhysicsPrior(category=coco_cls, V=v, M=m, S=s, I=i, source='offline', confidence=0.6)
    return prior.risk_score()


def train_mapper(mapper_type, risks, betas, lambdas):
    if mapper_type == 'linear':
        m = LinearMapper()
    elif mapper_type == 'polynomial':
        m = PolynomialMapper(degree=2)
    elif mapper_type == 'mlp':
        m = MLPMapper(hidden_dim=8)
    m.fit(np.array(risks), np.array(betas), np.array(lambdas))
    return m


def cosine_sim(a, b):
    """计算两个向量的余弦相似度"""
    a_np = a.cpu().numpy() if hasattr(a, 'cpu') else np.array(a)
    b_np = b.cpu().numpy() if hasattr(b, 'cpu') else np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-8))


def compute_domain_similarity(protos_source, protos_target):
    """计算域级相似度：源域所有类别原型均值 vs 目标域所有类别原型均值"""
    src_vecs = torch.stack([p for p in protos_source.values()])
    tgt_vecs = torch.stack([p for p in protos_target.values()])
    src_mean = src_vecs.mean(dim=0)
    tgt_mean = tgt_vecs.mean(dim=0)
    src_mean = src_mean / src_mean.norm()
    tgt_mean = tgt_mean / tgt_mean.norm()
    return cosine_sim(src_mean, tgt_mean)


def compute_class_similarity(protos_source, src_cls, protos_target, tgt_cls):
    """计算类别级相似度：源域某类别原型 vs 目标域某类别原型"""
    if src_cls not in protos_source or tgt_cls not in protos_target:
        return None
    return cosine_sim(protos_source[src_cls], protos_target[tgt_cls])


def evaluate_strategy(test_data, test_categories, strategy, mappers_dict, mapper_type='linear',
                      tau=1.0, protos_dior=None, protos_nwpu=None, protos_voc=None):
    """
    评估一种测试策略
    strategy: 'merged' | 'dior_only' | 'nwpu_only' | 'avg' | 'sim_weighted' | 'sim_weighted_cls'
    """
    import torch
    gains = []
    params_log = {}

    # 预计算域级相似度（所有策略都需要，作为fallback）
    sim_dior_domain = None
    sim_nwpu_domain = None
    w_dior_domain = 0.5
    w_nwpu_domain = 0.5
    if protos_dior and protos_voc and protos_nwpu:
        sim_dior_domain = compute_domain_similarity(protos_dior, protos_voc)
        sim_nwpu_domain = compute_domain_similarity(protos_nwpu, protos_voc)
        e_d = np.exp(tau * sim_dior_domain)
        e_n = np.exp(tau * sim_nwpu_domain)
        w_dior_domain = e_d / (e_d + e_n)
        w_nwpu_domain = e_n / (e_d + e_n)
        if strategy in ('sim_weighted', 'sim_weighted_cls'):
            print(f'    [域级相似度] DIOR-VOC={sim_dior_domain:.4f}, NWPU-VOC={sim_nwpu_domain:.4f}, '
                  f'w_dior={w_dior_domain:.4f}, w_nwpu={w_nwpu_domain:.4f}')

    for name in test_categories:
        cd = test_data[name]
        risk = get_risk(cd.coco_cls)

        if strategy == 'merged':
            beta, lam = mappers_dict['merged'].predict(risk)
        elif strategy == 'dior_only':
            beta, lam = mappers_dict['dior'].predict(risk)
        elif strategy == 'nwpu_only':
            beta, lam = mappers_dict['nwpu'].predict(risk)
        elif strategy == 'avg':
            b1, l1 = mappers_dict['dior'].predict(risk)
            b2, l2 = mappers_dict['nwpu'].predict(risk)
            beta, lam = (b1 + b2) / 2, (l1 + l2) / 2
        elif strategy == 'sim_weighted':
            # 域级相似度加权
            b1, l1 = mappers_dict['dior'].predict(risk)
            b2, l2 = mappers_dict['nwpu'].predict(risk)
            beta = w_dior_domain * b1 + w_nwpu_domain * b2
            lam = w_dior_domain * l1 + w_nwpu_domain * l2
        elif strategy == 'sim_weighted_cls':
            # 类别级相似度加权
            voc_cls = cd.src_cls  # TEST_TARGETS中的src_cls是VOC类别名
            src_cls_dior = VOC_TO_SRC_CLS.get(voc_cls, None)
            if src_cls_dior is not None:
                # 有对应源域类别，用类别级相似度
                sim_d = compute_class_similarity(protos_dior, src_cls_dior, protos_voc, voc_cls)
                sim_n = compute_class_similarity(protos_nwpu, src_cls_dior, protos_voc, voc_cls)
                if sim_d is not None and sim_n is not None:
                    e_d = np.exp(tau * sim_d)
                    e_n = np.exp(tau * sim_n)
                    w_d = e_d / (e_d + e_n)
                    w_n = e_n / (e_d + e_n)
                else:
                    w_d, w_n = w_dior_domain, w_nwpu_domain
            else:
                # 无对应源域类别，用域级相似度
                w_d, w_n = w_dior_domain, w_nwpu_domain
            b1, l1 = mappers_dict['dior'].predict(risk)
            b2, l2 = mappers_dict['nwpu'].predict(risk)
            beta = w_d * b1 + w_n * b2
            lam = w_d * l1 + w_n * l2

        beta = float(np.clip(beta, 0, 6))
        lam = float(np.clip(lam, 0, 2))
        gain = cd.evaluate(beta, lam)
        gains.append(gain)
        params_log[name] = {'risk': float(risk), 'beta': float(beta), 'lambda': float(lam), 'gain': float(gain)}

    return {
        'strategy': strategy,
        'mapper_type': mapper_type,
        'tau': tau if 'sim' in strategy else None,
        'mean_gain': float(np.mean(gains)),
        'n_positive': sum(1 for g in gains if g > 0),
        'n_total': len(gains),
        'gains': {n: float(g) for n, g in zip(test_categories, gains)},
        'params': params_log,
    }


def main():
    import torch
    t0 = time.time()
    print('=' * 100)
    print('A-CSPC AutoTuner Sim-Weighted: 基于域相似度的源域映射加权实验')
    print('=' * 100)

    # ── 1. 构建源域原型 ──
    print('\n[1/6] 构建源域原型 ...')
    protos_dior = build_protos('dior')
    protos_nwpu = build_protos('nwpu')
    protos_voc = build_protos('voc')
    protos_for_neg = dict(protos_dior); protos_for_neg.update(protos_nwpu)
    print(f'  DIOR: {len(protos_dior)}类, NWPU: {len(protos_nwpu)}类, VOC: {len(protos_voc)}类')

    # 预计算域级相似度
    sim_dior_voc = compute_domain_similarity(protos_dior, protos_voc)
    sim_nwpu_voc = compute_domain_similarity(protos_nwpu, protos_voc)
    print(f'  域级相似度: DIOR-VOC={sim_dior_voc:.4f}, NWPU-VOC={sim_nwpu_voc:.4f}')

    # ── 2. 加载开发集数据 ──
    print('\n[2/6] 加载开发集数据 ...')
    dev_data = {}
    for name, coco_cls, src_cls in DEV_TARGETS_DIOR:
        cd = load_class_data(name, coco_cls, src_cls, protos_dior, 'dior')
        if cd: dev_data[name] = cd
    for name, coco_cls, src_cls in DEV_TARGETS_NWPU:
        cd = load_class_data(name, coco_cls, src_cls, protos_nwpu, 'nwpu', neg_protos=protos_for_neg)
        if cd: dev_data[name] = cd
    print(f'  开发集: {len(dev_data)}类')

    # ── 3. 开发集网格搜索 ──
    print('\n[3/6] 开发集网格搜索 ...')
    gs = GridSearchTuner(beta_range=(0.0, 4.0), lambda_range=(0.0, 2.0), n_beta=17, n_lambda=9)
    optimals = []
    for name, cd in dev_data.items():
        risk = get_risk(cd.coco_cls)
        print(f'  [GS] {name} (risk={risk:.3f}) ...')
        opt = gs.search(name, risk, lambda b, l: cd.evaluate(b, l))
        optimals.append(opt)
        print(f'    → β*={opt.beta_opt:.2f}, λ*={opt.lambda_opt:.2f}, gain*={opt.gain_opt:+.4f}')

    dior_opts = [o for o in optimals if o.category.startswith('DIOR')]
    nwpu_opts = [o for o in optimals if o.category.startswith('NWPU')]

    # ── 4. 按源域分组训练映射函数 ──
    print('\n[4/6] 按源域分组训练映射函数 ...')
    mapper_types = ['linear', 'polynomial']
    all_results = {}

    for mt in mapper_types:
        print(f'\n  --- 映射类型: {mt} ---')
        m_merged = train_mapper(mt, [o.risk for o in optimals], [o.beta_opt for o in optimals], [o.lambda_opt for o in optimals])
        m_dior = train_mapper(mt, [o.risk for o in dior_opts], [o.beta_opt for o in dior_opts], [o.lambda_opt for o in dior_opts])
        m_nwpu = train_mapper(mt, [o.risk for o in nwpu_opts], [o.beta_opt for o in nwpu_opts], [o.lambda_opt for o in nwpu_opts])

        if mt == 'linear':
            print(f'    merged: β={m_merged.a_beta:.3f}R+{m_merged.b_beta:.3f}')
            print(f'    dior:   β={m_dior.a_beta:.3f}R+{m_dior.b_beta:.3f}')
            print(f'    nwpu:   β={m_nwpu.a_beta:.3f}R+{m_nwpu.b_beta:.3f}')

        mappers_dict = {'merged': m_merged, 'dior': m_dior, 'nwpu': m_nwpu}

        # ── 5. 加载测试集数据 ──
        print('\n[5/6] 加载测试集数据 ...')
        test_data = {}
        for name, coco_cls, src_cls in TEST_TARGETS:
            cd = load_class_data(name, coco_cls, src_cls, protos_voc, 'voc')
            if cd: test_data[name] = cd
        test_categories = list(test_data.keys())
        print(f'  测试集: {len(test_data)}类')

        # ── 6. 测试集评估 ──
        print('\n[6/6] 测试集评估 ...')

        # 基线策略
        for strat in ['merged', 'dior_only', 'nwpu_only', 'avg']:
            print(f'\n  评估策略: {strat} ({mt})')
            res = evaluate_strategy(test_data, test_categories, strat, mappers_dict, mt,
                                    protos_dior=protos_dior, protos_nwpu=protos_nwpu, protos_voc=protos_voc)
            all_results[f'{mt}_{strat}'] = res
            print(f'    MEAN: {res["mean_gain"]:+.4f}  pos={res["n_positive"]}/{res["n_total"]}')

        # 域级相似度加权（扫描τ）
        taus = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
        for tau in taus:
            strat = 'sim_weighted'
            print(f'\n  评估策略: {strat} (τ={tau}, {mt})')
            res = evaluate_strategy(test_data, test_categories, strat, mappers_dict, mt, tau=tau,
                                    protos_dior=protos_dior, protos_nwpu=protos_nwpu, protos_voc=protos_voc)
            all_results[f'{mt}_{strat}_tau{tau}'] = res
            print(f'    MEAN: {res["mean_gain"]:+.4f}  pos={res["n_positive"]}/{res["n_total"]}')

        # 类别级相似度加权（扫描τ）
        for tau in taus:
            strat = 'sim_weighted_cls'
            print(f'\n  评估策略: {strat} (τ={tau}, {mt})')
            res = evaluate_strategy(test_data, test_categories, strat, mappers_dict, mt, tau=tau,
                                    protos_dior=protos_dior, protos_nwpu=protos_nwpu, protos_voc=protos_voc)
            all_results[f'{mt}_{strat}_tau{tau}'] = res
            print(f'    MEAN: {res["mean_gain"]:+.4f}  pos={res["n_positive"]}/{res["n_total"]}')

    # ── 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比（VOC→COCO 19类测试集）')
    print('=' * 100)
    print(f'{"映射类型":<12} {"策略":<28} {"均值增益":>10} {"正增益":>8} {"vs dior_only":>14}')
    print('-' * 75)
    for mt in mapper_types:
        dior_gain = all_results[f'{mt}_dior_only']['mean_gain']
        for key in sorted(all_results.keys()):
            if key.startswith(f'{mt}_'):
                r = all_results[key]
                strat_name = key.replace(f'{mt}_', '')
                vs_dior = r['mean_gain'] - dior_gain
                marker = ' ←超越DIOR!' if (vs_dior > 0 and 'dior_only' not in key) else ''
                print(f'{mt:<12} {strat_name:<28} {r["mean_gain"]:>+10.4f} {r["n_positive"]:>4d}/{r["n_total"]:<3d} {vs_dior:>+14.4f}{marker}')
        print('-' * 75)

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': 'DIOR 6类 + NWPU 5类 = 11类',
            'test_set': 'VOC→COCO 19类',
            'domain_similarity': {'dior_voc': sim_dior_voc, 'nwpu_voc': sim_nwpu_voc},
            'taus': taus,
        },
        'results': all_results,
    }

    out_path = DATA / 'tge_v2_auto_tune_sim_weighted_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
