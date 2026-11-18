# -*- coding: utf-8 -*-
r"""tge_v2_auto_tune_gate_v4.py — 门控v4：温度τ扫描 + 两级门控组合

在v3对比实验基础上，进一步优化：
  1. 原型相似度温度τ扫描：τ ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}
  2. 两级门控：
     - 第一级：Δnorm门控（Δnorm<τ_delta时β=0，只用λ）
     - 第二级：原型相似度选择DIOR/NWPU专家（Δnorm>=τ_delta时）
  3. 两级门控 + 逻辑回归组合
  4. 原型相似度 + Δnorm特征的逻辑回归（5参数）

运行: python tge_v2_auto_tune_gate_v4.py
输出: 实验数据/tge_v2_auto_tune_gate_v4_results.json
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
import torch.nn as nn
import torch.optim as optim
import unified_eval as U
import tge_probe as T

from acspc.auto_tuner import GridSearchTuner, LinearMapper
from acspc.coco_categories import get_offline_prior
from acspc.appg import PhysicsPrior

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

K_CALIB = 10
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
PRE_FILTER = 0.03
DELTA_GATE_THRESHOLD = 0.01

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

DEV_DIOR = [
    ('DIOR-COCO-airplane',   'airplane',   'airplane'),
    ('DIOR-COCO-boat',       'boat',       'ship'),
    ('DIOR-COCO-car',        'car',        'vehicle'),
    ('DIOR-COCO-truck',      'truck',      'vehicle'),
    ('DIOR-COCO-bus',        'bus',        'vehicle'),
    ('DIOR-COCO-motorcycle', 'motorcycle', 'vehicle'),
]

DEV_NWPU = [
    ('NWPU-COCO-airplane', 'airplane', 'airplane'),
    ('NWPU-COCO-boat',     'boat',     'ship'),
    ('NWPU-COCO-car',      'car',      'vehicle'),
    ('NWPU-COCO-truck',    'truck',    'vehicle'),
    ('NWPU-COCO-bus',      'bus',      'vehicle'),
]

DEV_VOC = [
    ('VOC-COCO-person', 'person', 'person'),
    ('VOC-COCO-bird',   'bird',   'bird'),
    ('VOC-COCO-cat',    'cat',    'cat'),
    ('VOC-COCO-dog',    'dog',    'dog'),
    ('VOC-COCO-horse',  'horse',  'horse'),
    ('VOC-COCO-sheep',  'sheep',  'sheep'),
    ('VOC-COCO-cow',    'cow',    'cow'),
]

TEST_TARGETS = [
    ('COCO-airplane',    'airplane',    'aeroplane'),
    ('COCO-boat',        'boat',        'boat'),
    ('COCO-car',         'car',         'car'),
    ('COCO-motorcycle',  'motorcycle',  'motorbike'),
    ('COCO-bus',         'bus',         'bus'),
    ('COCO-train',       'train',       'train'),
    ('COCO-bicycle',     'bicycle',     'bicycle'),
    ('COCO-bottle',      'bottle',      'bottle'),
    ('COCO-chair',       'chair',       'chair'),
    ('COCO-couch',       'couch',       'sofa'),
    ('COCO-potted plant','potted plant','pottedplant'),
    ('COCO-tv',          'tv',          'tvmonitor'),
]

TAU_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


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
    p = DATA / f'tge_v2_dets_COCO-COCO-{name}.pkl'
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
    def __init__(self, name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50, source):
        self.name = name; self.coco_cls = coco_cls; self.src_cls = src_cls
        self.per_img = per_img; self.F = F; self.P_src = P_src
        self.P_neg = P_neg; self.Delta = Delta; self.base50 = base50; self.source = source
        self.delta_norm = float(Delta.norm())
        if len(F) > 0:
            all_gd = np.concatenate([p['gd'] for p in per_img]) if per_img else np.array([])
            all_ci = np.concatenate([p['crop_idx'] for p in per_img]) if per_img else np.array([])
            if len(all_gd) > 0:
                weights = all_gd / all_gd.sum()
                self.target_proto = (F[all_ci] * weights[:, None]).sum(axis=0)
                self.target_proto = self.target_proto / (np.linalg.norm(self.target_proto) + 1e-8)
            else:
                self.target_proto = F.mean(axis=0)
                self.target_proto = self.target_proto / (np.linalg.norm(self.target_proto) + 1e-8)
        else:
            self.target_proto = np.zeros(512)

    def evaluate(self, beta, lam, use_delta_gate=True):
        if use_delta_gate and self.delta_norm < DELTA_GATE_THRESHOLD:
            beta = 0.0
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
        if len(images) >= n_max: break
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

def load_class_data(name, coco_cls, src_cls, protos, source, neg_protos=None):
    images, gts = load_coco_all(coco_cls)
    if len(images) < K_CALIB + 10:
        return None
    dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), coco_cls)
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
    m = LinearMapper()
    m.fit(np.array(risks), np.array(betas), np.array(lambdas))
    return m


# ── 原型相似度门控（可调节温度τ）──
def prototype_similarity_gate(cd, protos_dior, protos_nwpu, src_cls_dior, src_cls_nwpu, tau=0.1):
    target_proto = cd.target_proto
    sim_dior = float(np.dot(target_proto, protos_dior[src_cls_dior].cpu().numpy())) if src_cls_dior in protos_dior else 0.0
    sim_nwpu = float(np.dot(target_proto, protos_nwpu[src_cls_nwpu].cpu().numpy())) if src_cls_nwpu in protos_nwpu else 0.0
    w_dior = np.exp(sim_dior / tau)
    w_nwpu = np.exp(sim_nwpu / tau)
    total = w_dior + w_nwpu
    return w_dior / total, w_nwpu / total, sim_dior, sim_nwpu


# ── 逻辑回归门控（3参数）──
def train_logistic_gate(X_train, y_train, n_epochs=1000, lr=0.01):
    a = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    c = torch.tensor(0.0, requires_grad=True)
    optimizer = optim.Adam([a, b, c], lr=lr)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logit = a * X[:, 0] + b * X[:, 1] + c
        w = torch.sigmoid(logit)
        loss = -torch.mean(y * torch.log(w + 1e-8) + (1 - y) * torch.log(1 - w + 1e-8))
        loss.backward()
        optimizer.step()
    return float(a.item()), float(b.item()), float(c.item())


def logistic_gate_predict(R, delta_norm, a, b, c):
    w_dior = 1.0 / (1.0 + np.exp(-(a * R + b * delta_norm + c)))
    return w_dior, 1.0 - w_dior


def main():
    t0 = time.time()
    print('=' * 100)
    print('门控v4：温度τ扫描 + 两级门控（Δnorm + 原型相似度）')
    print('=' * 100)

    # ── 1. 构建原型 ──
    print('\n[1/5] 构建源域原型 ...')
    protos_dior = build_protos('dior')
    protos_nwpu = build_protos('nwpu')
    protos_voc = build_protos('voc')
    protos_for_neg = dict(protos_dior); protos_for_neg.update(protos_nwpu); protos_for_neg.update(protos_voc)
    print(f'  DIOR: {len(protos_dior)}类, NWPU: {len(protos_nwpu)}类, VOC: {len(protos_voc)}类')

    # ── 2. 加载数据 ──
    print('\n[2/5] 加载开发集和测试集数据 ...')
    dev_data = {}
    for name, coco_cls, src_cls in DEV_DIOR + DEV_NWPU + DEV_VOC:
        source = 'dior' if name.startswith('DIOR') else ('nwpu' if name.startswith('NWPU') else 'voc')
        protos = protos_dior if source == 'dior' else (protos_nwpu if source == 'nwpu' else protos_voc)
        cd = load_class_data(name, coco_cls, src_cls, protos, source, neg_protos=protos_for_neg)
        if cd: dev_data[name] = cd
    print(f'  开发集: {len(dev_data)}类')

    test_data = {}
    for name, coco_cls, src_cls in TEST_TARGETS:
        cd = load_class_data(name, coco_cls, src_cls, protos_voc, 'voc')
        if cd: test_data[name] = cd
    test_categories = list(test_data.keys())
    print(f'  测试集: {len(test_data)}类')

    # ── 3. 网格搜索 + 训练专家 ──
    print('\n[3/5] 开发集网格搜索 + 训练专家映射 ...')
    gs = GridSearchTuner(beta_range=(0.0, 4.0), lambda_range=(0.0, 2.0), n_beta=17, n_lambda=9)
    optimals = []
    for name, cd in dev_data.items():
        risk = get_risk(cd.coco_cls)
        opt = gs.search(name, risk, lambda b, l: cd.evaluate(b, l, use_delta_gate=True))
        optimals.append(opt)

    dior_opts = [o for o in optimals if o.category.startswith('DIOR')]
    nwpu_opts = [o for o in optimals if o.category.startswith('NWPU')]
    voc_opts = [o for o in optimals if o.category.startswith('VOC')]

    m_dior = train_mapper('linear', [o.risk for o in dior_opts], [o.beta_opt for o in dior_opts], [o.lambda_opt for o in dior_opts])
    m_nwpu = train_mapper('linear', [o.risk for o in nwpu_opts], [o.beta_opt for o in nwpu_opts], [o.lambda_opt for o in nwpu_opts])
    m_voc = train_mapper('linear', [o.risk for o in voc_opts], [o.beta_opt for o in voc_opts], [o.lambda_opt for o in voc_opts])
    print(f'  专家训练完成')

    # 训练逻辑回归门控
    X_train_logistic = []
    y_train_logistic = []
    for name, cd in dev_data.items():
        risk = get_risk(cd.coco_cls)
        b_d, l_d = m_dior.predict(risk)
        b_n, l_n = m_nwpu.predict(risk)
        gain_d = cd.evaluate(b_d, l_d, use_delta_gate=True)
        gain_n = cd.evaluate(b_n, l_n, use_delta_gate=True)
        X_train_logistic.append([risk, cd.delta_norm])
        y_train_logistic.append(1 if gain_d >= gain_n else 0)
    a_lr, b_lr, c_lr = train_logistic_gate(np.array(X_train_logistic), np.array(y_train_logistic))
    print(f'  逻辑回归门控: a={a_lr:.4f}, b={b_lr:.4f}, c={c_lr:.4f}')

    # ── 4. 测试集评估 ──
    print('\n[4/5] 测试集评估 ...')

    test_to_dior = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
                     'motorcycle': 'vehicle', 'bus': 'vehicle', 'train': None,
                     'bicycle': None, 'bottle': None, 'chair': None, 'couch': None,
                     'potted plant': None, 'tv': None}
    test_to_nwpu = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
                    'motorcycle': 'vehicle', 'bus': 'vehicle', 'train': None,
                    'bicycle': None, 'bottle': None, 'chair': None, 'couch': None,
                    'potted plant': None, 'tv': None}

    all_results = {}

    def evaluate_with_weights(cd, beta_d, lambda_d, beta_n, lambda_n, beta_v, lambda_v,
                               w_dior, w_nwpu, w_voc, use_delta_gate=True):
        beta = w_dior * beta_d + w_nwpu * beta_n + w_voc * beta_v
        lam = w_dior * lambda_d + w_nwpu * lambda_n + w_voc * lambda_v
        beta = float(np.clip(beta, 0, 6))
        lam = float(np.clip(lam, 0, 2))
        return cd.evaluate(beta, lam, use_delta_gate=use_delta_gate), beta, lam

    def evaluate_strategy(strategy, tau=0.1, use_delta_gate=True):
        gains = []
        params_log = {}
        for name in test_categories:
            cd = test_data[name]
            risk = get_risk(cd.coco_cls)
            b_d, l_d = m_dior.predict(risk)
            b_n, l_n = m_nwpu.predict(risk)
            b_v, l_v = m_voc.predict(risk)

            if strategy == 'dior_only':
                w = [1.0, 0.0, 0.0]
            elif strategy == 'voc_only':
                w = [0.0, 0.0, 1.0]
            elif strategy == 'avg':
                w = [1/3, 1/3, 1/3]
            elif strategy == 'proto_sim':
                src_d = test_to_dior.get(cd.coco_cls)
                src_n = test_to_nwpu.get(cd.coco_cls)
                if src_d and src_n:
                    w_d, w_n, sim_d, sim_n = prototype_similarity_gate(cd, protos_dior, protos_nwpu, src_d, src_n, tau=tau)
                    w = [w_d, w_n, 0.0]
                else:
                    w = [0.5, 0.5, 0.0]
            elif strategy == 'two_level_proto':
                # 两级门控：Δnorm门控 + 原型相似度
                if cd.delta_norm < DELTA_GATE_THRESHOLD:
                    # Δnorm小，直接β=0，用VOC专家的λ
                    w = [0.0, 0.0, 1.0]
                else:
                    src_d = test_to_dior.get(cd.coco_cls)
                    src_n = test_to_nwpu.get(cd.coco_cls)
                    if src_d and src_n:
                        w_d, w_n, _, _ = prototype_similarity_gate(cd, protos_dior, protos_nwpu, src_d, src_n, tau=tau)
                        w = [w_d, w_n, 0.0]
                    else:
                        w = [0.5, 0.5, 0.0]
            elif strategy == 'two_level_logistic':
                # 两级门控：Δnorm门控 + 逻辑回归
                if cd.delta_norm < DELTA_GATE_THRESHOLD:
                    w = [0.0, 0.0, 1.0]
                else:
                    w_d, w_n = logistic_gate_predict(risk, cd.delta_norm, a_lr, b_lr, c_lr)
                    w = [w_d, w_n, 0.0]
            elif strategy == 'logistic':
                w_d, w_n = logistic_gate_predict(risk, cd.delta_norm, a_lr, b_lr, c_lr)
                w = [w_d, w_n, 0.0]
            elif strategy == 'oracle':
                gain_d = cd.evaluate(b_d, l_d, use_delta_gate=use_delta_gate)
                gain_n = cd.evaluate(b_n, l_n, use_delta_gate=use_delta_gate)
                gain_v = cd.evaluate(b_v, l_v, use_delta_gate=use_delta_gate)
                gains_all = [gain_d, gain_n, gain_v]
                best_idx = int(np.argmax(gains_all))
                w = [0.0, 0.0, 0.0]
                w[best_idx] = 1.0

            gain, beta_final, lam_final = evaluate_with_weights(
                cd, b_d, l_d, b_n, l_n, b_v, l_v, w[0], w[1], w[2], use_delta_gate=use_delta_gate)
            gains.append(gain)
            params_log[name] = {
                'risk': float(risk), 'delta_norm': float(cd.delta_norm),
                'beta': float(beta_final), 'lambda': float(lam_final),
                'weights': {'DIOR': float(w[0]), 'NWPU': float(w[1]), 'VOC': float(w[2])},
                'gain': float(gain),
            }

        return {
            'strategy': strategy, 'tau': tau, 'use_delta_gate': use_delta_gate,
            'mean_gain': float(np.mean(gains)),
            'n_positive': sum(1 for g in gains if g > 0),
            'n_total': len(gains),
            'gains': {n: float(g) for n, g in zip(test_categories, gains)},
            'params': params_log,
        }

    # 基线
    for strat in ['dior_only', 'voc_only', 'avg', 'oracle']:
        print(f'  评估: {strat} ...')
        all_results[strat] = evaluate_strategy(strat)
        print(f'    MEAN: {all_results[strat]["mean_gain"]:+.4f}')

    # 温度τ扫描（原型相似度）
    print(f'\n  温度τ扫描（原型相似度）:')
    tau_results = {}
    for tau in TAU_GRID:
        res = evaluate_strategy('proto_sim', tau=tau)
        tau_results[str(tau)] = res
        all_results[f'proto_sim_tau{tau}'] = res
        print(f'    τ={tau:>5}: mean={res["mean_gain"]:+.4f}, pos={res["n_positive"]}/{res["n_total"]}')

    # 两级门控（用最佳τ）
    best_tau = max(TAU_GRID, key=lambda t: tau_results[str(t)]['mean_gain'])
    print(f'\n  最佳τ={best_tau}，评估两级门控:')

    for strat in ['two_level_proto', 'two_level_logistic', 'logistic']:
        print(f'  评估: {strat} (τ={best_tau}) ...')
        res = evaluate_strategy(strat, tau=best_tau)
        all_results[strat] = res
        print(f'    MEAN: {res["mean_gain"]:+.4f}, pos={res["n_positive"]}/{res["n_total"]}')

    # ── 5. 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比（VOC→COCO 12类测试集）')
    print('=' * 100)
    single_best_gain = all_results['dior_only']['mean_gain']
    oracle_gain = all_results['oracle']['mean_gain']
    print(f'最佳单专家(dior_only): +{single_best_gain:.4f}')
    print(f'Oracle上界: +{oracle_gain:.4f}')
    print()
    print(f'{"策略":<25} {"参数量":>6} {"均值增益":>10} {"正增益":>8} {"vs最佳单专家":>14} {"达到oracle":>12}')
    print('-' * 80)

    summary_strategies = [
        ('dior_only', 0),
        ('avg', 0),
        (f'proto_sim_tau{best_tau}', 0),
        ('two_level_proto', 0),
        ('logistic', 3),
        ('two_level_logistic', 3),
        ('oracle', 0),
    ]
    for strat, n_params in summary_strategies:
        r = all_results[strat]
        vs_best = r['mean_gain'] - single_best_gain
        pct_oracle = r['mean_gain'] / oracle_gain * 100 if oracle_gain > 0 else 0
        marker = ' ←超越最佳单专家!' if (vs_best > 0 and strat != 'oracle') else ''
        print(f'{strat:<25} {n_params:>6} {r["mean_gain"]:>+10.4f} '
              f'{r["n_positive"]:>4d}/{r["n_total"]:<3d} {vs_best:>+14.4f} {pct_oracle:>11.1f}%{marker}')

    # τ扫描详细
    print(f'\n温度τ扫描详细:')
    print(f'{"τ":>8} {"均值增益":>10} {"正增益":>8} {"vs最佳单专家":>14}')
    print('-' * 45)
    for tau in TAU_GRID:
        r = tau_results[str(tau)]
        vs_best = r['mean_gain'] - single_best_gain
        marker = ' ←最佳' if tau == best_tau else ''
        print(f'{tau:>8} {r["mean_gain"]:>+10.4f} {r["n_positive"]:>4d}/{r["n_total"]:<3d} {vs_best:>+14.4f}{marker}')

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': f'{len(dev_data)}类',
            'test_set': f'VOC→COCO {len(test_data)}类',
            'tau_grid': TAU_GRID,
            'best_tau': best_tau,
            'delta_gate_threshold': DELTA_GATE_THRESHOLD,
        },
        'gate_params': {
            'logistic': {'a': a_lr, 'b': b_lr, 'c': c_lr},
        },
        'tau_sweep': {str(t): {'mean_gain': tau_results[str(t)]['mean_gain'],
                                 'n_positive': tau_results[str(t)]['n_positive']} for t in TAU_GRID},
        'results': all_results,
    }

    out_path = DATA / 'tge_v2_auto_tune_gate_v4_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
