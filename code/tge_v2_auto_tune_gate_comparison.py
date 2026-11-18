# -*- coding: utf-8 -*-
r"""tge_v2_auto_tune_gate_comparison.py — 门控函数对比实验

对比多种门控设计：
  基线:
    - dior_only, nwpu_only, voc_only  （单专家）
    - merged, avg  （简单融合）
    - oracle_gate  （上界）

  方案A: 基于原型相似度的门控（0参数）
    - 计算目标域原型与DIOR/NWPU源域原型的余弦相似度
    - 用相似度归一化作为权重
    - Δnorm<τ时β=0，只融合λ

  方案B: 逻辑回归门控（3参数）
    - w_dior = sigmoid(a·R + b·Δnorm + c)
    - 用开发集训练（18样本，3参数，不会过拟合）
    - Δnorm<τ时β=0

  方案C: sigmoid独立权重门控（4参数）
    - w_dior = sigmoid(a·R + b)
    - w_nwpu = sigmoid(c·R + d)
    - 允许"都不信任"（权重和<1，剩余用默认参数）
    - Δnorm<τ时β=0

  方案D: v3神经网络门控（8→16→3 softmax，195参数）
    - 已有结果，直接复现用于对比

所有方案都使用外推方向门控（Δnorm<0.01时强制β=0）。

运行: python tge_v2_auto_tune_gate_comparison.py
输出: 实验数据/tge_v2_auto_tune_gate_comparison_results.json
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
        # 目标域原型：从检测框特征聚合（高置信度框）
        if len(F) > 0:
            # 用置信度加权平均
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


# ── 方案A：原型相似度门控（0参数）──
def prototype_similarity_gate(cd, protos_dior, protos_nwpu, src_cls_dior, src_cls_nwpu):
    """基于目标域原型与源域原型的余弦相似度计算权重"""
    target_proto = cd.target_proto
    # 计算与DIOR/NWPU源域原型的相似度
    if src_cls_dior in protos_dior:
        sim_dior = float(np.dot(target_proto, protos_dior[src_cls_dior].cpu().numpy()))
    else:
        sim_dior = 0.0
    if src_cls_nwpu in protos_nwpu:
        sim_nwpu = float(np.dot(target_proto, protos_nwpu[src_cls_nwpu].cpu().numpy()))
    else:
        sim_nwpu = 0.0
    # 温度缩放后softmax
    tau = 0.1
    w_dior = np.exp(sim_dior / tau)
    w_nwpu = np.exp(sim_nwpu / tau)
    total = w_dior + w_nwpu
    return w_dior / total, w_nwpu / total, sim_dior, sim_nwpu


# ── 方案B：逻辑回归门控（3参数）──
def train_logistic_gate(X_train, y_train, n_epochs=1000, lr=0.01):
    """训练逻辑回归：w_dior = sigmoid(a·R + b·Δnorm + c)"""
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
        # 二元交叉熵（y=1表示DIOR更好，y=0表示NWPU更好）
        loss = -torch.mean(y * torch.log(w + 1e-8) + (1 - y) * torch.log(1 - w + 1e-8))
        loss.backward()
        optimizer.step()
    return float(a.item()), float(b.item()), float(c.item()), float(loss.item())


def logistic_gate_predict(R, delta_norm, a, b, c):
    w_dior = 1.0 / (1.0 + np.exp(-(a * R + b * delta_norm + c)))
    return w_dior, 1.0 - w_dior


# ── 方案C：sigmoid独立权重门控（4参数）──
def train_sigmoid_independent_gate(X_train, gains_dior, gains_nwpu, n_epochs=1000, lr=0.01):
    """训练独立sigmoid：w_dior = sigmoid(a·R + b), w_nwpu = sigmoid(c·R + d)
    允许权重和<1（"都不信任"），剩余权重用默认参数"""
    a = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    c = torch.tensor(0.0, requires_grad=True)
    d = torch.tensor(0.0, requires_grad=True)
    optimizer = optim.Adam([a, b, c, d], lr=lr)
    X = torch.tensor(X_train, dtype=torch.float32)
    gd = torch.tensor(gains_dior, dtype=torch.float32)
    gn = torch.tensor(gains_nwpu, dtype=torch.float32)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        w_d = torch.sigmoid(a * X[:, 0] + b)
        w_n = torch.sigmoid(c * X[:, 0] + d)
        # 目标：让加权增益最大化
        pred_gain = w_d * gd + w_n * gn
        loss = -torch.mean(pred_gain)
        loss.backward()
        optimizer.step()
    return (float(a.item()), float(b.item()), float(c.item()), float(d.item())), float(loss.item())


def sigmoid_independent_predict(R, params):
    a, b, c, d = params
    w_dior = 1.0 / (1.0 + np.exp(-(a * R + b)))
    w_nwpu = 1.0 / (1.0 + np.exp(-(c * R + d)))
    return w_dior, w_nwpu


# ── v3神经网络门控（用于对比）──
class MoEGate3(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def train_nn_gate(X_train, y_train, n_epochs=500, lr=0.01, weight_decay=1e-4, seed=42):
    torch.manual_seed(seed)
    gate = MoEGate3(input_dim=8, hidden_dim=16)
    optimizer = optim.Adam(gate.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)
    gate.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = gate(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    gate.eval()
    return gate


def main():
    t0 = time.time()
    print('=' * 100)
    print('门控函数对比实验：原型相似度(0参) vs 逻辑回归(3参) vs sigmoid独立(4参) vs NN(195参)')
    print('=' * 100)

    # ── 1. 构建源域原型 ──
    print('\n[1/6] 构建源域原型 ...')
    protos_dior = build_protos('dior')
    protos_nwpu = build_protos('nwpu')
    protos_voc = build_protos('voc')
    protos_for_neg = dict(protos_dior); protos_for_neg.update(protos_nwpu); protos_for_neg.update(protos_voc)
    print(f'  DIOR: {len(protos_dior)}类, NWPU: {len(protos_nwpu)}类, VOC: {len(protos_voc)}类')

    # ── 2. 加载数据 ──
    print('\n[2/6] 加载开发集和测试集数据 ...')
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

    # ── 3. 开发集网格搜索 + 训练专家 ──
    print('\n[3/6] 开发集网格搜索 + 训练专家映射 ...')
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
    m_merged = train_mapper('linear', [o.risk for o in optimals], [o.beta_opt for o in optimals], [o.lambda_opt for o in optimals])
    print(f'  DIOR: β={m_dior.a_beta:.3f}R+{m_dior.b_beta:.3f}, NWPU: β={m_nwpu.a_beta:.3f}R+{m_nwpu.b_beta:.3f}, VOC: β={m_voc.a_beta:.3f}R+{m_voc.b_beta:.3f}')

    # ── 4. 训练各门控 ──
    print('\n[4/6] 训练各门控函数 ...')

    # 准备训练数据
    X_train_logistic = []  # [R, Δnorm]
    y_train_logistic = []  # 1=DIOR更好, 0=NWPU更好
    gains_dior_train = []
    gains_nwpu_train = []
    X_train_nn = []  # 8维
    y_train_nn = []  # 3分类

    for name, cd in dev_data.items():
        risk = get_risk(cd.coco_cls)
        b_d, l_d = m_dior.predict(risk)
        b_n, l_n = m_nwpu.predict(risk)
        b_v, l_v = m_voc.predict(risk)
        gain_d = cd.evaluate(b_d, l_d, use_delta_gate=True)
        gain_n = cd.evaluate(b_n, l_n, use_delta_gate=True)
        gain_v = cd.evaluate(b_v, l_v, use_delta_gate=True)

        # 逻辑回归训练数据（只在DIOR和NWPU间选择）
        X_train_logistic.append([risk, cd.delta_norm])
        y_train_logistic.append(1 if gain_d >= gain_n else 0)
        gains_dior_train.append(gain_d)
        gains_nwpu_train.append(gain_n)

        # NN训练数据（3分类）
        feat = [risk, b_d, b_n, b_v, l_d, l_n, l_v, cd.delta_norm]
        X_train_nn.append(feat)
        gains_all = [gain_d, gain_n, gain_v]
        y_train_nn.append(int(np.argmax(gains_all)))

    X_train_logistic = np.array(X_train_logistic)
    y_train_logistic = np.array(y_train_logistic)
    X_train_nn = np.array(X_train_nn)
    y_train_nn = np.array(y_train_nn)

    # 方案B：逻辑回归
    print('  训练逻辑回归门控（3参数）...')
    a_lr, b_lr, c_lr, loss_lr = train_logistic_gate(X_train_logistic, y_train_logistic)
    train_acc_lr = np.mean([1 if (logistic_gate_predict(X_train_logistic[i,0], X_train_logistic[i,1], a_lr, b_lr, c_lr)[0] > 0.5) == (y_train_logistic[i] == 1) else 0 for i in range(len(y_train_logistic))])
    print(f'    a={a_lr:.4f}, b={b_lr:.4f}, c={c_lr:.4f}, loss={loss_lr:.4f}, train_acc={train_acc_lr:.3f}')

    # 方案C：sigmoid独立权重
    print('  训练sigmoid独立权重门控（4参数）...')
    params_sig, loss_sig = train_sigmoid_independent_gate(X_train_logistic, gains_dior_train, gains_nwpu_train)
    print(f'    a={params_sig[0]:.4f}, b={params_sig[1]:.4f}, c={params_sig[2]:.4f}, d={params_sig[3]:.4f}, loss={loss_sig:.4f}')

    # 方案D：神经网络门控（5次平均）
    print('  训练神经网络门控（195参数，5次平均）...')
    nn_gates = []
    for seed in range(5):
        gate = train_nn_gate(X_train_nn, y_train_nn, seed=seed)
        nn_gates.append(gate)
        with torch.no_grad():
            pred = gate(torch.tensor(X_train_nn, dtype=torch.float32)).argmax(dim=1).numpy()
            acc = (pred == y_train_nn).mean()
        print(f'    seed={seed}: train_acc={acc:.3f}')

    # ── 5. 测试集评估 ──
    print('\n[5/6] 测试集评估 ...')

    # DIOR/NWPU的源域类别映射（用于原型相似度）
    # 测试类别 → DIOR源域类别 → NWPU源域类别
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
        """用给定权重融合专家参数并评估"""
        beta = w_dior * beta_d + w_nwpu * beta_n + w_voc * beta_v
        lam = w_dior * lambda_d + w_nwpu * lambda_n + w_voc * lambda_v
        beta = float(np.clip(beta, 0, 6))
        lam = float(np.clip(lam, 0, 2))
        return cd.evaluate(beta, lam, use_delta_gate=use_delta_gate), beta, lam

    def evaluate_strategy(strategy):
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
            elif strategy == 'nwpu_only':
                w = [0.0, 1.0, 0.0]
            elif strategy == 'voc_only':
                w = [0.0, 0.0, 1.0]
            elif strategy == 'merged':
                b_m, l_m = m_merged.predict(risk)
                b_d, l_d = b_m, l_m
                w = [1.0, 0.0, 0.0]
            elif strategy == 'avg':
                w = [1/3, 1/3, 1/3]
            elif strategy == 'proto_sim':
                # 方案A：原型相似度
                src_d = test_to_dior.get(cd.coco_cls)
                src_n = test_to_nwpu.get(cd.coco_cls)
                if src_d and src_n:
                    w_d, w_n, sim_d, sim_n = prototype_similarity_gate(cd, protos_dior, protos_nwpu, src_d, src_n)
                    w = [w_d, w_n, 0.0]
                else:
                    # 没有对应源域原型，用平均
                    w = [0.5, 0.5, 0.0]
            elif strategy == 'logistic':
                # 方案B：逻辑回归
                w_d, w_n = logistic_gate_predict(risk, cd.delta_norm, a_lr, b_lr, c_lr)
                w = [w_d, w_n, 0.0]
            elif strategy == 'sigmoid_indep':
                # 方案C：sigmoid独立权重
                w_d, w_n = sigmoid_independent_predict(risk, params_sig)
                # 权重和可能<1，剩余用VOC专家（β=0保守）
                w_remain = max(0, 1.0 - w_d - w_n)
                w = [w_d, w_n, w_remain]
            elif strategy == 'nn_gate':
                # 方案D：神经网络门控
                feat = np.array([risk, b_d, b_n, b_v, l_d, l_n, l_v, cd.delta_norm], dtype=np.float32)
                w_list = []
                for g in nn_gates:
                    with torch.no_grad():
                        ww = g(torch.tensor(feat).unsqueeze(0)).squeeze(0).numpy()
                    w_list.append(ww)
                w = [float(np.mean([ww[i] for ww in w_list])) for i in range(3)]
            elif strategy == 'oracle':
                gain_d = cd.evaluate(b_d, l_d, use_delta_gate=True)
                gain_n = cd.evaluate(b_n, l_n, use_delta_gate=True)
                gain_v = cd.evaluate(b_v, l_v, use_delta_gate=True)
                gains_all = [gain_d, gain_n, gain_v]
                best_idx = int(np.argmax(gains_all))
                w = [0.0, 0.0, 0.0]
                w[best_idx] = 1.0

            gain, beta_final, lam_final = evaluate_with_weights(
                cd, b_d, l_d, b_n, l_n, b_v, l_v, w[0], w[1], w[2])
            gains.append(gain)
            params_log[name] = {
                'risk': float(risk), 'delta_norm': float(cd.delta_norm),
                'beta': float(beta_final), 'lambda': float(lam_final),
                'weights': {'DIOR': float(w[0]), 'NWPU': float(w[1]), 'VOC': float(w[2])},
                'gain': float(gain),
            }

        return {
            'strategy': strategy,
            'mean_gain': float(np.mean(gains)),
            'n_positive': sum(1 for g in gains if g > 0),
            'n_total': len(gains),
            'gains': {n: float(g) for n, g in zip(test_categories, gains)},
            'params': params_log,
        }

    strategies = ['dior_only', 'nwpu_only', 'voc_only', 'merged', 'avg',
                  'proto_sim', 'logistic', 'sigmoid_indep', 'nn_gate', 'oracle']
    for strat in strategies:
        print(f'  评估: {strat} ...')
        res = evaluate_strategy(strat)
        all_results[strat] = res
        print(f'    MEAN: {res["mean_gain"]:+.4f}  pos={res["n_positive"]}/{res["n_total"]}')

    # ── 6. 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比（VOC→COCO 12类测试集，外推方向门控ON）')
    print('=' * 100)
    single_best = max(['dior_only', 'nwpu_only', 'voc_only'], key=lambda k: all_results[k]['mean_gain'])
    single_best_gain = all_results[single_best]['mean_gain']
    oracle_gain = all_results['oracle']['mean_gain']
    print(f'最佳单专家: {single_best} (+{single_best_gain:.4f})')
    print(f'Oracle上界: +{oracle_gain:.4f}')
    print()
    print(f'{"策略":<20} {"参数量":>6} {"均值增益":>10} {"正增益":>8} {"vs最佳单专家":>14} {"达到oracle":>12}')
    print('-' * 75)
    param_counts = {'dior_only': 0, 'nwpu_only': 0, 'voc_only': 0, 'merged': 0, 'avg': 0,
                    'proto_sim': 0, 'logistic': 3, 'sigmoid_indep': 4, 'nn_gate': 195, 'oracle': 0}
    for strat in strategies:
        r = all_results[strat]
        vs_best = r['mean_gain'] - single_best_gain
        pct_oracle = r['mean_gain'] / oracle_gain * 100 if oracle_gain > 0 else 0
        marker = ' ←超越最佳单专家!' if (vs_best > 0 and strat != 'oracle') else ''
        print(f'{strat:<20} {param_counts[strat]:>6} {r["mean_gain"]:>+10.4f} '
              f'{r["n_positive"]:>4d}/{r["n_total"]:<3d} {vs_best:>+14.4f} {pct_oracle:>11.1f}%{marker}')

    # 门控权重统计
    print('\n各门控权重统计:')
    for strat in ['proto_sim', 'logistic', 'sigmoid_indep', 'nn_gate']:
        if strat in all_results:
            for expert in ['DIOR', 'NWPU', 'VOC']:
                ws = [v['weights'][expert] for v in all_results[strat]['params'].values()]
                print(f'  {strat:>15} w_{expert}: mean={np.mean(ws):.3f}, range=[{np.min(ws):.3f}, {np.max(ws):.3f}]')

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': f'{len(dev_data)}类',
            'test_set': f'VOC→COCO {len(test_data)}类',
            'delta_gate_threshold': DELTA_GATE_THRESHOLD,
            'methods_compared': strategies,
        },
        'gate_params': {
            'logistic': {'a': a_lr, 'b': b_lr, 'c': c_lr, 'train_acc': float(train_acc_lr)},
            'sigmoid_indep': {'a': params_sig[0], 'b': params_sig[1], 'c': params_sig[2], 'd': params_sig[3]},
        },
        'results': all_results,
    }

    out_path = DATA / 'tge_v2_auto_tune_gate_comparison_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
