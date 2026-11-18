# -*- coding: utf-8 -*-
r"""tge_v2_auto_tune_moe_gate.py — MoE + 域感知门控网络实验

方案B：每个源域一个映射专家（DIOR专家 + NWPU专家），
      训练轻量门控网络动态选择/加权专家。

门控网络设计：
  输入特征（7维）：
    1. 类别风险评分 R
    2. DIOR专家预测的 β_dior
    3. NWPU专家预测的 β_nwpu
    4. DIOR专家预测的 λ_dior
    5. NWPU专家预测的 λ_nwpu
    6. |β_dior - β_nwpu| （参数差异度）
    7. |λ_dior - λ_nwpu|
  结构：7 → 16 (ReLU) → 2 (softmax) → [w_dior, w_nwpu]
  训练：开发集11类，标签=增益更高的专家，交叉熵损失
  正则：L2 weight decay + 小网络（仅72参数）

实验设计：
  开发集：DIOR→COCO 6类 + NWPU→COCO 5类 = 11类
  测试集：VOC→COCO 19类

  对比策略：
    1. dior_only:    仅用DIOR专家（最佳基线 +0.0459）
    2. nwpu_only:    仅用NWPU专家
    3. merged:       合并训练
    4. avg:          简单平均
    5. moe_gate:     MoE门控网络（方案B核心）
    6. oracle_gate:  最优专家选择（上界，测试时用真实增益选最佳）

  如果moe_gate > dior_only (+0.0459)，则方案B成功。

运行: python tge_v2_auto_tune_moe_gate.py
输出: 实验数据/tge_v2_auto_tune_moe_gate_results.json
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

from acspc.auto_tuner import GridSearchTuner, LinearMapper, PolynomialMapper, ClassOptimal
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
    m.fit(np.array(risks), np.array(betas), np.array(lambdas))
    return m


# ── 门控网络 ──
class MoEGate(nn.Module):
    """轻量门控网络：7维输入 → 2维softmax输出"""
    def __init__(self, input_dim=7, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def extract_gate_features(risk, beta_dior, beta_nwpu, lambda_dior, lambda_nwpu):
    """提取门控网络输入特征（7维）"""
    return np.array([
        risk,
        beta_dior, beta_nwpu,
        lambda_dior, lambda_nwpu,
        abs(beta_dior - beta_nwpu),
        abs(lambda_dior - lambda_nwpu),
    ], dtype=np.float32)


def train_gate(gate, X_train, y_train, n_epochs=500, lr=0.01, weight_decay=1e-4):
    """训练门控网络"""
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
    return float(loss.item())


def predict_weights(gate, features):
    """用门控网络预测专家权重"""
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        w = gate(x).squeeze(0).numpy()
    return float(w[0]), float(w[1])


def main():
    t0 = time.time()
    print('=' * 100)
    print('A-CSPC AutoTuner MoE-Gate: MoE + 域感知门控网络实验')
    print('=' * 100)

    # ── 1. 构建源域原型 ──
    print('\n[1/7] 构建源域原型 ...')
    protos_dior = build_protos('dior')
    protos_nwpu = build_protos('nwpu')
    protos_voc = build_protos('voc')
    protos_for_neg = dict(protos_dior); protos_for_neg.update(protos_nwpu)
    print(f'  DIOR: {len(protos_dior)}类, NWPU: {len(protos_nwpu)}类, VOC: {len(protos_voc)}类')

    # ── 2. 加载开发集数据 ──
    print('\n[2/7] 加载开发集数据 ...')
    dev_data = {}
    for name, coco_cls, src_cls in DEV_TARGETS_DIOR:
        cd = load_class_data(name, coco_cls, src_cls, protos_dior, 'dior')
        if cd: dev_data[name] = cd
    for name, coco_cls, src_cls in DEV_TARGETS_NWPU:
        cd = load_class_data(name, coco_cls, src_cls, protos_nwpu, 'nwpu', neg_protos=protos_for_neg)
        if cd: dev_data[name] = cd
    print(f'  开发集: {len(dev_data)}类')

    # ── 3. 开发集网格搜索 ──
    print('\n[3/7] 开发集网格搜索 ...')
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

    # ── 4. 训练两个专家 ──
    print('\n[4/7] 训练源域专家映射 ...')
    m_dior = train_mapper('linear', [o.risk for o in dior_opts],
                           [o.beta_opt for o in dior_opts], [o.lambda_opt for o in dior_opts])
    m_nwpu = train_mapper('linear', [o.risk for o in nwpu_opts],
                           [o.beta_opt for o in nwpu_opts], [o.lambda_opt for o in nwpu_opts])
    m_merged = train_mapper('linear', [o.risk for o in optimals],
                            [o.beta_opt for o in optimals], [o.lambda_opt for o in optimals])
    print(f'  DIOR专家: β={m_dior.a_beta:.3f}R+{m_dior.b_beta:.3f}')
    print(f'  NWPU专家: β={m_nwpu.a_beta:.3f}R+{m_nwpu.b_beta:.3f}')

    # ── 5. 准备门控网络训练数据 ──
    print('\n[5/7] 准备门控网络训练数据 ...')
    X_train = []
    y_train = []
    train_details = []

    for name, cd in dev_data.items():
        risk = get_risk(cd.coco_cls)
        b_d, l_d = m_dior.predict(risk)
        b_n, l_n = m_nwpu.predict(risk)

        # 用两个专家的参数分别评估增益
        gain_dior = cd.evaluate(b_d, l_d)
        gain_nwpu = cd.evaluate(b_n, l_n)

        # 特征
        feat = extract_gate_features(risk, b_d, b_n, l_d, l_n)
        X_train.append(feat)

        # 标签：增益更高的专家
        label = 0 if gain_dior >= gain_nwpu else 1
        y_train.append(label)

        train_details.append({
            'name': name, 'risk': risk,
            'beta_dior': b_d, 'beta_nwpu': b_n,
            'lambda_dior': l_d, 'lambda_nwpu': l_n,
            'gain_dior': gain_dior, 'gain_nwpu': gain_nwpu,
            'label': label,
        })
        print(f'  {name}: gain_dior={gain_dior:+.4f}, gain_nwpu={gain_nwpu:+.4f}, '
              f'label={"DIOR" if label==0 else "NWPU"}')

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f'  训练样本: {len(X_train)} (DIOR标签={sum(y_train==0)}, NWPU标签={sum(y_train==1)})')

    # 训练门控网络（多次取平均，减少随机性）
    print('\n  训练门控网络 (5次平均) ...')
    gates = []
    for seed in range(5):
        torch.manual_seed(seed)
        gate = MoEGate(input_dim=7, hidden_dim=16)
        loss = train_gate(gate, X_train, y_train, n_epochs=500, lr=0.01, weight_decay=1e-4)
        gates.append(gate)
        # 训练集准确率
        with torch.no_grad():
            pred = gate(torch.tensor(X_train, dtype=torch.float32)).argmax(dim=1).numpy()
            acc = (pred == y_train).mean()
        print(f'    seed={seed}: loss={loss:.4f}, train_acc={acc:.3f}')

    # ── 6. 加载测试集数据 ──
    print('\n[6/7] 加载测试集数据 ...')
    test_data = {}
    for name, coco_cls, src_cls in TEST_TARGETS:
        cd = load_class_data(name, coco_cls, src_cls, protos_voc, 'voc')
        if cd: test_data[name] = cd
    test_categories = list(test_data.keys())
    print(f'  测试集: {len(test_data)}类')

    # ── 7. 测试集评估 ──
    print('\n[7/7] 测试集评估 ...')
    all_results = {}

    def evaluate_strategy(strategy, gate=None):
        gains = []
        params_log = {}
        for name in test_categories:
            cd = test_data[name]
            risk = get_risk(cd.coco_cls)
            b_d, l_d = m_dior.predict(risk)
            b_n, l_n = m_nwpu.predict(risk)

            if strategy == 'dior_only':
                beta, lam = b_d, l_d
                w_d, w_n = 1.0, 0.0
            elif strategy == 'nwpu_only':
                beta, lam = b_n, l_n
                w_d, w_n = 0.0, 1.0
            elif strategy == 'merged':
                beta, lam = m_merged.predict(risk)
                w_d, w_n = 0.5, 0.5
            elif strategy == 'avg':
                beta, lam = (b_d + b_n) / 2, (l_d + l_n) / 2
                w_d, w_n = 0.5, 0.5
            elif strategy == 'moe_gate':
                feat = extract_gate_features(risk, b_d, b_n, l_d, l_n)
                # 5个门控网络平均
                w_d_list, w_n_list = [], []
                for g in gates:
                    wd, wn = predict_weights(g, feat)
                    w_d_list.append(wd); w_n_list.append(wn)
                w_d = float(np.mean(w_d_list))
                w_n = float(np.mean(w_n_list))
                beta = w_d * b_d + w_n * b_n
                lam = w_d * l_d + w_n * l_n
            elif strategy == 'oracle_gate':
                # 上界：用真实增益选最佳专家
                gain_d = cd.evaluate(b_d, l_d)
                gain_n = cd.evaluate(b_n, l_n)
                if gain_d >= gain_n:
                    beta, lam = b_d, l_d
                    w_d, w_n = 1.0, 0.0
                else:
                    beta, lam = b_n, l_n
                    w_d, w_n = 0.0, 1.0

            beta = float(np.clip(beta, 0, 6))
            lam = float(np.clip(lam, 0, 2))
            gain = cd.evaluate(beta, lam)
            gains.append(gain)
            params_log[name] = {
                'risk': float(risk), 'beta': float(beta), 'lambda': float(lam),
                'w_dior': float(w_d), 'w_nwpu': float(w_n), 'gain': float(gain),
            }

        return {
            'strategy': strategy,
            'mean_gain': float(np.mean(gains)),
            'n_positive': sum(1 for g in gains if g > 0),
            'n_total': len(gains),
            'gains': {n: float(g) for n, g in zip(test_categories, gains)},
            'params': params_log,
        }

    for strat in ['dior_only', 'nwpu_only', 'merged', 'avg', 'moe_gate', 'oracle_gate']:
        print(f'\n  评估策略: {strat}')
        res = evaluate_strategy(strat)
        all_results[strat] = res
        print(f'    MEAN: {res["mean_gain"]:+.4f}  pos={res["n_positive"]}/{res["n_total"]}')

    # ── 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比（VOC→COCO 19类测试集，线性映射专家）')
    print('=' * 100)
    dior_gain = all_results['dior_only']['mean_gain']
    print(f'{"策略":<16} {"均值增益":>10} {"正增益":>8} {"vs dior_only":>14} {"达到oracle":>12}')
    print('-' * 65)
    oracle_gain = all_results['oracle_gate']['mean_gain']
    for strat in ['dior_only', 'nwpu_only', 'merged', 'avg', 'moe_gate', 'oracle_gate']:
        r = all_results[strat]
        vs_dior = r['mean_gain'] - dior_gain
        pct_oracle = r['mean_gain'] / oracle_gain * 100 if oracle_gain > 0 else 0
        marker = ' ←超越DIOR!' if (vs_dior > 0 and strat != 'oracle_gate') else ''
        print(f'{strat:<16} {r["mean_gain"]:>+10.4f} {r["n_positive"]:>4d}/{r["n_total"]:<3d} '
              f'{vs_dior:>+14.4f} {pct_oracle:>11.1f}%{marker}')

    # 门控网络权重统计
    print('\n门控网络权重统计（moe_gate）:')
    w_diors = [v['w_dior'] for v in all_results['moe_gate']['params'].values()]
    print(f'  w_dior: mean={np.mean(w_diors):.3f}, min={np.min(w_diors):.3f}, max={np.max(w_diors):.3f}')
    print(f'  选择DIOR为主(w_dior>0.5): {sum(1 for w in w_diors if w > 0.5)}/{len(w_diors)}')

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': 'DIOR 6类 + NWPU 5类 = 11类',
            'test_set': 'VOC→COCO 19类',
            'gate_input_dim': 7, 'gate_hidden_dim': 16, 'n_gates': 5,
            'expert_type': 'linear',
        },
        'train_details': train_details,
        'results': all_results,
    }

    out_path = DATA / 'tge_v2_auto_tune_moe_gate_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
