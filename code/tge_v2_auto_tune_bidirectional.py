# -*- coding: utf-8 -*-
r"""tge_v2_auto_tune_bidirectional.py — 双向原型校正实验

目前A-CSPC是单向外推：p_hat = p_src + β * (d(target) - s(source))
本实验探索双向结合：同时利用正向和反向外推信息

方案：
  1. uni_forward    - 单向正向（基线）
  2. uni_backward   - 单向反向
  3. bi_select      - 双向选择：根据目标域特征与正/反原型相似度选择
  4. bi_fuse_sim    - 双向加权融合：权重由相似度决定
  5. bi_max         - 双向最大相似度：分别计算取最大
  6. bi_gate        - 双向 + 原型相似度门控（组合最佳方案）
  7. bi_oracle      - 双向oracle：逐类别选正向/反向最佳

运行: python tge_v2_auto_tune_bidirectional.py
输出: 实验数据/tge_v2_auto_tune_bidirectional_results.json
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

from acspc.auto_tuner import GridSearchTuner, LinearMapper
from acspc.coco_categories import get_offline_prior
from acspc.appg import PhysicsPrior

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

K_CALIB = 10
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
PRE_FILTER = 0.03
DELTA_GATE_THRESHOLD = 0.01
BEST_TAU = 0.01

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

def fused_arrays_bidirectional(per_img, F, W_forward, W_backward, mode='forward', target_proto=None):
    """双向融合的视觉分数计算

    mode:
      'forward'  - 只用正向
      'backward' - 只用反向
      'select'   - 根据target_proto与正/反原型相似度选择
      'fuse_sim' - 加权融合，权重由相似度决定
      'max'      - 分别计算取最大
    """
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis_f = np.maximum(0.0, F[p['crop_idx']] @ W_forward)
            vis_b = np.maximum(0.0, F[p['crop_idx']] @ W_backward)

            if mode == 'forward':
                vis = vis_f
            elif mode == 'backward':
                vis = vis_b
            elif mode == 'select':
                # 根据目标域原型与正/反原型的相似度选择
                sim_f = float(np.dot(target_proto, W_forward / (np.linalg.norm(W_forward) + 1e-8)))
                sim_b = float(np.dot(target_proto, W_backward / (np.linalg.norm(W_backward) + 1e-8)))
                vis = vis_f if sim_f >= sim_b else vis_b
            elif mode == 'fuse_sim':
                # 加权融合，权重由相似度决定（softmax温度0.1）
                sim_f = float(np.dot(target_proto, W_forward / (np.linalg.norm(W_forward) + 1e-8)))
                sim_b = float(np.dot(target_proto, W_backward / (np.linalg.norm(W_backward) + 1e-8)))
                tau = 0.1
                w_f = np.exp(sim_f / tau) / (np.exp(sim_f / tau) + np.exp(sim_b / tau))
                w_b = 1.0 - w_f
                vis = w_f * vis_f + w_b * vis_b
            elif mode == 'max':
                # 逐框取最大相似度
                vis = np.maximum(vis_f, vis_b)
            else:
                vis = vis_f

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

    def evaluate_bidirectional(self, beta, lam, mode='forward', use_delta_gate=True):
        """双向评估

        mode: forward, backward, select, fuse_sim, max
        """
        if use_delta_gate and self.delta_norm < DELTA_GATE_THRESHOLD:
            beta = 0.0

        # 正向原型
        P_forward = self.P_src + beta * self.Delta.to(U.DEVICE)
        P_forward = P_forward / P_forward.norm()
        W_forward = (P_forward.cpu().numpy() - lam * self.P_neg.cpu().numpy())

        # 反向原型
        P_backward = self.P_src - beta * self.Delta.to(U.DEVICE)
        P_backward = P_backward / P_backward.norm()
        W_backward = (P_backward.cpu().numpy() - lam * self.P_neg.cpu().numpy())

        fused_arrays_bidirectional(self.per_img, self.F, W_forward, W_backward,
                                     mode=mode, target_proto=self.target_proto)
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

def prototype_similarity_gate(cd, protos_dior, protos_nwpu, src_cls_dior, src_cls_nwpu, tau=0.01):
    target_proto = cd.target_proto
    sim_dior = float(np.dot(target_proto, protos_dior[src_cls_dior].cpu().numpy())) if src_cls_dior in protos_dior else 0.0
    sim_nwpu = float(np.dot(target_proto, protos_nwpu[src_cls_nwpu].cpu().numpy())) if src_cls_nwpu in protos_nwpu else 0.0
    w_dior = np.exp(sim_dior / tau)
    w_nwpu = np.exp(sim_nwpu / tau)
    total = w_dior + w_nwpu
    return w_dior / total, w_nwpu / total, sim_dior, sim_nwpu


def main():
    t0 = time.time()
    print('=' * 100)
    print('双向原型校正实验：单向正向 vs 单向反向 vs 双向选择/融合/最大')
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
        # 用正向模式搜索最优参数
        opt = gs.search(name, risk, lambda b, l: cd.evaluate_bidirectional(b, l, mode='forward'))
        optimals.append(opt)

    dior_opts = [o for o in optimals if o.category.startswith('DIOR')]
    nwpu_opts = [o for o in optimals if o.category.startswith('NWPU')]
    voc_opts = [o for o in optimals if o.category.startswith('VOC')]

    m_dior = LinearMapper(); m_dior.fit(np.array([o.risk for o in dior_opts]),
                                           np.array([o.beta_opt for o in dior_opts]),
                                           np.array([o.lambda_opt for o in dior_opts]))
    m_nwpu = LinearMapper(); m_nwpu.fit(np.array([o.risk for o in nwpu_opts]),
                                           np.array([o.beta_opt for o in nwpu_opts]),
                                           np.array([o.lambda_opt for o in nwpu_opts]))
    m_voc = LinearMapper(); m_voc.fit(np.array([o.risk for o in voc_opts]),
                                         np.array([o.beta_opt for o in voc_opts]),
                                         np.array([o.lambda_opt for o in voc_opts]))
    print(f'  专家训练完成')

    # ── 4. 测试集评估 ──
    print('\n[4/5] 测试集评估（7种双向模式）...')

    test_to_dior = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
                     'motorcycle': 'vehicle', 'bus': 'vehicle', 'train': None,
                     'bicycle': None, 'bottle': None, 'chair': None, 'couch': None,
                     'potted plant': None, 'tv': None}
    test_to_nwpu = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
                    'motorcycle': 'vehicle', 'bus': 'vehicle', 'train': None,
                    'bicycle': None, 'bottle': None, 'chair': None, 'couch': None,
                    'potted plant': None, 'tv': None}

    modes = ['forward', 'backward', 'select', 'fuse_sim', 'max']
    all_results = {}

    for mode in modes:
        print(f'\n  模式: {mode}')
        gains_dior = []
        gains_proto_gate = []
        gains_oracle = []
        params_log = {}

        for name in test_categories:
            cd = test_data[name]
            risk = get_risk(cd.coco_cls)
            b_d, l_d = m_dior.predict(risk)
            b_n, l_n = m_nwpu.predict(risk)
            b_v, l_v = m_voc.predict(risk)

            # dior_only
            gain_d = cd.evaluate_bidirectional(b_d, l_d, mode=mode)
            gains_dior.append(gain_d)

            # 原型相似度门控（DIOR+NWPU）
            src_d = test_to_dior.get(cd.coco_cls)
            src_n = test_to_nwpu.get(cd.coco_cls)
            if src_d and src_n:
                w_d, w_n, _, _ = prototype_similarity_gate(cd, protos_dior, protos_nwpu, src_d, src_n, tau=BEST_TAU)
            else:
                w_d, w_n = 0.5, 0.5
            beta_gate = float(np.clip(w_d * b_d + w_n * b_n, 0, 6))
            lam_gate = float(np.clip(w_d * l_d + w_n * l_n, 0, 2))
            gain_gate = cd.evaluate_bidirectional(beta_gate, lam_gate, mode=mode)
            gains_proto_gate.append(gain_gate)

            # oracle（逐类别选最佳专家+最佳方向）
            gains_all = []
            for b, l in [(b_d, l_d), (b_n, l_n), (b_v, l_v)]:
                for m in ['forward', 'backward']:
                    gains_all.append(cd.evaluate_bidirectional(b, l, mode=m))
            gain_oracle = max(gains_all)
            gains_oracle.append(gain_oracle)

            params_log[name] = {
                'risk': float(risk), 'delta_norm': float(cd.delta_norm),
                'gain_dior': float(gain_d), 'gain_gate': float(gain_gate),
                'gain_oracle': float(gain_oracle),
                'weights': {'DIOR': float(w_d), 'NWPU': float(w_n)},
            }

        all_results[f'{mode}_dior'] = {
            'mode': mode, 'strategy': 'dior_only',
            'mean_gain': float(np.mean(gains_dior)),
            'n_positive': sum(1 for g in gains_dior if g > 0),
            'n_total': len(gains_dior),
            'gains': {n: float(g) for n, g in zip(test_categories, gains_dior)},
        }
        all_results[f'{mode}_proto_gate'] = {
            'mode': mode, 'strategy': 'proto_sim_gate',
            'mean_gain': float(np.mean(gains_proto_gate)),
            'n_positive': sum(1 for g in gains_proto_gate if g > 0),
            'n_total': len(gains_proto_gate),
            'gains': {n: float(g) for n, g in zip(test_categories, gains_proto_gate)},
        }
        all_results[f'{mode}_oracle'] = {
            'mode': mode, 'strategy': 'oracle',
            'mean_gain': float(np.mean(gains_oracle)),
            'n_positive': sum(1 for g in gains_oracle if g > 0),
            'n_total': len(gains_oracle),
            'gains': {n: float(g) for n, g in zip(test_categories, gains_oracle)},
        }
        print(f'    dior_only:    {np.mean(gains_dior):+.4f} ({sum(1 for g in gains_dior if g>0)}/{len(gains_dior)})')
        print(f'    proto_gate:   {np.mean(gains_proto_gate):+.4f} ({sum(1 for g in gains_proto_gate if g>0)}/{len(gains_proto_gate)})')
        print(f'    oracle:       {np.mean(gains_oracle):+.4f} ({sum(1 for g in gains_oracle if g>0)}/{len(gains_oracle)})')

    # ── 5. 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比（VOC→COCO 12类测试集）')
    print('=' * 100)

    print(f'\n{"模式":<12} {"策略":<15} {"均值增益":>10} {"正增益":>8} {"vs正向dior":>12}')
    print('-' * 60)
    baseline = all_results['forward_dior']['mean_gain']
    for mode in modes:
        for strat in ['dior', 'proto_gate', 'oracle']:
            key = f'{mode}_{strat}'
            r = all_results[key]
            vs_base = r['mean_gain'] - baseline
            marker = ' ←最佳!' if (mode != 'forward' and strat != 'oracle' and vs_base > 0) else ''
            print(f'{mode:<12} {strat:<15} {r["mean_gain"]:>+10.4f} '
                  f'{r["n_positive"]:>4d}/{r["n_total"]:<3d} {vs_base:>+12.4f}{marker}')

    # 找出最佳非oracle方案
    best_key = None
    best_gain = -999
    for mode in modes:
        for strat in ['dior', 'proto_gate']:
            key = f'{mode}_{strat}'
            if all_results[key]['mean_gain'] > best_gain:
                best_gain = all_results[key]['mean_gain']
                best_key = key
    print(f'\n最佳非oracle方案: {best_key} = +{best_gain:.4f}')
    print(f'vs 正向dior_only: +{best_gain - baseline:+.4f}')

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': f'{len(dev_data)}类',
            'test_set': f'VOC→COCO {len(test_data)}类',
            'modes': modes,
            'best_tau': BEST_TAU,
            'delta_gate_threshold': DELTA_GATE_THRESHOLD,
        },
        'best_non_oracle': {'key': best_key, 'mean_gain': best_gain},
        'baseline_forward_dior': baseline,
        'results': all_results,
    }

    out_path = DATA / 'tge_v2_auto_tune_bidirectional_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
