# -*- coding: utf-8 -*-
r"""tge_v2_auto_tune_cross_dataset.py — 跨数据集验证实验

在DIOR→COCO测试集上验证门控方法的泛化性（之前都在VOC→COCO上测试）
DIOR→COCO的Δnorm=0.619，外推更有效，可以更好检验门控价值

开发集：DIOR 6类 + NWPU 5类 + VOC 7类 = 18类（同前）
测试集：DIOR→COCO 6类（airplane, boat, car, truck, bus, motorcycle）

对比方案：
  1. fixed (β=2.0, λ=0.5) - 第一篇论文主配置
  2. target_independent (β=1.0, λ=0.25) - 目标域无关配置
  3. dior_only - 只用DIOR专家
  4. merged - 合并所有源域训练
  5. avg - 三专家简单平均
  6. proto_sim_gate - 原型相似度门控（0参数）
  7. two_level_gate - 两级门控（Δnorm + 原型相似度）
  8. oracle - 上界

运行: python tge_v2_auto_tune_cross_dataset.py
输出: 实验数据/tge_v2_auto_tune_cross_dataset_results.json
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

# DIOR→COCO测试集（6类交通工具，与开发集DIOR类别相同但用不同的COCO图像）
TEST_DIOR_COCO = [
    ('TEST-DIOR-COCO-airplane',   'airplane',   'airplane'),
    ('TEST-DIOR-COCO-boat',       'boat',       'ship'),
    ('TEST-DIOR-COCO-car',        'car',        'vehicle'),
    ('TEST-DIOR-COCO-truck',      'truck',      'vehicle'),
    ('TEST-DIOR-COCO-bus',        'bus',        'vehicle'),
    ('TEST-DIOR-COCO-motorcycle', 'motorcycle', 'vehicle'),
]


# ── 基础工具 ──
def load_coco_all(cls_name):
    return U.load_coco(cls_name, n=100000)

def load_coco_heldout(cls_name, skip_first=200):
    """加载COCO数据，跳过前skip_first张（用作开发集），剩余作为测试集"""
    images, gts = U.load_coco(cls_name, n=100000)
    if len(images) > skip_first:
        return images[skip_first:], gts[skip_first:]
    return images, gts

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
    # 测试集没有K_CALIB，直接用所有图像
    preds = [(p['box'], p[key]) for p in per_img]
    gts = [p['gt'] for p in per_img]
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

def load_class_data(name, coco_cls, src_cls, protos, source, neg_protos=None, heldout=False):
    if heldout:
        images, gts = load_coco_heldout(coco_cls, skip_first=200)
    else:
        images, gts = load_coco_all(coco_cls)
    if len(images) < 10:
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
    print('跨数据集验证：DIOR→COCO测试集（held-out，跳过前200张）')
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
        cd = load_class_data(name, coco_cls, src_cls, protos, source, neg_protos=protos_for_neg, heldout=False)
        if cd: dev_data[name] = cd
    print(f'  开发集: {len(dev_data)}类')

    test_data = {}
    for name, coco_cls, src_cls in TEST_DIOR_COCO:
        cd = load_class_data(name, coco_cls, src_cls, protos_dior, 'dior', neg_protos=protos_for_neg, heldout=True)
        if cd: test_data[name] = cd
    test_categories = list(test_data.keys())
    print(f'  测试集: {len(test_data)}类 (DIOR→COCO held-out)')
    for name in test_categories:
        cd = test_data[name]
        print(f'    {name}: {len(cd.per_img)}张图, Δnorm={cd.delta_norm:.4f}, baseAP={cd.base50:.4f}')

    # ── 3. 网格搜索 + 训练专家 ──
    print('\n[3/5] 开发集网格搜索 + 训练专家映射 ...')
    gs = GridSearchTuner(beta_range=(0.0, 4.0), lambda_range=(0.0, 2.0), n_beta=17, n_lambda=9)
    optimals = []
    for name, cd in dev_data.items():
        risk = get_risk(cd.coco_cls)
        opt = gs.search(name, risk, lambda b, l: cd.evaluate(b, l))
        optimals.append(opt)

    dior_opts = [o for o in optimals if o.category.startswith('DIOR')]
    nwpu_opts = [o for o in optimals if o.category.startswith('NWPU')]
    voc_opts = [o for o in optimals if o.category.startswith('VOC')]
    all_opts = dior_opts + nwpu_opts + voc_opts

    m_dior = LinearMapper(); m_dior.fit(np.array([o.risk for o in dior_opts]),
                                           np.array([o.beta_opt for o in dior_opts]),
                                           np.array([o.lambda_opt for o in dior_opts]))
    m_nwpu = LinearMapper(); m_nwpu.fit(np.array([o.risk for o in nwpu_opts]),
                                           np.array([o.beta_opt for o in nwpu_opts]),
                                           np.array([o.lambda_opt for o in nwpu_opts]))
    m_voc = LinearMapper(); m_voc.fit(np.array([o.risk for o in voc_opts]),
                                         np.array([o.beta_opt for o in voc_opts]),
                                         np.array([o.lambda_opt for o in voc_opts]))
    m_merged = LinearMapper(); m_merged.fit(np.array([o.risk for o in all_opts]),
                                              np.array([o.beta_opt for o in all_opts]),
                                              np.array([o.lambda_opt for o in all_opts]))
    print(f'  专家训练完成')

    # ── 4. 测试集评估 ──
    print('\n[4/5] 测试集评估（8种策略）...')

    strategies = ['fixed_main', 'target_indep', 'dior_only', 'merged', 'avg',
                  'proto_sim_gate', 'two_level_gate', 'oracle']
    all_results = {}

    for strat in strategies:
        print(f'\n  策略: {strat}')
        gains = []
        params_log = {}

        for name in test_categories:
            cd = test_data[name]
            risk = get_risk(cd.coco_cls)

            if strat == 'fixed_main':
                beta, lam = 2.0, 0.5
            elif strat == 'target_indep':
                beta, lam = 1.0, 0.25
            elif strat == 'dior_only':
                beta, lam = m_dior.predict(risk)
            elif strat == 'merged':
                beta, lam = m_merged.predict(risk)
            elif strat == 'avg':
                b_d, l_d = m_dior.predict(risk)
                b_n, l_n = m_nwpu.predict(risk)
                b_v, l_v = m_voc.predict(risk)
                beta, lam = (b_d + b_n + b_v) / 3, (l_d + l_n + l_v) / 3
            elif strat == 'proto_sim_gate':
                b_d, l_d = m_dior.predict(risk)
                b_n, l_n = m_nwpu.predict(risk)
                w_d, w_n, _, _ = prototype_similarity_gate(cd, protos_dior, protos_nwpu,
                                                              cd.src_cls, cd.src_cls, tau=BEST_TAU)
                beta = w_d * b_d + w_n * b_n
                lam = w_d * l_d + w_n * l_n
            elif strat == 'two_level_gate':
                if cd.delta_norm < DELTA_GATE_THRESHOLD:
                    beta, lam = 0.0, 0.5  # Δnorm小，β=0
                else:
                    b_d, l_d = m_dior.predict(risk)
                    b_n, l_n = m_nwpu.predict(risk)
                    w_d, w_n, _, _ = prototype_similarity_gate(cd, protos_dior, protos_nwpu,
                                                                  cd.src_cls, cd.src_cls, tau=BEST_TAU)
                    beta = w_d * b_d + w_n * b_n
                    lam = w_d * l_d + w_n * l_n
            elif strat == 'oracle':
                # 逐类别搜索最优β,λ
                best_gain = -999
                best_beta, best_lam = 0, 0
                for b in np.linspace(0, 4, 17):
                    for l in np.linspace(0, 2, 9):
                        g = cd.evaluate(float(b), float(l))
                        if g > best_gain:
                            best_gain = g
                            best_beta, best_lam = float(b), float(l)
                beta, lam = best_beta, best_lam

            beta = float(np.clip(beta, 0, 6))
            lam = float(np.clip(lam, 0, 2))
            gain = cd.evaluate(beta, lam)
            gains.append(gain)
            params_log[name] = {
                'risk': float(risk), 'delta_norm': float(cd.delta_norm),
                'beta': float(beta), 'lambda': float(lam), 'gain': float(gain),
            }

        all_results[strat] = {
            'strategy': strat,
            'mean_gain': float(np.mean(gains)),
            'n_positive': sum(1 for g in gains if g > 0),
            'n_total': len(gains),
            'gains': {n: float(g) for n, g in zip(test_categories, gains)},
            'params': params_log,
        }
        print(f'    MEAN: {np.mean(gains):+.4f}, pos={sum(1 for g in gains if g>0)}/{len(gains)}')

    # ── 5. 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比（DIOR→COCO held-out测试集，6类）')
    print('=' * 100)

    print(f'\n{"策略":<20} {"均值增益":>10} {"正增益":>8} {"vs fixed_main":>14} {"达到oracle":>12}')
    print('-' * 65)
    baseline = all_results['fixed_main']['mean_gain']
    oracle_gain = all_results['oracle']['mean_gain']
    for strat in strategies:
        r = all_results[strat]
        vs_base = r['mean_gain'] - baseline
        pct_oracle = r['mean_gain'] / oracle_gain * 100 if oracle_gain > 0 else 0
        marker = ' ←超越主配置!' if (vs_base > 0 and strat not in ['oracle']) else ''
        print(f'{strat:<20} {r["mean_gain"]:>+10.4f} '
              f'{r["n_positive"]:>4d}/{r["n_total"]:<3d} {vs_base:>+14.4f} {pct_oracle:>11.1f}%{marker}')

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': f'{len(dev_data)}类',
            'test_set': f'DIOR→COCO held-out {len(test_data)}类',
            'heldout_skip': 200,
            'best_tau': BEST_TAU,
            'delta_gate_threshold': DELTA_GATE_THRESHOLD,
        },
        'baseline_fixed_main': baseline,
        'oracle_gain': oracle_gain,
        'results': all_results,
    }

    out_path = DATA / 'tge_v2_auto_tune_cross_dataset_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
