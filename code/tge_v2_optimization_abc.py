# -*- coding: utf-8 -*-
r"""tge_v2_optimization_abc.py — A-CSPC算法优化消融实验

三个优化方向：
  A. 逐框自适应β：β_k = base_beta * (1 + α * (1 - <f_k, p_src>))
  B. 难负样本动态选择：对每个候选框，选择top-m个最相似的负原型
  C. 校准后加权融合：温度校准 + 加权融合

实验设计：
  开发集：DIOR→COCO 6类（搜索超参数）
  测试集：VOC→COCO 12类（评估泛化）
  对比：基线 / +A / +B / +C / +A+B / +A+C / +B+C / +A+B+C / oracle

运行: python tge_v2_optimization_abc.py
输出: 实验数据/tge_v2_optimization_abc_results.json
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

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

K_CALIB = 10
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
PRE_FILTER = 0.03

VOC_ROOT = r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007'
VOC_IMG_DIR = os.path.join(VOC_ROOT, 'JPEGImages')
VOC_ANN_DIR = os.path.join(VOC_ROOT, 'Annotations')

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

def dataset_ap(per_img, key):
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    return U.compute_ap50_img(preds, gts)


class ClassData:
    def __init__(self, name, coco_cls, src_cls, per_img, F, P_src, P_neg, P_neg_all, Delta, base50):
        self.name = name; self.coco_cls = coco_cls; self.src_cls = src_cls
        self.per_img = per_img; self.F = F; self.P_src = P_src
        self.P_neg = P_neg; self.P_neg_all = P_neg_all; self.Delta = Delta; self.base50 = base50
        self.delta_norm = float(Delta.norm())

    def evaluate_baseline(self, beta, lam):
        """基线：固定β，全局负原型，几何平均融合"""
        P_hat = self.P_src + beta * self.Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - lam * self.P_neg.cpu().numpy())
        for p in self.per_img:
            n = len(p['gd'])
            if n:
                vis = np.maximum(0.0, self.F[p['crop_idx']] @ W)
                p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
            else:
                p['fused'] = np.zeros(0)
        return dataset_ap(self.per_img, 'fused') - self.base50

    def evaluate_optimized(self, beta, lam, use_A=False, use_B=False, use_C=False,
                            alpha_A=0.5, m_B=3, w_C=0.5, tau_det=1.0, tau_vis=1.0):
        """优化版：A逐框自适应β + B难负样本动态选择 + C校准加权融合"""
        P_src_np = self.P_src.cpu().numpy()
        Delta_np = self.Delta.cpu().numpy()
        P_neg_all_np = self.P_neg_all.cpu().numpy()  # (K, 512)
        P_neg_np = self.P_neg.cpu().numpy()

        for p in self.per_img:
            n = len(p['gd'])
            if n == 0:
                p['fused'] = np.zeros(0)
                continue

            f_k = self.F[p['crop_idx']]  # (n, 512)
            gd = p['gd']  # (n,)

            # ── 方向A：逐框自适应β ──
            if use_A:
                sim_src = f_k @ P_src_np  # (n,) 与源域原型的相似度
                # 相似度越低（域偏移越大），外推幅度越大
                beta_k = beta * (1.0 + alpha_A * (1.0 - sim_src))
                beta_k = np.clip(beta_k, 0, 6)
            else:
                beta_k = np.full(n, beta)

            # 逐框外推正原型
            P_hat_k = P_src_np[None, :] + beta_k[:, None] * Delta_np[None, :]  # (n, 512)
            norms = np.linalg.norm(P_hat_k, axis=1, keepdims=True) + 1e-8
            P_hat_k = P_hat_k / norms

            # ── 方向B：难负样本动态选择 ──
            if use_B:
                # 对每个候选框，选择top-m个最相似的负原型
                sim_neg_all = f_k @ P_neg_all_np.T  # (n, K)
                topm_idx = np.argpartition(-sim_neg_all, m_B-1, axis=1)[:, :m_B]  # (n, m_B)
                # 取top-m负原型的平均作为该框的负原型
                P_neg_dyn = np.zeros((n, 512))
                for i in range(n):
                    P_neg_dyn[i] = P_neg_all_np[topm_idx[i]].mean(axis=0)
                neg_norm = np.linalg.norm(P_neg_dyn, axis=1, keepdims=True) + 1e-8
                P_neg_dyn = P_neg_dyn / neg_norm
                vis_pos = np.sum(f_k * P_hat_k, axis=1)  # (n,)
                vis_neg = np.sum(f_k * P_neg_dyn, axis=1)  # (n,)
            else:
                vis_pos = np.sum(f_k * P_hat_k, axis=1)  # (n,)
                vis_neg = f_k @ P_neg_np  # (n,)

            vis = np.maximum(0.0, vis_pos - lam * vis_neg)

            # ── 方向C：校准后加权融合 ──
            if use_C:
                # 温度校准（用sigmoid将logit映射到概率）
                # 检测器分数已经是概率，做温度缩放
                gd_calib = np.clip(gd ** (1.0 / tau_det), 1e-6, 1.0)
                vis_calib = np.clip(vis ** (1.0 / tau_vis), 1e-6, 1.0)
                p['fused'] = w_C * gd_calib + (1 - w_C) * vis_calib
            else:
                p['fused'] = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))

        return dataset_ap(self.per_img, 'fused') - self.base50


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
    elif source == 'voc':
        for cls in VOC_TO_COCO.keys():
            images, gts = load_voc_class(cls, n_max=50)
            p = clip_proto(gt_crops(images, gts))
            if p is not None: protos[cls] = p
    return protos

def load_class_data(name, coco_cls, src_cls, protos, source):
    images, gts = load_coco_all(coco_cls)
    if len(images) < K_CALIB + 10:
        return None
    dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), coco_cls)
    per_img, F = collect_features(images, gts, dets)
    if src_cls not in protos:
        return None
    P_src = protos[src_cls].to(U.DEVICE)
    P_neg = T.get_neg_proto(protos, src_cls)
    # 所有负原型（用于方向B动态选择）
    neg_list = []
    for k, v in protos.items():
        if k != src_cls:
            neg_list.append(v.cpu().numpy())
    P_neg_all = torch.tensor(np.stack(neg_list), dtype=torch.float32)
    if source == 'dior':
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
    else:
        s_xs = T.text_emb(['a photo of {}'.format(src_cls)]).cpu()[0]
    d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
    Delta = d_y - s_xs
    base50 = dataset_ap(per_img, 'gd')
    return ClassData(name, coco_cls, src_cls, per_img, F, P_src, P_neg, P_neg_all, Delta, base50)


def main():
    t0 = time.time()
    print('=' * 100)
    print('A-CSPC优化消融实验：A逐框自适应β + B难负样本动态选择 + C校准加权融合')
    print('=' * 100)

    # ── 1. 构建原型 ──
    print('\n[1/5] 构建源域原型 ...')
    protos_dior = build_protos('dior')
    protos_voc = build_protos('voc')
    print(f'  DIOR: {len(protos_dior)}类, VOC: {len(protos_voc)}类')

    # ── 2. 加载数据 ──
    print('\n[2/5] 加载开发集和测试集数据 ...')
    dev_data = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        cd = load_class_data(name, coco_cls, src_cls, protos_dior, 'dior')
        if cd: dev_data[name] = cd
    print(f'  开发集: {len(dev_data)}类')

    test_data = {}
    for name, coco_cls, src_cls in TEST_TARGETS:
        cd = load_class_data(name, coco_cls, src_cls, protos_voc, 'voc')
        if cd: test_data[name] = cd
    test_categories = list(test_data.keys())
    print(f'  测试集: {len(test_data)}类')

    # ── 3. 开发集超参数搜索 ──
    print('\n[3/5] 开发集超参数搜索 ...')

    # 基线最优β,λ（网格搜索）
    print('  搜索基线最优β,λ ...')
    best_base_gain = -999
    best_beta, best_lam = 2.0, 0.5
    for b in np.linspace(0, 4, 9):
        for l in np.linspace(0, 2, 5):
            gains = [cd.evaluate_baseline(float(b), float(l)) for cd in dev_data.values()]
            mg = np.mean(gains)
            if mg > best_base_gain:
                best_base_gain = mg
                best_beta, best_lam = float(b), float(l)
    print(f'  基线最优: β={best_beta}, λ={best_lam}, dev_gain={best_base_gain:+.4f}')

    # 方向A：搜索alpha_A
    print('  搜索方向A alpha_A ...')
    best_A_gain = -999
    best_alpha_A = 0.5
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        gains = [cd.evaluate_optimized(best_beta, best_lam, use_A=True, alpha_A=alpha) for cd in dev_data.values()]
        mg = np.mean(gains)
        if mg > best_A_gain:
            best_A_gain = mg
            best_alpha_A = alpha
    print(f'  方向A最优: alpha_A={best_alpha_A}, dev_gain={best_A_gain:+.4f}')

    # 方向B：搜索m_B
    print('  搜索方向B m_B ...')
    best_B_gain = -999
    best_m_B = 3
    K_neg = len(protos_dior) - 1
    for m in [1, 2, 3, 5, 8, min(10, K_neg)]:
        gains = [cd.evaluate_optimized(best_beta, best_lam, use_B=True, m_B=m) for cd in dev_data.values()]
        mg = np.mean(gains)
        if mg > best_B_gain:
            best_B_gain = mg
            best_m_B = m
    print(f'  方向B最优: m_B={best_m_B}, dev_gain={best_B_gain:+.4f}')

    # 方向C：搜索w_C, tau_det, tau_vis（简化搜索）
    print('  搜索方向C w_C, tau ...')
    best_C_gain = -999
    best_w_C, best_tau_det, best_tau_vis = 0.5, 1.0, 1.0
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for td in [0.5, 1.0, 2.0]:
            for tv in [0.5, 1.0, 2.0]:
                gains = [cd.evaluate_optimized(best_beta, best_lam, use_C=True,
                                                  w_C=w, tau_det=td, tau_vis=tv) for cd in dev_data.values()]
                mg = np.mean(gains)
                if mg > best_C_gain:
                    best_C_gain = mg
                    best_w_C, best_tau_det, best_tau_vis = w, td, tv
    print(f'  方向C最优: w_C={best_w_C}, tau_det={best_tau_det}, tau_vis={best_tau_vis}, dev_gain={best_C_gain:+.4f}')

    # A+B组合搜索alpha_A和m_B（用各自的最优值作为起点）
    print('  搜索A+B组合 ...')
    best_AB_gain = -999
    best_AB_alpha, best_AB_m = best_alpha_A, best_m_B
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for m in [2, 3, 5]:
            gains = [cd.evaluate_optimized(best_beta, best_lam, use_A=True, use_B=True,
                                              alpha_A=alpha, m_B=m) for cd in dev_data.values()]
            mg = np.mean(gains)
            if mg > best_AB_gain:
                best_AB_gain = mg
                best_AB_alpha, best_AB_m = alpha, m
    print(f'  A+B最优: alpha={best_AB_alpha}, m={best_AB_m}, dev_gain={best_AB_gain:+.4f}')

    # A+B+C全组合（用各方向最优值）
    print('  评估A+B+C全组合 ...')
    gains_ABC = [cd.evaluate_optimized(best_beta, best_lam, use_A=True, use_B=True, use_C=True,
                                          alpha_A=best_AB_alpha, m_B=best_AB_m,
                                          w_C=best_w_C, tau_det=best_tau_det, tau_vis=best_tau_vis) for cd in dev_data.values()]
    print(f'  A+B+C dev_gain={np.mean(gains_ABC):+.4f}')

    # ── 4. 测试集评估 ──
    print('\n[4/5] 测试集评估（8种配置）...')

    configs = [
        ('baseline',       {'use_A': False, 'use_B': False, 'use_C': False}),
        ('A_only',         {'use_A': True,  'use_B': False, 'use_C': False, 'alpha_A': best_alpha_A}),
        ('B_only',         {'use_A': False, 'use_B': True,  'use_C': False, 'm_B': best_m_B}),
        ('C_only',         {'use_A': False, 'use_B': False, 'use_C': True,  'w_C': best_w_C, 'tau_det': best_tau_det, 'tau_vis': best_tau_vis}),
        ('A+B',            {'use_A': True,  'use_B': True,  'use_C': False, 'alpha_A': best_AB_alpha, 'm_B': best_AB_m}),
        ('A+C',            {'use_A': True,  'use_B': False, 'use_C': True,  'alpha_A': best_alpha_A, 'w_C': best_w_C, 'tau_det': best_tau_det, 'tau_vis': best_tau_vis}),
        ('B+C',            {'use_A': False, 'use_B': True,  'use_C': True,  'm_B': best_m_B, 'w_C': best_w_C, 'tau_det': best_tau_det, 'tau_vis': best_tau_vis}),
        ('A+B+C',          {'use_A': True,  'use_B': True,  'use_C': True,  'alpha_A': best_AB_alpha, 'm_B': best_AB_m, 'w_C': best_w_C, 'tau_det': best_tau_det, 'tau_vis': best_tau_vis}),
    ]

    all_results = {}
    for config_name, config_kwargs in configs:
        print(f'\n  配置: {config_name}')
        gains = []
        for name in test_categories:
            cd = test_data[name]
            gain = cd.evaluate_optimized(best_beta, best_lam, **config_kwargs)
            gains.append(gain)
        all_results[config_name] = {
            'mean_gain': float(np.mean(gains)),
            'n_positive': sum(1 for g in gains if g > 0),
            'n_total': len(gains),
            'gains': {n: float(g) for n, g in zip(test_categories, gains)},
        }
        print(f'    MEAN: {np.mean(gains):+.4f}, pos={sum(1 for g in gains if g>0)}/{len(gains)}')

    # oracle（逐类别网格搜索）
    print(f'\n  配置: oracle')
    oracle_gains = []
    for name in test_categories:
        cd = test_data[name]
        best_g = -999
        for b in np.linspace(0, 4, 9):
            for l in np.linspace(0, 2, 5):
                g = cd.evaluate_baseline(float(b), float(l))
                if g > best_g: best_g = g
        oracle_gains.append(best_g)
    all_results['oracle'] = {
        'mean_gain': float(np.mean(oracle_gains)),
        'n_positive': sum(1 for g in oracle_gains if g > 0),
        'n_total': len(oracle_gains),
        'gains': {n: float(g) for n, g in zip(test_categories, oracle_gains)},
    }
    print(f'    MEAN: {np.mean(oracle_gains):+.4f}')

    # ── 5. 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比（VOC→COCO 12类测试集）')
    print('=' * 100)

    baseline_gain = all_results['baseline']['mean_gain']
    oracle_gain = all_results['oracle']['mean_gain']
    print(f'\n基线(baseline): +{baseline_gain:.4f}')
    print(f'Oracle上界: +{oracle_gain:.4f}')
    print()
    print(f'{"配置":<15} {"均值增益":>10} {"正增益":>8} {"vs基线":>10} {"达到oracle":>12} {"相对提升":>10}')
    print('-' * 70)
    for config_name, _ in configs + [('oracle', {})]:
        r = all_results[config_name]
        vs_base = r['mean_gain'] - baseline_gain
        pct_oracle = r['mean_gain'] / oracle_gain * 100 if oracle_gain > 0 else 0
        rel_improve = vs_base / abs(baseline_gain) * 100 if baseline_gain != 0 else 0
        marker = ''
        if config_name not in ['baseline', 'oracle'] and vs_base > 0:
            marker = ' ←提升!'
        print(f'{config_name:<15} {r["mean_gain"]:>+10.4f} '
              f'{r["n_positive"]:>4d}/{r["n_total"]:<3d} {vs_base:>+10.4f} '
              f'{pct_oracle:>11.1f}% {rel_improve:>+9.1f}%{marker}')

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': f'DIOR→COCO {len(dev_data)}类',
            'test_set': f'VOC→COCO {len(test_data)}类',
            'best_beta': best_beta, 'best_lambda': best_lam,
            'best_alpha_A': best_alpha_A, 'best_m_B': best_m_B,
            'best_w_C': best_w_C, 'best_tau_det': best_tau_det, 'best_tau_vis': best_tau_vis,
            'best_AB_alpha': best_AB_alpha, 'best_AB_m': best_AB_m,
        },
        'dev_gains': {
            'baseline': best_base_gain, 'A': best_A_gain, 'B': best_B_gain, 'C': best_C_gain,
            'A+B': best_AB_gain, 'A+B+C': float(np.mean(gains_ABC)),
        },
        'results': all_results,
    }

    out_path = DATA / 'tge_v2_optimization_abc_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
