# -*- coding: utf-8 -*-
r"""tge_v2_optimization_adaptive_fusion.py — 逐框自适应融合权重消融实验

对比三种融合策略：
  1. baseline：几何平均 sqrt(s_det * s_vis)
  2. C_fixed：温度校准 + 固定加权融合（w全局固定）
  3. C_adaptive：温度校准 + 逐框自适应加权融合（w_k = sigmoid(a*s_det + b*sim_vis + c)）
  4. oracle

在两个测试集上评估：
  - VOC→COCO 12类
  - DIOR→COCO held-out 6类（skip=60）

运行: python tge_v2_optimization_adaptive_fusion.py
输出: 实验数据/tge_v2_optimization_adaptive_fusion_results.json
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
HELDOUT_SKIP = 60

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

TEST_VOC = [
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

def load_coco_split(cls_name, split='dev', skip=60):
    images, gts = U.load_coco(cls_name, n=100000)
    if split == 'dev':
        return images[:skip], gts[:skip]
    else:
        return images[skip:], gts[skip:]

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

def dataset_ap(per_img, key, use_kcalib=True):
    if use_kcalib and len(per_img) > K_CALIB:
        sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    else:
        sel = per_img
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    return U.compute_ap50_img(preds, gts)


class ClassData:
    def __init__(self, name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50):
        self.name = name; self.coco_cls = coco_cls; self.src_cls = src_cls
        self.per_img = per_img; self.F = F; self.P_src = P_src
        self.P_neg = P_neg; self.Delta = Delta; self.base50 = base50

    def _compute_vis(self, beta, lam):
        """计算所有候选框的视觉相似度和正原型相似度"""
        P_src_np = self.P_src.cpu().numpy()
        Delta_np = self.Delta.cpu().numpy()
        P_neg_np = self.P_neg.cpu().numpy()

        P_hat = P_src_np + beta * Delta_np
        P_hat = P_hat / (np.linalg.norm(P_hat) + 1e-8)

        all_f = []
        all_gd = []
        for p in self.per_img:
            if len(p['gd']) > 0:
                all_f.append(self.F[p['crop_idx']])
                all_gd.append(p['gd'])
        if not all_f:
            return None, None, None, None
        F_all = np.concatenate(all_f, axis=0)
        gd_all = np.concatenate(all_gd, axis=0)

        sim_pos = F_all @ P_hat  # 与正原型相似度
        sim_neg = F_all @ P_neg_np  # 与负原型相似度
        vis = np.maximum(0.0, sim_pos - lam * sim_neg)

        return F_all, gd_all, vis, sim_pos

    def evaluate_baseline(self, beta, lam, use_kcalib=True):
        """基线：几何平均融合"""
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
        return dataset_ap(self.per_img, 'fused', use_kcalib) - self.base50

    def evaluate_fixed_fusion(self, beta, lam, w, tau_det, tau_vis, use_kcalib=True):
        """C_fixed：温度校准 + 固定加权融合"""
        P_hat = self.P_src + beta * self.Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - lam * self.P_neg.cpu().numpy())
        for p in self.per_img:
            n = len(p['gd'])
            if n:
                vis = np.maximum(0.0, self.F[p['crop_idx']] @ W)
                gd_calib = np.clip(p['gd'] ** (1.0 / tau_det), 1e-6, 1.0)
                vis_calib = np.clip(vis ** (1.0 / tau_vis), 1e-6, 1.0)
                p['fused'] = w * gd_calib + (1 - w) * vis_calib
            else:
                p['fused'] = np.zeros(0)
        return dataset_ap(self.per_img, 'fused', use_kcalib) - self.base50

    def evaluate_adaptive_fusion(self, beta, lam, a, b, c, tau_det, tau_vis, use_kcalib=True):
        """C_adaptive：温度校准 + 逐框自适应加权融合
        w_k = sigmoid(a * s_det_k + b * sim_pos_k + c)
        """
        P_hat = self.P_src + beta * self.Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        P_hat_np = P_hat.cpu().numpy()
        P_neg_np = self.P_neg.cpu().numpy()

        for p in self.per_img:
            n = len(p['gd'])
            if n:
                f_k = self.F[p['crop_idx']]
                sim_pos = f_k @ P_hat_np
                sim_neg = f_k @ P_neg_np
                vis = np.maximum(0.0, sim_pos - lam * sim_neg)

                # 逐框自适应权重
                logit = a * p['gd'] + b * sim_pos + c
                w_k = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))

                gd_calib = np.clip(p['gd'] ** (1.0 / tau_det), 1e-6, 1.0)
                vis_calib = np.clip(vis ** (1.0 / tau_vis), 1e-6, 1.0)
                p['fused'] = w_k * gd_calib + (1 - w_k) * vis_calib
            else:
                p['fused'] = np.zeros(0)
        return dataset_ap(self.per_img, 'fused', use_kcalib) - self.base50


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

def load_class_data(name, coco_cls, src_cls, protos, source, split='all'):
    if split == 'dev':
        images, gts = load_coco_split(coco_cls, split='dev', skip=HELDOUT_SKIP)
    elif split == 'test':
        images, gts = load_coco_split(coco_cls, split='test', skip=HELDOUT_SKIP)
    else:
        images, gts = load_coco_all(coco_cls)
    if len(images) < 5:
        return None
    dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), coco_cls)
    if split == 'dev':
        dets = dets[:HELDOUT_SKIP]
    elif split == 'test':
        dets = dets[HELDOUT_SKIP:]
    per_img, F = collect_features(images, gts, dets)
    if src_cls not in protos:
        return None
    P_src = protos[src_cls].to(U.DEVICE)
    P_neg = T.get_neg_proto(protos, src_cls)
    if source == 'dior':
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
    else:
        s_xs = T.text_emb(['a photo of {}'.format(src_cls)]).cpu()[0]
    d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
    Delta = d_y - s_xs
    use_kcalib = (split != 'test')  # test集不跳过K_CALIB
    base50 = dataset_ap(per_img, 'gd', use_kcalib)
    cd = ClassData(name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50)
    cd.use_kcalib = use_kcalib
    return cd


def main():
    t0 = time.time()
    print('=' * 100)
    print('逐框自适应融合权重消融实验：baseline vs C_fixed vs C_adaptive')
    print('=' * 100)

    # ── 1. 构建原型 ──
    print('\n[1/6] 构建源域原型 ...')
    protos_dior = build_protos('dior')
    protos_voc = build_protos('voc')
    print(f'  DIOR: {len(protos_dior)}类, VOC: {len(protos_voc)}类')

    # ── 2. 加载数据 ──
    print('\n[2/6] 加载开发集和测试集数据 ...')
    dev_data = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        cd = load_class_data(name, coco_cls, src_cls, protos_dior, 'dior', split='dev')
        if cd: dev_data[name] = cd
    print(f'  开发集: {len(dev_data)}类 (DIOR→COCO 前{HELDOUT_SKIP}张)')

    test_voc = {}
    for name, coco_cls, src_cls in TEST_VOC:
        cd = load_class_data(name, coco_cls, src_cls, protos_voc, 'voc', split='all')
        if cd: test_voc[name] = cd
    print(f'  测试集1 (VOC→COCO): {len(test_voc)}类')

    test_dior = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        cd = load_class_data(name + '_test', coco_cls, src_cls, protos_dior, 'dior', split='test')
        if cd: test_dior[name] = cd
    print(f'  测试集2 (DIOR→COCO held-out): {len(test_dior)}类')

    # ── 3. 开发集超参数搜索 ──
    print('\n[3/6] 开发集超参数搜索 ...')

    # 基线最优β,λ
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

    # C_fixed：搜索w, tau_det, tau_vis
    print('  搜索C_fixed最优w,τ ...')
    best_fixed_gain = -999
    best_w, best_td, best_tv = 0.3, 0.5, 0.5
    for w in [0.2, 0.3, 0.4, 0.5]:
        for td in [0.5, 1.0, 2.0]:
            for tv in [0.5, 1.0, 2.0]:
                gains = [cd.evaluate_fixed_fusion(best_beta, best_lam, w, td, tv) for cd in dev_data.values()]
                mg = np.mean(gains)
                if mg > best_fixed_gain:
                    best_fixed_gain = mg
                    best_w, best_td, best_tv = w, td, tv
    print(f'  C_fixed最优: w={best_w}, τ_det={best_td}, τ_vis={best_tv}, dev_gain={best_fixed_gain:+.4f}')

    # C_adaptive：搜索a, b, c（τ用fixed的最优值）
    print('  搜索C_adaptive最优a,b,c ...')
    best_adapt_gain = -999
    best_a, best_b, best_c = 0.0, 0.0, 0.0
    for a in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for b in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            for c in [-1.0, 0.0, 1.0]:
                gains = [cd.evaluate_adaptive_fusion(best_beta, best_lam, a, b, c, best_td, best_tv) for cd in dev_data.values()]
                mg = np.mean(gains)
                if mg > best_adapt_gain:
                    best_adapt_gain = mg
                    best_a, best_b, best_c = a, b, c
    print(f'  C_adaptive最优: a={best_a}, b={best_b}, c={best_c}, dev_gain={best_adapt_gain:+.4f}')

    # ── 4. 测试集1评估（VOC→COCO）──
    print('\n[4/6] 测试集1评估（VOC→COCO 12类）...')
    results_voc = {}
    for config_name, eval_fn in [
        ('baseline', lambda cd: cd.evaluate_baseline(best_beta, best_lam)),
        ('C_fixed', lambda cd: cd.evaluate_fixed_fusion(best_beta, best_lam, best_w, best_td, best_tv)),
        ('C_adaptive', lambda cd: cd.evaluate_adaptive_fusion(best_beta, best_lam, best_a, best_b, best_c, best_td, best_tv)),
    ]:
        gains = [eval_fn(cd) for cd in test_voc.values()]
        results_voc[config_name] = {
            'mean_gain': float(np.mean(gains)),
            'n_positive': sum(1 for g in gains if g > 0),
            'n_total': len(gains),
            'gains': {n: float(g) for n, g in zip(test_voc.keys(), gains)},
        }
        print(f'  {config_name}: {np.mean(gains):+.4f}, pos={sum(1 for g in gains if g>0)}/{len(gains)}')

    # oracle
    oracle_gains = []
    for cd in test_voc.values():
        best_g = -999
        for b in np.linspace(0, 4, 9):
            for l in np.linspace(0, 2, 5):
                g = cd.evaluate_baseline(float(b), float(l))
                if g > best_g: best_g = g
        oracle_gains.append(best_g)
    results_voc['oracle'] = {
        'mean_gain': float(np.mean(oracle_gains)),
        'n_positive': sum(1 for g in oracle_gains if g > 0),
        'n_total': len(oracle_gains),
    }
    print(f'  oracle: {np.mean(oracle_gains):+.4f}')

    # ── 5. 测试集2评估（DIOR→COCO held-out）──
    print('\n[5/6] 测试集2评估（DIOR→COCO held-out 6类）...')
    results_dior = {}
    for config_name, eval_fn in [
        ('baseline', lambda cd: cd.evaluate_baseline(best_beta, best_lam, use_kcalib=False)),
        ('C_fixed', lambda cd: cd.evaluate_fixed_fusion(best_beta, best_lam, best_w, best_td, best_tv, use_kcalib=False)),
        ('C_adaptive', lambda cd: cd.evaluate_adaptive_fusion(best_beta, best_lam, best_a, best_b, best_c, best_td, best_tv, use_kcalib=False)),
    ]:
        gains = [eval_fn(cd) for cd in test_dior.values()]
        results_dior[config_name] = {
            'mean_gain': float(np.mean(gains)),
            'n_positive': sum(1 for g in gains if g > 0),
            'n_total': len(gains),
            'gains': {n: float(g) for n, g in zip(test_dior.keys(), gains)},
        }
        print(f'  {config_name}: {np.mean(gains):+.4f}, pos={sum(1 for g in gains if g>0)}/{len(gains)}')
        for n, g in zip(test_dior.keys(), gains):
            print(f'    {n}: {g:+.4f}')

    # oracle
    oracle_gains = []
    for cd in test_dior.values():
        best_g = -999
        for b in np.linspace(0, 4, 9):
            for l in np.linspace(0, 2, 5):
                g = cd.evaluate_baseline(float(b), float(l), use_kcalib=False)
                if g > best_g: best_g = g
        oracle_gains.append(best_g)
    results_dior['oracle'] = {
        'mean_gain': float(np.mean(oracle_gains)),
        'n_positive': sum(1 for g in oracle_gains if g > 0),
        'n_total': len(oracle_gains),
    }
    print(f'  oracle: {np.mean(oracle_gains):+.4f}')

    # ── 6. 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比')
    print('=' * 100)

    for test_name, results in [('VOC→COCO (12类)', results_voc), ('DIOR→COCO held-out (6类)', results_dior)]:
        print(f'\n--- {test_name} ---')
        baseline_gain = results['baseline']['mean_gain']
        oracle_gain = results['oracle']['mean_gain']
        print(f'  baseline: +{baseline_gain:.4f}, oracle: +{oracle_gain:.4f}')
        print(f'  {"配置":<15} {"均值增益":>10} {"正增益":>8} {"vs基线":>10} {"达到oracle":>12} {"相对提升":>10}')
        print('  ' + '-' * 65)
        for config_name in ['baseline', 'C_fixed', 'C_adaptive', 'oracle']:
            r = results[config_name]
            vs_base = r['mean_gain'] - baseline_gain
            pct_oracle = r['mean_gain'] / oracle_gain * 100 if oracle_gain > 0 else 0
            rel_improve = vs_base / abs(baseline_gain) * 100 if baseline_gain != 0 else 0
            marker = ''
            if config_name not in ['baseline', 'oracle'] and vs_base > 0:
                marker = ' ←提升!'
            print(f'  {config_name:<15} {r["mean_gain"]:>+10.4f} '
                  f'{r["n_positive"]:>4d}/{r["n_total"]:<3d} {vs_base:>+10.4f} '
                  f'{pct_oracle:>11.1f}% {rel_improve:>+9.1f}%{marker}')

    # 跨数据集一致性
    print('\n' + '=' * 100)
    print('跨数据集一致性对比')
    print('=' * 100)
    print(f'  {"配置":<15} {"VOC→COCO":>12} {"DIOR→COCO":>14} {"都提升?":>10}')
    print('  ' + '-' * 55)
    for config_name in ['C_fixed', 'C_adaptive']:
        voc_gain = results_voc[config_name]['mean_gain']
        dior_gain = results_dior[config_name]['mean_gain']
        voc_vs_base = voc_gain - results_voc['baseline']['mean_gain']
        dior_vs_base = dior_gain - results_dior['baseline']['mean_gain']
        both_up = '✓' if (voc_vs_base > 0 and dior_vs_base > 0) else '✗'
        print(f'  {config_name:<15} {voc_gain:>+12.4f} {dior_gain:>+14.4f} {both_up:>10}')

    # ── 保存结果 ──
    output = {
        'config': {
            'best_beta': best_beta, 'best_lambda': best_lam,
            'best_w': best_w, 'best_tau_det': best_td, 'best_tau_vis': best_tv,
            'best_a': best_a, 'best_b': best_b, 'best_c': best_c,
            'heldout_skip': HELDOUT_SKIP,
        },
        'dev_gains': {
            'baseline': best_base_gain,
            'C_fixed': best_fixed_gain,
            'C_adaptive': best_adapt_gain,
        },
        'results_voc': results_voc,
        'results_dior': results_dior,
    }

    out_path = DATA / 'tge_v2_optimization_adaptive_fusion_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
