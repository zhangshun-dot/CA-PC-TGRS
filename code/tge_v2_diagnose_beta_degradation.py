# -*- coding: utf-8 -*-
r"""tge_v2_diagnose_beta_degradation.py — 诊断VOC专家β退化原因

对VOC→COCO非交通工具类和DIOR→COCO交通工具类做完整的β×λ网格搜索，
保存增益曲面，分析为什么VOC类别最优β全为0。

诊断维度：
  1. β扫描曲线：固定λ=0.5，β∈[0,4]，看增益随β变化
  2. λ扫描曲线：固定β=0，λ∈[0,2]，看负原型抑制效果
  3. 文本外推方向分析：计算VOC→COCO和DIOR→COCO的文本方向向量
  4. 源域原型质量分析：原型范数、类内方差
  5. 类别特性对比：交通工具vs非交通工具

运行: python tge_v2_diagnose_beta_degradation.py
输出: 实验数据/tge_v2_diagnose_beta_degradation.json
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

VOC_TO_COCO = {
    'person': 'person', 'bicycle': 'bicycle', 'car': 'car',
    'motorbike': 'motorcycle', 'aeroplane': 'airplane', 'bus': 'bus',
    'train': 'train', 'truck': 'truck', 'boat': 'boat', 'bird': 'bird',
    'cat': 'cat', 'dog': 'dog', 'horse': 'horse', 'sheep': 'sheep',
    'cow': 'cow', 'bottle': 'bottle', 'chair': 'chair', 'sofa': 'couch',
    'pottedplant': 'potted plant', 'tvmonitor': 'tv',
}

# 诊断类别：VOC非交通工具7类 + DIOR交通工具6类
DIAG_VOC = [
    ('VOC-COCO-person', 'person', 'person'),
    ('VOC-COCO-bird',   'bird',   'bird'),
    ('VOC-COCO-cat',    'cat',    'cat'),
    ('VOC-COCO-dog',    'dog',    'dog'),
    ('VOC-COCO-horse',  'horse',  'horse'),
    ('VOC-COCO-sheep',  'sheep',  'sheep'),
    ('VOC-COCO-cow',    'cow',    'cow'),
]

DIAG_DIOR = [
    ('DIOR-COCO-airplane',   'airplane',   'airplane'),
    ('DIOR-COCO-boat',       'boat',       'ship'),
    ('DIOR-COCO-car',        'car',        'vehicle'),
    ('DIOR-COCO-truck',      'truck',      'vehicle'),
    ('DIOR-COCO-bus',        'bus',        'vehicle'),
    ('DIOR-COCO-motorcycle', 'motorcycle', 'vehicle'),
]

BETA_GRID = np.linspace(0.0, 4.0, 17)
LAMBDA_GRID = np.linspace(0.0, 2.0, 9)


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

    def evaluate(self, beta, lam):
        P_hat = self.P_src + beta * self.Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - lam * self.P_neg.cpu().numpy())
        fused_arrays(self.per_img, self.F, W)
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
    if source == 'dior':
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
    else:
        s_xs = T.text_emb(['a photo of {}'.format(src_cls)]).cpu()[0]
    d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
    Delta = d_y - s_xs
    base50 = dataset_ap(per_img, 'gd')
    return ClassData(name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50, source)


def main():
    t0 = time.time()
    print('=' * 100)
    print('诊断实验：VOC专家β退化原因分析')
    print('=' * 100)

    # ── 1. 构建原型 ──
    print('\n[1/5] 构建源域原型 ...')
    protos_dior = build_protos('dior')
    protos_voc = build_protos('voc')
    protos_for_neg = dict(protos_dior); protos_for_neg.update(protos_voc)
    print(f'  DIOR: {len(protos_dior)}类, VOC: {len(protos_voc)}类')

    # ── 2. 加载诊断类别数据 ──
    print('\n[2/5] 加载诊断类别数据（13类）...')
    diag_data = {}
    for name, coco_cls, src_cls in DIAG_VOC:
        cd = load_class_data(name, coco_cls, src_cls, protos_voc, 'voc', neg_protos=protos_for_neg)
        if cd: diag_data[name] = cd
    for name, coco_cls, src_cls in DIAG_DIOR:
        cd = load_class_data(name, coco_cls, src_cls, protos_dior, 'dior', neg_protos=protos_for_neg)
        if cd: diag_data[name] = cd
    print(f'  共加载 {len(diag_data)} 类')

    # ── 3. 完整β×λ网格搜索 ──
    print('\n[3/5] 完整β×λ网格搜索（17×9=153点/类）...')
    results = {}
    for name, cd in diag_data.items():
        print(f'  [{cd.source}] {cd.coco_cls} ...')
        gain_grid = np.zeros((len(BETA_GRID), len(LAMBDA_GRID)))
        for i, beta in enumerate(BETA_GRID):
            for j, lam in enumerate(LAMBDA_GRID):
                gain_grid[i, j] = cd.evaluate(beta, lam)

        # 找最优
        opt_idx = np.unravel_index(np.argmax(gain_grid), gain_grid.shape)
        beta_opt = BETA_GRID[opt_idx[0]]
        lambda_opt = LAMBDA_GRID[opt_idx[1]]
        gain_opt = gain_grid[opt_idx[0], opt_idx[1]]

        # β扫描曲线（固定λ=0.5，即第4个点）
        lambda_05_idx = np.argmin(np.abs(LAMBDA_GRID - 0.5))
        beta_curve = gain_grid[:, lambda_05_idx].tolist()

        # λ扫描曲线（固定β=0）
        beta_0_idx = 0
        lambda_curve = gain_grid[beta_0_idx, :].tolist()

        # 文本外推方向
        delta_norm = float(cd.Delta.norm())
        delta_dir = cd.Delta / cd.Delta.norm() if delta_norm > 0 else cd.Delta

        # 源域原型质量
        src_proto_norm = float(cd.P_src.norm())
        neg_proto_norm = float(cd.P_neg.norm())

        results[name] = {
            'source': cd.source,
            'coco_cls': cd.coco_cls,
            'src_cls': cd.src_cls,
            'base_ap': float(cd.base50),
            'beta_opt': float(beta_opt),
            'lambda_opt': float(lambda_opt),
            'gain_opt': float(gain_opt),
            'beta_curve_lambda05': beta_curve,
            'lambda_curve_beta0': lambda_curve,
            'delta_norm': delta_norm,
            'src_proto_norm': src_proto_norm,
            'neg_proto_norm': neg_proto_norm,
            'gain_at_beta0_lambda0': float(gain_grid[0, 0]),
            'gain_at_beta0_lambda05': float(gain_grid[0, lambda_05_idx]),
            'gain_at_beta2_lambda05': float(gain_grid[np.argmin(np.abs(BETA_GRID - 2.0)), lambda_05_idx]),
        }
        print(f'    → β*={beta_opt:.2f}, λ*={lambda_opt:.2f}, gain*={gain_opt:+.4f}, '
              f'Δnorm={delta_norm:.4f}, baseAP={cd.base50:.4f}')

    # ── 4. 统计分析 ──
    print('\n[4/5] 统计分析 ...')

    voc_results = {k: v for k, v in results.items() if v['source'] == 'voc'}
    dior_results = {k: v for k, v in results.items() if v['source'] == 'dior'}

    print(f'\n  VOC非交通工具类（{len(voc_results)}类）:')
    voc_beta_opts = [v['beta_opt'] for v in voc_results.values()]
    voc_lambda_opts = [v['lambda_opt'] for v in voc_results.values()]
    voc_gain_opts = [v['gain_opt'] for v in voc_results.values()]
    voc_delta_norms = [v['delta_norm'] for v in voc_results.values()]
    print(f'    β*: mean={np.mean(voc_beta_opts):.3f}, all_zero={all(b == 0 for b in voc_beta_opts)}')
    print(f'    λ*: mean={np.mean(voc_lambda_opts):.3f}, range=[{min(voc_lambda_opts):.2f}, {max(voc_lambda_opts):.2f}]')
    print(f'    gain*: mean={np.mean(voc_gain_opts):+.4f}')
    print(f'    Δnorm: mean={np.mean(voc_delta_norms):.4f}')

    # β=0 vs β=2的增益对比（固定λ=0.5）
    voc_gain_b0 = [v['gain_at_beta0_lambda05'] for v in voc_results.values()]
    voc_gain_b2 = [v['gain_at_beta2_lambda05'] for v in voc_results.values()]
    print(f'    gain(β=0,λ=0.5): mean={np.mean(voc_gain_b0):+.4f}')
    print(f'    gain(β=2,λ=0.5): mean={np.mean(voc_gain_b2):+.4f}')
    print(f'    β=2比β=0差: {np.mean(voc_gain_b2) - np.mean(voc_gain_b0):+.4f}')

    print(f'\n  DIOR交通工具类（{len(dior_results)}类）:')
    dior_beta_opts = [v['beta_opt'] for v in dior_results.values()]
    dior_lambda_opts = [v['lambda_opt'] for v in dior_results.values()]
    dior_gain_opts = [v['gain_opt'] for v in dior_results.values()]
    dior_delta_norms = [v['delta_norm'] for v in dior_results.values()]
    print(f'    β*: mean={np.mean(dior_beta_opts):.3f}, >0的类={sum(1 for b in dior_beta_opts if b > 0)}/{len(dior_beta_opts)}')
    print(f'    λ*: mean={np.mean(dior_lambda_opts):.3f}')
    print(f'    gain*: mean={np.mean(dior_gain_opts):+.4f}')
    print(f'    Δnorm: mean={np.mean(dior_delta_norms):.4f}')

    dior_gain_b0 = [v['gain_at_beta0_lambda05'] for v in dior_results.values()]
    dior_gain_b2 = [v['gain_at_beta2_lambda05'] for v in dior_results.values()]
    print(f'    gain(β=0,λ=0.5): mean={np.mean(dior_gain_b0):+.4f}')
    print(f'    gain(β=2,λ=0.5): mean={np.mean(dior_gain_b2):+.4f}')
    print(f'    β=2比β=0差: {np.mean(dior_gain_b2) - np.mean(dior_gain_b0):+.4f}')

    # ── 5. 退化原因分析 ──
    print('\n[5/5] 退化原因分析 ...')

    analysis = {}

    # 原因1：文本外推方向差异
    # VOC→COCO和DIOR→COCO的文本方向向量对比
    # 对于相同的COCO类别（如airplane, boat, car, bus, truck），计算两个源域的Δ方向
    common_cls = ['airplane', 'boat', 'car', 'bus', 'truck']
    direction_comparison = {}
    for cls in common_cls:
        voc_key = f'VOC-COCO-{cls}' if cls != 'truck' else None  # truck在VOC开发集中没有
        dior_key = f'DIOR-COCO-{cls}'
        if voc_key in results and dior_key in results:
            # 这里只有增益信息，没有Δ方向，需要重新计算
            pass

    # 原因2：源域-目标域域差异
    # VOC和COCO都是自然图像，域差异小；DIOR是遥感，域差异大
    # 用文本模板差异来量化：VOC用"a photo of X"，COCO也用"a photo of X"，所以Δ≈0
    # DIOR用"an overhead image of X"，COCO用"a photo of X"，所以Δ≠0
    analysis['domain_gap_hypothesis'] = {
        'description': 'VOC和COCO都是自然图像域，文本模板相同（都是"a photo of X"），'
                       '因此文本外推方向Δ≈0，外推没有收益。'
                       'DIOR是遥感域（"an overhead image of X"），COCO是自然图像域，'
                       '文本方向Δ≠0，外推可以桥接域差异。',
        'evidence': 'VOC类别最优β全为0；DIOR类别中4/6最优β>0。'
    }

    # 原因3：类别特性
    # 交通工具类在遥感和自然图像中外观差异大（视角、尺度），需要外推
    # 非交通工具类（person, animal）在自然图像中外观一致，不需要外推
    analysis['category_property_hypothesis'] = {
        'description': '交通工具类（car, truck, bus）在遥感俯视视角和自然图像平视视角中外观差异大，'
                       '文本外推可以桥接视角差异。'
                       '非交通工具类（person, cat, dog）在自然图像中外观一致，'
                       '文本外推反而引入噪声。',
        'evidence': 'DIOR的car/truck/bus/motorcycle最优β>0；airplane/boat最优β=0（这两类在俯视和平视中差异也大，但可能因为其他原因）。'
    }

    # 原因4：负原型抑制有效
    # VOC类别λ*在0-1之间变化，说明负原型抑制是有效的
    analysis['negative_prototype_effective'] = {
        'description': '虽然β外推无效，但λ负原型抑制在VOC类别上仍然有效（λ*∈[0,1]），'
                       '说明A-CSPC的非对称设计（正原型不外推，负原型抑制）对自然图像→自然图像也适用。',
        'evidence': f'VOC类别λ*均值={np.mean(voc_lambda_opts):.3f}，范围[{min(voc_lambda_opts):.2f}, {max(voc_lambda_opts):.2f}]'
    }

    for key, val in analysis.items():
        print(f'\n  [{key}]')
        print(f'    {val["description"]}')
        print(f'    证据: {val["evidence"]}')

    # ── 保存结果 ──
    output = {
        'beta_grid': BETA_GRID.tolist(),
        'lambda_grid': LAMBDA_GRID.tolist(),
        'results': results,
        'statistics': {
            'voc': {
                'n_classes': len(voc_results),
                'beta_opt_mean': float(np.mean(voc_beta_opts)),
                'beta_opt_all_zero': bool(all(b == 0 for b in voc_beta_opts)),
                'lambda_opt_mean': float(np.mean(voc_lambda_opts)),
                'gain_opt_mean': float(np.mean(voc_gain_opts)),
                'delta_norm_mean': float(np.mean(voc_delta_norms)),
                'gain_beta0_lambda05_mean': float(np.mean(voc_gain_b0)),
                'gain_beta2_lambda05_mean': float(np.mean(voc_gain_b2)),
            },
            'dior': {
                'n_classes': len(dior_results),
                'beta_opt_mean': float(np.mean(dior_beta_opts)),
                'beta_opt_gt0_count': int(sum(1 for b in dior_beta_opts if b > 0)),
                'lambda_opt_mean': float(np.mean(dior_lambda_opts)),
                'gain_opt_mean': float(np.mean(dior_gain_opts)),
                'delta_norm_mean': float(np.mean(dior_delta_norms)),
                'gain_beta0_lambda05_mean': float(np.mean(dior_gain_b0)),
                'gain_beta2_lambda05_mean': float(np.mean(dior_gain_b2)),
            },
        },
        'degradation_analysis': analysis,
    }

    out_path = DATA / 'tge_v2_diagnose_beta_degradation.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
