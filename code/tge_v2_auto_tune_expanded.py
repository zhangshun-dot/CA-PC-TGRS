# -*- coding: utf-8 -*-
r"""tge_v2_auto_tune_expanded.py — 扩大开发集的自动调参实验

在 tge_v2_auto_tune.py 基础上，将开发集从 DIOR→COCO 6类 扩大到
DIOR→COCO 6类 + NWPU→COCO 5类 = 11类，验证更大开发集是否能提升
自动调参映射的泛化性。

测试集保持 VOC→COCO 20类 不变。

开发集类别：
  DIOR→COCO: airplane, boat(ship), car(vehicle), truck(vehicle), bus(vehicle), motorcycle(vehicle)
  NWPU→COCO: airplane, boat(ship), car(vehicle), truck(vehicle), bus(vehicle)

运行: python tge_v2_auto_tune_expanded.py
输出: 实验数据/tge_v2_auto_tune_expanded_results.json
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
import unified_eval as U
import tge_probe as T

from acspc.auto_tuner import AutoTuner, ClassOptimal
from acspc.coco_categories import get_offline_prior
from acspc.appg import PhysicsPrior

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

K_CALIB = 10
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
PRE_FILTER = 0.03

# ── VOC 配置 ──
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

# ── NWPU 配置 ──
NWPU_ROOT = r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10'
NWPU_IMG_DIR = os.path.join(NWPU_ROOT, 'positive_image_set')
NWPU_ANN_DIR = os.path.join(NWPU_ROOT, 'ground_truth')

NWPU_ID2NAME = {
    1: 'airplane', 2: 'ship', 3: 'storage tank', 4: 'baseball diamond',
    5: 'tennis court', 6: 'basketball court', 7: 'ground track field',
    8: 'harbor', 9: 'bridge', 10: 'vehicle',
}

# ── 开发集：DIOR 6类 + NWPU 5类 = 11类 ──
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

# ── 测试集：VOC→COCO 20类（保持不变）──
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


# ── 基础工具函数 ──

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


def v2_cached_dets(images, prompt, name, detector='gdino-tiny'):
    p = DATA / f'tge_v2_dets_COCO-{name}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def collect_features(images, gts, dets):
    per_img = []
    crops_all = []
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


# ── 类别数据容器 ──

class ClassData:
    def __init__(self, name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50, source='dior'):
        self.name = name
        self.coco_cls = coco_cls
        self.src_cls = src_cls
        self.per_img = per_img
        self.F = F
        self.P_src = P_src
        self.P_neg = P_neg
        self.Delta = Delta
        self.base50 = base50
        self.source = source

    def evaluate(self, beta, lam):
        P_hat = self.P_src + beta * self.Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - lam * self.P_neg.cpu().numpy())
        fused_arrays(self.per_img, self.F, W)
        ap50 = dataset_ap(self.per_img, 'fused')
        return ap50 - self.base50


# ── NWPU 加载函数 ──

def load_nwpu_class(nwpu_class, n_max=100):
    """加载NWPU某个类别的图像和标注框。
    nwpu_class: 类别名称，如 'airplane', 'ship', 'vehicle'
    """
    class_id = None
    for cid, cname in NWPU_ID2NAME.items():
        if cname == nwpu_class:
            class_id = cid
            break
    if class_id is None:
        return [], []

    images, gts = [], []
    ann_files = sorted(glob.glob(os.path.join(NWPU_ANN_DIR, '*.txt')))
    count = 0
    for ann_file in ann_files:
        if count >= n_max:
            break
        try:
            boxes = []
            with open(ann_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 格式: (x1,y1),(x2,y2),class_id
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
                    images.append(img)
                    gts.append(boxes)
                    count += 1
        except Exception as e:
            continue
    return images, gts


def build_nwpu_protos():
    """构建NWPU源域原型库（只构建需要的类别）。"""
    print('[NWPU 源域 CLIP 原型]')
    protos = {}
    needed_classes = ['airplane', 'ship', 'vehicle']
    for cls in needed_classes:
        images, gts = load_nwpu_class(cls, n_max=50)
        if not images:
            print(f'  {cls}: 无图像，跳过')
            continue
        crops = gt_crops(images, gts)
        p = clip_proto(crops)
        if p is not None:
            protos[cls] = p
        print(f'  {cls:<14s}: {len(images)} 图, {len(crops)} 框')
    print(f'  原型池 {len(protos)} 类')
    return protos


# ── VOC 加载函数 ──

def load_voc_class(voc_class, n_max=100):
    images, gts = [], []
    ann_files = sorted(glob.glob(os.path.join(VOC_ANN_DIR, '*.xml')))
    count = 0
    for ann_file in ann_files:
        if count >= n_max:
            break
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            boxes = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name == voc_class:
                    bndbox = obj.find('bndbox')
                    x1 = float(bndbox.find('xmin').text)
                    y1 = float(bndbox.find('ymin').text)
                    x2 = float(bndbox.find('xmax').text)
                    y2 = float(bndbox.find('ymax').text)
                    boxes.append([x1, y1, x2, y2])
            if boxes:
                img_file = os.path.join(VOC_IMG_DIR, root.find('filename').text)
                if os.path.exists(img_file):
                    img = Image.open(img_file).convert('RGB')
                    images.append(img)
                    gts.append(boxes)
                    count += 1
        except Exception:
            continue
    return images, gts


def build_voc_protos():
    print('[VOC 20类 CLIP 源原型]')
    protos = {}
    for voc_cls in VOC_TO_COCO.keys():
        images, gts = load_voc_class(voc_cls, n_max=50)
        if not images:
            print(f'  {voc_cls}: 无图像，跳过')
            continue
        crops = gt_crops(images, gts)
        p = clip_proto(crops)
        if p is not None:
            protos[voc_cls] = p
        print(f'  {voc_cls:<14s}: {len(images)} 图, {len(crops)} 框')
    print(f'  原型池 {len(protos)} 类')
    return protos


# ── 源域原型构建 ──

def build_source_prototypes(source='dior'):
    if source == 'dior':
        protos = {}
        for cls in U.DIOR_ID2NAME.values():
            images, gts = U.load_dior(cls)
            p = clip_proto(gt_crops(images, gts))
            if p is not None:
                protos[cls] = p
        return protos
    elif source == 'nwpu':
        return build_nwpu_protos()
    elif source == 'voc':
        return build_voc_protos()
    return {}


def load_class_data(name, coco_cls, src_cls, protos, source='dior'):
    images, gts = load_coco_all(coco_cls)
    if len(images) < K_CALIB + 10:
        return None
    dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
    per_img, F = collect_features(images, gts, dets)

    if src_cls not in protos:
        print(f'  !! {src_cls} not in protos, skipping {name}')
        return None

    P_src = protos[src_cls].to(U.DEVICE)
    P_neg = T.get_neg_proto(protos, src_cls)

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


# ── 主实验 ──

def main():
    t0 = time.time()
    print('=' * 100)
    print('A-CSPC AutoTuner Expanded: 扩大开发集（DIOR 6类 + NWPU 5类 = 11类）')
    print('=' * 100)

    # ── 1. 构建源域原型 ──
    print('\n[1/6] 构建 DIOR 源域原型 ...')
    protos_dior = build_source_prototypes('dior')
    print(f'  DIOR protos: {len(protos_dior)} classes')

    print('\n[1/6] 构建 NWPU 源域原型 ...')
    protos_nwpu = build_source_prototypes('nwpu')
    print(f'  NWPU protos: {len(protos_nwpu)} classes')

    print('\n[1/6] 构建 VOC 源域原型（测试集用）...')
    protos_voc = build_source_prototypes('voc')
    print(f'  VOC protos: {len(protos_voc)} classes')

    # ── 2. 加载开发集数据 ──
    print('\n[2/6] 加载开发集数据（DIOR→COCO 6类）...')
    dev_data = {}
    for name, coco_cls, src_cls in DEV_TARGETS_DIOR:
        cd = load_class_data(name, coco_cls, src_cls, protos_dior, 'dior')
        if cd is not None:
            dev_data[name] = cd
            print(f'  {name:<22} base={cd.base50:.4f} 框={sum(len(p["gd"]) for p in cd.per_img)}')

    print('\n[2/6] 加载开发集数据（NWPU→COCO 5类）...')
    # 合并DIOR+NWPU原型池用于计算负原型（NWPU只有3类，topk(3)会越界）
    protos_merged = dict(protos_dior)
    protos_merged.update(protos_nwpu)
    for name, coco_cls, src_cls in DEV_TARGETS_NWPU:
        cd = load_class_data(name, coco_cls, src_cls, protos_merged, 'nwpu')
        if cd is not None:
            dev_data[name] = cd
            print(f'  {name:<22} base={cd.base50:.4f} 框={sum(len(p["gd"]) for p in cd.per_img)}')
    print(f'  开发集总计: {len(dev_data)} classes')

    # ── 3. 加载测试集数据 ──
    print('\n[3/6] 加载测试集数据（VOC→COCO 20类）...')
    test_data = {}
    for name, coco_cls, src_cls in TEST_TARGETS:
        cd = load_class_data(name, coco_cls, src_cls, protos_voc, 'voc')
        if cd is not None:
            test_data[name] = cd
            print(f'  {name:<22} base={cd.base50:.4f} 框={sum(len(p["gd"]) for p in cd.per_img)}')
    print(f'  测试集: {len(test_data)} classes')

    # ── 4. 开发集网格搜索 + 映射训练 ──
    print('\n[4/6] 开发集网格搜索 + 映射训练 ...')

    dev_categories = list(dev_data.keys())
    dev_risks = [get_risk(dev_data[n].coco_cls) for n in dev_categories]

    def make_eval_fn(cat_name):
        cd = dev_data[cat_name]
        return lambda beta, lam: cd.evaluate(beta, lam)

    tuner = AutoTuner(
        beta_range=(0.0, 4.0),
        lambda_range=(0.0, 2.0),
        n_beta=17,
        n_lambda=9,
        polynomial_degree=2,
        mlp_hidden=8,
    )

    print('\n  --- 开发集网格搜索 (11类) ---')
    optimals = tuner.search_development_set(dev_categories, dev_risks, make_eval_fn)

    print('\n  --- 映射训练 ---')
    mappers = tuner.train_mappers()

    # ── 5. 测试集评估 ──
    print('\n[5/6] 测试集评估（VOC→COCO 20类）...')

    test_categories = list(test_data.keys())
    test_risks = [get_risk(test_data[n].coco_cls) for n in test_categories]

    results = {}
    methods = ['fixed', 'alpha_scan', 'auto_linear', 'auto_polynomial', 'auto_mlp', 'oracle']

    for method in methods:
        print(f'\n  --- {method} ---')
        gains = []
        params_log = {}
        for name in test_categories:
            cd = test_data[name]
            risk = get_risk(cd.coco_cls)
            if method == 'fixed':
                beta, lam = 1.0, 0.25
            elif method == 'alpha_scan':
                beta_eff = 1.0 * (1 + 1.5 * risk)
                lambda_eff = 0.25 * (1 + 1.5 * risk)
                beta = min(max(beta_eff, 0), 6)
                lam = min(max(lambda_eff, 0), 2)
            elif method in ('auto_linear', 'auto_polynomial', 'auto_mlp'):
                mapper_key = method.replace('auto_', '')
                beta, lam = tuner.predict(mapper_key, risk)
            elif method == 'oracle':
                best_gain = -1e9
                beta, lam = 1.0, 0.25
                for b in np.linspace(0, 4, 17):
                    for l in np.linspace(0, 2, 9):
                        g = cd.evaluate(b, l)
                        if g > best_gain:
                            best_gain = g
                            beta, lam = b, l
            gain = cd.evaluate(beta, lam)
            gains.append(gain)
            params_log[name] = {'risk': float(risk), 'beta': float(beta), 'lambda': float(lam), 'gain': float(gain)}
            print(f'    {name:<22} risk={risk:.3f} β={beta:.2f} λ={lam:.2f} gain={gain:+.4f}')
        results[method] = {
            'mean_gain': float(np.mean(gains)),
            'n_positive': sum(1 for g in gains if g > 0),
            'n_total': len(gains),
            'gains': {n: float(g) for n, g in zip(test_categories, gains)},
            'params': params_log,
        }
        print(f'  MEAN: {results[method]["mean_gain"]:+.4f}  pos={results[method]["n_positive"]}/{results[method]["n_total"]}')

    # ── 6. 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比（VOC→COCO 20类测试集，开发集=DIOR 6类+NWPU 5类=11类）')
    print('=' * 100)
    print(f'{"方法":<20} {"均值增益":>10} {"正增益类数":>12} {"相对fixed提升":>14} {"相对oracle":>12}')
    print('-' * 100)
    fixed_mean = results['fixed']['mean_gain']
    oracle_mean = results['oracle']['mean_gain']
    for method in methods:
        r = results[method]
        rel_fixed = r['mean_gain'] - fixed_mean
        rel_oracle = r['mean_gain'] / oracle_mean if oracle_mean != 0 else 0
        print(f'{method:<20} {r["mean_gain"]:>+10.4f} {r["n_positive"]:>6d}/{r["n_total"]:<4d} '
              f'{rel_fixed:>+14.4f} {rel_oracle:>11.1%}')
    print('=' * 100)

    # ── 与原始6类开发集结果对比 ──
    orig_path = DATA / 'tge_v2_auto_tune_results.json'
    if orig_path.exists():
        print('\n--- 与原始6类开发集结果对比 ---')
        with open(orig_path, 'r') as f:
            orig = json.load(f)
        print(f'{"方法":<20} {"6类开发集":>12} {"11类开发集":>12} {"变化":>10}')
        print('-' * 60)
        for method in methods:
            orig_mean = orig['test_set_results'][method]['mean_gain']
            new_mean = results[method]['mean_gain']
            diff = new_mean - orig_mean
            print(f'{method:<20} {orig_mean:>+12.4f} {new_mean:>+12.4f} {diff:>+10.4f}')

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': 'DIOR→COCO 6类 + NWPU→COCO 5类 = 11类',
            'dev_set_dior': [x[0] for x in DEV_TARGETS_DIOR],
            'dev_set_nwpu': [x[0] for x in DEV_TARGETS_NWPU],
            'test_set': 'VOC→COCO 20类',
            'beta_range': [0.0, 4.0],
            'lambda_range': [0.0, 2.0],
            'grid_size': '17x9',
            'polynomial_degree': 2,
            'mlp_hidden': 8,
        },
        'development_set': {
            'categories': dev_categories,
            'risks': dev_risks,
            'optimals': [
                {'category': o.category, 'risk': o.risk, 'beta_opt': o.beta_opt,
                 'lambda_opt': o.lambda_opt, 'gain_opt': o.gain_opt}
                for o in optimals
            ],
            'mappers': {
                'linear': {
                    'a_beta': float(mappers['linear'].a_beta),
                    'b_beta': float(mappers['linear'].b_beta),
                    'a_lambda': float(mappers['linear'].a_lambda),
                    'b_lambda': float(mappers['linear'].b_lambda),
                },
                'polynomial': {
                    'degree': 2,
                    'coeffs_beta': mappers['polynomial'].coeffs_beta.tolist(),
                    'coeffs_lambda': mappers['polynomial'].coeffs_lambda.tolist(),
                },
            },
        },
        'test_set_results': results,
        'summary': {
            method: {
                'mean_gain': results[method]['mean_gain'],
                'n_positive': results[method]['n_positive'],
                'n_total': results[method]['n_total'],
                'vs_fixed': results[method]['mean_gain'] - fixed_mean,
                'vs_oracle': results[method]['mean_gain'] / oracle_mean if oracle_mean != 0 else 0,
            }
            for method in results
        },
    }

    out_path = DATA / 'tge_v2_auto_tune_expanded_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
