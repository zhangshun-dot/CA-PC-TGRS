# -*- coding: utf-8 -*-
r"""tge_v2_auto_tune.py — 物理先验指导的自动调参实验

第二阶段核心实验：把人工扫alpha变成自动搜索最优映射函数。

实验设计：
  开发集：DIOR→COCO 6类（airplane, boat, car, truck, bus, motorcycle）
  测试集：VOC→COCO 20类（动物/交通工具/室内物体/person）

  对比方法：
    1. fixed:       固定参数 (β=1.0, λ=0.25)
    2. alpha_scan:  人工扫alpha的线性映射 (最佳alpha=1.5)
    3. auto_linear: 自动搜索的线性映射 (开发集最小二乘拟合)
    4. auto_poly:   自动搜索的多项式映射 (degree=2)
    5. auto_mlp:    自动搜索的MLP映射 (2层隐藏层)
    6. oracle:      每个类别单独网格搜索最优 (上限)

运行: python tge_v2_auto_tune.py
输出: 实验数据/tge_v2_auto_tune_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import json
import pickle
import time
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import numpy as np
import unified_eval as U
import tge_probe as T

# AutoTuner imports
from acspc.auto_tuner import AutoTuner, ClassOptimal
from acspc.coco_categories import get_offline_prior
from acspc.appg import PhysicsPrior

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 基础配置 ──
K_CALIB = 10
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
PRE_FILTER = 0.03

# ── VOC 配置 ──
VOC_ROOT = r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007'
VOC_IMG_DIR = os.path.join(VOC_ROOT, 'JPEGImages')
VOC_ANN_DIR = os.path.join(VOC_ROOT, 'Annotations')
SRC_TMPL_VOC = 'a photo of {}'

# VOC→COCO 类别映射 (20类)
VOC_TO_COCO = {
    'person': 'person',
    'bicycle': 'bicycle',
    'car': 'car',
    'motorbike': 'motorcycle',
    'aeroplane': 'airplane',
    'bus': 'bus',
    'train': 'train',
    'truck': 'truck',
    'boat': 'boat',
    'bird': 'bird',
    'cat': 'cat',
    'dog': 'dog',
    'horse': 'horse',
    'sheep': 'sheep',
    'cow': 'cow',
    'bottle': 'bottle',
    'chair': 'chair',
    'sofa': 'couch',
    'pottedplant': 'potted plant',
    'tvmonitor': 'tv',
}

# 开发集：DIOR→COCO 6类
DEV_TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]

# 测试集：VOC→COCO 20类
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


# ── 数据加载与特征提取（复用 tge_v2_appg.py 的逻辑）──

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
    """单个类别的预加载数据，用于快速评估不同 (β, λ)"""
    def __init__(self, name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50):
        self.name = name
        self.coco_cls = coco_cls
        self.src_cls = src_cls
        self.per_img = per_img
        self.F = F
        self.P_src = P_src
        self.P_neg = P_neg
        self.Delta = Delta
        self.base50 = base50

    def evaluate(self, beta, lam):
        """评估给定 (β, λ) 的增益"""
        P_hat = self.P_src + beta * self.Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - lam * self.P_neg.cpu().numpy())
        fused_arrays(self.per_img, self.F, W)
        ap50 = dataset_ap(self.per_img, 'fused')
        return ap50 - self.base50


def load_voc_class(voc_class, n_max=100):
    """加载VOC某个类别的图像和标注框。"""
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
    """构建VOC 20类的CLIP源原型。"""
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


def build_source_prototypes(source='dior'):
    """构建源域原型库"""
    protos = {}
    if source == 'dior':
        for cls in U.DIOR_ID2NAME.values():
            images, gts = U.load_dior(cls)
            p = clip_proto(gt_crops(images, gts))
            if p is not None:
                protos[cls] = p
    elif source == 'voc':
        return build_voc_protos()
    return protos


def load_class_data(name, coco_cls, src_cls, protos, source='dior'):
    """加载单个类别的数据"""
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

    if source == 'dior':
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
    else:  # voc
        s_xs = T.text_emb(['a photo of {}'.format(src_cls)]).cpu()[0]
    d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
    Delta = d_y - s_xs

    base50 = dataset_ap(per_img, 'gd')

    return ClassData(name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50)


def get_risk(coco_cls):
    """获取类别的物理先验风险评分"""
    v, m, s, i = get_offline_prior(coco_cls)
    prior = PhysicsPrior(category=coco_cls, V=v, M=m, S=s, I=i, source='offline', confidence=0.6)
    return prior.risk_score()


# ── 主实验 ──

def main():
    t0 = time.time()
    print('=' * 100)
    print('A-CSPC AutoTuner: 物理先验指导的自动调参实验')
    print('=' * 100)

    # ── 1. 构建源域原型 ──
    print('\n[1/5] 构建 DIOR 源域原型 ...')
    protos_dior = build_source_prototypes('dior')
    print(f'  DIOR protos: {len(protos_dior)} classes')

    print('\n[1/5] 构建 VOC 源域原型 ...')
    protos_voc = build_source_prototypes('voc')
    print(f'  VOC protos: {len(protos_voc)} classes')

    # ── 2. 加载开发集数据（DIOR→COCO 6类）──
    print('\n[2/5] 加载开发集数据（DIOR→COCO 6类）...')
    dev_data = {}
    for name, coco_cls, src_cls in DEV_TARGETS:
        cd = load_class_data(name, coco_cls, src_cls, protos_dior, 'dior')
        if cd is not None:
            dev_data[name] = cd
            print(f'  {name:<16} base={cd.base50:.4f} 框={sum(len(p["gd"]) for p in cd.per_img)}')
    print(f'  开发集: {len(dev_data)} classes')

    # ── 3. 加载测试集数据（VOC→COCO 20类）──
    print('\n[3/5] 加载测试集数据（VOC→COCO 20类）...')
    test_data = {}
    for name, coco_cls, src_cls in TEST_TARGETS:
        cd = load_class_data(name, coco_cls, src_cls, protos_voc, 'voc')
        if cd is not None:
            test_data[name] = cd
            print(f'  {name:<20} base={cd.base50:.4f} 框={sum(len(p["gd"]) for p in cd.per_img)}')
    print(f'  测试集: {len(test_data)} classes')

    # ── 4. 开发集网格搜索 + 映射训练 ──
    print('\n[4/5] 开发集网格搜索 + 映射训练 ...')

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

    print('\n  --- 开发集网格搜索 ---')
    optimals = tuner.search_development_set(dev_categories, dev_risks, make_eval_fn)

    print('\n  --- 映射训练 ---')
    mappers = tuner.train_mappers()

    # ── 5. 测试集评估 ──
    print('\n[5/5] 测试集评估（VOC→COCO 20类）...')

    test_categories = list(test_data.keys())
    test_risks = [get_risk(test_data[n].coco_cls) for n in test_categories]

    def make_test_eval_fn(cat_name):
        cd = test_data[cat_name]
        return lambda beta, lam: cd.evaluate(beta, lam)

    results = {}

    # 5.1 固定参数
    print('\n  --- fixed (β=1.0, λ=0.25) ---')
    fixed_gains = []
    for name in test_categories:
        cd = test_data[name]
        gain = cd.evaluate(1.0, 0.25)
        fixed_gains.append(gain)
        print(f'    {name:<20} gain={gain:+.4f}')
    results['fixed'] = {
        'mean_gain': float(np.mean(fixed_gains)),
        'n_positive': sum(1 for g in fixed_gains if g > 0),
        'n_total': len(fixed_gains),
        'gains': {n: float(g) for n, g in zip(test_categories, fixed_gains)},
    }
    print(f'  MEAN: {results["fixed"]["mean_gain"]:+.4f}  pos={results["fixed"]["n_positive"]}/{results["fixed"]["n_total"]}')

    # 5.2 人工alpha扫描（alpha=1.5）
    print('\n  --- alpha_scan (α=1.5, 线性映射) ---')
    alpha_gains = []
    for name in test_categories:
        cd = test_data[name]
        risk = get_risk(cd.coco_cls)
        beta_eff = 1.0 * (1 + 1.5 * risk)
        lambda_eff = 0.25 * (1 + 1.5 * risk)
        beta_eff = min(max(beta_eff, 0), 6)
        lambda_eff = min(max(lambda_eff, 0), 2)
        gain = cd.evaluate(beta_eff, lambda_eff)
        alpha_gains.append(gain)
        print(f'    {name:<20} risk={risk:.3f} β={beta_eff:.2f} λ={lambda_eff:.2f} gain={gain:+.4f}')
    results['alpha_scan'] = {
        'alpha': 1.5,
        'mean_gain': float(np.mean(alpha_gains)),
        'n_positive': sum(1 for g in alpha_gains if g > 0),
        'n_total': len(alpha_gains),
        'gains': {n: float(g) for n, g in zip(test_categories, alpha_gains)},
    }
    print(f'  MEAN: {results["alpha_scan"]["mean_gain"]:+.4f}  pos={results["alpha_scan"]["n_positive"]}/{results["alpha_scan"]["n_total"]}')

    # 5.3 自动线性映射
    print('\n  --- auto_linear (开发集最小二乘) ---')
    linear_gains = []
    for name in test_categories:
        cd = test_data[name]
        risk = get_risk(cd.coco_cls)
        beta_pred, lambda_pred = tuner.predict('linear', risk)
        gain = cd.evaluate(beta_pred, lambda_pred)
        linear_gains.append(gain)
        print(f'    {name:<20} risk={risk:.3f} β={beta_pred:.2f} λ={lambda_pred:.2f} gain={gain:+.4f}')
    results['auto_linear'] = {
        'mean_gain': float(np.mean(linear_gains)),
        'n_positive': sum(1 for g in linear_gains if g > 0),
        'n_total': len(linear_gains),
        'gains': {n: float(g) for n, g in zip(test_categories, linear_gains)},
    }
    print(f'  MEAN: {results["auto_linear"]["mean_gain"]:+.4f}  pos={results["auto_linear"]["n_positive"]}/{results["auto_linear"]["n_total"]}')

    # 5.4 自动多项式映射
    print('\n  --- auto_polynomial (degree=2) ---')
    poly_gains = []
    for name in test_categories:
        cd = test_data[name]
        risk = get_risk(cd.coco_cls)
        beta_pred, lambda_pred = tuner.predict('polynomial', risk)
        gain = cd.evaluate(beta_pred, lambda_pred)
        poly_gains.append(gain)
        print(f'    {name:<20} risk={risk:.3f} β={beta_pred:.2f} λ={lambda_pred:.2f} gain={gain:+.4f}')
    results['auto_polynomial'] = {
        'degree': 2,
        'mean_gain': float(np.mean(poly_gains)),
        'n_positive': sum(1 for g in poly_gains if g > 0),
        'n_total': len(poly_gains),
        'gains': {n: float(g) for n, g in zip(test_categories, poly_gains)},
    }
    print(f'  MEAN: {results["auto_polynomial"]["mean_gain"]:+.4f}  pos={results["auto_polynomial"]["n_positive"]}/{results["auto_polynomial"]["n_total"]}')

    # 5.5 自动MLP映射
    print('\n  --- auto_mlp (2层隐藏层) ---')
    mlp_gains = []
    for name in test_categories:
        cd = test_data[name]
        risk = get_risk(cd.coco_cls)
        beta_pred, lambda_pred = tuner.predict('mlp', risk)
        gain = cd.evaluate(beta_pred, lambda_pred)
        mlp_gains.append(gain)
        print(f'    {name:<20} risk={risk:.3f} β={beta_pred:.2f} λ={lambda_pred:.2f} gain={gain:+.4f}')
    results['auto_mlp'] = {
        'hidden_dim': 8,
        'mean_gain': float(np.mean(mlp_gains)),
        'n_positive': sum(1 for g in mlp_gains if g > 0),
        'n_total': len(mlp_gains),
        'gains': {n: float(g) for n, g in zip(test_categories, mlp_gains)},
    }
    print(f'  MEAN: {results["auto_mlp"]["mean_gain"]:+.4f}  pos={results["auto_mlp"]["n_positive"]}/{results["auto_mlp"]["n_total"]}')

    # 5.6 Oracle（测试集每个类别单独最优）
    print('\n  --- oracle (测试集每类单独最优) ---')
    oracle_gains = []
    oracle_params = {}
    for name in test_categories:
        cd = test_data[name]
        best_gain = -1e9
        best_beta, best_lam = 1.0, 0.25
        for beta in np.linspace(0, 4, 17):
            for lam in np.linspace(0, 2, 9):
                gain = cd.evaluate(beta, lam)
                if gain > best_gain:
                    best_gain = gain
                    best_beta, best_lam = beta, lam
        oracle_gains.append(best_gain)
        oracle_params[name] = {'beta': float(best_beta), 'lambda': float(best_lam), 'gain': float(best_gain)}
        print(f'    {name:<20} β*={best_beta:.2f} λ*={best_lam:.2f} gain*={best_gain:+.4f}')
    results['oracle'] = {
        'mean_gain': float(np.mean(oracle_gains)),
        'n_positive': sum(1 for g in oracle_gains if g > 0),
        'n_total': len(oracle_gains),
        'gains': {n: float(g) for n, g in zip(test_categories, oracle_gains)},
        'params': oracle_params,
    }
    print(f'  MEAN: {results["oracle"]["mean_gain"]:+.4f}  pos={results["oracle"]["n_positive"]}/{results["oracle"]["n_total"]}')

    # ── 汇总对比表 ──
    print('\n' + '=' * 100)
    print('汇总对比（VOC→COCO 20类测试集）')
    print('=' * 100)
    print(f'{"方法":<20} {"均值增益":>10} {"正增益类数":>12} {"相对fixed提升":>14} {"相对oracle":>12}')
    print('-' * 100)
    fixed_mean = results['fixed']['mean_gain']
    oracle_mean = results['oracle']['mean_gain']
    for method in ['fixed', 'alpha_scan', 'auto_linear', 'auto_polynomial', 'auto_mlp', 'oracle']:
        r = results[method]
        rel_fixed = r['mean_gain'] - fixed_mean
        rel_oracle = r['mean_gain'] / oracle_mean if oracle_mean != 0 else 0
        print(f'{method:<20} {r["mean_gain"]:>+10.4f} {r["n_positive"]:>6d}/{r["n_total"]:<4d} '
              f'{rel_fixed:>+14.4f} {rel_oracle:>11.1%}')
    print('=' * 100)

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': 'DIOR→COCO 6类',
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
                {
                    'category': o.category,
                    'risk': o.risk,
                    'beta_opt': o.beta_opt,
                    'lambda_opt': o.lambda_opt,
                    'gain_opt': o.gain_opt,
                }
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

    out_path = DATA / 'tge_v2_auto_tune_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
