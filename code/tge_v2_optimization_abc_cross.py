# -*- coding: utf-8 -*-
r"""tge_v2_optimization_abc_cross.py — B+C优化跨数据集验证

在DIOR→COCO held-out测试集上验证B+C组合的泛化性
（跳过前200张COCO图像作为开发集，剩余作为测试集）

对比：baseline / B_only / C_only / B+C / oracle
开发集：DIOR→COCO 前200张/类（搜索超参数）
测试集：DIOR→COCO 200张之后/类（评估泛化）

运行: python tge_v2_optimization_abc_cross.py
输出: 实验数据/tge_v2_optimization_abc_cross_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import json
import pickle
import time
import os
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

DEV_DIOR = [
    ('DIOR-COCO-airplane',   'airplane',   'airplane'),
    ('DIOR-COCO-boat',       'boat',       'ship'),
    ('DIOR-COCO-car',        'car',        'vehicle'),
    ('DIOR-COCO-truck',      'truck',      'vehicle'),
    ('DIOR-COCO-bus',        'bus',        'vehicle'),
    ('DIOR-COCO-motorcycle', 'motorcycle', 'vehicle'),
]


# ── 基础工具 ──
def load_coco_split(cls_name, split='dev', skip=200):
    """加载COCO数据，分为dev（前skip张）和test（skip之后）"""
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

def dataset_ap(per_img, key):
    """测试集AP：不跳过K_CALIB（因为测试集是独立的）"""
    preds = [(p['box'], p[key]) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return U.compute_ap50_img(preds, gts)


class ClassData:
    def __init__(self, name, coco_cls, src_cls, per_img, F, P_src, P_neg, P_neg_all, Delta, base50):
        self.name = name; self.coco_cls = coco_cls; self.src_cls = src_cls
        self.per_img = per_img; self.F = F; self.P_src = P_src
        self.P_neg = P_neg; self.P_neg_all = P_neg_all; self.Delta = Delta; self.base50 = base50

    def evaluate_baseline(self, beta, lam):
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

    def evaluate_optimized(self, beta, lam, use_B=False, use_C=False,
                            m_B=3, w_C=0.5, tau_det=1.0, tau_vis=1.0):
        P_src_np = self.P_src.cpu().numpy()
        Delta_np = self.Delta.cpu().numpy()
        P_neg_all_np = self.P_neg_all.cpu().numpy()
        P_neg_np = self.P_neg.cpu().numpy()

        for p in self.per_img:
            n = len(p['gd'])
            if n == 0:
                p['fused'] = np.zeros(0)
                continue

            f_k = self.F[p['crop_idx']]
            gd = p['gd']

            # 固定β外推（方向A不使用，因为验证发现价值有限）
            P_hat = P_src_np + beta * Delta_np
            P_hat = P_hat / (np.linalg.norm(P_hat) + 1e-8)
            vis_pos = f_k @ P_hat

            # 方向B：难负样本动态选择
            if use_B:
                sim_neg_all = f_k @ P_neg_all_np.T
                topm_idx = np.argpartition(-sim_neg_all, m_B-1, axis=1)[:, :m_B]
                P_neg_dyn = np.zeros((n, 512))
                for i in range(n):
                    P_neg_dyn[i] = P_neg_all_np[topm_idx[i]].mean(axis=0)
                neg_norm = np.linalg.norm(P_neg_dyn, axis=1, keepdims=True) + 1e-8
                P_neg_dyn = P_neg_dyn / neg_norm
                vis_neg = np.sum(f_k * P_neg_dyn, axis=1)
            else:
                vis_neg = f_k @ P_neg_np

            vis = np.maximum(0.0, vis_pos - lam * vis_neg)

            # 方向C：校准后加权融合
            if use_C:
                gd_calib = np.clip(gd ** (1.0 / tau_det), 1e-6, 1.0)
                vis_calib = np.clip(vis ** (1.0 / tau_vis), 1e-6, 1.0)
                p['fused'] = w_C * gd_calib + (1 - w_C) * vis_calib
            else:
                p['fused'] = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))

        return dataset_ap(self.per_img, 'fused') - self.base50


def build_protos():
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None: protos[cls] = p
    return protos

def load_class_data(name, coco_cls, src_cls, protos, split='dev'):
    images, gts = load_coco_split(coco_cls, split=split, skip=HELDOUT_SKIP)
    if len(images) < 5:
        return None
    dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), coco_cls)
    # 注意：dets是全量图像的，需要根据split截取
    if split == 'dev':
        dets = dets[:HELDOUT_SKIP]
    else:
        dets = dets[HELDOUT_SKIP:]
    per_img, F = collect_features(images, gts, dets)
    if src_cls not in protos:
        return None
    P_src = protos[src_cls].to(U.DEVICE)
    P_neg = T.get_neg_proto(protos, src_cls)
    neg_list = []
    for k, v in protos.items():
        if k != src_cls:
            neg_list.append(v.cpu().numpy())
    P_neg_all = torch.tensor(np.stack(neg_list), dtype=torch.float32)
    s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
    d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
    Delta = d_y - s_xs
    base50 = dataset_ap(per_img, 'gd')
    return ClassData(name, coco_cls, src_cls, per_img, F, P_src, P_neg, P_neg_all, Delta, base50)


def main():
    t0 = time.time()
    print('=' * 100)
    print(f'B+C优化跨数据集验证：DIOR→COCO held-out（跳过前{HELDOUT_SKIP}张）')
    print('=' * 100)

    # ── 1. 构建原型 ──
    print('\n[1/5] 构建源域原型 ...')
    protos_dior = build_protos()
    print(f'  DIOR: {len(protos_dior)}类')

    # ── 2. 加载数据 ──
    print('\n[2/5] 加载开发集和测试集数据 ...')
    dev_data = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        cd = load_class_data(name, coco_cls, src_cls, protos_dior, split='dev')
        if cd: dev_data[name] = cd
    print(f'  开发集: {len(dev_data)}类 (前{HELDOUT_SKIP}张/类)')

    test_data = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        cd = load_class_data(name + '_test', coco_cls, src_cls, protos_dior, split='test')
        if cd: test_data[name] = cd
    test_categories = list(test_data.keys())
    print(f'  测试集: {len(test_data)}类 ({HELDOUT_SKIP}张之后)')
    for name in test_categories:
        cd = test_data[name]
        print(f'    {name}: {len(cd.per_img)}张图, baseAP={cd.base50:.4f}')

    # ── 3. 开发集超参数搜索 ──
    print('\n[3/5] 开发集超参数搜索 ...')

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

    # 方向C：搜索w_C, tau_det, tau_vis
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

    # B+C组合（用各自最优值）
    print('  评估B+C组合 ...')
    gains_BC = [cd.evaluate_optimized(best_beta, best_lam, use_B=True, use_C=True,
                                         m_B=best_m_B, w_C=best_w_C,
                                         tau_det=best_tau_det, tau_vis=best_tau_vis) for cd in dev_data.values()]
    print(f'  B+C dev_gain={np.mean(gains_BC):+.4f}')

    # ── 4. 测试集评估 ──
    print('\n[4/5] 测试集评估（held-out）...')

    configs = [
        ('baseline', {'use_B': False, 'use_C': False}),
        ('B_only',   {'use_B': True,  'use_C': False, 'm_B': best_m_B}),
        ('C_only',   {'use_B': False, 'use_C': True,  'w_C': best_w_C, 'tau_det': best_tau_det, 'tau_vis': best_tau_vis}),
        ('B+C',      {'use_B': True,  'use_C': True,  'm_B': best_m_B, 'w_C': best_w_C, 'tau_det': best_tau_det, 'tau_vis': best_tau_vis}),
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
        for n, g in zip(test_categories, gains):
            print(f'      {n}: {g:+.4f}')

    # oracle
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
    print(f'汇总对比（DIOR→COCO held-out测试集，{len(test_data)}类，跳过前{HELDOUT_SKIP}张）')
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

    # 与VOC测试集结果对比
    print('\n' + '=' * 100)
    print('跨数据集一致性对比')
    print('=' * 100)
    print(f'{"配置":<15} {"VOC→COCO":>12} {"DIOR→COCO held-out":>20} {"一致性":>10}')
    print('-' * 60)
    voc_results = {
        'baseline': 0.0347, 'B_only': 0.0416, 'C_only': 0.0649,
        'B+C': 0.0721, 'oracle': 0.0775,
    }
    for config_name in ['baseline', 'B_only', 'C_only', 'B+C', 'oracle']:
        dior_gain = all_results[config_name]['mean_gain']
        voc_gain = voc_results[config_name]
        consistent = '✓' if (dior_gain - baseline_gain) * (voc_gain - voc_results['baseline']) > 0 else '✗'
        print(f'{config_name:<15} {voc_gain:>+12.4f} {dior_gain:>+20.4f} {consistent:>10}')

    # ── 保存结果 ──
    output = {
        'config': {
            'dev_set': f'DIOR→COCO 前{HELDOUT_SKIP}张/类, {len(dev_data)}类',
            'test_set': f'DIOR→COCO {HELDOUT_SKIP}张之后/类, {len(test_data)}类',
            'heldout_skip': HELDOUT_SKIP,
            'best_beta': best_beta, 'best_lambda': best_lam,
            'best_m_B': best_m_B,
            'best_w_C': best_w_C, 'best_tau_det': best_tau_det, 'best_tau_vis': best_tau_vis,
        },
        'dev_gains': {
            'baseline': best_base_gain, 'B': best_B_gain, 'C': best_C_gain,
            'B+C': float(np.mean(gains_BC)),
        },
        'voc_results_for_comparison': voc_results,
        'results': all_results,
    }

    out_path = DATA / 'tge_v2_optimization_abc_cross_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
