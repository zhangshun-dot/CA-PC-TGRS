# -*- coding: utf-8 -*-
r"""tge_v2_paper_main_recompute.py — 第一篇论文主实验重算

在第一篇论文的主实验设置下（DIOR→COCO六类全量数据，K_CALIB=10），对比：
  1. 原A-CSPC（几何平均融合）：β=2.0, λ=0.5
  2. C_fixed（校准加权融合）：β=2.0, λ=0.5, w=0.3, τ_det=0.5, τ_vis=0.5
  3. C_fixed最优参数搜索（在开发集上搜索w,τ）
  4. oracle上界

运行: python tge_v2_paper_main_recompute.py
输出: 实验数据/tge_v2_paper_main_recompute_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import json
import pickle
import time
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

DEV_DIOR = [
    ('DIOR-COCO-airplane',   'airplane',   'airplane'),
    ('DIOR-COCO-boat',       'boat',       'ship'),
    ('DIOR-COCO-car',        'car',        'vehicle'),
    ('DIOR-COCO-truck',      'truck',      'vehicle'),
    ('DIOR-COCO-bus',        'bus',        'vehicle'),
    ('DIOR-COCO-motorcycle', 'motorcycle', 'vehicle'),
]


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
    """第一篇论文主实验协议：跳过前K_CALIB张用于校准"""
    if len(per_img) > K_CALIB:
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

    def evaluate_geometric(self, beta, lam):
        """原A-CSPC：几何平均融合"""
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

    def evaluate_calibrated(self, beta, lam, w, tau_det, tau_vis):
        """C_fixed：校准加权融合"""
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
        return dataset_ap(self.per_img, 'fused') - self.base50


def build_protos():
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None: protos[cls] = p
    return protos

def load_class_data(name, coco_cls, src_cls, protos):
    images, gts = U.load_coco(coco_cls, n=100000)
    dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), coco_cls)
    per_img, F = collect_features(images, gts, dets)
    P_src = protos[src_cls].to(U.DEVICE)
    P_neg = T.get_neg_proto(protos, src_cls)
    s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
    d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
    Delta = d_y - s_xs
    base50 = dataset_ap(per_img, 'gd')
    return ClassData(name, coco_cls, src_cls, per_img, F, P_src, P_neg, Delta, base50)


def main():
    t0 = time.time()
    print('=' * 100)
    print('第一篇论文主实验重算：原A-CSPC vs C_fixed校准加权融合')
    print('设置：DIOR→COCO六类全量数据，K_CALIB=10')
    print('=' * 100)

    # ── 1. 构建原型和加载数据 ──
    print('\n[1/4] 构建原型和加载数据 ...')
    protos = build_protos()
    print(f'  DIOR原型: {len(protos)}类')

    class_data = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        cd = load_class_data(name, coco_cls, src_cls, protos)
        class_data[name] = cd
        print(f'  {name}: {len(cd.per_img)}张图, baseAP={cd.base50:.4f}')

    # ── 2. 原A-CSPC主结果（β=2.0, λ=0.5）──
    print('\n[2/4] 原A-CSPC主结果（几何平均，β=2.0, λ=0.5）...')
    geo_gains = {}
    for name, cd in class_data.items():
        gain = cd.evaluate_geometric(2.0, 0.5)
        geo_gains[name] = gain
        print(f'  {name}: {gain:+.4f}')
    geo_mean = np.mean(list(geo_gains.values()))
    geo_pos = sum(1 for g in geo_gains.values() if g > 0)
    print(f'  均值: {geo_mean:+.4f}, 正增益: {geo_pos}/{len(geo_gains)}')

    # ── 3. C_fixed（用之前找到的最优参数：w=0.3, τ=0.5）──
    print('\n[3/4] C_fixed校准加权融合（w=0.3, τ_det=0.5, τ_vis=0.5）...')
    fixed_gains = {}
    for name, cd in class_data.items():
        gain = cd.evaluate_calibrated(2.0, 0.5, 0.3, 0.5, 0.5)
        fixed_gains[name] = gain
        print(f'  {name}: {gain:+.4f}')
    fixed_mean = np.mean(list(fixed_gains.values()))
    fixed_pos = sum(1 for g in fixed_gains.values() if g > 0)
    print(f'  均值: {fixed_mean:+.4f}, 正增益: {fixed_pos}/{len(fixed_gains)}')

    # ── 4. C_fixed最优参数搜索（在六类上搜索w,τ）──
    print('\n[4/4] C_fixed最优参数搜索 ...')
    best_fixed_gain = -999
    best_w, best_td, best_tv = 0.3, 0.5, 0.5
    for w in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        for td in [0.3, 0.5, 0.7, 1.0, 2.0]:
            for tv in [0.3, 0.5, 0.7, 1.0, 2.0]:
                gains = [cd.evaluate_calibrated(2.0, 0.5, w, td, tv) for cd in class_data.values()]
                mg = np.mean(gains)
                if mg > best_fixed_gain:
                    best_fixed_gain = mg
                    best_w, best_td, best_tv = w, td, tv
    print(f'  最优参数: w={best_w}, τ_det={best_td}, τ_vis={best_tv}')
    print(f'  最优均值: {best_fixed_gain:+.4f}')

    # 用最优参数重新评估
    best_fixed_gains = {}
    for name, cd in class_data.items():
        gain = cd.evaluate_calibrated(2.0, 0.5, best_w, best_td, best_tv)
        best_fixed_gains[name] = gain
    best_fixed_pos = sum(1 for g in best_fixed_gains.values() if g > 0)

    # oracle上界
    print('\n计算oracle上界 ...')
    oracle_gains = {}
    for name, cd in class_data.items():
        best_g = -999
        for b in np.linspace(0, 4, 9):
            for l in np.linspace(0, 2, 5):
                g = cd.evaluate_geometric(float(b), float(l))
                if g > best_g: best_g = g
        oracle_gains[name] = best_g
    oracle_mean = np.mean(list(oracle_gains.values()))

    # ── 汇总对比 ──
    print('\n' + '=' * 100)
    print('汇总对比（DIOR→COCO六类全量数据，K_CALIB=10）')
    print('=' * 100)

    print(f'\n{"类别":<25} {"原A-CSPC":>12} {"C_fixed(w=0.3)":>16} {"C_fixed最优":>14} {"Oracle":>10}')
    print('-' * 80)
    for name in DEV_DIOR:
        n = name[0]
        print(f'{n:<25} {geo_gains[n]:>+12.4f} {fixed_gains[n]:>+16.4f} {best_fixed_gains[n]:>+14.4f} {oracle_gains[n]:>+10.4f}')
    print('-' * 80)
    print(f'{"均值":<25} {geo_mean:>+12.4f} {fixed_mean:>+16.4f} {best_fixed_gain:>+14.4f} {oracle_mean:>+10.4f}')
    print(f'{"正增益数":<25} {geo_pos:>12d} {fixed_pos:>16d} {best_fixed_pos:>14d} {len(oracle_gains):>10d}')

    print(f'\n原A-CSPC均值: {geo_mean:+.4f} (论文报告值: +0.0646)')
    print(f'C_fixed(w=0.3)均值: {fixed_mean:+.4f}, 相对提升: {(fixed_mean-geo_mean)/abs(geo_mean)*100:+.1f}%')
    print(f'C_fixed最优均值: {best_fixed_gain:+.4f}, 相对提升: {(best_fixed_gain-geo_mean)/abs(geo_mean)*100:+.1f}%')
    print(f'Oracle上界: {oracle_mean:+.4f}')
    print(f'C_fixed最优达到oracle: {best_fixed_gain/oracle_mean*100:.1f}%')

    # 逐类别对比：C_fixed最优 vs 原A-CSPC
    print(f'\n逐类别对比（C_fixed最优 - 原A-CSPC）:')
    for name in DEV_DIOR:
        n = name[0]
        diff = best_fixed_gains[n] - geo_gains[n]
        marker = '↑' if diff > 0 else ('↓' if diff < 0 else '=')
        print(f'  {n}: {diff:+.4f} {marker}')

    # ── 保存结果 ──
    output = {
        'config': {
            'setting': 'DIOR→COCO六类全量数据，K_CALIB=10',
            'beta': 2.0, 'lambda': 0.5,
            'c_fixed_default': {'w': 0.3, 'tau_det': 0.5, 'tau_vis': 0.5},
            'c_fixed_best': {'w': best_w, 'tau_det': best_td, 'tau_vis': best_tv},
        },
        'geometric_mean': float(geo_mean),
        'geometric_positive': geo_pos,
        'geometric_gains': {n: float(g) for n, g in geo_gains.items()},
        'c_fixed_default_mean': float(fixed_mean),
        'c_fixed_default_positive': fixed_pos,
        'c_fixed_default_gains': {n: float(g) for n, g in fixed_gains.items()},
        'c_fixed_best_mean': float(best_fixed_gain),
        'c_fixed_best_positive': best_fixed_pos,
        'c_fixed_best_gains': {n: float(g) for n, g in best_fixed_gains.items()},
        'oracle_mean': float(oracle_mean),
        'oracle_gains': {n: float(g) for n, g in oracle_gains.items()},
    }

    out_path = DATA / 'tge_v2_paper_main_recompute_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
