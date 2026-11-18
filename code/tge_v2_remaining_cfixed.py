# -*- coding: utf-8 -*-
r"""tge_v2_remaining_cfixed.py — 剩余实验重跑（C_fixed融合）

包含：
  1. β扫描（0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0）
  2. λ扫描（0, 0.25, 0.5, 1.0, 1.5, 2.0）
  3. DIOR→VOC跨数据集（6类）
  4. NWPU→COCO多源域（5类）

运行: python tge_v2_remaining_cfixed.py
输出: 实验数据/tge_v2_remaining_cfixed_results.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
import json
import pickle
import time
import glob
import re
import xml.etree.ElementTree as ET
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

VOC_ROOT = r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007'
VOC_IMG_DIR = os.path.join(VOC_ROOT, 'JPEGImages')
VOC_ANN_DIR = os.path.join(VOC_ROOT, 'Annotations')

CF_W = 0.35
CF_TAU_DET = 2.0
CF_TAU_VIS = 2.0

DEV_DIOR = [
    ('airplane',   'airplane',   'airplane'),
    ('boat',       'boat',       'ship'),
    ('car',        'car',        'vehicle'),
    ('truck',      'truck',      'vehicle'),
    ('bus',        'bus',        'vehicle'),
    ('motorcycle', 'motorcycle', 'vehicle'),
]

NWPU_CLASSES = [
    ('airplane', 'airplane', 'airplane'),
    ('boat',     'boat',     'ship'),
    ('car',      'car',      'vehicle'),
    ('truck',    'truck',    'vehicle'),
    ('bus',      'bus',      'vehicle'),
]


def clip_proto(crops):
    if not crops: return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()

def gt_crops(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10: continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops

def v2_cached_dets(images, prompt, name, prefix='tge_v2_dets_COCO-COCO'):
    p = DATA / f'{prefix}-{name}.pkl'
    if p.exists(): return pickle.load(open(p, 'rb'))
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
            if x2 <= x1 + 10 or y2 <= y1 + 10: continue
            if float(sc) < PRE_FILTER: continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                        'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64)})
    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return per_img, F

def dataset_ap(per_img, key, use_kcalib=True):
    if use_kcalib and len(per_img) > K_CALIB:
        sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    else:
        sel = per_img
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    return U.compute_ap50_img(preds, gts)

def evaluate_cfixed(per_img, F, P_pos_ext, P_neg, beta, lam):
    for p in per_img:
        n = len(p['gd'])
        if n:
            f_k = F[p['crop_idx']]
            sim_pos = f_k @ P_pos_ext
            sim_neg = f_k @ P_neg
            vis = np.maximum(0.0, sim_pos - lam * sim_neg)
            gd_calib = np.clip(p['gd'] ** (1.0 / CF_TAU_DET), 1e-6, 1.0)
            vis_calib = np.clip(vis ** (1.0 / CF_TAU_VIS), 1e-6, 1.0)
            p['fused'] = CF_W * gd_calib + (1 - CF_W) * vis_calib
        else:
            p['fused'] = np.zeros(0)
    return dataset_ap(per_img, 'fused')

def load_voc_class(voc_class, n_max=100):
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

def load_nwpu_class(cls_name, n_max=200):
    """加载NWPU数据"""
    nwpu_root = r'C:\Users\张三\modeling\nwpu_data\NWPU-VHR-10'
    img_dir = os.path.join(nwpu_root, 'positive image set')
    ann_dir = os.path.join(nwpu_root, 'ground truth')
    # NWPU类别映射
    nwpu_cls_map = {'airplane': 1, 'ship': 2, 'vehicle': 3, 'bridge': 5,
                    'basketball court': 6, 'ground track field': 7, 'harbor': 8,
                    'storage tank': 9, 'tennis court': 10, 'baseball diamond': 11}
    cls_id = nwpu_cls_map.get(cls_name)
    if cls_id is None:
        return [], []
    images, gts = [], []
    for ann_file in sorted(glob.glob(os.path.join(ann_dir, '*.txt'))):
        if len(images) >= n_max: break
        try:
            with open(ann_file, 'r') as f:
                lines = f.readlines()
            boxes = []
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 5 and int(parts[4]) == cls_id:
                    x1, y1, x2, y2 = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    boxes.append([x1, y1, x2, y2])
            if boxes:
                base = os.path.splitext(os.path.basename(ann_file))[0]
                img_file = os.path.join(img_dir, base + '.jpg')
                if os.path.exists(img_file):
                    images.append(Image.open(img_file).convert('RGB')); gts.append(boxes)
        except Exception:
            continue
    return images, gts


def main():
    t0 = time.time()
    print('=' * 100)
    print('剩余实验重跑（C_fixed融合，w=0.35, τ=2.0）')
    print('=' * 100)

    all_results = {}

    # ── 1. β扫描 ──
    print('\n[1/4] β扫描（λ=0.5固定）...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None: protos[cls] = p

    class_data = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        images, gts = U.load_coco(coco_cls, n=100000)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), coco_cls)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].cpu().numpy()
        P_neg = T.get_neg_proto(protos, src_cls).cpu().numpy()
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0].numpy()
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0].numpy()
        Delta = d_y - s_xs
        base50 = dataset_ap(per_img, 'gd')
        class_data[coco_cls] = {'per_img': per_img, 'F': F, 'base50': base50,
                                  'P_src': P_src, 'P_neg': P_neg, 'Delta': Delta}

    beta_values = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    beta_results = {}
    for beta in beta_values:
        gains = []
        for coco_cls, cd in class_data.items():
            P_pos_ext = cd['P_src'] + beta * cd['Delta']
            P_pos_ext = P_pos_ext / (np.linalg.norm(P_pos_ext) + 1e-8)
            ap = evaluate_cfixed(cd['per_img'], cd['F'], P_pos_ext, cd['P_neg'], beta, 0.5)
            gains.append(ap - cd['base50'])
        beta_results[str(beta)] = float(np.mean(gains))
        print(f'  β={beta}: {np.mean(gains):+.4f}')
    all_results['beta_sweep'] = beta_results

    # ── 2. λ扫描 ──
    print('\n[2/4] λ扫描（β=2.0固定）...')
    lambda_values = [0, 0.25, 0.5, 1.0, 1.5, 2.0]
    lambda_results = {}
    for lam in lambda_values:
        gains = []
        for coco_cls, cd in class_data.items():
            P_pos_ext = cd['P_src'] + 2.0 * cd['Delta']
            P_pos_ext = P_pos_ext / (np.linalg.norm(P_pos_ext) + 1e-8)
            ap = evaluate_cfixed(cd['per_img'], cd['F'], P_pos_ext, cd['P_neg'], 2.0, lam)
            gains.append(ap - cd['base50'])
        lambda_results[str(lam)] = float(np.mean(gains))
        print(f'  λ={lam}: {np.mean(gains):+.4f}')
    all_results['lambda_sweep'] = lambda_results

    # ── 3. DIOR→VOC跨数据集 ──
    print('\n[3/4] DIOR→VOC跨数据集 ...')
    voc_map = {'airplane': 'aeroplane', 'boat': 'boat', 'car': 'car',
               'truck': 'truck', 'bus': 'bus', 'motorcycle': 'motorbike'}
    voc_results = {}
    for name, coco_cls, src_cls in DEV_DIOR:
        voc_cls = voc_map[coco_cls]
        images, gts = load_voc_class(voc_cls, n_max=100)
        if len(images) < 5:
            print(f'  {coco_cls}: VOC数据不足，跳过')
            continue
        # 用VOC检测缓存
        cache_path = DATA / f'tge_probe_dets_VOC-{voc_cls}.pkl'
        if cache_path.exists():
            dets = pickle.load(open(cache_path, 'rb'))
        else:
            dets = U.gdino_detect(images, TGT_TMPL.format(coco_cls), 0.0)
            pickle.dump(dets, open(cache_path, 'wb'))
        if len(dets) > len(images):
            dets = dets[:len(images)]
        per_img, F = collect_features(images, gts, dets)
        cd = class_data.get(coco_cls)
        if cd is None:
            continue
        P_pos_ext = cd['P_src'] + 2.0 * cd['Delta']
        P_pos_ext = P_pos_ext / (np.linalg.norm(P_pos_ext) + 1e-8)
        base50 = dataset_ap(per_img, 'gd', use_kcalib=False)
        ap = evaluate_cfixed(per_img, F, P_pos_ext, cd['P_neg'], 2.0, 0.5)
        # 注意：evaluate_cfixed内部用use_kcalib=True，需要重新计算
        for p in per_img:
            n = len(p['gd'])
            if n:
                f_k = F[p['crop_idx']]
                sim_pos = f_k @ P_pos_ext
                sim_neg = f_k @ cd['P_neg']
                vis = np.maximum(0.0, sim_pos - 0.5 * sim_neg)
                gd_calib = np.clip(p['gd'] ** (1.0 / CF_TAU_DET), 1e-6, 1.0)
                vis_calib = np.clip(vis ** (1.0 / CF_TAU_VIS), 1e-6, 1.0)
                p['fused'] = CF_W * gd_calib + (1 - CF_W) * vis_calib
            else:
                p['fused'] = np.zeros(0)
        ap_voc = dataset_ap(per_img, 'fused', use_kcalib=False)
        gain = ap_voc - base50
        voc_results[coco_cls] = float(gain)
        print(f'  {coco_cls}: base={base50:.4f}, gain={gain:+.4f}')
    all_results['dior_to_voc'] = {
        'mean': float(np.mean(list(voc_results.values()))) if voc_results else 0,
        'gains': voc_results,
    }
    print(f'  均值: {all_results["dior_to_voc"]["mean"]:+.4f}')

    # ── 4. NWPU→COCO多源域 ──
    print('\n[4/4] NWPU→COCO多源域 ...')
    # 构建NWPU原型
    nwpu_protos = {}
    for cls_name in ['airplane', 'ship', 'vehicle']:
        images, gts = load_nwpu_class(cls_name, n_max=200)
        p = clip_proto(gt_crops(images, gts))
        if p is not None: nwpu_protos[cls_name] = p
    print(f'  NWPU原型: {list(nwpu_protos.keys())}')

    nwpu_results = {}
    for name, coco_cls, nwpu_cls in NWPU_CLASSES:
        if nwpu_cls not in nwpu_protos:
            print(f'  {coco_cls}: NWPU原型不存在，跳过')
            continue
        images, gts = U.load_coco(coco_cls, n=100000)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), coco_cls, prefix='tge_v2_dets_COCO-NWPU-COCO')
        per_img, F = collect_features(images, gts, dets)
        P_src = nwpu_protos[nwpu_cls].cpu().numpy()
        # NWPU负原型：用DIOR+NWPU合并池
        all_neg = []
        for k, v in protos.items():
            if k != nwpu_cls: all_neg.append(v.cpu().numpy())
        for k, v in nwpu_protos.items():
            if k != nwpu_cls: all_neg.append(v.cpu().numpy())
        P_neg_pool = np.stack(all_neg)
        sims = P_src @ P_neg_pool.T
        topk_idx = np.argsort(-sims)[:3]
        P_neg = P_neg_pool[topk_idx].mean(axis=0)
        P_neg = P_neg / (np.linalg.norm(P_neg) + 1e-8)

        s_xs = T.text_emb([SRC_TMPL.format(nwpu_cls)]).cpu()[0].numpy()
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0].numpy()
        Delta = d_y - s_xs
        P_pos_ext = P_src + 2.0 * Delta
        P_pos_ext = P_pos_ext / (np.linalg.norm(P_pos_ext) + 1e-8)

        base50 = dataset_ap(per_img, 'gd')
        for p in per_img:
            n = len(p['gd'])
            if n:
                f_k = F[p['crop_idx']]
                sim_pos = f_k @ P_pos_ext
                sim_neg = f_k @ P_neg
                vis = np.maximum(0.0, sim_pos - 0.5 * sim_neg)
                gd_calib = np.clip(p['gd'] ** (1.0 / CF_TAU_DET), 1e-6, 1.0)
                vis_calib = np.clip(vis ** (1.0 / CF_TAU_VIS), 1e-6, 1.0)
                p['fused'] = CF_W * gd_calib + (1 - CF_W) * vis_calib
            else:
                p['fused'] = np.zeros(0)
        ap = dataset_ap(per_img, 'fused')
        gain = ap - base50
        nwpu_results[coco_cls] = float(gain)
        print(f'  {coco_cls}: base={base50:.4f}, gain={gain:+.4f}')
    all_results['nwpu_to_coco'] = {
        'mean': float(np.mean(list(nwpu_results.values()))) if nwpu_results else 0,
        'gains': nwpu_results,
    }
    print(f'  均值: {all_results["nwpu_to_coco"]["mean"]:+.4f}')

    # 保存
    output = {
        'config': {'fusion': 'C_fixed', 'w': CF_W, 'tau_det': CF_TAU_DET, 'tau_vis': CF_TAU_VIS},
        'results': all_results,
    }
    out_path = DATA / 'tge_v2_remaining_cfixed_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'总耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
