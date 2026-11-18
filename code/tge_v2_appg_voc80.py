# -*- coding: utf-8 -*-
r"""tge_v2_appg_voc80.py — A-CSPC + APPG: VOC→COCO 20类扩展验证

用 VOC2007 作为源域，跑 20 个 COCO 类别的跨域原型校正实验，
验证 APPG 物理先验自适应参数在更多类别上的普适性。

VOC→COCO 类别映射 (20类):
  aeroplane→airplane, bicycle→bicycle, bird→bird, boat→boat,
  bottle→bottle, bus→bus, car→car, cat→cat, chair→chair,
  cow→cow, diningtable→dining table, dog→dog, horse→horse,
  motorbike→motorcycle, person→person, pottedplant→potted plant,
  sheep→sheep, sofa→couch, train→train, tvmonitor→tv

运行: python tge_v2_appg_voc80.py --alpha 1.5
输出: 实验数据/tge_v2_appg_voc80_alpha1.5.json
"""
import sys, os, glob, json, pickle, argparse, time, xml.etree.ElementTree as ET
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         fused_arrays, dataset_ap, recall_proposals,
                         FROZEN_BETA, FROZEN_LAM, K_CALIB, TGT_TMPL, DATA, PRE_FILTER)

# APPG imports
from acspc.appg import PhysicsPriorLibrary, PhysicsPrior
from acspc.coco_categories import get_offline_prior

VOC_ROOT = r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007'
VOC_IMG_DIR = os.path.join(VOC_ROOT, 'JPEGImages')
VOC_ANN_DIR = os.path.join(VOC_ROOT, 'Annotations')

# VOC→COCO 类别映射
VOC_TO_COCO = {
    'aeroplane': 'airplane',
    'bicycle': 'bicycle',
    'bird': 'bird',
    'boat': 'boat',
    'bottle': 'bottle',
    'bus': 'bus',
    'car': 'car',
    'cat': 'cat',
    'chair': 'chair',
    'cow': 'cow',
    'diningtable': 'dining table',
    'dog': 'dog',
    'horse': 'horse',
    'motorbike': 'motorcycle',
    'person': 'person',
    'pottedplant': 'potted plant',
    'sheep': 'sheep',
    'sofa': 'couch',
    'train': 'train',
    'tvmonitor': 'tv',
}

# 源域文本模板 (自然图像)
SRC_TMPL_VOC = 'a photo of {}'


def load_voc_class(voc_class, n_max=100):
    """加载VOC某个类别的图像和标注框。

    Returns:
        (images, gts) 列表
    """
    images, gts = [], []
    # 遍历所有XML标注，找包含该类别的图像
    ann_files = sorted(glob.glob(os.path.join(VOC_ANN_DIR, '*.xml')))
    count = 0
    for ann_file in ann_files:
        if count >= n_max:
            break
        try:
            tree = ET.parse(ann_file)
            root = tree.getroot()
            # 检查是否包含目标类别
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
        except Exception as e:
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


def build_prior_library(categories):
    """构建物理先验库。"""
    print('[APPG] 构建物理先验库')
    lib = PhysicsPriorLibrary()
    for coco_cls in categories:
        v, m, s, i = get_offline_prior(coco_cls)
        prior = PhysicsPrior(category=coco_cls, V=v, M=m, S=s, I=i,
                             source='offline', confidence=0.6)
        lib.add(prior)
        print(f'  {coco_cls:<16s}: risk={prior.risk_score():.4f}')
    return lib


def get_adaptive_params(category, prior_lib, beta_base, lam_base, alpha):
    """根据物理先验风险计算自适应 β/λ。"""
    prior = prior_lib.get(category)
    risk = prior.risk_score() if prior else 0.5
    beta_eff = min(max(beta_base * (1.0 + alpha * risk), 0.0), 6.0)
    lambda_eff = min(max(lam_base * (1.0 + alpha * risk), 0.0), 2.0)
    return beta_eff, lambda_eff, risk


def cached_dets_coco(images, prompt, name):
    """加载或生成COCO检测缓存（优先用tge_probe缓存）。"""
    # 先试tge_probe缓存
    p1 = DATA / f'tge_probe_dets_COCO-{name}.pkl'
    if p1.exists():
        print(f'    [缓存] {p1.name}')
        return pickle.load(open(p1, 'rb'))
    # 再试tge_v2缓存
    p2 = DATA / f'tge_v2_dets_COCO-{name}.pkl'
    if p2.exists():
        print(f'    [缓存] {p2.name}')
        return pickle.load(open(p2, 'rb'))
    # 都没有，重新检测
    print(f'    [检测] {name} (无缓存)')
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p1, 'wb'))
    return dets


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC + APPG: VOC→COCO 20类扩展')
    ap.add_argument('--alpha', type=float, default=1.5, help='风险调制系数')
    ap.add_argument('--max-classes', type=int, default=20, help='最大类别数')
    args = ap.parse_args()

    print('=' * 100)
    print('A-CSPC + APPG: VOC→COCO 20类扩展验证')
    print('=' * 100)
    print(f'配置: α={args.alpha} β_base={FROZEN_BETA} λ_base={FROZEN_LAM}')

    # 选择类别
    categories = list(VOC_TO_COCO.values())[:args.max_classes]
    voc_classes = list(VOC_TO_COCO.keys())[:args.max_classes]
    print(f'类别数: {len(categories)}')

    # 构建先验库
    prior_lib = build_prior_library(categories)

    # 构建VOC源原型
    protos = build_voc_protos()

    # 跑每个类别
    print(f'\n{"="*100}')
    print('实验')
    print(f'{"="*100}')

    rows = {}
    for voc_cls, coco_cls in zip(voc_classes, categories):
        if voc_cls not in protos:
            print(f'  !! {voc_cls}→{coco_cls}: 无源原型，跳过')
            continue

        print(f'\n  [{voc_cls}→{coco_cls}]')
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print(f'    图太少 ({len(images)}), 跳过')
            continue

        prompt = TGT_TMPL.format(coco_cls)
        dets = cached_dets_coco(images, prompt, coco_cls)
        per_img, F = collect_features(images, gts, dets)

        P_src = protos[voc_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, voc_cls)
        s_xs = T.text_emb([SRC_TMPL_VOC.format(voc_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs

        # 自适应参数
        beta_eff, lambda_eff, risk = get_adaptive_params(
            coco_cls, prior_lib, FROZEN_BETA, FROZEN_LAM, args.alpha)

        # baseline
        base50 = dataset_ap(per_img, 'gd')

        # 固定参数 A-CSPC
        P_hat_fixed = P_src + FROZEN_BETA * Delta.to(U.DEVICE)
        P_hat_fixed = P_hat_fixed / P_hat_fixed.norm()
        W_fixed = (P_hat_fixed.cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
        fused_arrays(per_img, F, W_fixed)
        fixed50 = dataset_ap(per_img, 'fused')

        # 自适应参数 A-CSPC
        P_hat_adapt = P_src + beta_eff * Delta.to(U.DEVICE)
        P_hat_adapt = P_hat_adapt / P_hat_adapt.norm()
        W_adapt = (P_hat_adapt.cpu().numpy() - lambda_eff * P_neg.cpu().numpy())
        fused_arrays(per_img, F, W_adapt)
        adapt50 = dataset_ap(per_img, 'fused')

        rec = recall_proposals(per_img)

        rows[coco_cls] = {
            'voc_cls': voc_cls,
            'n_pool': len(images), 'n_test': len(images) - K_CALIB,
            'recall': round(rec, 4),
            'risk': round(risk, 4),
            'beta_eff': round(beta_eff, 4),
            'lambda_eff': round(lambda_eff, 4),
            'base50': round(base50, 4),
            'fixed50': round(fixed50, 4),
            'adapt50': round(adapt50, 4),
            'gain_fixed': round(fixed50 - base50, 4),
            'gain_adapt': round(adapt50 - base50, 4),
            'adapt_vs_fixed': round(adapt50 - fixed50, 4),
        }
        print(f'    risk={risk:.3f} β={beta_eff:.3f} λ={lambda_eff:.3f} | '
              f'base={base50:.4f} fixed={fixed50:.4f} adapt={adapt50:.4f} | '
              f'Δfixed={fixed50-base50:+.4f} Δadapt={adapt50-base50:+.4f} '
              f'diff={adapt50-fixed50:+.4f}')

    # 汇总
    print(f'\n{"="*100}')
    print('VOC→COCO 20类汇总')
    print(f'{"="*100}')
    print(f'{"类别":<16} {"Risk":>6} {"Base":>8} {"Fixed":>8} {"Adapt":>8} '
          f'{"ΔFixed":>9} {"ΔAdapt":>9} {"Diff":>9}')
    print(f'{"-"*100}')
    for coco_cls, r in rows.items():
        print(f'{coco_cls:<16} {r["risk"]:>6.3f} {r["base50"]:>8.4f} '
              f'{r["fixed50"]:>8.4f} {r["adapt50"]:>8.4f} '
              f'{r["gain_fixed"]:>+9.4f} {r["gain_adapt"]:>+9.4f} '
              f'{r["adapt_vs_fixed"]:>+9.4f}')
    print(f'{"-"*100}')

    mean_row = {k: round(float(np.mean([rows[c][k] for c in rows])), 4)
                 for k in ['base50', 'fixed50', 'adapt50', 'gain_fixed', 'gain_adapt',
                           'adapt_vs_fixed', 'risk']}
    n_pos_fixed = sum(1 for c in rows.values() if c['gain_fixed'] > 0)
    n_pos_adapt = sum(1 for c in rows.values() if c['gain_adapt'] > 0)
    n_improve = sum(1 for c in rows.values() if c['adapt_vs_fixed'] > 0)

    print(f'{"MEAN":<16} {mean_row["risk"]:>6.3f} {mean_row["base50"]:>8.4f} '
          f'{mean_row["fixed50"]:>8.4f} {mean_row["adapt50"]:>8.4f} '
          f'{mean_row["gain_fixed"]:>+9.4f} {mean_row["gain_adapt"]:>+9.4f} '
          f'{mean_row["adapt_vs_fixed"]:>+9.4f}')
    print(f'\n正增益类别: fixed={n_pos_fixed}/{len(rows)}, adapt={n_pos_adapt}/{len(rows)}')
    print(f'自适应优于固定: {n_improve}/{len(rows)}')
    print(f'{"="*100}')

    # 保存
    out_path = DATA / f'tge_v2_appg_voc80_alpha{args.alpha}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {'alpha': args.alpha, 'beta_base': FROZEN_BETA,
                       'lam_base': FROZEN_LAM, 'source': 'VOC2007', 'target': 'COCO'},
            'classes': rows,
            'mean': mean_row,
            'stats': {
                'n_classes': len(rows),
                'n_pos_fixed': n_pos_fixed,
                'n_pos_adapt': n_pos_adapt,
                'n_improve': n_improve,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至: {out_path}')
    print(f'耗时: {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
