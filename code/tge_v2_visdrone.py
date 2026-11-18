# -*- coding: utf-8 -*-
r"""tge_v2_visdrone.py — v2 冻结协议 §2.2: 第 4 源域 VisDrone → COCO (β=1.0, λ=0.25)

VisDrone 作为独立无人机源域, 类映射为**同名细到细** (与 DIOR 的 vehicle 粗到细
结构不同): car→car, truck→truck, bus→bus, motor→motorcycle。
负原型池 = VisDrone 全部 10 个 GT 类 (car/van/truck/bus/motor/pedestrian/people/
bicycle/tricycle/awning-tricycle), 检验小且部分近义(van~truck)的负原型池稳健性。

VisDrone 标注格式 (Detection): 每图一个 txt, 每行
  bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
category: 1=pedestrian 2=people 3=bicycle 4=car 5=van 6=truck 7=tricycle
          8=awning-tricycle 9=bus 10=motor
GT 的 score=1; 忽略 category=0 (ignored-regions)。

⚠️ 需要先设置 VISDRONE_DIR (用户提供路径/下载后填入)。检测复用 tge_v2 的 COCO 缓存。
运行: python tge_v2_visdrone.py   (~10 分钟, 主要是 COCO 特征一次编码)
输出: 实验数据/tge_v2_visdrone.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, pickle, argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import unified_eval as U
import tge_probe as T
from tge_v2_main import (load_coco_all, clip_proto, gt_crops, collect_features,
                         fused_arrays, dataset_ap, recall_proposals,
                         v2_cached_dets, FROZEN_BETA, FROZEN_LAM, K_CALIB,
                         SRC_TMPL, TGT_TMPL, DATA, PRE_FILTER)

# ── VisDrone 根目录 (含 images/ 与 annotations/) ──
VISDRONE_DIR = Path(r'C:\Users\张三\Downloads\VisDrone2019-DET-val\VisDrone2019-DET-val')

VISDRONE_CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                    'tricycle', 'awning-tricycle', 'bus', 'motor']
VISDRONE_CLS_ID = {c: i + 1 for i, c in enumerate(VISDRONE_CLASSES)}  # 1-10

# (COCO 评估类, VisDrone 源类)
TARGETS = [
    ('COCO-car',        'car'),
    ('COCO-truck',      'truck'),
    ('COCO-bus',        'bus'),
    ('COCO-motorcycle', 'motor'),
]


def load_visdrone(cls_name, n=20):
    cls_id = VISDRONE_CLS_ID[cls_name]
    ann_dir = VISDRONE_DIR / 'annotations'
    img_dir = VISDRONE_DIR / 'images'
    images, gts = [], []
    if not ann_dir.exists():
        print('  ❌ 标注目录不存在: %s (请设置 VISDRONE_DIR)' % ann_dir)
        return images, gts
    for txt in sorted(ann_dir.glob('*.txt')):
        boxes = []
        for line in txt.read_text(encoding='utf-8', errors='ignore').splitlines():
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            try:
                l, t, w, h, score, cat = [float(parts[i]) for i in range(5)] + [int(parts[5])]
            except Exception:
                continue
            if int(cat) != cls_id or score < 1:
                continue
            boxes.append([int(l), int(t), int(l + w), int(t + h)])
        if not boxes:
            continue
        img_path = img_dir / (txt.stem + '.jpg')
        if not img_path.exists():
            continue
        images.append(Image.open(img_path).convert('RGB'))
        gts.append(boxes)
        if len(images) >= n:
            break
    return images, gts


def main():
    global FROZEN_BETA, FROZEN_LAM
    import time
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC v2 VisDrone 源域 (可指定 β/λ)')
    ap.add_argument('--beta', type=float, default=FROZEN_BETA)
    ap.add_argument('--lam', type=float, default=FROZEN_LAM)
    args = ap.parse_args()
    FROZEN_BETA, FROZEN_LAM = args.beta, args.lam
    tag = 'b%g_l%g' % (FROZEN_BETA, FROZEN_LAM)
    out_path = DATA / ('tge_v2_visdrone_%s.json' % tag)
    print('配置: β=%.2f λ=%.2f  →  输出 %s' % (FROZEN_BETA, FROZEN_LAM, out_path.name))
    if not (VISDRONE_DIR / 'annotations').exists():
        print('❌ VisDrone 未就绪: %s' % VISDRONE_DIR)
        print('   请先设置 VISDRONE_DIR 为实际路径 (含 images/ 与 annotations/ 子目录)。')
        return

    print('[1/3] VisDrone %d 类源原型 ...' % len(VISDRONE_CLASSES))
    protos = {}
    for cls in VISDRONE_CLASSES:
        images, gts = load_visdrone(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
            print('  %-18s N=%2d 图, 原型 ✓' % (cls, len(images)))
        else:
            print('  %-18s 无图, 原型 ✗' % cls)

    print('[2/3] COCO 4 类: 复用检测缓存 + 特征一次编码 ...')
    class_data = []
    for name, vis_cls in TARGETS:
        coco_cls = name.split('-')[1]
        images, gts = load_coco_all(coco_cls)
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[vis_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, vis_cls)
        s_xs = T.text_emb([SRC_TMPL.format(vis_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        P_hat = P_src + FROZEN_BETA * Delta.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W = (P_hat.cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
        fused_arrays(per_img, F, W)
        a_base = dataset_ap(per_img, 'gd')
        a_ours = dataset_ap(per_img, 'fused')
        rec = recall_proposals(per_img)
        class_data.append({'name': name, 'base': round(a_base, 4), 'ours': round(a_ours, 4),
                           'gain': round(a_ours - a_base, 4), 'recall': round(rec, 4),
                           'n_test': len(images) - K_CALIB})
        print('  %-16s base=%.4f A-CSPC=%.4f (%+.4f) rec=%.4f' %
              (name, a_base, a_ours, a_ours - a_base, rec))

    print('[3/3] 汇总写盘 ...')
    mean_b = float(np.mean([d['base'] for d in class_data]))
    mean_o = float(np.mean([d['ours'] for d in class_data]))
    out = {'config': {'source': 'VisDrone (10 类池, 同名细到细映射)',
                      'target': 'COCO val2017 全量', 'beta': FROZEN_BETA,
                      'neg_lam': FROZEN_LAM,
                      'protocol': 'VisDrone->COCO, β/λ 沿用冻结值, 复用 tge_v2 COCO 缓存'},
           'classes': {d['name']: {k: d[k] for k in ['base', 'ours', 'gain', 'recall', 'n_test']}
                       for d in class_data},
           'mean': {'baseline': round(mean_b, 4), 'ours': round(mean_o, 4),
                    'gain': round(mean_o - mean_b, 4)}}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n' + '=' * 80)
    print('%-16s %9s %9s %9s %9s' % ('class', 'base', 'A-CSPC', '增益', 'recall'))
    print('-' * 80)
    for d in class_data:
        print('%-16s %9.4f %9.4f %+9.4f %9.4f' % (d['name'], d['base'], d['ours'],
                                                  d['gain'], d['recall']))
    print('-' * 80)
    print('%-16s %9.4f %9.4f %+9.4f %9s' % ('mean', mean_b, mean_o, mean_o - mean_b, '—'))
    print('=' * 80)
    print('→ %s   (耗时 %.1f 分钟)' % (out_path.name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
