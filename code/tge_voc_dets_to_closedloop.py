# -*- coding: utf-8 -*-
"""tge_voc_dets_to_closedloop.py — 把 TGE 已有的 GDINO VOC 检测缓存转成 balanced 闭环的 per_img/F 格式

背景: 元要求用"完整 balanced CA-PC 闭环"重跑 DIOR→VOC, 以与 DIOR→COCO 的 balanced 口径统一
(现有 tge_probe_voc.json 是 TGE 前身配置, 协议不统一)。

做法: TGE 已用 GDINO + prompt 'a photo of {voc_cls}' 在 VOC2007 上产出检测框
(tge_probe_dets_VOC-{cls}.pkl, 每图 900 候选框 + scores, N=100/类, 与 run_bidirectional.detect_encode
同检测器同 prompt)。本脚本只做格式转换 + 同款过滤 (PRE_FILTER=0.03 + 小框>10px) + CLIP 重编码,
产出 probe_voc_cache/fwd_dior2voc_{cls}_off20_n{N}.pkl, 与 probe_bidir_cache 完全同构,
供 run_closed_loop_balanced_voc.py 直接消费 (CPU 重算 AP, 不重跑 GPU 检测)。

2026-10-07: 小样本探针 N=20 (与 run_bidirectional_rtdetr.py 探针同规)。

用法 (GPU, CLIP 重编码更快):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cuda ^
  "C:/Users/张三/vg/Scripts/python.exe" -u tge_voc_dets_to_closedloop.py
"""
import os, sys, pickle, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path

import run_bidirectional as RB

RB.DEVICE = os.environ.get('AC_SPC_DET_DEVICE', 'cuda')
PRE_FILTER = RB.PRE_FILTER
N = int(os.environ.get('VOC_N', '20'))  # 小样本探针

VOC_DIR = Path(r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007')
DET_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
OUT_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_voc_cache')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (eval_cls 缓存键即 VOC 类名, voc_cls, src_cls DIOR)
TARGETS = [
    ('aeroplane', 'aeroplane', 'airplane'),
    ('boat', 'boat', 'ship'),
    ('car', 'car', 'vehicle'),
    ('bus', 'bus', 'vehicle'),
    ('motorbike', 'motorbike', 'vehicle'),
]


def load_voc(voc_cls, n=N):
    """VOC2007 trainval: 含该类 GT 的图按 xml 字典序取前 n 张 (与 TGE cached_dets 生成顺序一致)。"""
    ann_dir = VOC_DIR / 'Annotations'
    img_dir = VOC_DIR / 'JPEGImages'
    images, gts = [], []
    for xml_path in sorted(ann_dir.glob('*.xml')):
        root = ET.parse(xml_path).getroot()
        boxes = []
        for obj in root.findall('object'):
            if obj.find('name').text != voc_cls:
                continue
            b = obj.find('bndbox')
            boxes.append([float(b.find('xmin').text), float(b.find('ymin').text),
                          float(b.find('xmax').text), float(b.find('ymax').text)])
        if not boxes:
            continue
        img_path = img_dir / root.find('filename').text
        if not img_path.exists():
            continue
        images.append(Image.open(str(img_path)).convert('RGB'))
        gts.append(boxes)
        if len(images) >= n:
            break
    return images, gts


def convert():
    t0 = time.time()
    print(f'[convert] N={N}  PRE_FILTER={PRE_FILTER}  CLIP device={RB.DEVICE}')
    for eval_cls, voc_cls, src_cls in TARGETS:
        det_path = DET_DIR / f'tge_probe_dets_VOC-{voc_cls}.pkl'
        if not det_path.exists():
            print(f'  !! {det_path.name} 缺失, 跳过')
            continue
        dets = pickle.load(open(det_path, 'rb'))
        images, gts = load_voc(voc_cls)
        if len(images) > len(dets):
            print(f'  !! {voc_cls}: {len(images)} 图 > {len(dets)} dets, 截断')
            images, gts = images[:len(dets)], gts[:len(dets)]
        per_img, crops_all = [], []
        for i, (boxes, scores) in enumerate(dets[:len(images)]):
            bl, sl, ci = [], [], []
            for box, sc in zip(boxes, scores):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(images[i].width, x2), min(images[i].height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                if float(sc) < PRE_FILTER:
                    continue
                bl.append([x1, y1, x2, y2])
                sl.append(float(sc))
                ci.append(len(crops_all))
                crops_all.append(images[i].crop((x1, y1, x2, y2)))
            per_img.append({
                'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                'gt': np.array(gts[i], dtype=np.float64) if gts[i] else np.zeros((0, 4)),
                'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
            })
        F = RB.clip_encode_imgs(crops_all) if crops_all else np.zeros((0, 512), dtype=np.float64)
        key = f'fwd_dior2voc_{eval_cls}_off20_n{len(images)}'
        pickle.dump({'per_img': per_img, 'F': F}, open(OUT_DIR / f'{key}.pkl', 'wb'), protocol=4)
        nbox = int(sum(len(p['gd']) for p in per_img))
        print(f'  {key}: imgs={len(images)} kept_boxes={nbox} F={F.shape}')
    print(f'[convert] done {time.time()-t0:.0f}s -> {OUT_DIR}')


if __name__ == '__main__':
    convert()
