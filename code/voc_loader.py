# -*- coding: utf-8 -*-
"""VOC数据集加载模块
用于第二篇论文扩大样本量：增加VOC目标域
"""
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from pathlib import Path
import pickle

# VOC2007路径
VOC_DIR = Path(r'C:\Users\张三\Downloads\VOCdevkit\VOC2007')
VOC_IMG_DIR = VOC_DIR / 'JPEGImages'
VOC_ANN_DIR = VOC_DIR / 'Annotations'
VOC_IMGSETS_DIR = VOC_DIR / 'ImageSets' / 'Main'

# VOC类别 → COCO类别名称映射
VOC2COCO = {
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

# 和遥感对应的VOC类别
REMOTE_VOC_CLASSES = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train']


def parse_voc_xml(xml_path):
    """解析VOC XML标注文件，返回GT框列表和类别列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    names = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        names.append(name)
    return boxes, names


def load_voc(cls_name, n=200):
    """加载VOC某个类别的图片和GT框
    Args:
        cls_name: VOC类别名称（如'aeroplane', 'car'）
        n: 最大图片数
    Returns:
        images: PIL Image列表
        gts: GT框列表，每个元素是[[x1,y1,x2,y2], ...]
    """
    # 先检查ImageSets/Main/{cls_name}_test.txt是否存在
    test_file = VOC_IMGSETS_DIR / f'{cls_name}_test.txt'
    img_ids = []
    if test_file.exists():
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and int(parts[1]) == 1:
                    img_ids.append(parts[0])
    else:
        # 如果没有ImageSets，直接扫描Annotations目录
        for xml_path in sorted(VOC_ANN_DIR.glob('*.xml')):
            try:
                boxes, names = parse_voc_xml(xml_path)
                if cls_name in names:
                    img_ids.append(xml_path.stem)
            except Exception:
                continue

    images, gts = [], []
    for img_id in img_ids:
        img_path = VOC_IMG_DIR / f'{img_id}.jpg'
        xml_path = VOC_ANN_DIR / f'{img_id}.xml'
        if not img_path.exists() or not xml_path.exists():
            continue
        try:
            boxes, names = parse_voc_xml(xml_path)
            cls_boxes = [b for b, n in zip(boxes, names) if n == cls_name]
            if not cls_boxes:
                continue
            im = Image.open(img_path).convert('RGB')
            images.append(im)
            gts.append(cls_boxes)
            if len(images) >= n:
                break
        except Exception:
            continue
    return images, gts


def load_target_voc(voc_cls, n=200, detector=None, pre_filter=0.03):
    """加载VOC目标域数据（类似load_target_coco）
    Args:
        voc_cls: VOC类别名称
        n: 最大图片数
        detector: 检测器函数（如果为None，用GDINO）
        pre_filter: 预过滤阈值
    Returns:
        dict: {'per_img': [...], 'features': F, 'n_crops': N}
    """
    import sys
    sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
    import tge_probe as T
    import unified_eval as U

    images, gts = load_voc(voc_cls, n=n)
    if not images:
        return None

    # 检测（用缓存）
    cache_p = Path(r'C:\Users\张三\Desktop\实验数据') / f'tge_v2_dets_VOC-{voc_cls}.pkl'
    if cache_p.exists():
        dets = pickle.load(open(cache_p, 'rb'))
    else:
        dets = U.gdino_detect(images, voc_cls, 0.0)
        pickle.dump(dets, open(cache_p, 'wb'))

    per_img = []
    crops_all = []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < pre_filter:
                continue
            bl.append([x1, y1, x2, y2])
            sl.append(float(sc))
            ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })

    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return {'per_img': per_img, 'features': F, 'n_crops': len(crops_all)}


if __name__ == '__main__':
    # 测试
    print("Testing VOC loader...")
    for cls in ['aeroplane', 'car', 'bus']:
        imgs, gts = load_voc(cls, n=5)
        print(f"  {cls}: {len(imgs)} images, avg boxes={np.mean([len(g) for g in gts]):.1f}")
    print("Done!")
