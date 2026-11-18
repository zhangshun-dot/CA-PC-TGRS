# -*- coding: utf-8 -*-
"""tge_probe_voc.py — 第二正向目标域: DIOR → VOC2007 (5 类, 与主表同协议)

期刊补强 (多域对验证): 主结论 (外推修复失效类) 在第二个正向跨域对上复现。
类映射 (DIOR 源原型 → VOC 目标类):
  airplane → aeroplane, ship → boat, vehicle → car/bus/motorbike (粗→细, 同主表结构)
行: baseline / B2(CLIP 真实原型) / TGE(λ=0 纯外推) / A-CSPC(外推+负原型 λ=0.5)
协议与 tge_probe_fs 完全一致 (N=100/类, pre-filter 0.03, 小框过滤, 逐图 AP50, sqrt(gd*vis))。
检测 prompt 'a photo of {voc_cls}', 缓存 tge_probe_dets_VOC-*.pkl。
输出: 实验数据/tge_probe_voc.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_voc.json'
VOC_DIR = Path(r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007')

BETA = 2.0
N_VOC = 100
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

# (评估名, VOC 目标类, DIOR 源原型类)
TARGETS = [
    ('VOC-aeroplane',  'aeroplane',  'airplane'),
    ('VOC-boat',       'boat',       'ship'),
    ('VOC-car',        'car',        'vehicle'),
    ('VOC-bus',        'bus',        'vehicle'),
    ('VOC-motorbike',  'motorbike',  'vehicle'),
]


def load_voc(cls_name, n=N_VOC):
    """VOC2007 trainval: 该类所有含 GT 的图 (xml 解析), 取前 n 张。"""
    ann_dir = VOC_DIR / 'Annotations'
    img_dir = VOC_DIR / 'JPEGImages'
    images, gts = [], []
    for xml_path in sorted(ann_dir.glob('*.xml')):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            if obj.find('name').text != cls_name:
                continue
            b = obj.find('bndbox')
            boxes.append([float(b.find('xmin').text), float(b.find('ymin').text),
                          float(b.find('xmax').text), float(b.find('ymax').text)])
        if not boxes:
            continue
        img_path = img_dir / (root.find('filename').text)
        if not img_path.exists():
            continue
        images.append(__import__('PIL.Image', fromlist=['Image']).open(str(img_path)).convert('RGB'))
        gts.append(boxes)
        if len(images) >= n:
            break
    return images, gts


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


def eval_class(images, gts, dets, P_cal, P_neg=None):
    """逐图 AP50 (同 tge_probe_fs): P_neg=None → TGE 行 (无负原型)。"""
    boxes_all, gd_all, feats, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(img.crop((x1, y1, x2, y2)))
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            img_ids.append(ii)
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
    if feats:
        F = T.encode_clip_batch(feats).cpu().numpy()
        if P_neg is not None:
            vis = np.maximum(0.0, F @ P_cal.cpu().numpy()
                             - U.NEG_LAM * (F @ P_neg.cpu().numpy()))
        else:
            vis = np.maximum(0.0, F @ P_cal.cpu().numpy())
        fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
        ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    else:
        ap = base
    return base, ap


def main():
    if not (VOC_DIR / 'JPEGImages').exists():
        print('❌ VOC2007 未就绪, 先跑 tge_download_voc.py')
        return

    print('[1/2] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/2] VOC 5 类评估 (N=%d, 逐图 AP50) ...' % N_VOC)
    rows = {}
    for name, voc_cls, src_cls in TARGETS:
        images, gts = load_voc(voc_cls)
        if len(images) < 20:
            print('  !! %s 图太少 (%d), 跳过' % (name, len(images)))
            continue
        dets = T.cached_dets(images, TGT_TMPL.format(voc_cls), name)

        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(voc_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()

        base, _ = eval_class(images, gts, dets, P_src, P_neg)
        _, ap_b2 = eval_class(images, gts, dets, P_src, P_neg)
        _, ap_tge = eval_class(images, gts, dets, P_hat, None)
        _, ap_cspc = eval_class(images, gts, dets, P_hat, P_neg)

        rows[name] = {'baseline': round(base, 4), 'b2': round(ap_b2, 4),
                      'tge_lam0': round(ap_tge, 4), 'acspc': round(ap_cspc, 4),
                      'tge_gain': round(ap_tge - base, 4),
                      'acspc_gain': round(ap_cspc - base, 4),
                      'n_images': len(images)}
        print('  %-14s N=%3d base=%.4f  B2=%.4f  TGE=%.4f  A-CSPC=%.4f  [TGE %+.4f | A %+.4f]'
              % (name, len(images), base, ap_b2, ap_tge, ap_cspc,
                 ap_tge - base, ap_cspc - base))

    if not rows:
        print('❌ 无有效类')
        return

    mean_row = {k: round(float(np.mean([rows[n][k] for n in rows])), 4)
                for k in ['baseline', 'b2', 'tge_lam0', 'acspc', 'tge_gain', 'acspc_gain']}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_voc': N_VOC, 'beta': BETA, 'neg_lam': U.NEG_LAM,
                              'protocol': 'DIOR->VOC2007, N=100/类, 逐图 AP50, 后同主表口径'},
                   'classes': rows, 'mean': mean_row}, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 100)
    print('%-14s %9s %9s %9s %9s %9s %9s' %
          ('class', 'baseline', 'B2', 'TGE(λ0)', 'A-CSPC', 'TGE增益', 'A增益'))
    print('-' * 100)
    for name in [t[0] for t in TARGETS]:
        r = rows.get(name)
        if not r:
            continue
        print('%-14s %9.4f %9.4f %9.4f %9.4f %+9.4f %+9.4f' %
              (name, r['baseline'], r['b2'], r['tge_lam0'], r['acspc'],
               r['tge_gain'], r['acspc_gain']))
    print('-' * 100)
    print('%-14s %9.4f %9.4f %9.4f %9.4f %+9.4f %+9.4f   <- mean' %
          ('mean', mean_row['baseline'], mean_row['b2'], mean_row['tge_lam0'],
           mean_row['acspc'], mean_row['tge_gain'], mean_row['acspc_gain']))
    print('=' * 100)
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
