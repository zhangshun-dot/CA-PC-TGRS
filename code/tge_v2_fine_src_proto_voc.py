# -*- coding: utf-8 -*-
r"""tge_v2_fine_src_proto_voc.py — VOC 细类别感知源原型构造实验

DIOR -> VOC2007, car/bus/motorbike 映射到 vehicle (粗->细)
P_src_fine(c) = norm(P_vehicle + gamma * (text_emb("a photo of c") - text_emb("a photo of vehicle")))
s_xs 保持粗 "an overhead image of vehicle"
beta=2.0, lambda=0.5 固定

扫描 gamma in {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
aeroplane/boat 保持原细到细原型 (G=0)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import xml.etree.ElementTree as ET
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
VOC_DIR = Path(r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007')

BETA = 2.0
LAMBDA = 0.5
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
N_VOC = 100
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
PRE_FILTER = 0.03

TARGETS = [
    ('VOC-aeroplane', 'aeroplane', 'airplane', False),
    ('VOC-boat',      'boat',      'ship',     False),
    ('VOC-car',       'car',       'vehicle',  True),
    ('VOC-bus',       'bus',       'vehicle',  True),
    ('VOC-motorbike', 'motorbike', 'vehicle',  True),
]


def load_voc(cls_name, n=N_VOC):
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
        images.append(Image.open(str(img_path)).convert('RGB'))
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


def cached_dets(images, prompt, name):
    p = DATA / f'tge_probe_dets_{name}.pkl'
    if p.exists():
        print(f'    [缓存] {p.name}')
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def collect_features(images, gts, dets):
    """收集所有检测框的 CLIP 特征, 返回扁平数组."""
    boxes_all, gd_all, feats, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > PRE_FILTER
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
    if feats:
        F = T.encode_clip_batch(feats).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    return boxes_all, gd, F, img_ids, len(images)


def compute_ap_with_W(boxes_all, gd, F, img_ids, n_img, gts, W):
    """给定 W (numpy array), 计算 AP50."""
    if len(boxes_all) == 0:
        return 0.0
    vis = np.maximum(0.0, F @ W)
    fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
    preds = U.split_by_img(boxes_all, fused, img_ids, n_img)
    return U.compute_ap50_img(preds, gts)


def main():
    t0 = time.time()
    print('=' * 90)
    print('VOC 细类别感知源原型构造实验 (β=2.0, λ=0.5 固定)')
    print('=' * 90)

    if not (VOC_DIR / 'JPEGImages').exists():
        print('VOC2007 未就绪')
        return

    # [1] DIOR 源原型
    print('[1/4] DIOR 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
    P_vehicle = protos['vehicle'].to(U.DEVICE)

    # [2] 文本嵌入
    print('[2/4] 文本嵌入 ...')
    d_y = {}
    for name, voc_cls, src_cls, needs_fine in TARGETS:
        d_y[voc_cls] = T.text_emb([TGT_TMPL.format(voc_cls)]).cpu()[0]
    s_xs_vehicle = T.text_emb([SRC_TMPL.format('vehicle')]).cpu()[0]
    text_vehicle = T.text_emb([TGT_TMPL.format('vehicle')]).cpu()[0]
    text_c = {}
    for name, voc_cls, src_cls, needs_fine in TARGETS:
        if needs_fine:
            text_c[voc_cls] = T.text_emb([TGT_TMPL.format(voc_cls)]).cpu()[0]

    # [3] VOC 检测 + 特征
    print('[3/4] VOC 5 类检测 + 特征 ...')
    class_data = []
    for name, voc_cls, src_cls, needs_fine in TARGETS:
        images, gts = load_voc(voc_cls)
        if len(images) < 20:
            print('  !! %s 图太少 (%d), 跳过' % (name, len(images)))
            continue
        dets = cached_dets(images, TGT_TMPL.format(voc_cls), name)
        boxes_all, gd, F, img_ids, n_img = collect_features(images, gts, dets)
        P_neg = T.get_neg_proto(protos, src_cls).cpu().numpy()
        # baseline AP
        base_preds = U.split_by_img(boxes_all, gd, img_ids, n_img)
        base50 = U.compute_ap50_img(base_preds, gts)
        class_data.append({'name': name, 'voc_cls': voc_cls, 'src_cls': src_cls,
                           'needs_fine': needs_fine, 'boxes_all': boxes_all, 'gd': gd,
                           'F': F, 'img_ids': img_ids, 'n_img': n_img, 'gts': gts,
                           'P_neg': P_neg, 'base50': round(base50, 4)})
        print('  %-16s N=%3d 框=%6d  base=%.4f  fine=%s' %
              (name, n_img, len(boxes_all), base50, needs_fine))

    # [4] gamma 扫描
    print('[4/4] gamma 扫描 ...')
    results = {}
    for d in class_data:
        name = d['name']
        voc_cls = d['voc_cls']
        base50 = d['base50']
        gamma_results = {}

        if d['needs_fine']:
            Delta = d_y[voc_cls] - s_xs_vehicle
        else:
            s_xs = T.text_emb([SRC_TMPL.format(d['src_cls'])]).cpu()[0]
            Delta = d_y[voc_cls] - s_xs

        for gamma in GAMMAS:
            if d['needs_fine']:
                offset = (text_c[voc_cls] - text_vehicle).to(U.DEVICE)
                P_src = P_vehicle + gamma * offset
                P_src = P_src / P_src.norm()
            else:
                P_src = protos[d['src_cls']].to(U.DEVICE)

            P_hat = P_src + BETA * Delta.to(U.DEVICE)
            P_hat = P_hat / P_hat.norm()
            W = P_hat.cpu().numpy() - LAMBDA * d['P_neg']

            ap50 = compute_ap_with_W(d['boxes_all'], d['gd'], d['F'],
                                      d['img_ids'], d['n_img'], d['gts'], W)
            gamma_results[gamma] = {'ap50': round(ap50, 4), 'gain': round(ap50 - base50, 4)}

        best_gamma = max(GAMMAS, key=lambda g: gamma_results[g]['gain'])
        results[name] = {'base50': base50, 'gamma_scan': gamma_results,
                         'best_gamma': best_gamma, 'best_gain': gamma_results[best_gamma]['gain'],
                         'gamma0_gain': gamma_results[0.0]['gain'], 'needs_fine': d['needs_fine']}
        gains_str = '  '.join(f'γ={g}:{gamma_results[g]["gain"]:+.3f}' for g in GAMMAS)
        print('  %-16s base=%.4f  %s  | 最优γ=%.2f (%+.4f)' %
              (name, base50, gains_str, best_gamma, gamma_results[best_gamma]['gain']))

    # 汇总
    print('\n' + '=' * 90)
    print('VOC 汇总表')
    print('=' * 90)
    header = '%-16s %8s' % ('类别', 'base')
    for g in GAMMAS:
        header += ' %8s' % ('γ=%.2f' % g)
    header += ' %10s %10s' % ('最优γ', '最优增益')
    print(header)
    print('-' * 130)
    for name, r in results.items():
        gains = [r['gamma_scan'][g]['gain'] for g in GAMMAS]
        row = '%-16s %8.4f' % (name, r['base50'])
        for g in gains:
            row += ' %+8.4f' % g
        row += ' %10.2f %+10.4f' % (r['best_gamma'], r['best_gain'])
        print(row)

    # 细原型 3 类均值
    fine_results = {k: v for k, v in results.items() if v['needs_fine']}
    if fine_results:
        n_fine = len(fine_results)
        print('\n--- 细原型 3 类 (car/bus/motorbike) ---')
        for g in GAMMAS:
            avg = sum(r['gamma_scan'][g]['gain'] for r in fine_results.values()) / n_fine
            print('  γ=%.2f 均值增益: %+.4f' % (g, avg))
        best_avg = sum(r['best_gain'] for r in fine_results.values()) / n_fine
        print('  逐类最优γ 均值: %+.4f' % best_avg)

    # 保存
    out = {'config': {'beta': BETA, 'lambda': LAMBDA, 'gammas': GAMMAS,
                       'protocol': 'DIOR->VOC2007 fine source prototype, scheme A'},
           'results': results}
    out_path = DATA / 'tge_v2_fine_src_proto_voc.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n→ %s  (耗时 %.1f 分钟)' % (out_path.name, (time.time()-t0)/60))


if __name__ == '__main__':
    main()
