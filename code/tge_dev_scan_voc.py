# -*- coding: utf-8 -*-
r"""tge_dev_scan_voc.py — v2 冻结协议 §1: β/λ 开发域选参 (DIOR→VOC2007, 零目标域信息)

把 (β, λ) 的确定从 DIOR→COCO 目标域扫描改为 DIOR→VOC2007 开发域扫描:
  - 开发域 = VOC (航拍→自然, 与 COCO 同转移类型)
  - 网格 β ∈ {0.5, 1.0, 2.0, 4.0} × λ ∈ {0, 0.25, 0.5, 0.75, 1.0}
  - 选择准则 (协议 §1):
      1) 5 类平均增益 (A-CSPC − baseline) 最大
      2) 并列时取 λ 较小者 (保守优先)
      3) 要求 5/5 类非负增益, 否则记为"开发域选择失败"并照实报告
  - 冻结 (β*, λ*) 用于全部测试域 (DIOR/NWPU/DOTA/VisDrone → COCO)
  - 本脚本**只看 VOC GT**, 不看任何目标域 (COCO) 结果

实现: 每类候选框 CLIP 特征 F 只编码一次, 在 (β, λ) 网格上复用
      (检测缓存复用 tge_probe_voc 的 tge_probe_dets_VOC-*.pkl)。

输出: 实验数据/tge_dev_scan_voc.json
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
OUT = DATA / 'tge_dev_scan_voc.json'
VOC_DIR = Path(r'C:\Users\张三\modeling\voc_data\VOCdevkit\VOC2007')

N_VOC = 100
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
BETAS = [0.5, 1.0, 2.0, 4.0]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# (评估名, VOC 目标类, DIOR 源原型类)
TARGETS = [
    ('VOC-aeroplane',  'aeroplane',  'airplane'),
    ('VOC-boat',       'boat',       'ship'),
    ('VOC-car',        'car',        'vehicle'),
    ('VOC-bus',        'bus',        'vehicle'),
    ('VOC-motorbike',  'motorbike',  'vehicle'),
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
        from PIL import Image
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


def collect_features(images, gts, dets):
    """pre-filter + 小框过滤 → (boxes_all, gd, F, img_ids); F 只编码一次。"""
    crops, boxes_all, gd_all, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            img_ids.append(ii)
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    if crops:
        F = T.encode_clip_batch(crops).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    return np.array(boxes_all), gd, F, img_ids


def fused_ap(boxes, gd, F, img_ids, n_img, gts, P_hat, neg_lam):
    """sqrt(gd * max(0, F·P̂ − λ·F·P_neg)) → 逐图 AP50。P_hat/P_neg 已含 λ 处理。"""
    vis = np.maximum(0.0, F @ P_hat)
    fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
    return U.compute_ap50_img(U.split_by_img(boxes, fused, img_ids, n_img), gts)


def main():
    if not (VOC_DIR / 'JPEGImages').exists():
        print('❌ VOC2007 未就绪')
        return

    print('[1/3] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/3] VOC 5 类: 特征一次性编码 ...')
    class_data = []
    for name, voc_cls, src_cls in TARGETS:
        images, gts = load_voc(voc_cls)
        if len(images) < 20:
            print('  !! %s 图太少 (%d), 跳过' % (name, len(images)))
            continue
        dets = T.cached_dets(images, TGT_TMPL.format(voc_cls), name)  # 复用 voc 缓存
        boxes, gd, F, img_ids = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(voc_cls)]).cpu()[0]
        Delta = d_y - s_xs
        class_data.append({'name': name, 'images': images, 'gts': gts, 'boxes': boxes,
                           'gd': gd, 'F': F, 'img_ids': img_ids, 'P_src': P_src,
                           'P_neg': P_neg, 'Delta': Delta, 'n_img': len(images)})
        print('  %-14s N=%3d 框=%6d' % (name, len(images), len(boxes)))

    print('[3/3] β×λ 网格 ...')
    grid = {}
    per_class = {}
    for b in BETAS:
        for lam in LAMBDAS:
            aps = {}
            for d in class_data:
                P_hat = d['P_src'] + b * d['Delta'].to(U.DEVICE)
                P_hat = P_hat / P_hat.norm()
                W = (P_hat.cpu().numpy() - lam * d['P_neg'].cpu().numpy())
                # 一次性预乘: F @ W 等价于 F·P̂ − λ·F·P_neg
                ap = fused_ap(d['boxes'], d['gd'], d['F'], d['img_ids'],
                              d['n_img'], d['gts'], W, lam)
                aps[d['name']] = round(float(ap), 4)
            grid[f'b{b}_l{lam}'] = aps

    # 每类 baseline (仅 gd)
    base = {}
    for d in class_data:
        base[d['name']] = round(float(U.compute_ap50_img(
            U.split_by_img(d['boxes'], d['gd'], d['img_ids'], d['n_img']), d['gts'])), 4)
    per_class['baseline'] = base

    # 选择准则 (协议 §1): 在满足 5/5 非负的配置中取 mean_gain 最大;
    # 并列取 λ 最小 (保守优先); 若无一配置满足 5/5 非负 → 记录开发域选择失败。
    results = []
    for key, aps in grid.items():
        gains = {c: aps[c] - base[c] for c in base}
        mean_gain = float(np.mean(list(gains.values())))
        all_pos = all(g >= 0 for g in gains.values())
        lam = float(key.split('l')[1])
        results.append({'key': key, 'mean_gain': round(mean_gain, 4),
                        'all_positive': all_pos, 'lambda': lam,
                        'aps': aps, 'gains': gains})
    results.sort(key=lambda r: (-r['mean_gain'], r['lambda']))
    passing = [r for r in results if r['all_positive']]
    best = passing[0] if passing else None
    if best is not None:
        chosen = {'beta': float(best['key'].split('b')[1].split('_')[0]),
                  'lambda': best['lambda'], 'mean_gain': best['mean_gain'],
                  'all_positive': True, 'selected_from': 'passing (5/5 non-neg)'}
    else:
        chosen = {'beta': None, 'lambda': None, 'mean_gain': None,
                  'all_positive': False, 'selected_from': 'FAIL (no 5/5 non-neg config)'}

    print('\n' + '=' * 110)
    print('%-16s %10s %10s %8s' % ('(β,λ)', '5类均值增益', '全部类非负', 'λ'))
    print('-' * 110)
    for r in results:
        mark = '  <- 选中' if best is not None and r['key'] == best['key'] else ''
        print('%-16s %+10.4f %10s %8s%s' % (r['key'], r['mean_gain'], r['all_positive'],
                                            r['lambda'], mark))
    print('=' * 110)
    if best is not None:
        print('选择: β=%.1f λ=%.1f, mean_gain=%+.4f (5/5 非负集合内最优)' %
              (chosen['beta'], chosen['lambda'], chosen['mean_gain']))
    else:
        print('⚠️ 开发域选择失败: 无配置满足 5/5 非负, 按协议 §1 照实报告')

    out = {'config': {'protocol': 'v2 frozen §1: DIOR->VOC2007 dev scan, N=100/类, 逐图 AP50, '
                                  '只看 VOC GT, 不看目标域',
                      'betas': BETAS, 'lambdas': LAMBDAS, 'n_voc': N_VOC},
           'baseline': base, 'grid': grid,
           'ranking': results[:20],
           'chosen': chosen}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
