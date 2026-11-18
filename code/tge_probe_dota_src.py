# -*- coding: utf-8 -*-
"""tge_probe_dota_src.py — 第三源域验证: DOTA v1.5 → COCO (与主表 DIOR→COCO 同构)

源域独立性矩阵: DIOR (主表, 20 类) / NWPU VHR-10 (tge_probe_nwpu, 10 类) / DOTA v1.5 (本脚本)。
三个独立遥感源域 → 同一 COCO 目标, 主结论 (A-CSPC 增益) 在第三个源域上是否复现。

协议与主表完全一致 (tge_probe_fs): N=100/类, K=10, 后 90 图测试,
逐图 IoU 匹配 + dataset-level 11-point AP50; GDINO th=0 → pre-filter 0.03;
β=2, λ=0.5 不调参 (基于 DIOR→COCO 定, 未针对 DOTA 调整); sqrt(gd*vis)。

类映射 (DOTA 源 → COCO 目标, 与 DIOR/NWPU 同构):
  plane → airplane (同名), ship → boat (同名), vehicle (large+small 合并) → car/truck/bus/motorcycle
负原型池 = DOTA v1.5 全部 16 类 (完整源域结构, 同 DIOR 20 / NWPU 10)。
COCO 检测结果复用主表缓存 (tge_probe_dets_COCO-*.pkl, 检测与源域无关)。
外推公式同主表: P_hat = P_src + β·(d_y − s_xs), s='an overhead image of {dota_cls}',
d='a photo of {coco_cls}' (DOTA 遥感 → COCO 自然)。

输出: 实验数据/tge_probe_dota_src.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_dota_src.json'

K = 10
BETA = 2.0
N_COCO = 100
N_DOTA = 10            # 每类 10 张大图 (与 tge_probe_dota 一致), 每图 GT 框数十个
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
IMG_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
LBL_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'

# DOTA v1.5 全部 16 类 (负原型池完整)
DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
    'container-crane',
]

# (name, coco_cls, dota_src_cls)
TARGETS = [
    ('COCO-airplane',   'airplane',   'plane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


def load_dota(dota_cls, n=N_DOTA):
    """DOTA 类 → (images, gts) 轴对齐框 (quad min/max, 同 tge_probe_dota)。"""
    quads_by_img = defaultdict(list)
    for fn in Path(LBL_DIR).glob('*.txt'):
        img_name = fn.stem + '.png'
        img_path = Path(IMG_DIR) / img_name
        if not img_path.exists():
            continue
        with open(fn) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                q = list(map(float, parts[:8]))
                if parts[8] == dota_cls:
                    xs, ys = q[0::2], q[1::2]
                    quads_by_img[img_name].append([min(xs), min(ys), max(xs), max(ys)])
            except Exception:
                continue
    images, gts = [], []
    for img_name, boxes in quads_by_img.items():
        try:
            im = Image.open(Path(IMG_DIR) / img_name).convert('RGB')
        except Exception:
            continue
        images.append(im)
        gts.append(boxes)
        if len(images) >= n:
            break
    return images, gts


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


def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def collect_boxes(images, dets, gts):
    """GDINO 分数 (概率尺度) + pre-filter 0.03 + 小框过滤, 一次 CLIP 编码 (同主表)。"""
    crops_all, per_img = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl = [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < U.PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2])
            sl.append(float(sc))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': (np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0)
                               if sl else np.zeros(0)),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4))})
    if crops_all:
        print('    ... %d 个 crop 待 CLIP 编码 (bs=256, %d 批) ...'
              % (len(crops_all), (len(crops_all) + 255) // 256), flush=True)
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    fi = 0
    for p in per_img:
        n = len(p['gd'])
        p['F'] = F[fi:fi + n] if n else np.zeros((0, 512))
        fi += n
    return per_img


def score_ours(per_img, W_hat):
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, p['F'] @ W_hat)
            p['ours'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['ours'] = np.zeros(0)


def ap_90(per_img, key):
    sel = [per_img[i] for i in range(K, len(per_img))]
    return U.compute_ap50_img([(p['box'], p[key]) for p in sel], [p['gt'] for p in sel])


def recall_pf(per_img):
    n_gt, hit = 0, 0
    for p in per_img:
        gts = p['gt']
        boxes = p['box']
        n_gt += len(gts)
        if len(boxes) == 0:
            continue
        matched = np.zeros(len(gts), dtype=bool)
        for b in boxes:
            if matched.all():
                break
            for j, gb in enumerate(gts):
                if matched[j]:
                    continue
                if U.iou(b.tolist(), gb.tolist()) >= 0.5:
                    matched[j] = True
        hit += int(matched.sum())
    return hit / n_gt if n_gt else float('nan')


def main():
    import faulthandler
    faulthandler.enable()
    t0 = time.time()
    if not Path(IMG_DIR).exists():
        print('❌ DOTA v1.5 数据未就绪: %s' % IMG_DIR)
        return

    print('[1/3] DOTA v1.5 16 类源原型 ...')
    protos = {}
    for cls in DOTA_CLASSES:
        images, gts = load_dota(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p
            print('  %-20s N=%2d 图, 原型 ✓' % (cls, len(images)))
        else:
            print('  %-20s 无图, 原型 ✗' % cls)

    # vehicle 合并原型: large-vehicle + small-vehicle
    imgs_v, gts_v = [], []
    for vc in ('large-vehicle', 'small-vehicle'):
        im2, gt2 = load_dota(vc)
        imgs_v += im2
        gts_v += gt2
    protos['vehicle'] = clip_proto(gt_crops(imgs_v, gts_v))
    print('  %-20s N=%2d 图 (large+small 合并), 原型 ✓' % ('vehicle', len(imgs_v)))

    print('[2/3] COCO 6 类测试: GDINO 检测 (主表缓存) + CLIP 特征一次性编码 ...')
    class_data = []
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)  # 命中主表缓存
        per_img = collect_boxes(images, dets, gts)
        n_boxes = sum(len(p['gd']) for p in per_img)
        print('    %-16s N=%3d 框=%6d (pre-filter 0.03 后)' % (name, len(images), n_boxes),
              flush=True)
        class_data.append({'name': name, 'coco_cls': coco_cls, 'src_cls': src_cls,
                           'per_img': per_img, 'n_test': len(images) - K})

    print('[3/3] A-CSPC 重打分 (β=%.1f, λ=%.1f 固定, 未针对 DOTA 调整) ...' % (BETA, U.NEG_LAM))
    rows = {}
    for d in class_data:
        name, coco_cls, src_cls = d['name'], d['coco_cls'], d['src_cls']
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W_hat = (P_hat.cpu().numpy() - U.NEG_LAM * P_neg.cpu().numpy())
        score_ours(d['per_img'], W_hat)
        ap_base = ap_90(d['per_img'], 'gd')
        ap_ours = ap_90(d['per_img'], 'ours')
        rows[name] = {'src_cls': src_cls, 'baseline': round(ap_base, 4),
                      'acspc': round(ap_ours, 4), 'gain': round(ap_ours - ap_base, 4),
                      'recall': round(recall_pf(d['per_img']), 4)}
        print('  %-16s base=%.4f  A-CSPC=%.4f  [%+.4f]  rec=%.4f'
              % (name, ap_base, ap_ours, ap_ours - ap_base, rows[name]['recall']))

    mean_b = float(np.mean([rows[n]['baseline'] for n in rows]))
    mean_o = float(np.mean([rows[n]['acspc'] for n in rows]))
    out = {'config': {'source': 'DOTA v1.5 (16 类, 每类 10 大图)', 'target': 'COCO val2017',
                      'n_coco': N_COCO, 'k': K, 'beta': BETA, 'neg_lam': U.NEG_LAM,
                      'pre_filter': U.PRE_FILTER,
                      'protocol': 'GDINO boxes + A-CSPC rerank, N=100/类, K=10, 后90图, '
                                  '逐图 AP50, pre-filter 0.03, sqrt(gd*vis), 与主表完全一致',
                      'note': 'β/λ 沿用 DIOR->COCO 主表固定值, 未针对 DOTA 调整 (第三源域验证)'},
           'classes': rows,
           'mean': {'baseline': round(mean_b, 4), 'acspc': round(mean_o, 4),
                    'gain': round(mean_o - mean_b, 4)}}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 90)
    print('DOTA 源 -> COCO 测试 (第三源域): %-14s %9s %9s %9s %9s' %
          ('class', 'baseline', 'A-CSPC', '增益', '召回'))
    print('-' * 90)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-16s %9.4f %9.4f %+9.4f %9.4f' % (name, r['baseline'], r['acspc'],
                                                  r['gain'], r['recall']))
    print('-' * 90)
    print('%-16s %9.4f %9.4f %+9.4f %9s   <- mean' % ('mean', mean_b, mean_o,
                                                      mean_o - mean_b, '—'))
    print('=' * 90)
    print('对照: DIOR->COCO +0.0652 (主表) | NWPU->COCO +0.0589 (tge_probe_nwpu)')
    print('elapsed %.1f s -> %s' % (time.time() - t0, OUT))


if __name__ == '__main__':
    main()
