# -*- coding: utf-8 -*-
"""tge_probe_dota.py — 跨数据集验证: DIOR→DOTA (同域跨数据集) + COCO→DOTA (反向)

线1 DIOR→DOTA: 源原型 DIOR 6 类 → DOTA 映射类 (同域俯视, 数据集偏移)
   预期: 文本域偏移 Δ≈0 → 外推退化为保守模式; 关键看 B2 (源原型+负原型) > baseline
   → "非对称构造在数据集偏移下稳健" (选择性外推的保守场景)

线2 COCO→DOTA: 源 COCO 原型 → DOTA plane/ship/vehicle (反向跨域, 第二个目标域)
   预期: 同 COCO→DIOR 反向 (外推推偏, B2 有效) → 反向规律普适性

协议: GDINO th=0 -> pre-filter 0.03 -> dataset-level AP50 (统一, 方法间公平)
      DOTA quad → axis-aligned bbox; 每类 N=10 图 (大图, 与历史 exp_dota 一致)
输出: 实验数据/tge_probe_dota.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_dota.json'

BETA = 2.0
N_PER_CLASS = 10
IMG_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
LBL_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

DOTA_CLASSES = ['plane', 'ship', 'storage-tank', 'harbor', 'bridge',
                'large-vehicle', 'small-vehicle']
DOTA_TO_DIOR = {'plane': 'airplane', 'ship': 'ship', 'storage-tank': 'storage tank',
                'harbor': 'harbor', 'bridge': 'bridge',
                'large-vehicle': 'vehicle', 'small-vehicle': 'vehicle'}


def load_dota(dota_cls, n=N_PER_CLASS):
    """DOTA 类 → (images, gts) 轴对齐框 (quad min/max)。"""
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


def proto_from_gts(images, gts):
    crops = []
    for img, g in zip(images, gts):
        for box in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 > x1 + 10 and y2 > y1 + 10:
                crops.append(img.crop((x1, y1, x2, y2)))
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def eval_ap(images, gts, dets, P_cal, P_neg=None, neg_lam=U.NEG_LAM):
    """逐图 AP50 (2026-08-23: 全局拼接匹配改为 per-image 匹配 + 全局排序)。"""
    crops, boxes_all, gd_all, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            img_ids.append(ii)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                crops.append(None)
            else:
                crops.append(img.crop((x1, y1, x2, y2)))
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
    real_idx = [i for i, c in enumerate(crops) if c is not None]
    F = np.zeros((len(crops), 512))
    if real_idx:
        F[real_idx] = T.encode_clip_batch([crops[i] for i in real_idx]).cpu().numpy()
    P = P_cal.cpu().numpy()
    vis = np.maximum(0.0, F @ P)
    if P_neg is not None:
        vis = vis - neg_lam * (F @ P_neg.cpu().numpy())
        vis = np.maximum(0.0, vis)
    fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
    ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    return base, ap


def main():
    # ── 线1: DIOR→DOTA ──
    print('═' * 70)
    print('线1: DIOR→DOTA (同域跨数据集, 保守模式验证)')
    print('═' * 70)
    print('[1/3] DIOR 源原型 6 类 ...')
    dior = {}
    for c in ['airplane', 'ship', 'vehicle', 'storage tank', 'harbor', 'bridge']:
        im, gt = U.load_dior(c)
        dior[c] = proto_from_gts(im, gt)
        print(f'  {c}: ok')

    rows1 = {}
    for dota_cls in DOTA_CLASSES:
        dior_cls = DOTA_TO_DIOR[dota_cls]
        images, gts = load_dota(dota_cls)
        if not images:
            print(f'  !! {dota_cls}: 无图, 跳过')
            continue
        print(f'\n  [{dota_cls}→{dior_cls}] {len(images)} 图')
        dets = U.gdino_detect(images, f'{SRC_TMPL.format(dior_cls)}', 0.0)
        P_src = dior[dior_cls]
        # 负原型 (DIOR 其余 5 类均值)
        others = torch.stack([dior[c] for c in dior if c != dior_cls]).mean(dim=0)
        P_neg = others / others.norm()
        # 外推 (同域 → Δ 应 ≈ 0: s 和 d 模板同为 overhead 系? 不: d 是 photo)
        s_xs = T.text_emb([SRC_TMPL.format(dior_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(dior_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = (P_hat / P_hat.norm()).cpu()

        base, ap_b2 = eval_ap(images, gts, dets, P_src, P_neg)
        _, ap_ext = eval_ap(images, gts, dets, P_hat, P_neg)
        rows1[dota_cls] = {'baseline': round(base, 4), 'B2_src_neg': round(ap_b2, 4),
                           'extrap': round(ap_ext, 4)}
        print(f'    base={base:.4f}  B2(源+负)={ap_b2:.4f}  extrap={ap_ext:.4f}')

    # ── 线2: COCO→DOTA ──
    print('\n' + '═' * 70)
    print('线2: COCO→DOTA (反向跨域, 第二个目标域)')
    print('═' * 70)
    print('[2/3] COCO 源原型 ...')
    coco = {}
    im, gt = U.load_coco('airplane', n=500)
    coco['airplane'] = proto_from_gts(im, gt)
    im, gt = U.load_coco('boat', n=500)
    coco['boat'] = proto_from_gts(im, gt)
    imgs_all, gts_all = [], []
    for c in ['car', 'truck', 'bus', 'motorcycle']:
        im2, gt2 = U.load_coco(c, n=500)
        imgs_all += im2
        gts_all += gt2
    coco['vehicle'] = proto_from_gts(imgs_all, gts_all)

    rows2 = {}
    for dota_cls, coco_key in [('plane', 'airplane'), ('ship', 'boat'), ('large-vehicle', 'vehicle')]:
        images, gts = load_dota(dota_cls)
        if not images:
            continue
        print(f'\n  [{dota_cls}←COCO {coco_key}] {len(images)} 图')
        dets = U.gdino_detect(images, f'{SRC_TMPL.format(dota_cls)}', 0.0)
        P_src = coco[coco_key]
        others = torch.stack([coco[k] for k in coco if k != coco_key]).mean(dim=0)
        P_neg = others / others.norm()
        s_y = T.text_emb([SRC_TMPL.format(dota_cls)]).cpu()[0]
        d_xs = T.text_emb([TGT_TMPL.format(coco_key)]).cpu()[0]
        P_hat = P_src + BETA * (s_y - d_xs).to(U.DEVICE)
        P_hat = (P_hat / P_hat.norm()).cpu()

        base, ap_b2 = eval_ap(images, gts, dets, P_src, P_neg)
        _, ap_ext = eval_ap(images, gts, dets, P_hat, P_neg)
        rows2[dota_cls] = {'baseline': round(base, 4), 'B2_src_neg': round(ap_b2, 4),
                           'extrap': round(ap_ext, 4)}
        print(f'    base={base:.4f}  B2(源+负)={ap_b2:.4f}  extrap={ap_ext:.4f}')

    # ── 汇总 ──
    print('\n' + '=' * 78)
    print(f'{"线1 DIOR→DOTA":<18s} {"baseline":>9s} {"B2":>9s} {"extrap":>9s}')
    print('-' * 78)
    for c, r in rows1.items():
        print(f'{c:<18s} {r["baseline"]:>9.4f} {r["B2_src_neg"]:>9.4f} {r["extrap"]:>9.4f}')
    if rows1:
        print(f'{"均值":<18s} ' + ' '.join(
            f'{np.mean([r[k] for r in rows1.values()]):>9.4f}' for k in
            ['baseline', 'B2_src_neg', 'extrap']))
    print('-' * 78)
    print(f'{"线2 COCO→DOTA":<18s} {"baseline":>9s} {"B2":>9s} {"extrap":>9s}')
    print('-' * 78)
    for c, r in rows2.items():
        print(f'{c:<18s} {r["baseline"]:>9.4f} {r["B2_src_neg"]:>9.4f} {r["extrap"]:>9.4f}')
    if rows2:
        print(f'{"均值":<18s} ' + ' '.join(
            f'{np.mean([r[k] for r in rows2.values()]):>9.4f}' for k in
            ['baseline', 'B2_src_neg', 'extrap']))

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'beta': BETA, 'n_per_class': N_PER_CLASS,
                              'protocol': 'GDINO th=0 -> pre-filter 0.03 -> image-safe per-image AP50 '
                                          '(2026-08-23 修正)'},
                   'dior_to_dota': rows1, 'coco_to_dota': rows2}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
