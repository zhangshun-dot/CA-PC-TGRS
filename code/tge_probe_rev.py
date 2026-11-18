# -*- coding: utf-8 -*-
"""tge_probe_rev.py — 域对验证: COCO → DIOR 反向 (方法通用性)

正向 (DIOR→COCO) 已验证: 粗→细外推 (vehicle→bus/moto) + 同名域转移 (airplane)。
反向验证方法不挑域对方向:
  - DIOR-airplane  ← COCO-airplane   (同名域转移, photo→overhead)
  - DIOR-ship      ← COCO-boat       (语义对应, 类别名不同)
  - DIOR-vehicle   ← COCO {car,truck,bus,moto} 合并 (细类聚合→粗类)

协议与正向完全对称 (GDINO th=0 -> pre-filter 0.03 -> AP50, sqrt(gd*vis), λ=0.5):
  评估 prompt 用目标域前缀 "an overhead image of {y}" (正向用 "a photo of")
  外推: P̂ = P_src + β·Δ_rev, Δ_rev = s(y) − d(x_s)  (s=overhead 域, d=photo 域)
  β=2.0 固定 (主口径), 对照: baseline / B2 源原型直移 / T4 纯文本 s(y)

注意: DIOR full20 每类仅 20 图, AP50 方差大, 同口径比较仍有效 (如实报告)。
输出: 实验数据/tge_probe_rev.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_rev.json'

BETA = 2.0
N_COCO_SRC = 500      # 源域 COCO 图数
SRC_TMPL = 'an overhead image of {}'   # 目标域 (DIOR) 前缀 —— 注意: 反向时 s=DIOR, d=COCO

# (dior 目标类, 源 COCO 类集合)
TARGETS = [
    ('DIOR-airplane', ['airplane']),
    ('DIOR-ship',     ['boat']),
    ('DIOR-vehicle',  ['car', 'truck', 'bus', 'motorcycle']),
]


def coco_proto(coco_cls_list):
    """COCO 源原型: 若干类的 GT 框 crop → CLIP 均值 (聚合类取全部 GT 框)。"""
    crops = []
    for c in coco_cls_list:
        images, gts = U.load_coco(c, n=N_COCO_SRC)
        for img, g in zip(images, gts):
            for gt in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 > x1 + 10 and y2 > y1 + 10:
                    crops.append(img.crop((x1, y1, x2, y2)))
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def main():
    # 1) COCO 源原型 (3 个: airplane / boat / vehicle-聚合)
    print('[1/3] COCO 源原型 (CLIP 空间) ...')
    src_key = {'DIOR-airplane': 'airplane', 'DIOR-ship': 'boat', 'DIOR-vehicle': 'vehicle'}
    protos = {k: coco_proto(cls_list) for k, cls_list in
              [('airplane', ['airplane']), ('boat', ['boat']),
               ('vehicle', ['car', 'truck', 'bus', 'motorcycle'])]}
    for k, p in protos.items():
        print(f'  {k:<12s} 原型就绪' if p is not None else f'  {k} 构建失败!')

    # 2) 评估 DIOR 3 类
    print('\n[2/3] DIOR 3 类反向评估 ...')
    rows = {}
    for name, coco_cls_list in TARGETS:
        dior_cls = name.split('-')[1]
        sk = src_key[name]
        P_src = protos[sk]
        images, gts = U.load_dior(dior_cls)
        dets = T.cached_dets(images, SRC_TMPL.format(dior_cls), name)

        # 特征 + 标签 (img_ids: 逐图 AP50 需要)
        crops, boxes_all, gd_all, lab_all, img_ids = [], [], [], [], []
        for ii, ((boxes, scores), img, g) in enumerate(zip(dets, images, gts)):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                lab = 1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in g) else 0
                boxes_all.append([x1, y1, x2, y2])
                gd_all.append(float(sc))
                lab_all.append(lab)
                img_ids.append(ii)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    crops.append(None)
                else:
                    crops.append(img.crop((x1, y1, x2, y2)))
        real_idx = [i for i, c in enumerate(crops) if c is not None]
        F = np.zeros((len(crops), 512))          # 全量框行 (小框=0 特征, vis=0 同正向协议)
        if real_idx:
            F[real_idx] = T.encode_clip_batch([crops[i] for i in real_idx]).cpu().numpy()
        gd = np.clip(np.array(gd_all), 1e-6, 1.0)
        gt_flat = [g for gs in gts for g in gs]

        # 负原型: 其余源类均值 (仅 3 类, 不能用 topk(3))
        others = torch.stack([protos[k] for k in protos if k != sk]).mean(dim=0)
        P_neg = others / others.norm()

        # 文本偏移 (反向): Δ_rev = s(y) − d(x_s)
        s_y = T.text_emb([SRC_TMPL.format(dior_cls)]).cpu()[0]
        if len(coco_cls_list) == 1:
            d_xs = T.text_emb([T.TGT_TMPL.format(coco_cls_list[0])]).cpu()[0]
        else:  # 聚合类: 源类文本平均
            d_xs = T.text_emb([T.TGT_TMPL.format(c) for c in coco_cls_list]).mean(dim=0).cpu()
        d_rev = s_y - d_xs

        def fused_ap(P_vec, beta=0.0, extra=None, neg_lam=U.NEG_LAM):
            if extra is not None:
                P = (P_vec.cpu() + beta * extra.cpu())
            else:
                P = P_vec.cpu()
            P = P / (P.norm() + 1e-8)
            vis = np.maximum(0.0, F @ P.numpy() - neg_lam * (F @ P_neg.cpu().numpy()))
            fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
            return U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)

        # 逐图 AP50 (2026-08-23 修正)
        ap_base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
        ap_b2 = fused_ap(P_src)                            # 源原型直移
        ap_extrap = fused_ap(P_src, BETA, d_rev)           # 反向外推 joint@2
        ap_text = fused_ap(s_y)                            # 纯文本目标域原型

        rows[name] = {'baseline': round(ap_base, 4), 'B2_src': round(ap_b2, 4),
                      'extrap': round(ap_extrap, 4), 'T4_text': round(ap_text, 4),
                      'n_images': len(images), 'n_boxes': len(boxes_all)}
        print(f'  {name:<16s} N={len(images)}图 框={len(boxes_all)}  '
              f'base={ap_base:.4f}  B2={ap_b2:.4f}  extrap={ap_extrap:.4f}  T4={ap_text:.4f}')

    # 3) 汇总
    print('\n' + '=' * 84)
    print(f'{"类":<16s} {"baseline":>9s} {"B2 src":>9s} {"extrap":>9s} {"T4 text":>9s}')
    print('-' * 84)
    for name, _ in TARGETS:
        r = rows[name]
        print(f'{name:<16s} {r["baseline"]:>9.4f} {r["B2_src"]:>9.4f} '
              f'{r["extrap"]:>9.4f} {r["T4_text"]:>9.4f}')
    print('-' * 84)
    for label, keys in [('均值', ['baseline', 'B2_src', 'extrap', 'T4_text'])]:
        print(f'{label:<16s}' + ''.join(f'{np.mean([rows[n][k] for n in rows]):>9.4f}' for k in keys))

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'beta': BETA, 'neg_lam': U.NEG_LAM, 'n_coco_src': N_COCO_SRC,
                              'protocol': 'GDINO th=0 -> pre-filter 0.03 -> image-safe per-image AP50 '
            '(2026-08-23 修正, COCO→DIOR 反向)'},
                   'classes': rows}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
