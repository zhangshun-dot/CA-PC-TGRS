# -*- coding: utf-8 -*-
"""tge_probe_rev_signcheck.py — 反向文本偏移"符号口径"对照 (复用 tge_probe_rev.py 完全相同的协议)

动机: tge_probe_rev.py 用的是 Δ_rev = s(y) − d(x_s) = overhead(目标类) − photo(源类),
      而论文 Eq.1 定义的是  Δ = d(x_c) − s(x_s(c)) = photo(目标类) − overhead(源类) = −Δ_rev。
      两者互为相反数 ⇒ 反向任务下"完整外推"的结果取决于用哪个符号。
本脚本在同一份缓存/同一份特征上, 同时算两套符号, 只要一次编码。

变体 (全部 λ=0.5 负原型, β=2.0):
  baseline      : 原始检测器分
  B2_src        : unit(P_src)                                   (= 论文 β=0 保守)
  extrap_rev    : unit(P_src + β·Δ_rev),  Δ_rev  = overhead(目标) − photo(源)   → 同 tge_probe_rev.py
  extrap_eq1    : unit(P_src + β·Δ_eq1),  Δ_eq1  = photo(目标) − overhead(源)   → 论文 Eq.1 字面
  T4_text       : unit(photo(目标类))                            (= 论文 T4)

输出: 实验数据/tge_probe_rev_signcheck.json
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
OUT = DATA / 'tge_probe_rev_signcheck.json'

BETA = 2.0
N_COCO_SRC = 500
SRC_TMPL = 'an overhead image of {}'   # 目标域(DIOR) 前缀
TGT_TMPL = 'a photo of {}'             # 源域(COCO) 前缀

TARGETS = [
    ('DIOR-airplane', ['airplane']),
    ('DIOR-ship',     ['boat']),
    ('DIOR-vehicle',  ['car', 'truck', 'bus', 'motorcycle']),
]
SRC_KEY = {'DIOR-airplane': 'airplane', 'DIOR-ship': 'boat', 'DIOR-vehicle': 'vehicle'}


def coco_proto(coco_cls_list):
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
    print('[1/3] COCO 源原型 ...')
    protos = {k: coco_proto(v) for k, v in
              [('airplane', ['airplane']), ('boat', ['boat']),
               ('vehicle', ['car', 'truck', 'bus', 'motorcycle'])]}
    for k, p in protos.items():
        print(f'  {k:<10s} ok' if p is not None else f'  {k} FAILED')

    print('\n[2/3] DIOR 3 类反向评估 (两套符号对照) ...')
    rows = {}
    for name, coco_cls_list in TARGETS:
        dior_cls = name.split('-')[1]
        sk = SRC_KEY[name]
        P_src = protos[sk]
        neg_others = torch.stack([protos[k] for k in protos if k != sk]).mean(dim=0)
        P_neg = neg_others / neg_others.norm()

        images, gts = U.load_dior(dior_cls)
        dets = T.cached_dets(images, SRC_TMPL.format(dior_cls), name)

        crops, boxes_all, gd_all, img_ids = [], [], [], []
        for ii, ((boxes, scores), img, g) in enumerate(zip(dets, images, gts)):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                boxes_all.append([x1, y1, x2, y2]); gd_all.append(float(sc)); img_ids.append(ii)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    crops.append(None)
                else:
                    crops.append(img.crop((x1, y1, x2, y2)))
        real_idx = [i for i, c in enumerate(crops) if c is not None]
        F = np.zeros((len(crops), 512))
        if real_idx:
            F[real_idx] = T.encode_clip_batch([crops[i] for i in real_idx]).cpu().numpy()
        gd = np.clip(np.array(gd_all), 1e-6, 1.0)

        s_y = T.text_emb([SRC_TMPL.format(dior_cls)]).cpu()[0]      # overhead(目标类)
        if len(coco_cls_list) == 1:
            d_xs = T.text_emb([TGT_TMPL.format(coco_cls_list[0])]).cpu()[0]
        else:
            d_xs = T.text_emb([TGT_TMPL.format(c) for c in coco_cls_list]).mean(dim=0).cpu()
        d_rev = s_y - d_xs            # overhead(目标) − photo(源)   [tge_probe_rev 口径]
        d_eq1 = d_xs - s_y            # photo(目标)  − overhead(源)  [论文 Eq.1 字面]

        neg_vec = P_neg.cpu().numpy()

        def fused_ap(P_vec, beta=0.0, extra=None, lam=U.NEG_LAM):
            P = P_vec.cpu() if extra is None else (P_vec.cpu() + beta * extra.cpu())
            P = P / (P.norm() + 1e-8)
            vis = np.maximum(0.0, F @ P.numpy() - lam * (F @ neg_vec))
            fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
            return U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)

        ap_base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
        ap_b2 = fused_ap(P_src)
        ap_rev = fused_ap(P_src, BETA, d_rev)
        ap_eq1 = fused_ap(P_src, BETA, d_eq1)
        ap_t4 = fused_ap(s_y)

        rows[name] = {'baseline': round(ap_base, 4), 'B2_src': round(ap_b2, 4),
                      'extrap_rev': round(ap_rev, 4), 'extrap_eq1': round(ap_eq1, 4),
                      'T4_text': round(ap_t4, 4), 'n_images': len(images)}
        print(f'  {name:<16s} N={len(images)} base={ap_base:.4f} B2={ap_b2:.4f} '
              f'extrap_rev={ap_rev:.4f} extrap_eq1={ap_eq1:.4f} T4={ap_t4:.4f}')

    print('\n[3/3] 3 类均值增益')
    keys = ['B2_src', 'extrap_rev', 'extrap_eq1', 'T4_text']
    mean_gain = {}
    for k in keys:
        g = [rows[n][k] - rows[n]['baseline'] for n in rows]
        mean_gain[k] = round(float(np.mean(g)), 4)
        pos = sum(1 for x in g if x > 0)
        print(f'  {k:<12s} mean gain = {np.mean(g):+.4f}   ({pos}/3 positive)')
    print(f"  baseline mean AP = {np.mean([rows[n]['baseline'] for n in rows]):.4f}")
    for v, lab in [(0.0418, 'β=0  +0.0418'), (-0.0244, 'full −0.0244'), (-0.0793, 'T4  −0.0793')]:
        for k, mv in mean_gain.items():
            if abs(mv - v) < 5e-5:
                print(f'  ✓ 论文值 {lab}  ← {k}')

    json.dump({'config': {'beta': BETA, 'neg_lam': U.NEG_LAM, 'n_coco_src': N_COCO_SRC,
                          'protocol': 'COCO→DIOR 反向, 3 类, DIOR full20 N=20, 两套 Δ 符号对照'},
               'classes': rows, 'mean_gain': mean_gain},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n-> {OUT}')


if __name__ == '__main__':
    main()
