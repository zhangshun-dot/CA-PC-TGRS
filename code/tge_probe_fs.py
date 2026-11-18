# -*- coding: utf-8 -*-
"""tge_probe_fs.py — 阶段2 线A 完整: CLIP 空间两段式 (外推零样本 + K=10 校准)

把 final_method_table.py 的 CDVM few-shot 校准机制搬到 CLIP 空间,
零样本段的正原型换成语言外推原型 (joint@2.0, 不挑参)。

协议与 final_method_table 完全一致 (N=100, 前 K=10 图校准、后 90 图测试,
pre-filter 0.03, sqrt(gd*vis) 融合, dataset-level AP50), 以便与
final_results_12class.json 的旧 CDVM 增益 (+0.0745) 直接对比。

列:
  baseline       — GDINO 原始分数
  zs_src         — CLIP 真实源原型 (零样本, 旧 CDVM 的 CLIP 空间版)
  zs_extrap      — 外推原型 joint@2 (零样本, 阶段1/线A 已验证 6/6 正)
  fs_calib_src   — 0.5·P_src + 0.5·P_tgt   (校准, 无外推: 分离"外推"与"校准")
  fs_calib_extrap— 0.5·P̂ + 0.5·P_tgt       (两段式: 外推 + 校准, 主线)

输出: 实验数据/tge_probe_fs.json + 控制台表
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
OUT = DATA / 'tge_probe_fs.json'

K = 10
ALPHA = 0.5          # P_src 权重 (与 final_method_table 一致)
BETA = 2.0           # 外推步长 (固定默认口径, 不挑参)
N_COCO = 100         # 与主表协议一致
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


def clip_proto(crops):
    if not crops:
        return None
    F = T.encode_clip_batch(crops)
    p = F.mean(dim=0)
    return p / p.norm()


def gt_crops(images, gts):
    """所有 GT 框 crop (同 final_method_table extract_fewshot_proto 的框规)。"""
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def eval_class(images, gts, dets, P_cal, P_neg):
    """同 final_method_table.eval_class: 跳过小框, sqrt(gd*vis) 融合, 标准逐图 AP50。

    2026-08-23: 全局拼接匹配有跨图误匹配 (COCO 同尺寸图坐标可比),
    全部改用 U.compute_ap50_img (逐图匹配 + 全局排序)。
    """
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
        vis = np.maximum(0.0, F @ P_cal.cpu().numpy()
                         - U.NEG_LAM * (F @ P_neg.cpu().numpy()))
        fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
        ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    else:
        ap = base
    return base, ap


def main():
    print('[1/2] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        crops = gt_crops(images, gts)
        p = clip_proto(crops)
        if p is not None:
            protos[cls] = p

    print('[2/2] COCO 6 类两段式 (N=100, K=10 校准, 后 90 测试) ...')
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)

        P_src = protos[src_cls]
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        d_xs = T.text_emb([TGT_TMPL.format(src_cls)]).cpu()[0]
        d_joint = d_y - s_xs
        P_hat = P_src + BETA * d_joint.to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()

        # 切分: 前 K 图校准, 后 90 图测试 (同 final_method_table)
        fs_imgs, fs_gts = images[:K], gts[:K]
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]
        P_tgt = clip_proto(gt_crops(fs_imgs, fs_gts))

        base, _ = eval_class(test_imgs, test_gts, test_dets, P_src, P_neg)
        _, ap_zs_src = eval_class(test_imgs, test_gts, test_dets, P_src, P_neg)
        _, ap_zs_extrap = eval_class(test_imgs, test_gts, test_dets, P_hat, P_neg)
        # T4 列 (N=100 主表精确值, 2026-08-23 补): 纯文本目标域原型 d(y) 直接打分
        # (text_emb 已归一化, 同 tge_probe_ap 的 T4 口径: 带负原型)
        P_text = d_y.to(U.DEVICE)
        _, ap_zs_text = eval_class(test_imgs, test_gts, test_dets, P_text, P_neg)

        if P_tgt is not None:
            P_cal_src = (ALPHA * P_src + (1 - ALPHA) * P_tgt)
            P_cal_src = P_cal_src / P_cal_src.norm()
            P_cal_hat = (ALPHA * P_hat + (1 - ALPHA) * P_tgt)
            P_cal_hat = P_cal_hat / P_cal_hat.norm()
            _, ap_fs_src = eval_class(test_imgs, test_gts, test_dets, P_cal_src, P_neg)
            _, ap_fs_extrap = eval_class(test_imgs, test_gts, test_dets, P_cal_hat, P_neg)
        else:
            ap_fs_src, ap_fs_extrap = ap_zs_src, ap_zs_extrap

        rows[name] = {'baseline': round(base, 4), 'zs_src': round(ap_zs_src, 4),
                      'zs_extrap': round(ap_zs_extrap, 4),
                      'zs_text': round(ap_zs_text, 4),
                      'fs_src': round(ap_fs_src, 4), 'fs_extrap': round(ap_fs_extrap, 4)}
        print(f'  {name:<16s} base={base:.4f}  zs_src={ap_zs_src:.4f}  '
              f'zs_text={ap_zs_text:.4f}  zs_extrap={ap_zs_extrap:.4f}  '
              f'fs_src={ap_fs_src:.4f}  fs_extrap={ap_fs_extrap:.4f}')

    print('\n' + '=' * 100)
    print(f'{"类":<16s} {"baseline":>9s} {"zs_src":>9s} {"zs_text":>9s} {"zs_extrap":>9s} {"fs_src":>9s} {"fs_extrap":>9s}')
    print('-' * 100)
    for name, coco_cls, src_cls in TARGETS:
        r = rows[name]
        print(f'{name:<16s} {r["baseline"]:>9.4f} {r["zs_src"]:>9.4f} {r["zs_text"]:>9.4f} '
              f'{r["zs_extrap"]:>9.4f} {r["fs_src"]:>9.4f} {r["fs_extrap"]:>9.4f}')
    for label, keys in [('跨域均值', ['baseline', 'zs_src', 'zs_text', 'zs_extrap', 'fs_src', 'fs_extrap'])]:
        print('-' * 100)
        print(f'{label:<16s}' + ''.join(f'{np.mean([rows[n][k] for n in rows]):>9.4f}' for k in keys))

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'k': K, 'alpha': ALPHA, 'beta': BETA, 'n_coco': N_COCO,
                              'neg_lam': U.NEG_LAM,
                              'protocol': 'gdino th=0 -> pre-filter 0.03 -> image-safe per-image AP50 '
                                          '(per-image matching + global PR, 2026-08-23 修正), '
                                          '前K图校准后90图测试'},
                   'classes': rows}, f, indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
