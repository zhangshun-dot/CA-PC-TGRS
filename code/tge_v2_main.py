# -*- coding: utf-8 -*-
r"""tge_v2_main.py — v2 冻结协议 §1/§4: 全量 COCO 主表 (β=1.0, λ=0.25, 全量 val2017)

冻结配置 (2026-08-26, 来自 tge_dev_scan_voc.json):
  β* = 1.0, λ* = 0.25   (DIOR→VOC 开发域选参, 5/5 非负, 目标域无关)

本脚本计算每类(6 类)在全量可用非 crowd 含 GT 图像上的:
  baseline (GDINO-tiny 分数)
  B2       (CLIP 源原型 + 负原型, λ=0.25)
  T4       (CLIP 纯文本 d(y) + 负原型, λ=0.25)
  A-CSPC   (P̂ = norm(P_src + 1.0·Δ), λ=0.25)
指标:
  AP50 (逐图匹配 + 全局排序 + 11 点)     ← 主表口径
  AP@[0.5:0.95] (COCO 风格 101 点, 10 IoU) ← v2 补充口径
  Recall@proposal (逐图 per-GT, IoU≥0.5)
统计:
  bootstrap 按图配对, R=1000 × 3 种子 (42/20260823/7)
  逐类 p 单边 (count+1)/(R+1), 补 Benjamini-Hochberg FDR 校正
  K=10 校准图 (仅供 few-shot 参照行), 零样本在 images[K:] 评估

检测缓存: tge_v2_dets_COCO-*.pkl (全新缓存, 全量检测, ~0.7-1s/图)。
主表检测器 = GDINO-tiny; GDINO-base / OWLv2 矩阵见 tge_v2_detectors.py。

运行: python tge_v2_main.py   (~25-35 分钟, 含全量检测 + CLIP 编码)
输出: 实验数据/tge_v2_main.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# ── v2 冻结配置（可用 --beta / --lam 覆盖，输出文件名带配置后缀）──
FROZEN_BETA = 1.0
FROZEN_LAM = 0.25
K_CALIB = 10            # 校准图数 (few-shot 参照行用), 零样本在 images[K:]
R_BOOT = 1000
SEEDS = [42, 20260823, 7]
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'
DETECTOR = 'gdino-tiny'
PRE_FILTER = 0.03

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


# ── 数据 ──

def load_coco_all(cls_name):
    """加载该类全部可用非 crowd 含 GT 图像 (val2017, 不截断)。"""
    return U.load_coco(cls_name, n=100000)


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


def v2_cached_dets(images, prompt, name):
    p = DATA / f'tge_v2_dets_COCO-{name}.pkl'
    if p.exists():
        print(f'    [缓存] 检测结果 {p.name}')
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets


def collect_features(images, gts, dets):
    """pre-filter + 小框过滤 → per-image (box, gd, fused, gt)。F 一次编码。"""
    per_img = []
    crops_all = []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
                        'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
                        'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
                        'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64)})
    if crops_all:
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))
    return per_img, F


def fused_arrays(per_img, F, W):
    """给每图加 fused 分数列: fused = sqrt(gd * clip(max(0, F·W), 1e-6, 1))。"""
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, F[p['crop_idx']] @ W)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)
    return per_img


def dataset_ap(per_img, key, coco_style=False):
    """数据集级 AP (只含 test 图, 即 idx>=K_CALIB)。coco_style=True → AP@[0.5:0.95]。"""
    sel = [per_img[i] for i in range(K_CALIB, len(per_img))]
    preds = [(p['box'], p[key]) for p in sel]
    gts = [p['gt'] for p in sel]
    if coco_style:
        return U.compute_ap_coco(preds, gts)
    return U.compute_ap50_img(preds, gts)


def recall_proposals(per_img):
    """逐图 per-GT recall@proposal (IoU≥0.5, 含小框, pre-filter 后)。"""
    n_gt, hit = 0, 0
    for p in per_img:
        gts, boxes = p['gt'], p['box']
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


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR 校正, 返回 q 值 (未排序保持原序)。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(n)
    out[order] = q
    return out


# ── 主流程 ──

def main():
    global FROZEN_BETA, FROZEN_LAM
    import time
    t0 = time.time()
    ap = argparse.ArgumentParser(description='A-CSPC v2 全量主表 (可指定 β/λ)')
    ap.add_argument('--beta', type=float, default=FROZEN_BETA)
    ap.add_argument('--lam', type=float, default=FROZEN_LAM)
    args = ap.parse_args()
    FROZEN_BETA, FROZEN_LAM = args.beta, args.lam
    tag = 'b%g_l%g' % (FROZEN_BETA, FROZEN_LAM)
    out_path = DATA / ('tge_v2_main_%s.json' % tag)
    print('配置: β=%.2f λ=%.2f  →  输出 %s' % (FROZEN_BETA, FROZEN_LAM, out_path.name))
    print('[1/3] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        p = clip_proto(gt_crops(images, gts))
        if p is not None:
            protos[cls] = p

    print('[2/3] COCO 6 类全量检测 + 特征 ...')
    class_data = []
    for name, coco_cls, src_cls in TARGETS:
        images, gts = load_coco_all(coco_cls)
        if len(images) < K_CALIB + 10:
            print('  !! %s 图太少 (%d), 跳过' % (name, len(images)))
            continue
        dets = v2_cached_dets(images, TGT_TMPL.format(coco_cls), name)
        per_img, F = collect_features(images, gts, dets)
        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        Delta = d_y - s_xs
        class_data.append({'name': name, 'per_img': per_img, 'F': F, 'gts': gts,
                           'P_src': P_src, 'P_neg': P_neg, 'Delta': Delta,
                           'd_y': d_y, 'coco_cls': coco_cls,
                           'n_pool': len(images), 'n_test': len(images) - K_CALIB})
        n_box = sum(len(p['gd']) for p in per_img)
        print('  %-16s N_pool=%3d N_test=%3d 框=%6d' % (name, len(images),
                                                        len(images) - K_CALIB, n_box))

    print('[3/3] 打分 + AP + bootstrap ...')
    rows = {}
    for d in class_data:
        P_src, P_neg = d['P_src'], d['P_neg']
        # baseline (gd)
        base50 = dataset_ap(d['per_img'], 'gd')
        base_coco = dataset_ap(d['per_img'], 'gd', coco_style=True)
        # B2: 源原型 + λ=0.25 负原型
        W_b2 = (P_src.cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
        fused_arrays(d['per_img'], d['F'], W_b2)
        b2_50 = dataset_ap(d['per_img'], 'fused')
        b2_coco = dataset_ap(d['per_img'], 'fused', coco_style=True)
        # T4: 纯文本 d(y) + λ=0.25 负原型
        d_y = d['d_y']
        W_t4 = (d_y.cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
        fused_arrays(d['per_img'], d['F'], W_t4)
        t4_50 = dataset_ap(d['per_img'], 'fused')
        t4_coco = dataset_ap(d['per_img'], 'fused', coco_style=True)
        # A-CSPC: P̂ = norm(P_src + β·Δ), λ=0.25
        P_hat = P_src + FROZEN_BETA * d['Delta'].to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        W_c = (P_hat.cpu().numpy() - FROZEN_LAM * P_neg.cpu().numpy())
        fused_arrays(d['per_img'], d['F'], W_c)
        ours_50 = dataset_ap(d['per_img'], 'fused')
        ours_coco = dataset_ap(d['per_img'], 'fused', coco_style=True)
        rec = recall_proposals(d['per_img'])
        rows[d['name']] = {'n_pool': d['n_pool'], 'n_test': d['n_test'],
                           'recall': round(rec, 4),
                           'base50': round(base50, 4), 'b2_50': round(b2_50, 4),
                           't4_50': round(t4_50, 4), 'ours50': round(ours_50, 4),
                           'gain50': round(ours_50 - base50, 4),
                           'base_coco': round(base_coco, 4), 'ours_coco': round(ours_coco, 4),
                           'gain_coco': round(ours_coco - base_coco, 4),
                           'ours_fused_arrays': None}
        print('  %-16s base50=%.4f ours50=%.4f (%+.4f) | APcoco %+.4f | rec=%.4f' %
              (d['name'], base50, ours_50, ours_50 - base50, ours_coco - base_coco, rec))

    # bootstrap (按图配对, 3 种子)
    boot = {}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        gains_mean = []
        for _ in range(R_BOOT):
            dsum = 0.0
            for d in class_data:
                n_t = len(d['per_img']) - K_CALIB
                idx = rng.randint(K_CALIB, len(d['per_img']), size=n_t)
                sel = [d['per_img'][i] for i in idx]
                a_base = U.compute_ap50_img([(p['box'], p['gd']) for p in sel],
                                            [p['gt'] for p in sel])
                a_ours = U.compute_ap50_img([(p['box'], p['fused']) for p in sel],
                                            [p['gt'] for p in sel])
                dsum += (a_ours - a_base)
            gains_mean.append(dsum / len(class_data))
        gains_mean = np.array(gains_mean)
        boot[str(seed)] = {'mean': float(np.mean(gains_mean)),
                           'ci95': [float(np.percentile(gains_mean, 2.5)),
                                    float(np.percentile(gains_mean, 97.5))],
                           'p_neg_frac': float(np.mean(gains_mean <= 0))}

    # 汇总 + BH-FDR (逐类 p 从 seed=42 的类别内重采样近似, 此处用 seed42 的逐类)
    mean_row = {k: round(float(np.mean([rows[c][k] for c in rows])), 4)
                for k in ['base50', 'b2_50', 't4_50', 'ours50', 'gain50',
                          'base_coco', 'ours_coco', 'gain_coco']}
    out = {'config': {'protocol': 'v2 frozen: β=1.0 λ=0.25 (DIOR→VOC dev scan), full val2017, '
                                  'K=10 calib, per-image AP50 + COCO AP@0.5:0.95, '
                                  'R=1000 × 3 seeds, BH-FDR',
                      'detector': DETECTOR, 'pre_filter': PRE_FILTER,
                      'beta': FROZEN_BETA, 'neg_lam': FROZEN_LAM,
                      'seeds': SEEDS, 'r_boot': R_BOOT},
           'classes': {k: {kk: vv for kk, vv in v.items() if kk != 'ours_fused_arrays'}
                       for k, v in rows.items()},
           'mean': mean_row, 'bootstrap': boot}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n' + '=' * 100)
    print('%-16s %9s %9s %9s %9s %9s %9s %9s' %
          ('class', 'base50', 'B2', 'T4', 'A-CSPC', 'Δ50', 'Δcoco', 'recall'))
    print('-' * 100)
    for c in rows:
        r = rows[c]
        print('%-16s %9.4f %9.4f %9.4f %9.4f %+9.4f %+9.4f %9.4f' %
              (c, r['base50'], r['b2_50'], r['t4_50'], r['ours50'],
               r['gain50'], r['gain_coco'], r['recall']))
    print('-' * 100)
    print('%-16s %9.4f %9.4f %9.4f %9.4f %+9.4f %+9.4f' %
          ('mean', mean_row['base50'], mean_row['b2_50'], mean_row['t4_50'],
           mean_row['ours50'], mean_row['gain50'], mean_row['gain_coco']))
    for seed in SEEDS:
        b = boot[str(seed)]
        print('boot seed=%d: mean_gain=%+.4f CI=[%+.4f, %+.4f] 负比例=%.4f' %
              (seed, b['mean'], b['ci95'][0], b['ci95'][1], b['p_neg_frac']))
    print('=' * 100)
    print('→ %s   (耗时 %.1f 分钟)' % (out_path.name, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
