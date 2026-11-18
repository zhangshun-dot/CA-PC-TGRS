# -*- coding: utf-8 -*-
"""tge_probe_reverse.py — 反向跨域: COCO 源原型 → DIOR 测试 (自然→遥感)

主表 (DIOR/NWPU→COCO) 只覆盖遥感→自然一个方向。本实验补自然→遥感,
回答: A-CSPC 增益的方向对称性 (外推公式 Δ = d(y) − s(x) 对反向域同样成立?)。

协议与主表完全一致 (tge_probe_fs): N=100/类, K=10, 后 90 图测试,
逐图 IoU 匹配 + dataset-level 11-point AP50; GDINO th=0 → pre-filter 0.03
(GDINO scores 是概率, 直接过滤, 同主表); β=2, λ=0.5 不调参; sqrt(gd*vis)。

方向反转 (源=COCO 自然场景, 目标=DIOR 遥感):
  源模板 s(x) = 'a photo of {coco_cls}'       (COCO 域)
  目标模板 d(y) = 'an overhead image of {dior_cls}'  (DIOR 域)
  P_hat = P_src + β·(d_y − s_xs)   # 与主表同构, 仅模板互换

类映射 (COCO 源 → DIOR 目标, 与主表互为镜像):
  airplane → airplane (同名), boat → ship (细→粗),
  car/truck/bus/motorcycle → vehicle (每源类独立一行, 细→粗 4 行)

负原型池 = COCO 6 类 (源域), 注: vehicle 任务的负原型池含 4 个车辆类,
抑制偏弱 —— COCO 作为源域的固有局限 (6 类 vs DIOR 20 / NWPU 10), 记录不调参。

输出: 实验数据/tge_probe_reverse.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_reverse.json'

K = 10
BETA = 2.0
N_DIOR = 100
SRC_TMPL = 'a photo of {}'                 # 源域 = COCO (自然)
TGT_TMPL = 'an overhead image of {}'       # 目标域 = DIOR (遥感)

# (name, dior_cls, coco_src_cls)
TARGETS = [
    ('DIOR-airplane',           'airplane', 'airplane'),
    ('DIOR-ship',               'ship',     'boat'),
    ('DIOR-vehicle-car',        'vehicle',  'car'),
    ('DIOR-vehicle-truck',      'vehicle',  'truck'),
    ('DIOR-vehicle-bus',        'vehicle',  'bus'),
    ('DIOR-vehicle-motorcycle', 'vehicle',  'motorcycle'),
]
COCO_CLASSES = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']


def build_protos_coco():
    protos = {}
    for cls in COCO_CLASSES:
        images, gts = U.load_coco(cls, n=100)
        crops = []
        for img, g in zip(images, gts):
            for gt in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                crops.append(img.crop((x1, y1, x2, y2)))
        F = T.encode_clip_batch(crops)
        p = F.mean(dim=0)
        protos[cls] = p / p.norm()
        print('  %-14s N=%3d 图, %4d crops, 原型 ✓' % (cls, len(images), len(crops)))
    return protos


def cached_dets(images, prompt, name):
    """DIOR GDINO 检测缓存 (检测与源域无关, 只按目标类分)。"""
    p = DATA / f'tge_probe_dets_{name}.pkl'
    if p.exists():
        print('    [缓存] DIOR 检测结果 %s' % p.name, flush=True)
        return pickle.load(open(p, 'rb'))
    dets = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(dets, open(p, 'wb'))
    return dets


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
    """A-CSPC 分数: ours = sqrt(gd * clip(max(0, F@W_hat), 1e-6, 1))。"""
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
    print('[1/3] COCO 6 类源原型 (自然域) ...')
    protos = build_protos_coco()

    print('[2/3] DIOR 3 类测试: GDINO 检测 (缓存) + CLIP 特征一次性编码 ...')
    class_data = []
    seen_dior = {}
    for name, dior_cls, coco_cls in TARGETS:
        if dior_cls in seen_dior:  # 共享检测结果 (vehicle 4 行同检测)
            images, gts, dets = seen_dior[dior_cls]
        else:
            images, gts = U.load_dior(dior_cls, n=N_DIOR)
            dets = cached_dets(images, TGT_TMPL.format(dior_cls), 'DIOR-' + dior_cls)
            seen_dior[dior_cls] = (images, gts, dets)
        per_img = collect_boxes(images, dets, gts)
        n_boxes = sum(len(p['gd']) for p in per_img)
        print('    %-24s N=%3d 框=%5d (pre-filter 0.03 后)' % (name, len(images), n_boxes),
              flush=True)
        class_data.append({'name': name, 'dior_cls': dior_cls, 'coco_cls': coco_cls,
                           'per_img': per_img, 'n_test': len(images) - K})

    print('[3/3] TGE(λ=0 纯外推) + A-CSPC 重打分 (β=%.1f, λ=%.1f 固定) ...' % (BETA, U.NEG_LAM))
    rows = {}
    for d in class_data:
        name, dior_cls, coco_cls = d['name'], d['dior_cls'], d['coco_cls']
        P_src = protos[coco_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, coco_cls)
        s_xs = T.text_emb([SRC_TMPL.format(coco_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(dior_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()
        # TGE (λ=0): 纯外推, 不扣负原型 —— 隔离"负原型池污染"与"外推失效"
        score_ours(d['per_img'], P_hat.cpu().numpy())
        ap_tge = ap_90(d['per_img'], 'ours')
        # A-CSPC: 外推 + 负原型抑制
        W_hat = (P_hat.cpu().numpy() - U.NEG_LAM * P_neg.cpu().numpy())
        score_ours(d['per_img'], W_hat)
        ap_base = ap_90(d['per_img'], 'gd')
        ap_ours = ap_90(d['per_img'], 'ours')
        rows[name] = {'src_cls': coco_cls, 'baseline': round(ap_base, 4),
                      'tge_lam0': round(ap_tge, 4), 'acspc': round(ap_ours, 4),
                      'tge_gain': round(ap_tge - ap_base, 4),
                      'acspc_gain': round(ap_ours - ap_base, 4),
                      'recall': round(recall_pf(d['per_img']), 4)}
        print('  %-24s base=%.4f  TGE(λ0)=%.4f  A-CSPC=%.4f  [TGE %+.4f | A %+.4f]  rec=%.4f'
              % (name, ap_base, ap_tge, ap_ours, ap_tge - ap_base, ap_ours - ap_base,
                 rows[name]['recall']))

    mean_b = float(np.mean([rows[n]['baseline'] for n in rows]))
    mean_t = float(np.mean([rows[n]['tge_lam0'] for n in rows]))
    mean_o = float(np.mean([rows[n]['acspc'] for n in rows]))
    out = {'config': {'source': 'COCO val2017 (6 类)', 'target': 'DIOR (airplane/ship/vehicle)',
                      'direction': 'natural -> remote sensing (主表反向)',
                      'n_dior': N_DIOR, 'k': K, 'beta': BETA, 'neg_lam': U.NEG_LAM,
                      'pre_filter': U.PRE_FILTER,
                      'protocol': 'GDINO boxes + TGE/A-CSPC rerank, N=100/类, K=10, 后90图, '
                                  '逐图 AP50, pre-filter 0.03, sqrt(gd*vis), 与主表一致 (模板互换)',
                      'question': 'A-CSPC 增益的方向对称性 (自然->遥感 反向是否成立); '
                                  'TGE(λ=0) 行隔离负原型池污染'},
           'classes': rows,
           'mean': {'baseline': round(mean_b, 4), 'tge_lam0': round(mean_t, 4),
                    'acspc': round(mean_o, 4), 'tge_gain': round(mean_t - mean_b, 4),
                    'acspc_gain': round(mean_o - mean_b, 4)}}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 110)
    print('COCO 源 -> DIOR 测试 (反向跨域): %-14s %9s %9s %9s %10s %10s %9s' %
          ('class', 'baseline', 'TGE(λ0)', 'A-CSPC', 'TGE增益', 'A增益', '召回'))
    print('-' * 110)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-24s %9.4f %9.4f %9.4f %+10.4f %+10.4f %9.4f' %
              (name, r['baseline'], r['tge_lam0'], r['acspc'],
               r['tge_gain'], r['acspc_gain'], r['recall']))
    print('-' * 110)
    print('%-24s %9.4f %9.4f %9.4f %+10.4f %+10.4f %9s   <- mean' %
          ('mean', mean_b, mean_t, mean_o, mean_t - mean_b, mean_o - mean_b, '—'))
    print('=' * 110)
    print('正向对照 (主表 DIOR->COCO): baseline 0.3809 / A-CSPC 0.4461 (+0.0652); 逐类全正')
    print('elapsed %.1f s -> %s' % (time.time() - t0, OUT))


if __name__ == '__main__':
    main()
