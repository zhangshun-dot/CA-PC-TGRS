# -*- coding: utf-8 -*-
r"""tge_v2_doubao_precheck.py — 豆包"非数据驱动"建议的廉价前检

不重抽任何特征，不占 GPU 训练。只用两份已有产物：
  - 目标域框特征 F (n_box,512)，CLIP ViT-B/32   ← 实验数据/tge_v2_coco_feats_6cls.pkl
  - 源域类原型 (CLIP 同空间，GT 裁剪均值)        ← 现算，只编 SRC_MAP 用到的类

三段：
P1 文本方向共线性：d(car) 与 d(truck) 等的两两 cos，及差分项的相对模长
     —— 判"类间文本方向投影"是不是又一次被共线吃掉的路线
P2 ★逐框判别力：文本类间 margin 能否比源域原型相似度更好地分开 TP/FP
     —— 这是 Group B 的直接前检，不需要重算 AP
P3 池级聚合对照（#8 族）：mean vs top-k 造目标原型后的逐框 AUC

输出: 实验数据/tge_v2_doubao_precheck.json
运行: C:\python-study\python39\python.exe tge_v2_doubao_precheck.py
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch

import unified_eval as U
import tge_probe as T
import tge_probe_nwpu as NW
import tge_probe_dota_src as DS
from tge_v2_main import clip_proto, gt_crops, SRC_TMPL, TGT_TMPL, DATA

FEAT_CACHE = DATA / 'tge_v2_coco_feats_6cls.pkl'
OUT = DATA / 'tge_v2_doubao_precheck.json'

TARGETS = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
SRC_MAP = {
    'DIOR':     {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
                 'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle'},
    'NWPU':     {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
                 'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle'},
    'DOTA':     {'airplane': 'plane', 'boat': 'ship', 'car': 'vehicle',
                 'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle'},
    'VisDrone': {'airplane': None, 'boat': None, 'car': 'car',
                 'truck': 'truck', 'bus': 'bus', 'motorcycle': 'motor'},
}


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def avg_rank(v):
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind='mergesort')
    rank = np.empty(len(v), dtype=np.float64)
    sv = v[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        rank[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return rank


def auc(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = avg_rank(np.concatenate([pos, neg]))
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(neg)))


def iou_pair(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=np.float64)[:, None, :]
    b = np.asarray(b, dtype=np.float64)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    bb = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    return inter / np.maximum(aa + bb - inter, 1e-9)


def main():
    t0 = time.time()
    print('=' * 78)
    print('豆包非数据驱动建议 · 廉价前检')
    print('=' * 78)

    cache = pickle_load(FEAT_CACHE)
    print('缓存类:', list(cache.keys()))

    texts = {}
    texts['tgt'] = {c: unit(T.text_emb([TGT_TMPL.format(c)]).cpu().numpy()[0]) for c in TARGETS}
    need_src = set()
    for s, m in SRC_MAP.items():
        for c, sc in m.items():
            if sc:
                need_src.add((s, sc))
    for s, sc in sorted(need_src):
        texts.setdefault('src', {})[(s, sc)] = unit(T.text_emb([SRC_TMPL.format(sc)]).cpu().numpy()[0])
    print('文本嵌入: 目标 %d 类 / 源 %d 个 (src,cls)' % (len(texts['tgt']), len(texts['src'])))

    # ---------------- P1 文本方向共线性 ----------------
    print('\n' + '-' * 78)
    print('P1  目标类文本方向两两 cos（共线 ⇒ 差分项是残差）')
    print('-' * 78)
    dt = texts['tgt']
    p1 = {'cos': {}, 'residual_ratio': {}}
    print('        ' + ''.join('%11s' % c[:10] for c in TARGETS))
    for a in TARGETS:
        row = []
        for b in TARGETS:
            c = float(dt[a] @ dt[b]); row.append('%11.4f' % c)
            if a < b:
                p1['cos']['%s|%s' % (a, b)] = c
                p1['residual_ratio']['%s-%s' % (a, b)] = float(np.linalg.norm(dt[a] - dt[b]))
        print('%8s' % a[:8] + ''.join(row))
    rr = [v for k, v in p1['residual_ratio'].items()]
    print('\n  差分项相对模长 ||d(a)-d(b)|| :  min %.4f  中位 %.4f  max %.4f'
          % (min(rr), float(np.median(rr)), max(rr)))
    print('  （对照：原型间只减共线 = 0 增益；正交分量占模长 0.14 时扛了 78%% 增益）')
    p1['residual_ratio_stats'] = {'min': min(rr), 'median': float(np.median(rr)), 'max': max(rr)}

    # ---------------- 源域原型（只编用到的类） ----------------
    print('\n' + '-' * 78)
    print('构造源域原型（只编 SRC_MAP 用到的类）...')
    print('-' * 78)
    protos = {}
    for s in SRC_MAP:
        protos[s] = {}
    t1 = time.time()
    for cls in ('airplane', 'ship', 'vehicle'):
        try:
            im, gt = U.load_dior(cls)
            protos['DIOR'][cls] = unit(clip_proto(gt_crops(im, gt)).cpu().numpy())
        except Exception as e:
            print('  DIOR/%s 失败 %s' % (cls, e))
    print('  DIOR  完成 %.0fs' % (time.time() - t1)); t1 = time.time()
    for cls in ('airplane', 'ship', 'vehicle'):
        try:
            im, gt = NW.load_nwpu(cls)
            protos['NWPU'][cls] = unit(clip_proto(gt_crops(im, gt)).cpu().numpy())
        except Exception as e:
            print('  NWPU/%s 失败 %s' % (cls, e))
    print('  NWPU  完成 %.0fs' % (time.time() - t1)); t1 = time.time()
    for cls in ('plane', 'ship'):
        try:
            im, gt = DS.load_dota(cls)
            protos['DOTA'][cls] = unit(clip_proto(gt_crops(im, gt)).cpu().numpy())
        except Exception as e:
            print('  DOTA/%s 失败 %s' % (cls, e))
    try:
        iv, gv = [], []
        for vc in ('large-vehicle', 'small-vehicle'):
            im2, gt2 = DS.load_dota(vc); iv += im2; gv += gt2
        protos['DOTA']['vehicle'] = unit(clip_proto(gt_crops(iv, gv)).cpu().numpy())
    except Exception as e:
        print('  DOTA/vehicle 失败 %s' % e)
    print('  DOTA  完成 %.0fs' % (time.time() - t1)); t1 = time.time()
    try:
        from tge_v2_visdrone import load_visdrone
        for cls in ('car', 'truck', 'bus', 'motor'):
            im, gt = load_visdrone(cls)
            if len(im) == 0:
                continue
            protos['VisDrone'][cls] = unit(clip_proto(gt_crops(im, gt)).cpu().numpy())
    except Exception as e:
        print('  VisDrone 失败 %s' % e)
    print('  VisDrone 完成 %.0fs' % (time.time() - t1))
    print('  原型数:', {s: sorted(v.keys()) for s, v in protos.items()})

    # ---------------- P2 逐框判别力 ----------------
    print('\n' + '=' * 78)
    print('P2  逐框 AUC（TP vs FP, IoU>=0.5）—— 文本类间 margin 能否赢过源原型相似度')
    print('=' * 78)
    p2 = {}
    hdr = '%-10s %-13s %6s %8s %8s %8s %8s'
    print(hdr % ('src', 'target', 'n_box', 'det', 'base', 'txt', 'margin'))
    for src, m in SRC_MAP.items():
        for cls in TARGETS:
            sc = m[cls]
            if not sc or sc not in protos.get(src, {}):
                continue
            key = '%s-%s' % (src, cls)
            d = cache['COCO-%s' % cls]
            F = np.asarray(d['F'], dtype=np.float64)
            F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
            p_src = protos[src][sc]
            rows, gds, labs = [], [], []
            for e in d['per_img']:
                ci = np.asarray(e['crop_idx'])
                bx = np.asarray(e['box'], dtype=np.float64)
                gt = np.asarray(e['gt'], dtype=np.float64) if len(e['gt']) else np.zeros((0, 4))
                io = iou_pair(bx, gt)
                mx = io.max(axis=1) if io.size else np.zeros(len(bx))
                rows.append(ci); gds.append(np.asarray(e['gd'], dtype=np.float64))
                labs.append((mx >= 0.5).astype(np.float64))
            idx = np.concatenate(rows)
            gd = np.concatenate(gds)
            lab = np.concatenate(labs)
            Fc = F[idx]
            pos, neg = lab > 0.5, lab < 0.5
            if pos.sum() < 5 or neg.sum() < 5:
                continue
            s_det = gd
            s_base = Fc @ p_src
            s_txt = Fc @ dt[cls]
            others = np.stack([dt[c] for c in TARGETS if c != cls])
            comp = (Fc @ others.T).max(axis=1)
            s_marg = s_txt - comp
            rec = {'n_box': int(len(gd)), 'n_pos': int(pos.sum()),
                   'auc_det': auc(s_det[pos], s_det[neg]),
                   'auc_base': auc(s_base[pos], s_base[neg]),
                   'auc_txt': auc(s_txt[pos], s_txt[neg]),
                   'auc_margin': auc(s_marg[pos], s_marg[neg])}
            p2[key] = rec
            print(hdr % (src, cls, rec['n_box'], '%.4f' % rec['auc_det'], '%.4f' % rec['auc_base'],
                         '%.4f' % rec['auc_txt'], '%.4f' % rec['auc_margin']))

    print('\n  宏平均（等权于上表格子）:')
    for k in ('auc_det', 'auc_base', 'auc_txt', 'auc_margin'):
        vals = [v[k] for v in p2.values() if v[k] == v[k]]
        print('    %-12s %.4f  (n=%d)' % (k, float(np.mean(vals)), len(vals)))
    print('\n  只取 DIOR（纯 vehicle 上位类，无子类区分）:')
    sub = [v for kk, v in p2.items() if kk.startswith('DIOR')]
    for k in ('auc_det', 'auc_base', 'auc_txt', 'auc_margin'):
        vals = [v[k] for v in sub if v[k] == v[k]]
        print('    %-12s %.4f  (n=%d)' % (k, float(np.mean(vals)), len(vals)))
    print('\n  只取 VisDrone（源域自带 car/truck/bus/motor 子类）:')
    sub = [v for kk, v in p2.items() if kk.startswith('VisDrone')]
    for k in ('auc_det', 'auc_base', 'auc_txt', 'auc_margin'):
        vals = [v[k] for v in sub if v[k] == v[k]]
        print('    %-12s %.4f  (n=%d)' % (k, float(np.mean(vals)), len(vals)))

    # ---------------- P3 池级聚合对照 ----------------
    print('\n' + '=' * 78)
    print('P3  池级聚合（#8 族对照）：用不同聚合造目标原型后的逐框 AUC')
    print('  注：这是"对框聚合"，不是"框内 patch 聚合"——patch 级要重抽特征，本次做不到')
    print('=' * 78)
    p3 = {}
    FRACS = [0.05, 0.10, 0.25, 1.0]
    print('%-10s %-12s %s' % ('target', 'n_box', '  '.join('top%3d%%' % (f * 100) for f in FRACS)))
    for cls in TARGETS:
        key = 'COCO-%s' % cls
        d = cache[key]
        F = np.asarray(d['F'], dtype=np.float64)
        F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
        gds, labs = [], []
        for e in d['per_img']:
            bx = np.asarray(e['box'], dtype=np.float64)
            gt = np.asarray(e['gt'], dtype=np.float64) if len(e['gt']) else np.zeros((0, 4))
            io = iou_pair(bx, gt)
            mx = io.max(axis=1) if io.size else np.zeros(len(bx))
            gds.append(np.asarray(e['gd'], dtype=np.float64))
            labs.append((mx >= 0.5).astype(np.float64))
        gd = np.concatenate(gds); lab = np.concatenate(labs)
        pos, neg = lab > 0.5, lab < 0.5
        if pos.sum() < 5 or neg.sum() < 5:
            continue
        row = []
        for f in FRACS:
            k = max(1, int(round(len(gd) * f)))
            sel = np.argsort(-gd)[:k]
            p_hat = unit(F[sel].mean(axis=0))
            s = F @ p_hat
            a = auc(s[pos], s[neg]); row.append(a)
            p3.setdefault('top%d' % int(f * 100), {})[cls] = a
        print('%-10s %-12d %s' % (cls, len(gd), '  '.join('%.4f' % a for a in row)))
    for kk, vv in p3.items():
        print('  %-8s 宏平均 %.4f' % (kk, float(np.mean(list(vv.values())))))

    res = {'P1_text_cosine': p1, 'P2_auc': p2, 'P3_pool_agg': p3,
           'elapsed_s': time.time() - t0}
    json.dump(res, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s   (%.0fs)' % (OUT, time.time() - t0))


def pickle_load(p):
    import pickle
    return pickle.load(open(p, 'rb'))


if __name__ == '__main__':
    main()
