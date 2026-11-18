# -*- coding: utf-8 -*-
r"""R2#4 补充指标：precision / recall / F1 / matched-box IoU + 多随机 split CI。

复用 confirm_split_cache 的逐框缓存与 confirm_devtest_global_split.py 的
打分/匹配口径（beta=2.0, lam=0.5，论文主配置；IoU 分母 1e-8，逐图贪心，
pre-filter 0.03，前 10 张 calibration 排除）。不重跑检测器。

定义（在固定 score 工作点上，取数据集级默认阈值 = detector pre-filter 之上
再做 score>=tau；这里给出两类口径）：
  * OP-F1：在全集 PR 曲线上取 F1 最大的工作点，报该点 P/R/F1（阈值无关，稳健）。
  * OP-AP50recall：在 AP50 的 11 点插值召回最靠近的点上报 P/R（与主表呼应）。
  * boxIoU：所有 TP 匹配的 IoU 均值（定位精度）。

显著性：图级配对 bootstrap（同一 resample 索引同时抽 base/corrected），
报告 delta 的 95% CI 与 P(delta<=0)。5 个随机全局 split（seed 0..4）同
confirm 脚本，dev/test 按 COCO image id 不重叠切分。
"""
import os, sys, json, pickle
import numpy as np

sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
DATA = r'C:\Users\张三\Desktop\实验数据'
CACHE = os.path.join(DATA, 'confirm_split_cache')
OUT = os.path.join(DATA, 'r2_extra_metrics_20261118.json')

CLASSES = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
K_CALIB = 10
BETA, LAM = 2.0, 0.5

def iou_mat(b, g):
    x1 = np.maximum(b[:, None, 0], g[None, :, 0]); y1 = np.maximum(b[:, None, 1], g[None, :, 1])
    x2 = np.minimum(b[:, None, 2], g[None, :, 2]); y2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    a2 = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-8)

def make_W(blob, beta, lam):
    P = blob['P_src'] + beta * blob['Delta']
    P = P / np.linalg.norm(P)
    return P - lam * blob['P_neg']

def per_image_eval(blob, idx, W):
    """逐图贪心匹配；返回每图的 (score 降序, tp, iou_matched)，供全局 PR 与指标。"""
    sc_all, tp_all, iou_all = [], [], []
    F = blob['F']
    for i in idx:
        p = blob['per_img'][i]
        gd = p['gd']; gt = p['gt']; box = p['box']
        n = len(gd)
        if n == 0:
            continue
        if W is None:            # base = 原始 detector score（与主表 base 口径一致）
            s = np.asarray(gd, dtype=np.float64).copy()
        else:
            w = F[p['crop_idx']] @ W
            s = np.sqrt(gd * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        order = np.argsort(-s)
        tp = np.zeros(len(order), dtype=np.int8)
        iou_m = np.zeros(len(order), dtype=np.float64)
        if len(gt):
            IOU = iou_mat(box, gt)
            matched = np.zeros(len(gt), dtype=bool)
            for k, ii in enumerate(order):
                row = IOU[ii].copy(); row[matched] = -1.0
                j = int(np.argmax(row))
                if row[j] >= 0.5:
                    tp[k] = 1; iou_m[k] = row[j]; matched[j] = True
        sc_all.append(s[order]); tp_all.append(tp); iou_all.append(iou_m)
    if not sc_all:
        return (np.zeros(0), np.zeros(0, np.int8), np.zeros(0))
    return (np.concatenate(sc_all), np.concatenate(tp_all), np.concatenate(iou_all))

def ap50(sc, tp, n_gt):
    if n_gt == 0:
        return 0.0
    order = np.argsort(-sc, kind='stable'); t = tp[order]
    tc = np.cumsum(t); fc = np.cumsum(1 - t)
    rec = tc / n_gt; prec = tc / (tc + fc + 1e-8)
    # 11-point 插值：分母恒为 11，达不到的召回点插值为 0（不能只对有效点取均值）
    ap = 0.0
    for th in np.linspace(0, 1, 11):
        m = rec >= th
        if m.any():
            ap += float(prec[m].max())
    return ap / 11.0

def metrics(sc, tp, iou_m, n_gt):
    """OP-F1（PR 曲线最大 F1）+ 该点 P/R/F1 + matched IoU。"""
    if n_gt == 0 or len(sc) == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, box_iou=0.0, op_thresh=None, n_tp=0)
    order = np.argsort(-sc, kind='stable'); t = tp[order].astype(np.float64); im = iou_m[order]
    tc = np.cumsum(t); fc = np.cumsum(1 - t)
    rec = tc / n_gt; prec = tc / (tc + fc + 1e-8)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    k = int(np.argmax(f1))
    tp_mask = t[:k+1] > 0
    box_iou = float(im[:k+1][tp_mask].mean()) if tp_mask.any() else 0.0
    return dict(precision=float(prec[k]), recall=float(rec[k]), f1=float(f1[k]),
                box_iou=box_iou, op_thresh=float(sc[order][k]), n_tp=int(tc[k]))

def load():
    blobs = {}
    for c in CLASSES:
        with open(os.path.join(CACHE, c + '.pkl'), 'rb') as f:
            blobs[c] = pickle.load(f)
    return blobs

def agg(blobs, idx_map, W_map):
    """返回六类池化的 (sc,tp,iou) 与每类 AP。"""
    pooled = {a: [] for a in ('b', 'c')}
    per_cls = {}
    for c in CLASSES:
        idx = idx_map[c]; blob = blobs[c]
        n_gt = sum(len(blob['per_img'][i]['gt']) for i in idx)
        sb, tb, ib = per_image_eval(blob, idx, None)
        sc, tc, ic = per_image_eval(blob, idx, W_map[c])
        pooled['b'].append((sb, tb, ib)); pooled['c'].append((sc, tc, ic))
        per_cls[c] = dict(n_gt=n_gt,
                          base_ap=ap50(sb, tb, n_gt),
                          capc_ap=ap50(sc, tc, n_gt))
    out = {}
    for a in ('b', 'c'):
        sc = np.concatenate([x[0] for x in pooled[a]])
        tp = np.concatenate([x[1] for x in pooled[a]])
        im = np.concatenate([x[2] for x in pooled[a]])
        out[a] = (sc, tp, im)
    n_gt_all = sum(pc['n_gt'] for pc in per_cls.values())
    return out, per_cls, n_gt_all

def _f1_only(sc, tp, n_gt):
    if n_gt == 0 or len(sc) == 0:
        return 0.0
    order = np.argsort(-sc, kind='stable'); t = tp[order].astype(np.float64)
    tc_ = np.cumsum(t); fc = np.cumsum(1 - t)
    rec = tc_ / n_gt; prec = tc_ / (tc_ + fc + 1e-8)
    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=(prec+rec) > 0)
    return float(f1.max())

def _iou_tp(sc, tp, iou_m, n_gt):
    """box-IoU 与全量表一致：取 OP-F1 工作点上的 TP 匹配框平均 IoU。"""
    if n_gt == 0 or len(sc) == 0:
        return 0.0
    order = np.argsort(-sc, kind='stable'); t = tp[order].astype(np.float64); im = iou_m[order]
    tc_ = np.cumsum(t); fc = np.cumsum(1 - t)
    rec = tc_ / n_gt; prec = tc_ / (tc_ + fc + 1e-8)
    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros_like(prec), where=(prec+rec) > 0)
    k = int(np.argmax(f1))
    mask = t[:k+1] > 0
    return float(im[:k+1][mask].mean()) if mask.any() else 0.0

def ci(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return dict(mean=float(arr.mean()), lo95=float(lo), hi95=float(hi),
                p_le_0=float((arr <= 0).mean()), n_boot=int(len(arr)))

def boot_ci(blobs, idx_map, W_map, B=1000, seed=0):
    """图级配对 bootstrap（六类合并）：F1 / boxIoU 的 delta 95% CI 与 P(delta<=0)。
    同一张图的 base 与 CA-PC 用同一 resample 索引，消除图难度差异。"""
    ib, ic = [], []
    for c in CLASSES:
        blob = blobs[c]; idx = idx_map[c]; F = blob['F']
        for i in idx:
            p = blob['per_img'][i]; gd = p['gd']; gt = p['gt']; box = p['box']
            n = len(gd)
            if n == 0:
                continue
            w = F[p['crop_idx']] @ W_map[c]
            s_c = np.sqrt(gd * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
            iou_big = iou_mat(box, gt) if len(gt) else None
            def build(s):
                order = np.argsort(-s)
                tp = np.zeros(len(order), np.int8); iom = np.zeros(len(order))
                if len(gt):
                    matched = np.zeros(len(gt), bool)
                    for k, ii in enumerate(order):
                        row = iou_big[ii].copy(); row[matched] = -1.0
                        j = int(np.argmax(row))
                        if row[j] >= 0.5:
                            tp[k] = 1; iom[k] = row[j]; matched[j] = True
                return (np.asarray(s, float)[order], tp, iom, len(gt))
            ib.append(build(gd)); ic.append(build(s_c))
    n = len(ib)
    rng = np.random.default_rng(seed)
    d_f1 = np.empty(B); d_iou = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, n, n)
        ng = int(sum(ib[i][3] for i in pick))
        sb = np.concatenate([ib[i][0] for i in pick]); tb = np.concatenate([ib[i][1] for i in pick]); mb = np.concatenate([ib[i][2] for i in pick])
        sc_ = np.concatenate([ic[i][0] for i in pick]); tc_ = np.concatenate([ic[i][1] for i in pick]); mc = np.concatenate([ic[i][2] for i in pick])
        d_f1[b] = _f1_only(sc_, tc_, ng) - _f1_only(sb, tb, ng)
        d_iou[b] = _iou_tp(sc_, tc_, mc, ng) - _iou_tp(sb, tb, mb, ng)
    return dict(f1=ci(d_f1), box_iou=ci(d_iou))

def main():
    blobs = load()
    W_map = {c: make_W(blobs[c], BETA, LAM) for c in CLASSES}
    # ---- 全量校验：复现 confirm 脚本的 base / gain ----
    idx_full = {c: list(range(K_CALIB, len(blobs[c]['per_img']))) for c in CLASSES}
    out, per_cls, ngt = agg(blobs, idx_full, W_map)
    sc_b, tp_b, im_b = out['b']; sc_c, tp_c, im_c = out['c']
    base_ap = ap50(sc_b, tp_b, ngt); capc_ap = ap50(sc_c, tp_c, ngt)
    print('=== 全量校验 ===')
    print(f'  base AP50={base_ap:.4f}  CA-PC AP50={capc_ap:.4f}  gain={capc_ap-base_ap:+.4f}')
    print('  confirm 脚本对照: base 六类均值≈0.3108  gain≈+0.0646')
    for c in CLASSES:
        pc = per_cls[c]
        print(f"   {c:<11} base={pc['base_ap']:.4f} capc={pc['capc_ap']:.4f} gain={pc['capc_ap']-pc['base_ap']:+.4f}")

    print('=== 全量 OP-F1 / P / R / boxIoU ===')
    mb = metrics(sc_b, tp_b, im_b, ngt); mc = metrics(sc_c, tp_c, im_c, ngt)
    print('  base ', mb)
    print('  CA-PC', mc)
    print('  delta F1=%+.4f  boxIoU=%+.4f' % (mc['f1']-mb['f1'], mc['box_iou']-mb['box_iou']))

    # ---- 5 个随机全局 split ----
    all_ids = np.unique(np.concatenate([blobs[c]['img_ids'][K_CALIB:] for c in CLASSES]))
    print(f'\n=== 5 随机 split (唯一 image id={len(all_ids)}) ===')
    splits = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(all_ids))
        dev_ids = set(all_ids[perm[:len(perm)//2]].tolist())
        test_ids = all_ids[perm[len(perm)//2:]]
        test_set = set(test_ids.tolist())
        idx_map = {}
        for c in CLASSES:
            tm = np.array([int(i) in test_set for i in blobs[c]['img_ids']])
            idx_map[c] = list(np.where(tm)[0])
        o, pcs, ng = agg(blobs, idx_map, W_map)
        b = metrics(o['b'][0], o['b'][1], o['b'][2], ng)
        cc = metrics(o['c'][0], o['c'][1], o['c'][2], ng)
        apb = ap50(o['b'][0], o['b'][1], ng); apc = ap50(o['c'][0], o['c'][1], ng)
        splits.append(dict(seed=seed, n_test_img=len(test_ids), n_gt=ng,
                           base=dict(ap50=apb, **b), capc=dict(ap50=apc, **cc),
                           d_f1=cc['f1']-b['f1'], d_box_iou=cc['box_iou']-b['box_iou'],
                           d_ap50=apc-apb))
        print(f"  seed={seed} n={len(test_ids):3d} base F1={b['f1']:.4f} "
              f"CA-PC F1={cc['f1']:.4f} dF1={cc['f1']-b['f1']:+.4f} "
              f"dBoxIoU={cc['box_iou']-b['box_iou']:+.4f} dAP={apc-apb:+.4f}")

    def stat(key):
        vals = [s[key] for s in splits]
        return dict(mean=float(np.mean(vals)), std=float(np.std(vals, ddof=1)),
                    lo=float(np.min(vals)), hi=float(np.max(vals)), n_pos=int(sum(v>0 for v in vals)))
    summary = dict(d_f1=stat('d_f1'), d_box_iou=stat('d_box_iou'), d_ap50=stat('d_ap50'))

    # ---- 图级配对 bootstrap（全集，B=1000, seed=20261118）----
    print('\n=== 配对 bootstrap (B=1000, seed=20261118) ===')
    boot = boot_ci(blobs, idx_full, W_map, B=1000, seed=20261118)
    print('  dF1    :', boot['f1'])
    print('  dBoxIoU:', boot['box_iou'])
    print('\n=== summary ==='); print(json.dumps(summary, indent=2, ensure_ascii=False))

    result = dict(config=dict(beta=BETA, lam=LAM, k_calib=K_CALIB,
                              metric='OP-F1 on pooled PR + matched-box IoU',
                              splits='5 random global COCO-image-id splits, seed 0..4',
                              bootstrap='paired image-level, B=1000, seed=20261118'),
                  full=dict(base_ap50=base_ap, capc_ap50=capc_ap, gain=capc_ap-base_ap,
                            base=mb, capc=mc, per_class=per_cls),
                  splits=splits, summary=summary, bootstrap=boot)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print('\nSaved:', OUT)

if __name__ == '__main__':
    main()
