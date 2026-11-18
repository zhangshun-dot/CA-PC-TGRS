# -*- coding: utf-8 -*-
"""tge_bootstrap_ci.py — 主表显著性: 6 类配对图像级 bootstrap (N=100 主口径, 逐图 AP50)

对 tge_probe_fs.py 的零样本主口径 (baseline vs zs_extrap, 后 90 图测试)
做配对重采样: 每轮从 90 图中带替换抽 90 图, baseline 与 Ours 用同一
重采样集重算标准逐图 AP50 (per-image 匹配 + 全局排序) → ΔAP 分布 → 95% CI + 单边 p。

2026-08-23 协议修正: 评估全部改用逐图匹配 AP50 (U.compute_ap50_img),
全局拼接匹配已被审计为跨图误匹配 (COCO 同尺寸图坐标可比)。

关键优化:
  * CLIP 特征一次性预编码 (vis = F@W 线性), bootstrap 循环内只有矩阵乘
  * 每图预计算"框×该图GT"IoU 子矩阵, 重采样轮内逐图匹配向量化

对账: 脚本先输出全量后 90 图 AP, 应与 (逐图版) tge_probe_fs.json 逐位一致 (4 位小数)。
注意: 请先重跑 tge_probe_fs.py 生成逐图版主表, 再跑本脚本对账。
p 值定义: p = (Δ<=0 的轮数 + 1) / (R + 1)  (单边, 避免 0)

输出: 实验数据/tge_bootstrap_ci.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import time
from pathlib import Path
import numpy as np
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_bootstrap_ci.json'
FS_JSON = DATA / 'tge_probe_fs.json'

BETA = 2.0
N_COCO = 100
K = 10                      # 前 K 图校准段 (零样本列只用后 90 图测试, 与主表一致)
R = 1000                    # bootstrap 轮数
SEED = 42
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


def iou_pairwise(boxes, gts):
    """(N,4) x (M,4) -> (N,M) IoU 矩阵 (与 U.iou 同公式, +1e-8 分母保护)。"""
    b = np.asarray(boxes, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    xx1 = np.maximum(b[:, None, 0], g[None, :, 0])
    yy1 = np.maximum(b[:, None, 1], g[None, :, 1])
    xx2 = np.minimum(b[:, None, 2], g[None, :, 2])
    yy2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / (ab[:, None] + ag[None, :] - inter + 1e-8)


def ap50_from_parts(per_img, idx, key):
    """标准逐图 AP50: 对重采样的图集合, 每图内贪心匹配, 全局排序画 PR。

    与 U.compute_ap50_img 完全同口径 (匹配条件 IoU>=0.5, 11-point max precision)。
    per_img: list of {'box', 'gd', 'ours', 'gt', 'iou'} (每图框×该图GT IoU 预计算)
    idx: 重采样图索引 (相对 per_img)
    key: 'gd' 或 'ours' (每图的分数数组)
    """
    n_gt = sum(len(per_img[i]['gt']) for i in idx)
    evals = []  # (score, tp)
    for i in idx:
        p = per_img[i]
        iou_m = p['iou']
        scores = p[key]
        if len(scores) == 0:
            continue
        matched = np.zeros(iou_m.shape[1], dtype=bool)
        for j in np.argsort(-scores):
            row = iou_m[j].copy()
            row[matched] = -1.0
            if row.max() >= 0.5:
                evals.append((float(scores[j]), 1))
                matched[int(row.argmax())] = True
            else:
                evals.append((float(scores[j]), 0))
    if n_gt == 0:
        return 0.0 if evals else 1.0
    if not evals:
        return 0.0
    evals.sort(key=lambda x: -x[0])
    tp = np.array([e[1] for e in evals])
    tc = np.cumsum(tp)
    fc = np.cumsum(1 - tp)
    rec = tc / n_gt
    prec = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = rec >= t
        if m.any():
            ap += float(prec[m].max())
    return ap / 11.0


def collect_data(images, dets, gts, W_hat):
    """收集 per-image 评估数据 (与 tge_probe_fs.eval_class 同协议)。

    pre-filter 0.03 -> 小框(<=10px)过滤 -> 每图 (boxes, gd) + CLIP 特征编码一次
    W_hat 为 (512,) 权重向量 (P_hat - lam*P_neg), vis = max(0, F@W_hat)
    ours = sqrt(gd * clip(vis, 1e-6, 1))
    """
    crops_all = []
    per_img = []
    for (boxes, scores), img, g in zip(dets, images, gts):
        keep = scores > U.PRE_FILTER
        bl, sl = [], []
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            bl.append([x1, y1, x2, y2])
            sl.append(float(sc))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': (np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0)
                   if sl else np.zeros(0)),
            'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
        })

    if crops_all:
        F = T.encode_clip_batch(crops_all).cpu().numpy()
    else:
        F = np.zeros((0, 512))

    fi = 0
    for p in per_img:
        n = len(p['gd'])
        if n:
            vis = np.maximum(0.0, F[fi:fi + n] @ W_hat)
            p['ours'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['ours'] = np.zeros(0)
        p['iou'] = iou_pairwise(p['box'], p['gt'])
        fi += n
    return per_img


def main():
    t0 = time.time()
    print('[1/4] DIOR 20 类 CLIP 源原型 ...')
    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        images, gts = U.load_dior(cls)
        crops = []
        for img, g in zip(images, gts):
            for box in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 > x1 + 10 and y2 > y1 + 10:
                    crops.append(img.crop((x1, y1, x2, y2)))
        if not crops:
            continue
        F = T.encode_clip_batch(crops)
        p = F.mean(dim=0)
        protos[cls] = p / p.norm()
    print('    %d 类原型就绪' % len(protos))

    # 读主表对账 (先重跑 tge_probe_fs.py 得到逐图版主表!)
    fs = json.load(open(FS_JSON, encoding='utf-8'))['classes']

    print('[2/4] COCO 6 类: 特征预编码 (一次性) ...')
    class_data = []
    reconciled = True
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = T.cached_dets(images, TGT_TMPL.format(coco_cls), name)

        P_src = protos[src_cls].to(U.DEVICE)
        P_neg = T.get_neg_proto(protos, src_cls)
        s_xs = T.text_emb([SRC_TMPL.format(src_cls)]).cpu()[0]
        d_y = T.text_emb([TGT_TMPL.format(coco_cls)]).cpu()[0]
        P_hat = P_src + BETA * (d_y - s_xs).to(U.DEVICE)
        P_hat = P_hat / P_hat.norm()

        W_hat = (P_hat.cpu().numpy() - U.NEG_LAM * P_neg.cpu().numpy())
        per_img = collect_data(images, dets, gts, W_hat)

        # 全量对账 (后 90 图, 逐图 AP50, 与主表同协议)
        test_idx = list(range(K, len(images)))
        sel = [per_img[i] for i in test_idx]
        im_preds_b = [(p['box'], p['gd']) for p in sel]
        im_preds_o = [(p['box'], p['ours']) for p in sel]
        im_gts = [p['gt'] for p in sel]
        ap_base = U.compute_ap50_img(im_preds_b, im_gts)
        ap_ours = U.compute_ap50_img(im_preds_o, im_gts)
        exp_b, exp_o = fs[name]['baseline'], fs[name]['zs_extrap']
        ok_b, ok_o = abs(ap_base - exp_b) < 5e-4, abs(ap_ours - exp_o) < 5e-4
        print('  %-16s base=%.4f (%.4f %s)  ours=%.4f (%.4f %s)'
              % (name, ap_base, exp_b, 'OK' if ok_b else 'MISMATCH',
                 ap_ours, exp_o, 'OK' if ok_o else 'MISMATCH'))
        if not (ok_b and ok_o):
            reconciled = False

        class_data.append({'per_img': per_img, 'name': name,
                           'n_test': len(images) - K,
                           'full_ap': {'baseline': round(ap_base, 4),
                                       'zs_extrap': round(ap_ours, 4)}})

    print('[3/4] 配对 bootstrap (R=%d, seed=%d, 逐图 AP50) ...' % (R, SEED))
    if not reconciled:
        print('  !! 警告: 对账存在 MISMATCH, 结果不可信, 先修 bug 再跑 !!')
    rng = np.random.default_rng(SEED)
    n_cls = len(class_data)
    deltas = np.zeros((n_cls, R))
    for r in range(R):
        idx = rng.integers(0, class_data[0]['n_test'], size=class_data[0]['n_test'])
        for ci, d in enumerate(class_data):
            im_idx = [i + K for i in idx]
            ap_b = ap50_from_parts(d['per_img'], im_idx, 'gd')
            ap_o = ap50_from_parts(d['per_img'], im_idx, 'ours')
            deltas[ci, r] = ap_o - ap_b
        if (r + 1) % 200 == 0:
            print('    round %d/%d' % (r + 1, R))

    print('[4/4] 汇总写盘 ...')
    rows = {}
    mean6 = deltas.mean(axis=0)
    for ci, d in enumerate(class_data):
        dd = deltas[ci]
        n_le = int((dd <= 0).sum())
        rows[d['name']] = {
            'full_ap': d['full_ap'],
            'boot': {
                'mean_delta': round(float(dd.mean()), 4),
                'ci95_low': round(float(np.percentile(dd, 2.5)), 4),
                'ci95_high': round(float(np.percentile(dd, 97.5)), 4),
                'p_delta_le_0': round((n_le + 1) / (R + 1), 4),
                'n_pos_rounds': int((dd > 0).sum()),
            },
            'n_test_images': d['n_test'],
            'n_gt_total': sum(len(p['gt']) for p in d['per_img']),
        }

    n_le6 = int((mean6 <= 0).sum())
    out = {
        'config': {'n_coco': N_COCO, 'k': K, 'beta': BETA, 'neg_lam': U.NEG_LAM,
                   'r': R, 'seed': SEED,
                   'protocol': 'paired image-level bootstrap on N=100 main-table '
                               'protocol (90 test images), image-safe per-image AP50 '
                               '(per-image matching + global PR, COCO official)',
                   'p_definition': '(n_rounds_delta_le_0 + 1) / (R + 1), one-sided'},
        'mean6': {'mean_delta': round(float(mean6.mean()), 4),
                  'ci95_low': round(float(np.percentile(mean6, 2.5)), 4),
                  'ci95_high': round(float(np.percentile(mean6, 97.5)), 4),
                  'p_delta_le_0': round((n_le6 + 1) / (R + 1), 4)},
        'classes': rows,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 100)
    print('%-16s %10s %8s %8s %8s' % ('class', 'dAP_mean', 'CI_low', 'CI_high', 'p(d<=0)'))
    print('-' * 100)
    for name in [t[0] for t in TARGETS]:
        b = rows[name]['boot']
        print('%-16s %+10.4f %+8.4f %+8.4f %8.4f' % (name, b['mean_delta'], b['ci95_low'],
                                                      b['ci95_high'], b['p_delta_le_0']))
    m = out['mean6']
    print('-' * 100)
    print('%-16s %+10.4f %+8.4f %+8.4f %8.4f   <- 6-class mean' % (u'mean6', m['mean_delta'],
                                                                   m['ci95_low'], m['ci95_high'],
                                                                   m['p_delta_le_0']))
    print('=' * 100)
    print('elapsed %.1f s -> %s' % (time.time() - t0, OUT))


if __name__ == '__main__':
    main()
