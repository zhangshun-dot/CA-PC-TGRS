# -*- coding: utf-8 -*-
r"""tge_v2_src_combine_probe.py — 让源域真正出力：三种"源域参与的 margin"

上游读数（tge_v2_source_probe.json）：
  · paper 与 margin 逐类互补（airplane/truck/motorcycle 归 paper，boat/car 归 margin，bus 平）
  · P_neg（纯源域产物）净贡献 +0.026 ~ +0.057，是除文本外推之外的最大单项
  · 四个源域整体都略低于无源域的 margin（DIOR 最接近，−0.0015）
  · VisDrone（唯一自带细粒度类）反而最差 —— 细粒度源域没有让源域更强

本脚本试三种"源域必须参与"的算子，看能否把 dior 的 paper 顶到 margin 之上：
  src_margin  vis = max(0, f·P_src_cls − max_{c≠src_cls} f·P_c)     ← 纯源域类间 margin（不用目标文本）
  hyb_txt     vis = max(0, f·(P̂ − λP_neg) − μ·max_{c≠tgt} f·d_c)   ← 源域通道 + 目标文本竞争
  hyb_src     vis = max(0, f·(P̂ − λP_neg) − μ·max_{c≠src_cls} f·P_c) ← 源域通道 + 源域竞争

输出: 实验数据/tge_v2_src_combine_probe.json
运行: C:\python-study\python39\python.exe tge_v2_src_combine_probe.py
"""
import sys, json, time, pickle
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
import tge_probe_nwpu as NW
import tge_probe_dota_src as DS
from tge_v2_main import clip_proto, gt_crops, SRC_TMPL, TGT_TMPL, DATA, K_CALIB

CACHE = DATA / 'tge_v2_coco_feats_6cls.pkl'
PROTO_PKL = DATA / 'tge_v2_src_protos.pkl'
OUT = DATA / 'tge_v2_src_combine_probe.json'
BETA, LAM = 2.0, 0.5
MUS = [0.25, 0.5, 1.0]
ALLT = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
COMMON4 = ['car', 'truck', 'bus', 'motorcycle']
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
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def iou_mats(bl, gl):
    out = []
    for b, g in zip(bl, gl):
        b = np.asarray(b, float); g = np.asarray(g, float)
        if len(b) == 0 or len(g) == 0:
            out.append(np.zeros((len(b), len(g)))); continue
        x1 = np.maximum(b[:, None, 0], g[None, :, 0]); y1 = np.maximum(b[:, None, 1], g[None, :, 1])
        x2 = np.minimum(b[:, None, 2], g[None, :, 2]); y2 = np.minimum(b[:, None, 3], g[None, :, 3])
        it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        a2 = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        out.append(it / (a1[:, None] + a2[None, :] - it + 1e-8))
    return out


def ap50(mats, sl, n_gt):
    if n_gt == 0:
        return 0.0 if any(len(s) for s in sl) else 1.0
    ev = []
    for M, s in zip(mats, sl):
        if len(s) == 0:
            continue
        s = np.asarray(s); mt = [False] * M.shape[1]
        for i in np.argsort(-s):
            bi, bj = 0.0, -1
            for j in range(M.shape[1]):
                if not mt[j] and M[i, j] > bi:
                    bi, bj = M[i, j], j
            if bi >= 0.5 and bj >= 0:
                ev.append((float(s[i]), 1)); mt[bj] = True
            else:
                ev.append((float(s[i]), 0))
    if not ev:
        return 0.0
    ev.sort(key=lambda x: -x[0])
    tp = np.array([e[1] for e in ev]); tc, fc = np.cumsum(tp), np.cumsum(1 - tp)
    rec = tc / n_gt; prec = tc / (tc + fc + 1e-8)
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0
                          for t in np.linspace(0, 1, 11)]))


def build_pools():
    if PROTO_PKL.exists():
        print('  [缓存] %s' % PROTO_PKL.name)
        return pickle.load(open(PROTO_PKL, 'rb'))
    p = {'DIOR': {}, 'NWPU': {}, 'DOTA': {}, 'VisDrone': {}}
    for cls in U.DIOR_ID2NAME.values():
        try:
            im, gt = U.load_dior(cls); q = clip_proto(gt_crops(im, gt))
            if q is not None:
                p['DIOR'][cls] = unit(q.cpu().numpy())
        except Exception as e:
            print('   DIOR/%s %s' % (cls, e))
    for cls in NW.NWPU_CLASSES:
        try:
            im, gt = NW.load_nwpu(cls); q = clip_proto(gt_crops(im, gt))
            if q is not None:
                p['NWPU'][cls] = unit(q.cpu().numpy())
        except Exception as e:
            print('   NWPU/%s %s' % (cls, e))
    for cls in DS.DOTA_CLASSES:
        try:
            im, gt = DS.load_dota(cls); q = clip_proto(gt_crops(im, gt))
            if q is not None:
                p['DOTA'][cls] = unit(q.cpu().numpy())
        except Exception as e:
            print('   DOTA/%s %s' % (cls, e))
    try:
        iv, gv = [], []
        for vc in ('large-vehicle', 'small-vehicle'):
            im2, gt2 = DS.load_dota(vc); iv += im2; gv += gt2
        p['DOTA']['vehicle'] = unit(clip_proto(gt_crops(iv, gv)).cpu().numpy())
    except Exception as e:
        print('   DOTA/vehicle %s' % e)
    try:
        from tge_v2_visdrone import load_visdrone, VISDRONE_CLASSES
        for cls in VISDRONE_CLASSES:
            im, gt = load_visdrone(cls)
            if len(im) == 0:
                continue
            q = clip_proto(gt_crops(im, gt))
            if q is not None:
                p['VisDrone'][cls] = unit(q.cpu().numpy())
    except Exception as e:
        print('   VisDrone %s' % e)
    pickle.dump(p, open(PROTO_PKL, 'wb'))
    return p


def main():
    t0 = time.time()
    cache = pickle.load(open(CACHE, 'rb'))
    texts = {c: unit(T.text_emb([TGT_TMPL.format(c)]).cpu().numpy()[0]) for c in ALLT}
    print('源域原型池...'); protos = build_pools()
    for s, d in protos.items():
        print('  %-9s %2d 类' % (s, len(d)))
    negp = {}
    for s in protos:
        negp[s] = {c: T.get_neg_proto({k: torch.from_numpy(v).cuda() for k, v in protos[s].items()},
                                     c).cpu().numpy() for c in protos[s]}

    res = {}
    for src in SRC_MAP:
        print('\n' + '=' * 82)
        print('源域 %s   (model: paper | margin | src_margin | hyb_txt μ | hyb_src μ)' % src)
        print('=' * 82)
        res[src] = {}
        for cls in ALLT:
            sc = SRC_MAP[src].get(cls)
            if not sc or sc not in protos.get(src, {}):
                continue
            P_src, P_neg = protos[src][sc], negp[src][sc]
            others_src = np.stack([v for k, v in protos[src].items() if k != sc])
            d = cache['COCO-%s' % cls]
            pi = d['per_img'][K_CALIB:]
            gds = [np.asarray(e['gd'], float) for e in pi]
            gts = [np.asarray(e['gt'], float) if len(e['gt']) else np.zeros((0, 4)) for e in pi]
            boxes = [np.asarray(e['box'], float) for e in pi]
            mats = iou_mats(boxes, gts); n_gt = int(sum(len(g) for g in gts))
            F = np.asarray(d['F'], float)
            Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
            Fc = [Fn[np.asarray(e['crop_idx'])] if len(e['gd']) else np.zeros((0, Fn.shape[1]))
                  for e in pi]
            Delta = texts[cls] - unit(T.text_emb([SRC_TMPL.format(sc)]).cpu().numpy()[0])
            W_pa = unit(P_src + BETA * Delta) - LAM * P_neg
            others_t = np.stack([texts[c] for c in ALLT if c != cls])

            def run(fn):
                sl = []
                for i in range(len(pi)):
                    gd = gds[i]
                    if len(gd) == 0:
                        sl.append(np.zeros(0)); continue
                    sl.append(np.sqrt(gd * np.clip(fn(Fc[i]), 1e-6, 1.0)))
                return ap50(mats, sl, n_gt)

            r = {'baseline': ap50(mats, gds, n_gt)}
            r['paper'] = run(lambda X: np.maximum(0.0, X @ W_pa))
            r['margin'] = run(lambda X: np.maximum(0.0,
                                                   X @ texts[cls] - (X @ others_t.T).max(axis=1)))
            r['src_margin'] = run(lambda X: np.maximum(0.0,
                                                       X @ P_src - (X @ others_src.T).max(axis=1)))
            for mu in MUS:
                r['hyb_txt_%g' % mu] = run(
                    lambda X, mu=mu: np.maximum(0.0, X @ W_pa - mu * (X @ others_t.T).max(axis=1)))
                r['hyb_src_%g' % mu] = run(
                    lambda X, mu=mu: np.maximum(0.0, X @ W_pa - mu * (X @ others_src.T).max(axis=1)))
            res[src][cls] = r
            keys = list(r.keys())
            print('%-12s' % cls + ''.join('%14s' % k for k in keys))
            b = r['baseline']
            print('%-12s' % '  Δ' + ''.join('%14s' % ('' if k == 'baseline' else '%+.4f' % (r[k] - b))
                                           for k in keys))

    print('\n' + '=' * 82)
    print('car/truck/bus/motorcycle 四类宏平均')
    print('=' * 82)
    macro = {}
    for src in SRC_MAP:
        if not res.get(src):
            continue
        sub = [res[src][c] for c in COMMON4 if c in res[src]]
        if len(sub) < 4:
            continue
        m = {k: float(np.mean([v[k] for v in sub])) for k in sub[0]}
        b = m['baseline']
        macro[src] = m
        print('%-9s' % src + ''.join('%14s' % k for k in m))
        print('%-9s' % '  Δ' + ''.join('%14s' % ('' if k == 'baseline' else '%+.4f' % (m[k] - b))
                                      for k in m))
    print('\n★ 相对无源域 margin 的差（>0 才算源域真的出力）:')
    best = {}
    for src, m in macro.items():
        cand = {k: m[k] - m['margin'] for k in m if k not in ('baseline', 'margin')}
        bk = max(cand, key=cand.get)
        best[src] = (bk, cand[bk])
        print('  %-9s margin %+0.4f | 最好通道 %-12s %+0.4f  → 差 %+0.4f  %s'
              % (src, m['margin'] - m['baseline'], bk, m[bk] - m['baseline'], cand[bk],
                 '✓源域赢' if cand[bk] > 0 else '✗仍输'))

    json.dump({'per_class': res, 'macro_common4': macro, 'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
