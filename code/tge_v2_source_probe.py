# -*- coding: utf-8 -*-
r"""tge_v2_source_probe.py — 换源域，量"源域到底贡献了多少"

动机：DIOR 源域下 car/truck/bus/motorcycle 四个目标类**共用同一个源原型 vehicle**，
所以源原型结构上不可能提供子类判别 —— 这解释了 src 通道 −0.0031。
VisDrone 源域自带 car/truck/bus/motor 四个细粒度类，是唯一能让源原型具备子类判别力的源域。

四个源域 × 逐类 AP，通道分解（全部沿用论文协议：sqrt(clip(gd)*clip(vis))，11-point AP50）：
  baseline   只有 gd
  src        W = P_src                                        ← 纯源图像证据
  extrap     W = unit(P_src + BETA*Delta)                     ← 源原型 + 文本外推（无负原型）
  paper      W = unit(P_src + BETA*Delta) - LAM*P_neg         ← 论文（+源域负原型结构）
  neg_only   vis = max(0, -f·P_neg)                           ← 纯源域类间结构
  txt        W = d_tgt                                        ← 源域无关参照
  margin     vis = max(0, cos(f,d_tgt) - max_{c≠tgt} cos(f,d_c))  ← 源域无关参照

关键读数：paper − extrap = **负原型（纯源域产物）的净贡献**；
          paper vs margin  = 源域整体 vs 无源域。

输出: 实验数据/tge_v2_source_probe.json
运行: C:\python-study\python39\python.exe tge_v2_source_probe.py
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
OUT = DATA / 'tge_v2_source_probe.json'
BETA, LAM = 2.0, 0.5
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
ALLT = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
CH = ['baseline', 'src', 'extrap', 'paper', 'neg_only', 'txt', 'margin']


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
    pools = {}
    protos = {}
    protos['DIOR'] = {}
    for cls in U.DIOR_ID2NAME.values():
        try:
            im, gt = U.load_dior(cls)
            p = clip_proto(gt_crops(im, gt))
            if p is not None:
                protos['DIOR'][cls] = unit(p.cpu().numpy())
        except Exception as e:
            print('   DIOR/%s %s' % (cls, e))
    protos['NWPU'] = {}
    for cls in NW.NWPU_CLASSES:
        try:
            im, gt = NW.load_nwpu(cls)
            p = clip_proto(gt_crops(im, gt))
            if p is not None:
                protos['NWPU'][cls] = unit(p.cpu().numpy())
        except Exception as e:
            print('   NWPU/%s %s' % (cls, e))
    protos['DOTA'] = {}
    for cls in DS.DOTA_CLASSES:
        try:
            im, gt = DS.load_dota(cls)
            p = clip_proto(gt_crops(im, gt))
            if p is not None:
                protos['DOTA'][cls] = unit(p.cpu().numpy())
        except Exception as e:
            print('   DOTA/%s %s' % (cls, e))
    try:
        iv, gv = [], []
        for vc in ('large-vehicle', 'small-vehicle'):
            im2, gt2 = DS.load_dota(vc); iv += im2; gv += gt2
        protos['DOTA']['vehicle'] = unit(clip_proto(gt_crops(iv, gv)).cpu().numpy())
    except Exception as e:
        print('   DOTA/vehicle %s' % e)
    protos['VisDrone'] = {}
    try:
        from tge_v2_visdrone import load_visdrone, VISDRONE_CLASSES
        for cls in VISDRONE_CLASSES:
            im, gt = load_visdrone(cls)
            if len(im) == 0:
                continue
            p = clip_proto(gt_crops(im, gt))
            if p is not None:
                protos['VisDrone'][cls] = unit(p.cpu().numpy())
    except Exception as e:
        print('   VisDrone %s' % e)
    for s, d in protos.items():
        print('  %-9s %2d 类 %s' % (s, len(d), sorted(d.keys())))
    return protos


def main():
    t0 = time.time()
    cache = pickle.load(open(CACHE, 'rb'))
    texts = {c: unit(T.text_emb([TGT_TMPL.format(c)]).cpu().numpy()[0]) for c in ALLT}
    print('构造四个源域原型池...')
    protos = build_pools()
    print('原型池构造完成 %.0fs' % (time.time() - t0))

    negp = {}
    for s in protos:
        try:
            negp[s] = {}
            for c in protos[s]:
                negp[s][c] = T.get_neg_proto(
                    {k: torch.from_numpy(v).cuda() for k, v in protos[s].items()}, c).cpu().numpy()
        except Exception as e:
            print('  P_neg %s 失败 %s' % (s, e))

    res = {}
    for src in SRC_MAP:
        print('\n' + '=' * 78)
        print('源域 %s' % src)
        print('=' * 78)
        print('%-12s %8s %8s %8s %8s %8s %8s %8s' % (('class',) + tuple(CH)))
        res[src] = {}
        for cls in ALLT:
            sc = SRC_MAP[src].get(cls)
            if not sc or sc not in protos.get(src, {}):
                continue
            P_neg = negp[src][sc]
            P_src = protos[src][sc]
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
            W_ex = unit(P_src + BETA * Delta)
            W_pa = W_ex - LAM * P_neg
            others = np.stack([texts[c] for c in ALLT if c != cls])

            def run(fn):
                sl = []
                for i in range(len(pi)):
                    gd = gds[i]
                    if len(gd) == 0:
                        sl.append(np.zeros(0)); continue
                    sl.append(np.sqrt(gd * np.clip(fn(Fc[i], i), 1e-6, 1.0)))
                return ap50(mats, sl, n_gt)

            r = {}
            r['baseline'] = ap50(mats, gds, n_gt)
            r['src'] = run(lambda X, i: np.maximum(0.0, X @ P_src))
            r['extrap'] = run(lambda X, i: np.maximum(0.0, X @ W_ex))
            r['paper'] = run(lambda X, i: np.maximum(0.0, X @ W_pa))
            r['neg_only'] = run(lambda X, i: np.maximum(0.0, -(X @ P_neg)))
            r['txt'] = run(lambda X, i: np.maximum(0.0, X @ texts[cls]))
            r['margin'] = run(lambda X, i: np.maximum(0.0,
                                                      X @ texts[cls] - (X @ others.T).max(axis=1)))
            res[src][cls] = r
            print('%-12s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' %
                  ((cls,) + tuple(r[k] for k in CH)))

    print('\n' + '=' * 78)
    print('car/truck/bus/motorcycle 四类宏平均（唯一四源都覆盖的子集）')
    print('=' * 78)
    print('%-11s %8s %8s %8s %8s %8s %8s %8s' % (('src',) + tuple(CH)))
    macro = {}
    for src in SRC_MAP:
        if not res.get(src):
            continue
        sub = [res[src][c] for c in COMMON4 if c in res[src]]
        if len(sub) < 4:
            continue
        macro[src] = {k: float(np.mean([v[k] for v in sub])) for k in CH}
        b = macro[src]['baseline']
        print('%-11s' % src + ''.join('%8.4f' % macro[src][k] for k in CH))
        print('%-11s' % '  Δ vs base' + ''.join('%+8.4f' % (macro[src][k] - b) if k != 'baseline'
                                                else '%8s' % '' for k in CH))
    print('\n  ★ 源域净贡献 = paper − extrap（差值全部来自 P_neg，即纯源域产物）:')
    for src, m in macro.items():
        print('     %-9s paper %+0.4f  extrap %+0.4f  →  P_neg 净贡献 %+0.4f'
              % (src, m['paper'] - m['baseline'], m['extrap'] - m['baseline'],
                 m['paper'] - m['extrap']))
    print('\n  ★ 源域整体 vs 无源域（margin）:')
    for src, m in macro.items():
        d_ = m['paper'] - m['margin']
        print('     %-9s paper %+0.4f   margin %+0.4f   源域是否更强: %s (%+0.4f)'
              % (src, m['paper'] - m['baseline'], m['margin'] - m['baseline'],
                 '是' if d_ > 0 else '否', d_))

    json.dump({'per_class': res, 'macro_common4': macro, 'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
