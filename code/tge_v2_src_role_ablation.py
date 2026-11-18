# -*- coding: utf-8 -*-
r"""tge_v2_src_role_ablation.py — 源域在算子里到底扮演什么角色（拆到位）

上游读数（tge_v2_source_probe.json, 6类宏平均, DIOR）：
    baseline  0.0000
    src       -0.0031   W = P_src                     纯源域图像原型
    txt       +0.0357   W = d_tgt                     纯目标文本（零源域）
    extrap    +0.0204   W = unit(P_src + BETA*Delta)  源原型 + 文本外推
    paper     +0.0646   W = unit(P_src + BETA*Delta) - LAM*P_neg
    margin    +0.0641   vis = cos(f,d_tgt) - max_{c!=tgt} cos(f,d_c)   零源域

★ 读数矛盾点：txt(+0.0357) > extrap(+0.0204) —— 把源域原型 P_src 加进文本外推，
  反而掉 0.0153。而 paper - extrap = +0.0442 全部来自 P_neg。
  ⇒ 假设：源域的**类原型 P_src 是负债**，源域的**类间混淆结构 P_neg 才是资产**。

本脚本做决定性的消融：把 P_src 从算子里拿掉，只留源域的类间结构，
看能否既保住"源域参与"（P_neg 由源域原型池构造），又超过 paper 和零源域 margin。

通道（λ 为负原型强度，μ 为目标文本竞争强度）：
  baseline        gd
  txt_pure        W = d_tgt
  delta           W = unit(Delta)                       文本外推方向本身（零源域）
  l<λ>            W = unit(Delta) - λ*P_neg             ★ 去掉 P_src，保留源域结构
  l<λ>n           W = unit(unit(Delta) - λ*P_neg)       （归一化版）
  ln<λ>_t<μ>      W = unit(Delta) - λ*P_neg - μ*max_{c!=tgt}(X·d_c)   再叠目标文本竞争
  src_ref         W = P_src（纯源域原型，参照）
  paper           W = unit(P_src + 2*Delta) - 0.5*P_neg（论文现状）
  margin          vis = cos(f,d_tgt) - max_{c!=tgt} cos(f,d_c)（零源域上界参照）

输出: 实验数据/tge_v2_src_role_ablation.json
运行: C:\python-study\python39\python.exe tge_v2_src_role_ablation.py
"""
import sys, json, time, pickle
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U
import tge_probe as T
from tge_v2_main import SRC_TMPL, TGT_TMPL, DATA, K_CALIB

CACHE = DATA / 'tge_v2_coco_feats_6cls.pkl'
PROTO_PKL = DATA / 'tge_v2_src_protos.pkl'
OUT = DATA / 'tge_v2_src_role_ablation.json'
BETA, LAM = 2.0, 0.5
LAMS = [0.25, 0.5, 1.0, 2.0]
MUS = [0.25, 0.5]
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


def main():
    t0 = time.time()
    cache = pickle.load(open(CACHE, 'rb'))
    texts = {c: unit(T.text_emb([TGT_TMPL.format(c)]).cpu().numpy()[0]) for c in ALLT}
    print('源域原型池 [缓存]...')
    protos = pickle.load(open(PROTO_PKL, 'rb'))
    negp = {s: {c: T.get_neg_proto({k: torch.from_numpy(v).cuda() for k, v in protos[s].items()},
                                  c).cpu().numpy() for c in protos[s]} for s in protos}
    src_txt = {}
    print('源域模板文本...')
    for s in protos:
        for c in protos[s]:
            src_txt[(s, c)] = unit(T.text_emb([SRC_TMPL.format(c)]).cpu().numpy()[0])

    res = {}
    for src in SRC_MAP:
        print('\n' + '=' * 100)
        print('源域 %s' % src)
        print('=' * 100)
        res[src] = {}
        for cls in ALLT:
            sc = SRC_MAP[src].get(cls)
            if not sc or sc not in protos.get(src, {}):
                continue
            P_src, P_neg = protos[src][sc], negp[src][sc]
            dsrc = src_txt[(src, sc)]
            Delta = texts[cls] - dsrc
            W_pa = unit(P_src + BETA * Delta) - LAM * P_neg
            W_d = unit(Delta)
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
            r['txt_pure'] = run(lambda X: np.maximum(0.0, X @ texts[cls]))
            r['delta'] = run(lambda X: np.maximum(0.0, X @ W_d))
            for lam in LAMS:
                r['l%g' % lam] = run(lambda X, L=lam: np.maximum(0.0, X @ (W_d - L * P_neg)))
                Wn = unit(W_d - lam * P_neg)
                r['l%gn' % lam] = run(lambda X, W=Wn: np.maximum(0.0, X @ W))
            for lam in LAMS:
                for mu in MUS:
                    r['ln%g_t%g' % (lam, mu)] = run(
                        lambda X, L=lam, M=mu: np.maximum(
                            0.0, X @ (W_d - L * P_neg) - M * (X @ others_t.T).max(axis=1)))
            r['src_ref'] = run(lambda X: np.maximum(0.0, X @ P_src))
            r['paper'] = run(lambda X: np.maximum(0.0, X @ W_pa))
            r['margin'] = run(lambda X: np.maximum(0.0,
                                                   X @ texts[cls] - (X @ others_t.T).max(axis=1)))
            res[src][cls] = r

            keys = list(r.keys()); b = r['baseline']
            if cls == ALLT[0]:
                print('%-12s' % 'class' + ''.join('%12s' % k for k in keys))
            print('%-12s' % cls + ''.join('%12s' % ('%.4f' % r[k]) for k in keys))
            print('%-12s' % '  Δ' + ''.join('%12s' % ('' if k == 'baseline' else '%+.4f' % (r[k] - b))
                                           for k in keys))

    def macro(sub):
        out = {}
        for src in SRC_MAP:
            if not res.get(src):
                continue
            rows = [res[src][c] for c in sub if c in res.get(src, {})]
            if len(rows) < len(sub):
                continue
            out[src] = {k: float(np.mean([v[k] for v in rows])) for k in rows[0]}
        return out

    allk = None
    M = {}
    for tag, sub in (('common4', COMMON4), ('common6', ALLT)):
        m = macro(sub)
        M[tag] = m
        if not m:
            print('\n[%s] 无源域同时覆盖，跳过' % tag)
            continue
        allk = list(next(iter(m.values())).keys())
        print('\n' + '=' * 100)
        print('%s 宏平均（Δ vs baseline）' % ('car/truck/bus/motorcycle 四类' if tag == 'common4'
                                        else '六类'))
        print('=' * 100)
        print('%-10s' % 'src' + ''.join('%13s' % k for k in allk))
        for src, v in m.items():
            b = v['baseline']
            print('%-10s' % src + ''.join('%13s' % ('base' if k == 'baseline'
                                                    else '%+.4f' % (v[k] - b)) for k in allk))
        print('%-10s' % '  ★均值' + ''.join(
            '%13s' % ('-', ) if k == 'baseline'
            else '%+.4f' % np.mean([v[k] - v['baseline'] for v in m.values()]) for k in allk))

    json.dump({'per_class': res, 'macro': M, 'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
