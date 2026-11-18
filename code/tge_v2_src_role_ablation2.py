# -*- coding: utf-8 -*-
r"""tge_v2_src_role_ablation2.py — 源域角色消融（修正版：λ 尺度 + 去偏）

上一版（tge_v2_src_role_ablation.py）的 l* 通道全塌成 baseline，原因已查明：
  |P_neg| = 1.000，但 **X·P_neg ≈ +0.74 对几乎所有框成立**（近乎常数偏置），
  而 X·d_tgt ≈ +0.225。所以 λ=0.5 一减就全场负值 → clip(1e-6) → 排序退化成 gd 排序
  → AP 结构性地等于 baseline（不是"没效果"，是"梯度被抹平"）。

paper 之所以能活，是因为它的底 unit(P_src + BETA*Delta) 的点积基准更高（~0.4+），
扣掉 0.5*0.74=0.37 后还剩正值。

本脚本用正确的 λ 网格，并加一条**零标注去偏**通道（把 X·P_neg 的常数偏置减掉，
偏置量来自检测器自己的框池统计，不用任何 GT）：
  m = mean_i(X_i · P_neg)          （该类全部框的池统计，零 GT）

通道：
  baseline                 gd
  txt_pure     W=d_tgt                                    零源域参照
  margin       X·d_tgt − max_{c!=tgt} X·d_c               零源域参照
  extrap       W=unit(P_src + BETA*Delta)                 论文去掉负原型
  paper        W=unit(P_src+BETA*Delta) − 0.5*P_neg       论文现状
  l<λ>         X·d_tgt − λ*X·P_neg                        λ ∈ {0.05..0.25}（小 λ 才不塌）
  lc<λ>        X·d_tgt − λ*(X·P_neg − m)                  ★去偏版（零标注）
  pc<λ>        X·unit(P+2D) − λ*(X·P_neg − m)             ★去偏版，论文底
  src_ref      W=P_src

输出: 实验数据/tge_v2_src_role_ablation2.json
运行: C:\python-study\python39\python.exe tge_v2_src_role_ablation2.py
"""
import sys, json, time, pickle
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import tge_probe as T
from tge_v2_main import SRC_TMPL, TGT_TMPL, DATA, K_CALIB

CACHE = DATA / 'tge_v2_coco_feats_6cls.pkl'
PROTO_PKL = DATA / 'tge_v2_src_protos.pkl'
OUT = DATA / 'tge_v2_src_role_ablation2.json'
BETA, LAM_PAPER = 2.0, 0.5
SMALL = [0.05, 0.1, 0.15, 0.25]
PAPER_LAM = [0.0, 0.25, 0.5, 0.75, 1.0]
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
    protos = pickle.load(open(PROTO_PKL, 'rb'))
    negp = {s: {c: T.get_neg_proto({k: torch.from_numpy(v).cuda() for k, v in protos[s].items()},
                                  c).cpu().numpy() for c in protos[s]} for s in protos}
    print('源域原型 [缓存] + P_neg 就绪  %.0fs' % (time.time() - t0))

    res, diag = {}, {}
    for src in SRC_MAP:
        print('\n' + '=' * 104)
        print('源域 %s' % src)
        print('=' * 104)
        res[src] = {}; diag[src] = {}
        for cls in ALLT:
            sc = SRC_MAP[src].get(cls)
            if not sc or sc not in protos.get(src, {}):
                continue
            P_src, P_neg = protos[src][sc], negp[src][sc]
            dt = texts[cls]
            W_pa = unit(P_src + BETA * (dt - unit(T.text_emb([SRC_TMPL.format(sc)]).cpu().numpy()[0])))
            W_ex = W_pa + LAM_PAPER * P_neg            # 还原 extrap（paper = extrap − 0.5 P_neg）
            d = cache['COCO-%s' % cls]
            pi = d['per_img'][K_CALIB:]
            gds = [np.asarray(e['gd'], float) for e in pi]
            gts = [np.asarray(e['gt'], float) if len(e['gt']) else np.zeros((0, 4)) for e in pi]
            mats = iou_mats([np.asarray(e['box'], float) for e in pi], gts)
            n_gt = int(sum(len(g) for g in gts))
            F = np.asarray(d['F'], float)
            Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
            Fc = [Fn[np.asarray(e['crop_idx'])] if len(e['gd']) else np.zeros((0, Fn.shape[1]))
                  for e in pi]
            others_t = np.stack([texts[c] for c in ALLT if c != cls])
            allX = np.concatenate([x for x in Fc if len(x)]) if any(len(x) for x in Fc) \
                else np.zeros((1, Fn.shape[1]))
            m = float(np.mean(allX @ P_neg))           # 零标注池统计
            diag[src][cls] = {'m_Pneg': m,
                              'mean_dt': float(np.mean(allX @ dt)),
                              'std_dt': float(np.std(allX @ dt)),
                              'std_Pneg': float(np.std(allX @ P_neg))}

            def run(fn):
                sl = []
                for i in range(len(pi)):
                    gd = gds[i]
                    if len(gd) == 0:
                        sl.append(np.zeros(0)); continue
                    sl.append(np.sqrt(gd * np.clip(fn(Fc[i]), 1e-6, 1.0)))
                return ap50(mats, sl, n_gt)

            r = {'baseline': ap50(mats, gds, n_gt)}
            r['txt_pure'] = run(lambda X: np.maximum(0.0, X @ dt))
            r['margin'] = run(lambda X: np.maximum(0.0, X @ dt - (X @ others_t.T).max(axis=1)))
            r['extrap'] = run(lambda X: np.maximum(0.0, X @ W_ex))
            r['paper'] = run(lambda X: np.maximum(0.0, X @ W_pa))
            for lam in SMALL:
                r['l%g' % lam] = run(lambda X, L=lam: np.maximum(0.0, X @ dt - L * (X @ P_neg)))
                r['lc%g' % lam] = run(lambda X, L=lam: np.maximum(0.0, X @ dt - L * (X @ P_neg - m)))
                r['pc%g' % lam] = run(lambda X, L=lam: np.maximum(0.0, X @ W_ex - L * (X @ P_neg - m)))
                r['lc%g_m' % lam] = run(lambda X, L=lam: np.maximum(
                    0.0, X @ dt - L * (X @ P_neg - m) - (X @ others_t.T).max(axis=1)))
            r['src_ref'] = run(lambda X: np.maximum(0.0, X @ P_src))
            res[src][cls] = r

            keys = list(r.keys()); b = r['baseline']
            if cls == ALLT[0]:
                print('%-11s' % 'class' + ''.join('%11s' % k for k in keys))
            print('%-11s' % cls + ''.join('%11s' % ('%.4f' % r[k]) for k in keys))
            print('%-11s' % '  Δ' + ''.join('%11s' % ('' if k == 'baseline' else '%+.4f' % (r[k] - b))
                                            for k in keys))

    def macro(sub, tag):
        rows_all = {}
        for src in SRC_MAP:
            rows = [res[src][c] for c in sub if c in res.get(src, {})]
            if len(rows) < len(sub):
                continue
            rows_all[src] = {k: float(np.mean([v[k] for v in rows])) for k in rows[0]}
        if not rows_all:
            return {}
        ks = list(next(iter(rows_all.values())).keys())
        print('\n' + '=' * 104)
        print('%s —— Δ vs baseline' % tag)
        print('=' * 104)
        print('%-10s' % 'src' + ''.join('%12s' % k for k in ks))
        for src, v in rows_all.items():
            b = v['baseline']
            print('%-10s' % src + ''.join('%12s' % ('base' if k == 'baseline' else '%+.4f' % (v[k] - b))
                                          for k in ks))
        print('%-10s' % '  ★均值' + ''.join(
            '%12s' % ('-',) if k == 'baseline'
            else '%+.4f' % np.mean([v[k] - v['baseline'] for v in rows_all.values()]) for k in ks))
        return rows_all

    M = {'common4': macro(COMMON4, 'car/truck/bus/motorcycle 四类宏平均'),
         'common6': macro(ALLT, '六类宏平均')}
    json.dump({'per_class': res, 'macro': M, 'diag': diag, 'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
