# -*- coding: utf-8 -*-
r"""tge_v2_doubao_ap_probe.py — 换打分方向的 AP 对拍（前检 P2 的 AP 版）

前检 P2 说：源原型方向逐框 AUC 只有 0.3976（比抛硬币差），目标文本方向 0.8520。
这里直接看 AP：把 vis 通道的 W 换掉，其余（pre_filter、sqrt(gd*vis)、数据集级 11-point AP50）
与论文/4sources 协议逐行一致。

对比的 W：
  baseline      : 只有 gd
  src           : W = P_src
  paper(DIOR)   : W = normalize(P_src + 2*Δ) − 0.5*P_neg        （论文主配置 β=2,λ=0.5）
  txt           : W = d_tgt                                      （零目标域数据的文本方向）
  txt_norm      : W = d_tgt 但 vis 用 cos 归一（同一个 W，只是看尺度影响）
  margin        : vis = max(0, cos(f,d_tgt) − max_{c≠tgt} cos(f,d_c))  （豆包 B 的形式）

输出: 实验数据/tge_v2_doubao_ap_probe.json
运行: C:\python-study\python39\python.exe tge_v2_doubao_ap_probe.py
"""
import sys, json, time
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import unified_eval as U
import tge_probe as T
from tge_v2_main import clip_proto, gt_crops, SRC_TMPL, TGT_TMPL, DATA, K_CALIB

CACHE = DATA / 'tge_v2_coco_feats_6cls.pkl'
OUT = DATA / 'tge_v2_doubao_ap_probe.json'
TARGETS = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
DIOR_SRC = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle',
            'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle'}
BETA, LAM = 2.0, 0.5


def unit(x):
    x = np.asarray(x, dtype=np.float64); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def build_iou_mats(bl, gl):
    out = []
    for b, g in zip(bl, gl):
        b = np.asarray(b, float); g = np.asarray(g, float)
        if len(b) == 0 or len(g) == 0:
            out.append(np.zeros((len(b), len(g)))); continue
        x1 = np.maximum(b[:, None, 0], g[None, :, 0]); y1 = np.maximum(b[:, None, 1], g[None, :, 1])
        x2 = np.minimum(b[:, None, 2], g[None, :, 2]); y2 = np.minimum(b[:, None, 3], g[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        a1 = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        a2 = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        out.append(inter / (a1[:, None] + a2[None, :] - inter + 1e-8))
    return out


def ap50_fast(mats, scores_list, n_gt):
    if n_gt == 0:
        return 0.0 if any(len(s) for s in scores_list) else 1.0
    evals = []
    for M, s in zip(mats, scores_list):
        if len(s) == 0:
            continue
        s = np.asarray(s); matched = [False] * M.shape[1]
        for i in np.argsort(-s):
            bi, bj = 0.0, -1
            for j in range(M.shape[1]):
                if not matched[j] and M[i, j] > bi:
                    bi, bj = M[i, j], j
            if bi >= 0.5 and bj >= 0:
                evals.append((float(s[i]), 1)); matched[bj] = True
            else:
                evals.append((float(s[i]), 0))
    if not evals:
        return 0.0
    evals.sort(key=lambda x: -x[0])
    tp = np.array([e[1] for e in evals])
    tc, fc = np.cumsum(tp), np.cumsum(1 - tp)
    rec = tc / n_gt
    prec = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = rec >= t
        if m.any():
            ap += float(prec[m].max())
    return ap / 11.0


def main():
    t0 = time.time()
    import pickle
    cache = pickle.load(open(CACHE, 'rb'))

    protos = {}
    for cls in U.DIOR_ID2NAME.values():
        im, gt = U.load_dior(cls)
        p = clip_proto(gt_crops(im, gt))
        if p is not None:
            protos[cls] = unit(p.cpu().numpy())
    print('DIOR 原型池: %d 类 %s' % (len(protos), sorted(protos.keys())))
    import torch
    P_neg = T.get_neg_proto({k: torch.from_numpy(v).cuda() for k, v in protos.items()},
                            'vehicle').cpu().numpy()
    dt = {c: unit(T.text_emb([TGT_TMPL.format(c)]).cpu().numpy()[0]) for c in TARGETS}
    ds = {k: unit(T.text_emb([SRC_TMPL.format(k)]).cpu().numpy()[0]) for k in protos}

    rows = {}
    print('%-12s %8s %8s %8s %8s %8s %8s' %
          ('class', 'baseline', 'src', 'paper', 'txt', 'txt_n', 'margin'))
    for cls in TARGETS:
        key = 'COCO-%s' % cls
        d = cache[key]
        pi = d['per_img'][K_CALIB:]
        boxes = [np.asarray(e['box'], float) for e in pi]
        gds = [np.asarray(e['gd'], float) for e in pi]
        gts = [np.asarray(e['gt'], float) if len(e['gt']) else np.zeros((0, 4)) for e in pi]
        mats = build_iou_mats(boxes, gts)
        n_gt = int(sum(len(g) for g in gts))
        F = np.asarray(d['F'], dtype=np.float64)
        Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
        Fc = [Fn[np.asarray(e['crop_idx'])] if len(e['gd']) else np.zeros((0, Fn.shape[1]))
              for e in pi]

        p_src = protos[DIOR_SRC[cls]] if cls in ('airplane', 'boat') else protos['vehicle']
        Delta = dt[cls] - ds[DIOR_SRC[cls]]
        W_paper = unit(p_src + BETA * Delta) - LAM * P_neg
        others = np.stack([dt[c] for c in TARGETS if c != cls])

        def fused(gd, vis):
            return np.sqrt(gd * np.clip(vis, 1e-6, 1.0))

        def ap_with_vis(vis_fn):
            sl = []
            for i, e in enumerate(pi):
                gd = gds[i]
                if len(gd) == 0:
                    sl.append(np.zeros(0)); continue
                sl.append(fused(gd, vis_fn(Fc[i], i)))
            return ap50_fast(mats, sl, n_gt)

        r = {}
        r['baseline'] = ap50_fast(mats, gds, n_gt)
        r['src'] = ap_with_vis(lambda X, i: np.maximum(0.0, X @ p_src))
        r['paper'] = ap_with_vis(lambda X, i: np.maximum(0.0, X @ W_paper))
        r['txt'] = ap_with_vis(lambda X, i: np.maximum(0.0, X @ dt[cls]))
        r['txt_n'] = ap_with_vis(lambda X, i: np.maximum(0.0, X @ dt[cls]))
        r['margin'] = ap_with_vis(
            lambda X, i: np.maximum(0.0, X @ dt[cls] - (X @ others.T).max(axis=1)))
        rows[cls] = r
        print('%-12s %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' %
              (cls, r['baseline'], r['src'], r['paper'], r['txt'], r['txt_n'], r['margin']))
        print('             Δ vs baseline:  src %+0.4f  paper %+0.4f  txt %+0.4f  margin %+0.4f'
              % (r['src'] - r['baseline'], r['paper'] - r['baseline'],
                 r['txt'] - r['baseline'], r['margin'] - r['baseline']))

    print('\n=== 6 类宏平均 ===')
    macro = {}
    for k in ('baseline', 'src', 'paper', 'txt', 'txt_n', 'margin'):
        macro[k] = float(np.mean([rows[c][k] for c in TARGETS]))
    for k in ('baseline', 'src', 'paper', 'txt', 'txt_n', 'margin'):
        tag = '' if k == 'baseline' else '  (Δ %+0.4f)' % (macro[k] - macro['baseline'])
        print('  %-10s %.4f%s' % (k, macro[k], tag))
    print('\n  6/6 类为正的通道: %s' % ', '.join(
        k for k in ('src', 'paper', 'txt', 'txt_n', 'margin')
        if all(rows[c][k] > rows[c]['baseline'] for c in TARGETS)))
    print('  逐类为正计数: %s' % ', '.join(
        '%s=%d/6' % (k, sum(rows[c][k] > rows[c]['baseline'] for c in TARGETS))
        for k in ('src', 'paper', 'txt', 'margin')))

    json.dump({'per_class': rows, 'macro': macro, 'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('\n→ %s  (%.0fs)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
