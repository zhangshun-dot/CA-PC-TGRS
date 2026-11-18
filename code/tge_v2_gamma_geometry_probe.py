# -*- coding: utf-8 -*-
r"""tge_v2_gamma_geometry_probe.py — 第二篇(AutoProCorrector)的小样本先验证

目的
----
双因素方差分解（见 AutoProCorrector_失败根因诊断_2026-09-18.md §3）显示：
  逐格最优 γ 的方差里，类主效应 37.3% / 源主效应 17.7% / **类×源交互 45.0%**
⇒ 任何"只依赖类"或"只依赖源"的加性特征，天花板约 46.7%。
⇒ 要预测 γ，特征**必须落在 (源域×类) 这一格**。本脚本就在这一层找免标注信号。

γ 在 USC 里的作用（experiments/usc_rank1_affine_gpu_v2.py 第 58-66 行）：
    Z_affine = Z + γ·(Z·v)·v        # 缩放源域原型在域方向 v 上的投影
    Z_corr   = Z_affine + α·v
所以候选几何量（全部不需要目标域 GT）：

  源域原型 p_src : 源域 **GT 框**裁剪的 CLIP 均值（源域标注，允许）
  目标原型 p_tgt : 目标域 **检测器框**裁剪的 CLIP 均值（零目标标注！用 tge_v2_coco_feats_6cls.pkl）
  域方向   v     : normalize(mean_cells(p_tgt - p_src))，本脚本在 CLIP 原始空间近似 USC 的 PCA 空间

  f1 cos_src_tgt    = cos(p_src, p_tgt)              —— 原型对齐度
  f2 delta_norm     = ||p_tgt - p_src||              —— 域偏移幅度
  f3 rank1_frac     = Δ 的秩1 能量占比               —— 偏移各向异性
  f4 proj_v         = p_src·v                        —— γ 直接作用的量
  f5 align_src_v    = cos(p_src, v)
  f6 align_delta_v  = cos(Δ, v)                      —— 本类偏移与域方向是否同心
  f7 gamma_surrogate= argmax_γ cos(USC(p_src;γ,α=0), p_tgt)
                      —— **免标注内目标**：校正后的原型只要最靠近目标原型，不碰任何标签

验证对象：AutoProCorrector 的逐格 oracle γ（22 格，逐格真样本，不是重复值）。
  来源 results/usc_detstats_gamma_alpha_loo_results.json → class_results[*].oracle_gamma

运行: C:\python-study\python39\python.exe tge_v2_gamma_geometry_probe.py
输出: C:\Users\张三\Desktop\实验数据\tge_v2_gamma_geometry_probe.json
"""
import sys, os, json, time
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np

from tge_v2_lambda_4sources import (
    build_dior_protos, build_nwpu_protos, build_dota_protos, build_visdrone_protos,
    SRC_MAP, FEAT_CACHE,
)

AUTOPRO = r'C:\Users\张三\Desktop\AutoProCorrector\results\usc_detstats_gamma_alpha_loo_results.json'
DATA = os.path.dirname(FEAT_CACHE)
OUT = os.path.join(DATA, 'tge_v2_gamma_geometry_probe.json')

GAMMAS = [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0, 5.0]


def unit(x):
    n = np.linalg.norm(x)
    return x / (n + 1e-12)


def spearman(x, y):
    n = len(x)
    def rk(a):
        s = sorted(range(n), key=lambda i: a[i]); r = [0.0] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and a[s[j + 1]] == a[s[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[s[k]] = avg
            i = j + 1
        return r
    rx, ry = rk(list(x)), rk(list(y))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** .5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** .5
    return num / (dx * dy + 1e-12)


def perm_p(x, y, obs, n_perm=20000, seed=0):
    """精确置换检验（n=22 时渐近近似不够可信）"""
    rng = np.random.default_rng(seed)
    yy = list(y)
    c = 0
    for _ in range(n_perm):
        rng.shuffle(yy)
        if abs(spearman(x, yy)) >= abs(obs) - 1e-12:
            c += 1
    return (c + 1) / (n_perm + 1)


def usc_correct(p_src, v, gamma, alpha=0.0):
    Z = p_src.copy()
    return unit(Z + gamma * float(Z @ v) * v + alpha * v)


def main():
    t0 = time.time()

    # ── 1. 目标原型：检测器框 CLIP 均值（零目标域标注） ──
    import pickle
    cache = pickle.load(open(FEAT_CACHE, 'rb'))
    tgt_proto = {}
    for eval_name, blob in cache.items():
        p = blob['F'].mean(axis=0)
        tgt_proto[eval_name] = unit(p.astype(np.float64))
        print('  目标原型 %-16s 框=%6d  ||p||=%s' % (eval_name, blob['n_box'], '已归一'))

    # ── 2. 源域原型（源域 GT 框 CLIP 均值，源域标注允许） ──
    print('\n构建源域原型 ...')
    src_protos = {}
    for name, fn in [('DIOR', build_dior_protos), ('NWPU', build_nwpu_protos),
                     ('DOTA', build_dota_protos), ('VisDrone', build_visdrone_protos)]:
        src_protos[name] = fn()
        print('  %-9s %d 类' % (name, len(src_protos[name])))

    # ── 3. 逐格几何量 ──
    cells = []
    for src in ['DIOR', 'NWPU', 'DOTA', 'VisDrone']:
        mp = SRC_MAP[src]
        for coco_cls, src_cls in mp.items():
            if src_cls is None or src_cls not in src_protos[src] or coco_cls not in tgt_proto:
                continue
            _p = src_protos[src][src_cls]
            if hasattr(_p, 'detach'):
                _p = _p.detach().cpu().numpy()
            p_src = unit(np.asarray(_p, dtype=np.float64))
            p_tgt = tgt_proto[coco_cls]
            cells.append(dict(src=src, coco=coco_cls, src_cls=src_cls,
                              p_src=p_src, p_tgt=p_tgt))
    print('\n有效格数: %d' % len(cells))

    # 域方向 v：每个源域内 mean(p_tgt - p_src) 归一（原始空间近似 USC 的 PCA 空间 v）
    for src in ['DIOR', 'NWPU', 'DOTA', 'VisDrone']:
        sub = [c for c in cells if c['src'] == src]
        if not sub:
            continue
        shift = np.mean([c['p_tgt'] - c['p_src'] for c in sub], axis=0)
        v = unit(shift)
        for c in sub:
            c['v'] = v

    # ── 4. 特征 ──
    for c in cells:
        p_s, p_t, v = c['p_src'], c['p_tgt'], c['v']
        d = p_t - p_s
        c['f1_cos_src_tgt'] = float(p_s @ p_t)
        c['f2_delta_norm'] = float(np.linalg.norm(d))
        # Δ 的秩1 能量占比（各向异性）：单个向量的"秩1"用其自身 max-abs 分量的能量占比
        d2 = d ** 2
        c['f3_rank1_frac'] = float(d2.max() / (d2.sum() + 1e-12))
        c['f4_proj_v'] = float(p_s @ v)
        c['f5_align_src_v'] = float(p_s @ v)          # p_s 已归一 ⇒ 与 f4 同值，保留以示口径
        c['f6_align_delta_v'] = float(d @ v / (np.linalg.norm(d) + 1e-12))
        # 免标注内目标代理
        best_g, best_s = None, -1e9
        curve = {}
        for g in GAMMAS:
            s = float(usc_correct(p_s, v, g, 0.0) @ p_t)
            curve[g] = s
            if s > best_s:
                best_s, best_g = s, g
        c['f7_gamma_surrogate'] = float(best_g)
        c['surrogate_curve'] = curve
        c['surrogate_gain'] = float(best_s - curve[0.0])

    # ── 5. oracle γ ──
    ap = json.load(open(AUTOPRO, encoding='utf-8'))
    oracle = {}
    for row in ap['class_results']:
        oracle[(row['domain'], row['class'])] = (row['oracle_gamma'], row['oracle_alpha'])
    matched, missing = [], []
    for c in cells:
        # AutoProCorrector 用的键是裸类名（'airplane'），我们的 cell 存的是 'COCO-airplane'
        bare = c['coco'].replace('COCO-', '')
        k = (c['src'], bare)
        if k in oracle:
            c['oracle_gamma'], c['oracle_alpha'] = oracle[k]
            matched.append(c)
        else:
            missing.append(k)
    print('匹配上 oracle γ 的格: %d   未匹配: %s' % (len(matched), missing))
    if len(matched) < 3:
        raise SystemExit('匹配格数不足，检查键名口径')

    # ── 6. 相关性与置换检验 ──
    y = [c['oracle_gamma'] for c in matched]
    feats = ['f1_cos_src_tgt', 'f2_delta_norm', 'f3_rank1_frac', 'f4_proj_v',
             'f6_align_delta_v', 'f7_gamma_surrogate', 'surrogate_gain']
    print('\n' + '=' * 78)
    print('与逐格 oracle γ 的关系（n=%d，逐格真样本）' % len(matched))
    print('=' * 78)
    print('%-24s %10s %10s %10s' % ('特征', 'Spearman', '置换p', '方向命中'))
    results = {}
    for f in feats:
        x = [c[f] for c in matched]
        rho = spearman(x, y)
        p = perm_p(x, y, rho)
        # 方向命中：按特征排名前一半 vs 后一半的 γ 均值
        order = sorted(range(len(x)), key=lambda i: -x[i])
        hi = np.mean([y[i] for i in order[:len(x) // 2]])
        lo = np.mean([y[i] for i in order[len(x) // 2:]])
        print('%-24s %+10.4f %10.4f   hi=%+.3f lo=%+.3f' % (f, rho, p, hi, lo))
        results[f] = dict(rho=rho, perm_p=p, hi_mean=float(hi), lo_mean=float(lo))

    # 免标注代理的"能不能直接用"检查：预测 γ 落在哪一档 vs 真实最优
    pred = [c['f7_gamma_surrogate'] for c in matched]
    exact = sum(1 for i in range(len(pred)) if pred[i] == y[i])
    within1 = sum(1 for i in range(len(pred))
                  if abs(GAMMAS.index(pred[i]) - GAMMAS.index(y[i])) <= 1)
    print('\n代理 γ（免标注内目标）精确命中 %d/%d，相邻一档内 %d/%d'
          % (exact, len(pred), within1, len(pred)))

    # ── 7. 落盘 ──
    out = dict(
        n_cells=len(matched), features=results,
        surrogate_exact=exact, surrogate_within1=within1,
        cells=[{k: v for k, v in c.items() if k not in ('p_src', 'p_tgt', 'v')} for c in matched],
        note='目标原型来自检测器框（零目标域标注）；源域原型来自源域 GT 框（源域标注允许）；'
             '域方向 v 在 CLIP 原始空间近似 USC 的 PCA 空间。',
    )
    json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\n→ %s   (耗时 %.1f 分钟)' % (OUT, (time.time() - t0) / 60))


if __name__ == '__main__':
    main()
