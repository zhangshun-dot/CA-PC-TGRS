# -*- coding: utf-8 -*-
"""probe_kernel_test.py — 核函数检验

假说: 打分器 W=unit(P_src+βv)-λP_neg 的 AP50 对注入方向 v 的一阶响应
      是【同一个梯度向量 g】的核内积:  ΔAP ≈ β·||v||·<ĝ, v̂>  (+ 二阶过冲项)
  若成立, 则 6 个方向 × 18 个 cell 的全部散点应坍缩成 <ΔAP_i> vs cos(v̂_i, ĝ) 的一条直线。

本脚本只做纯文本层计算 (秒级), 用已落盘的 probe_dir_decomp.json 的 oracle 增益做被解释量。
"""
import os, sys, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:/Users/张三/Desktop/实验代码')
sys.path.insert(0, r'C:/Users/张三/Desktop/A-CSPC')
import numpy as np
import run_bidirectional as RB

VEC_KEYS = ['full', 'par', 'perp', 'tge', 'vocab', 'cond']
D = json.load(open(r'C:/Users/张三/Desktop/实验数据/probe_dir_decomp.json', encoding='utf-8'))

all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
E_over  = RB.text_embs([f'an overhead image of {c}' for c in all_names])
idx = {c: i for i, c in enumerate(all_names)}
def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x
t_bar = unit(E_photo.mean(0))
Bmat = np.stack([E_photo[idx[c]] - t_bar for c in all_names], axis=1)
U, S, _ = np.linalg.svd(Bmat, full_matrices=False)
U = U[:, S > S[0] * 1e-8]; P = U @ U.T

def dirs(src, ev):
    dc = E_photo[idx[ev]] - E_over[idx[src]]; mag = np.linalg.norm(dc)
    par = P @ dc; perp = dc - par
    return {
        'full':  unit(dc),
        'par':   unit(par) if np.linalg.norm(par) > 1e-9 else np.zeros_like(dc),
        'perp':  unit(perp) if np.linalg.norm(perp) > 1e-9 else np.zeros_like(dc),
        'tge':   unit(E_photo[idx[ev]] - t_bar),
        'vocab': unit(E_over[idx[ev]] - E_over[idx[src]]),
        'cond':  unit(E_photo[idx[ev]] - E_over[idx[ev]]),
    }, mag

def pear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

report = {}
for det, cells in D['per_det'].items():
    # 逐 cell 的增益向量 (oracle 峰值 - base)
    gains = {k: [] for k in VEC_KEYS}
    DIRS = {k: [] for k in VEC_KEYS}
    for c in cells:
        b = c['base']; ds, mag = dirs(c['src'], c['eval'])
        for k in VEC_KEYS:
            gains[k].append(c[k + '_star'] - b)
            DIRS[k].append(ds[k])
    # 估计 ĝ: 用增益加权的最小二乘拟合 (限制在方向张成的子空间里)
    A = np.stack([np.stack(DIRS[k]) for k in VEC_KEYS], axis=0)   # 6 x ncell x 512
    y = np.stack([np.array(gains[k]) for k in VEC_KEYS], axis=0)  # 6 x ncell
    # 把 (dir, cell) 拉平成样本: x_i = dir_i(cell), 目标 y_i(cell); 求 g 使 <g,x> ≈ y
    X = A.reshape(-1, 512); Y = y.reshape(-1)
    # ridge, 限制在 X 的行空间 -> 用 SVD 解
    Usv, Ssv, Vt = np.linalg.svd(X, full_matrices=False)
    keep = Ssv > Ssv[0] * 1e-8
    g_hat = Vt[keep].T @ ((Usv[:, keep].T @ Y) / Ssv[keep])
    g_hat = unit(g_hat)
    # 预测 vs 观测
    pred = X @ g_hat
    # 归一化每个 cell 的 <g,v> 到与增益同尺度: 逐 cell 单参数拟合 slope
    slopes = []
    Phi = []
    for ci in range(len(cells)):
        xi = np.stack([DIRS[k][ci] for k in VEC_KEYS]) @ g_hat
        yi = np.array([gains[k][ci] for k in VEC_KEYS])
        s = float(np.dot(xi, yi) / max(np.dot(xi, xi), 1e-12)); slopes.append(s)
        Phi.append((xi, yi, s))
    pred_all = np.concatenate([p[2] * p[0] for p in Phi])
    obs_all = np.concatenate([p[1] for p in Phi])
    r2 = 1 - np.sum((obs_all - pred_all) ** 2) / max(np.sum((obs_all - obs_all.mean()) ** 2), 1e-12)
    cos_mat = np.array([[float(np.dot(DIRS[a][ci], DIRS[b][ci]))
                         for b in VEC_KEYS] for a in VEC_KEYS for ci in [0]])  # 占位
    # 平均 |cos| 矩阵
    C = np.zeros((6, 6))
    for ci in range(len(cells)):
        for i, a in enumerate(VEC_KEYS):
            for j, b in enumerate(VEC_KEYS):
                C[i, j] += float(np.dot(DIRS[a][ci], DIRS[b][ci]))
    C /= len(cells)
    # 单变量检验: 增益 vs cos(v, g_hat)
    cosg = {k: [float(np.dot(DIRS[k][ci], g_hat)) for ci in range(len(cells))] for k in VEC_KEYS}
    report[det] = dict(
        g_hat=g_hat, r2=float(r2),
        cos_with_g={k: float(np.mean(cosg[k])) for k in VEC_KEYS},
        mean_gain={k: float(np.mean(gains[k])) for k in VEC_KEYS},
        slope=float(np.mean(slopes)),
        C=C,
        per_dir_corr={k: pear(np.concatenate([ [cosg[k][ci]]*1 for ci in range(len(cells))]),
                              np.array(gains[k])) for k in VEC_KEYS},
    )
    print(f"\n=== {det} ===")
    print(f"  ĝ 拟合 R^2 = {r2:.3f}  (18*6=108 个 (方向,cell) 点, 逐 cell 独立斜率)")
    print(f"  slope(增益/⟨ĝ,v⟩) 均值 = {np.mean(slopes):.4f}")
    print("  方向       <ĝ,v̂>均值   平均增益    逐cell corr(cos,gain)")
    for k in VEC_KEYS:
        print(f"  {k:<8} {np.mean(cosg[k]):>+8.3f}   {np.mean(gains[k]):>+8.4f}   "
              f"{pear(cosg[k], np.array(gains[k])):>+8.3f}")
    # 跨 cell 的 pooled 相关: 把 cos 与 gain 全部拉平
    cx = []; cy = []
    for k in VEC_KEYS:
        cx += cosg[k]; cy += list(gains[k])
    print(f"  ★ pooled corr( ⟨ĝ,v̂> , ΔAP ) = {pear(cx, cy):+.3f}   (n={len(cx)})")

# 方向之间的平均余弦矩阵 (gdino)
print("\n\n方向间平均余弦矩阵 (gdino):")
det = list(report.keys())[0]
C = report[det]['C']
print('        ' + ''.join(f'{k:>9}' for k in VEC_KEYS))
for i, a in enumerate(VEC_KEYS):
    print(f'  {a:<6}' + ''.join(f'{C[i,j]:>+9.3f}' for j in range(6)))
np.save(r'C:/Users/张三/Desktop/实验数据/kernel_g_hat.npy',
        np.stack([report[d]['g_hat'] for d in report]))
json.dump({d: {k: v for k, v in report[d].items() if k != 'C'} for d in report},
          open(r'C:/Users/张三/Desktop/实验数据/probe_kernel_test.json', 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
print('\n→ probe_kernel_test.json')
