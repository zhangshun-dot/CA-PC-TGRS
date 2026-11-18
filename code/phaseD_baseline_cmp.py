# -*- coding: utf-8 -*-
"""phaseD_baseline_cmp.py — z_k vs 手工图像统计 (决定"有没有创新点")

核心问题: 统一空间的条件分支 z_k 是否比"零学习、一行代码"的现成统计更强?
  手工基线: 拉普拉斯方差(模糊) / 暗通道均值(雾霾) / 对比度 / 亮度
关键判定(两点):
  1) 单估计器跨退化: z_k 在 res(训练过) 与 haze(未训练) 上都要与严重度相关;
     手工统计通常只在其对应退化上有效(拉普拉斯→模糊, 暗通道→雾霾)。
     若只有 z_k 在两者上都成立 → 真创新点(通用退化监测器)。
  2) 同域条件 OOD: 干净 DIOR vs 降级 DIOR(k=4 / haze重), AUC(z_k) vs AUC(手工基线)。
     这才是方法本该做的 OOD(不是 COCO-vs-DIOR 那种过易的域身份任务)。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import numpy as np
import torch
from PIL import Image, ImageFilter
import unified_eval as U
import unified_space_v2 as US2
import phaseB_aux_probe as P
from phaseA_severe import degrade_haze

DEVICE = 'cpu'
V4_PATH = U.SAVE_DIR / 'unified_space_v4.pt'
DIOR_CLS = ['airplane', 'ship', 'vehicle']
N_PER = 30
RES_K = [1, 2, 3, 4]
HAZE_T = [0.3, 0.5, 0.7]     # t 越小雾霾越重; 严重度 = 1-t


# ───────── 手工统计基线 ─────────
def lap_var(pil):
    """拉普拉斯方差(模糊度量): 清晰图高, 模糊图低。"""
    g = np.asarray(pil.convert('L'), dtype=np.float64)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def dark_mean(pil):
    """暗通道均值(雾霾度量, He et al.): 无雾近0, 有雾变亮。"""
    a = np.asarray(pil.convert('RGB'), dtype=np.uint8)
    dc = a.min(axis=2)
    dcp = Image.fromarray(dc).filter(ImageFilter.MinFilter(3))
    return float(np.asarray(dcp, dtype=np.float64).mean())


def contrast(pil):
    return float(np.asarray(pil.convert('L'), dtype=np.float64).std())


def brightness(pil):
    return float(np.asarray(pil.convert('L'), dtype=np.float64).mean())


def stats_all(pils):
    return {
        'lap_var': [lap_var(p) for p in pils],
        'dark_mean': [dark_mean(p) for p in pils],
        'contrast': [contrast(p) for p in pils],
        'brightness': [brightness(p) for p in pils],
    }


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main():
    net = P.V4Net().to(DEVICE)
    net.load_state_dict(torch.load(V4_PATH, map_location=DEVICE))
    net.eval()

    print('=' * 70)
    print('D 对照 — z_k vs 手工统计 (跨退化 + 同域条件OOD)')
    print('=' * 70)

    dior = {}
    for dc in DIOR_CLS:
        imgs, _ = U.load_dior(dc, n=N_PER)
        dior[dc] = imgs
    all_imgs = [p for dc in DIOR_CLS for p in dior[dc]]
    print(f'  DIOR 图: {len(all_imgs)}')

    out = {'cond_corr': {}, 'ood': {}}

    # ── 1) 各条件下: z_k 与手工统计 vs 严重度的相关 ──
    for cond_name, sev_list, deg_fn in [
        ('res', [(k - 1) / 3.0 for k in RES_K], lambda p, v: US2.degrade_res(p, v)),
        ('haze', [1.0 - t for t in HAZE_T], lambda p, v: degrade_haze(p, v)),
    ]:
        vals = RES_K if cond_name == 'res' else HAZE_T
        zk_all, sev_all = [], []
        est = {'lap_var': [], 'dark_mean': [], 'contrast': [], 'brightness': []}
        for v, sev in zip(vals, sev_list):
            degs = [deg_fn(p, v) for p in all_imgs]
            zc, zk = P.encode_full(net, degs, bs=16)
            s = stats_all(degs)
            zk_all += list(zk); sev_all += [sev] * len(all_imgs)
            for k in est:
                est[k] += s[k]
        r = {'z_k': pearson(zk_all, sev_all)}
        for k in est:
            r[k] = pearson(est[k], sev_all)
        out['cond_corr'][cond_name] = r
        print(f'\n[{cond_name}] 与严重度的 Pearson r:')
        for k, v in r.items():
            print(f'   {k:12s} r={v:+.3f}')

    # ── 2) 同域条件 OOD: clean vs degraded (DIOR 内部) ──
    for tag, deg_fn in [('res_k4', lambda p: US2.degrade_res(p, 4)),
                        ('haze_t0.3', lambda p: degrade_haze(p, 0.3))]:
        clean = all_imgs
        deg = [deg_fn(p) for p in all_imgs]
        zc0, zk0 = P.encode_full(net, clean, bs=16)
        zc1, zk1 = P.encode_full(net, deg, bs=16)
        s0, s1 = stats_all(clean), stats_all(deg)
        y = np.array([0] * len(zk0) + [1] * len(zk1))
        aucs = {'z_k': P.roc_auc(y, np.concatenate([zk0, zk1]))}
        # 手工统计定向: 越高=越退化
        for k, sign in [('lap_var', -1), ('dark_mean', +1), ('contrast', -1), ('brightness', +1)]:
            sc = np.concatenate([np.array(s0[k]), np.array(s1[k])]) * sign
            aucs[k] = P.roc_auc(y, sc)
        out['ood'][tag] = aucs
        print(f'\n[同域条件OOD clean vs {tag}] AUC (随机0.5):')
        for k, v in aucs.items():
            print(f'   {k:12s} AUC={v:.3f}')

    with open(U.SAVE_DIR / 'phaseD_baseline_cmp.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n[完成] -> 实验数据/phaseD_baseline_cmp.json')


if __name__ == '__main__':
    main()
