# -*- coding: utf-8 -*-
"""phaseB_aux_full.py — A+B 两支柱全量扩展 (有效再全量)

仅扩已验证的两支柱 (C/D 在 DIOR 仍 null, 收口):
  A 条件识别/监测: DIOR 放大 n + 加 haze 条件验泛化 (测条件监测能否推广到未训练退化)
  B 域 OOD: COCO-clean(放大n) vs DIOR(全量) 复现 AUC; DOTA/NWPU 多 RS 域 OOD 标记待补原始图

复用: phaseB_aux_probe 的 V4Net/encode_full/度量; unified_space_v2.degrade_res; phaseA_severe.degrade_haze
环境: GroundingDINO/venv (CPU)
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import numpy as np
import torch
import unified_eval as U
import unified_space_v2 as US2
import phaseB_aux_probe as P
from phaseA_severe import degrade_haze

DEVICE = 'cpu'
V4_PATH = U.SAVE_DIR / 'unified_space_v4.pt'
DIOR_CLS = ['airplane', 'ship', 'vehicle']
N_PER = 60
RES_K = [1, 2, 3, 4]
HAZE_T = [0.3, 0.5, 0.7]   # t=transmission; 越大越清晰, 雾霾严重度=(1-t)


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def acc_4way(zk, k_true):
    """zk 按全局四分位阈值分 4 档, 与真实 k(1..4) 比对。"""
    zk = np.asarray(zk, float); k_true = np.asarray(k_true, int)
    qs = np.quantile(zk, [0.25, 0.5, 0.75])
    pred = np.digitize(zk, qs) + 1          # 1..4
    return float((pred == k_true).mean())


def encode(pils):
    return P.encode_full(NET, pils, bs=16)


def try_load_rs_domain(name):
    """尝试加载 DOTA/NWPU 原始图供 B 多域 OOD; 无原始图则返回空 + 标记。"""
    cand = [
        rf'C:\Users\张三\Desktop\A-CSPC\explore\{name}_images',
        rf'C:\Users\张三\modeling\{name}_data',
        rf'C:\Users\张三\Desktop\{name}',
    ]
    for d in cand:
        if os.path.isdir(d):
            from glob import glob
            fs = glob(os.path.join(d, '*.jpg')) + glob(os.path.join(d, '*.png'))
            if fs:
                from PIL import Image
                return [Image.open(f).convert('RGB') for f in fs[:N_PER * 3]]
    return []


def main():
    global NET
    print('=' * 70)
    print('B\' A+B 全量扩展 — 条件监测(含haze泛化) + 跨域OOD')
    print('=' * 70)
    NET = P.V4Net().to(DEVICE)
    NET.load_state_dict(torch.load(V4_PATH, map_location=DEVICE))
    NET.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    out = {'A_condition': {}, 'B_ood': {}, 'notes': []}

    # ───────── A: 条件识别 (res + haze 泛化) ─────────
    print('\n[A] DIOR 条件识别: res k=1..4 + haze 泛化', flush=True)
    res_zk, res_sev = [], []
    res_per_cls = {}
    haze_zk, haze_sev = [], []
    for cls in DIOR_CLS:
        imgs, _ = U.load_dior(cls, n=N_PER)
        n = len(imgs)
        # res
        zk_res_cls = []
        for k in RES_K:
            degs = [US2.degrade_res(im, k) for im in imgs]
            zc, zk = encode(degs)
            res_zk += list(zk); res_sev += [(k - 1) / 3.0] * n
            zk_res_cls.append(zk)
        # 4-way acc per class (用该类 z_k vs 真实 k)
        kt = np.repeat(RES_K, n)
        res_per_cls[cls] = {
            'r_zk_k': pearson(np.concatenate(zk_res_cls), kt),
            'acc_4way': acc_4way(np.concatenate(zk_res_cls), kt),
        }
        # haze 泛化
        for t in HAZE_T:
            degs = [degrade_haze(im, t) for im in imgs]
            zc, zk = encode(degs)
            haze_zk += list(zk); haze_sev += [(1.0 - t)] * n   # 严重度=(1-t)
    out['A_condition'] = {
        'res': {
            'pooled_r_zk_k': pearson(res_zk, res_sev),
            'pooled_acc_4way': acc_4way(res_zk, (np.array(res_sev) * 3 + 1).astype(int)),
            'random_acc_4way': 0.25,
            'per_class': res_per_cls,
        },
        'haze_generalization': {
            'pooled_r_zk_hazeseverity': pearson(haze_zk, haze_sev),
            'note': 'v4 CondEncoder 仅 res 训练; haze 为未训练条件, 验泛化',
        },
    }
    print(f"  res: R(z_k,k)={out['A_condition']['res']['pooled_r_zk_k']:.3f} "
          f"acc4={out['A_condition']['res']['pooled_acc_4way']:.3f} (rand 0.25)")
    print(f"  haze: R(z_k,1-t)={out['A_condition']['haze_generalization']['pooled_r_zk_hazeseverity']:.3f}")

    # ───────── B: 跨域 OOD (放大 n 复现 + 多 RS 域尝试) ─────────
    print('\n[B] 跨域 OOD: COCO-clean vs DIOR (放大n)', flush=True)
    coco_imgs, dior_imgs = [], []
    for coco_cls, _ in P.COCO_MAP:
        im, _ = U.load_coco(coco_cls, n=N_PER)
        coco_imgs += im
    for cls in DIOR_CLS:
        im, _ = U.load_dior(cls, n=N_PER)
        dior_imgs += im
    zc_c, zk_c = encode(coco_imgs)
    zc_d, zk_d = encode(dior_imgs)
    X = np.vstack([np.hstack([zk_c[:, None], zc_c]), np.hstack([zk_d[:, None], zc_d])])
    y = np.array([0] * len(zk_c) + [1] * len(zk_d))
    w, b, mu, sd = P.logistic_fit(X, y)
    auc_all = P.roc_auc(y, P.logistic_pred(w, b, mu, sd, X))
    auc_zk = P.roc_auc(y, np.concatenate([zk_c, zk_d]))
    auc_zc = P.roc_auc(y, np.concatenate([np.linalg.norm(zc_c, axis=1), np.linalg.norm(zc_d, axis=1)]))
    out['B_ood']['coco_vs_dior'] = {
        'n_coco': len(zk_c), 'n_dior': len(zk_d),
        'auc_zk_zc_logistic': auc_all, 'auc_zk': auc_zk, 'auc_zc': auc_zc,
        'random': 0.5,
    }
    print(f"  COCO({len(zk_c)}) vs DIOR({len(zk_d)}): "
          f"AUC(zk+zc)={auc_all:.3f} AUC(zk)={auc_zk:.3f} AUC(zc)={auc_zc:.3f}")

    # 多 RS 域尝试
    for dom in ['dota', 'nwpu']:
        rimgs = try_load_rs_domain(dom)
        if rimgs:
            rzc, rzk = encode(rimgs)
            X2 = np.vstack([np.hstack([zk_d[:, None], zc_d]), np.hstack([rzk[:, None], rzc])])
            y2 = np.array([0] * len(zk_d) + [1] * len(rzk))
            w2, b2, mu2, sd2 = P.logistic_fit(X2, y2)
            out['B_ood'][f'dior_vs_{dom}'] = {
                'n_dior': len(zk_d), 'n_' + dom: len(rzk),
                'auc_zk_zc_logistic': P.roc_auc(y2, P.logistic_pred(w2, b2, mu2, sd2, X2)),
            }
            print(f"  DIOR vs {dom}({len(rzk)}): AUC={out['B_ood'][f'dior_vs_{dom}']['auc_zk_zc_logistic']:.3f}")
        else:
            out['B_ood'][f'{dom}_status'] = 'pending_external_raw_images'
            out['notes'].append(f'{dom} 原始图缺失, 多RS域OOD待补数据')

    with open(U.SAVE_DIR / 'phaseB_aux_full.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n[完成] -> 实验数据/phaseB_aux_full.json')


if __name__ == '__main__':
    main()
