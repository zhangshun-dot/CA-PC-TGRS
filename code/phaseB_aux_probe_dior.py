# -*- coding: utf-8 -*-
"""phaseB_aux_probe_dior.py — B' 全套 A/B/C/D 在真实 DIOR 硬域 (元选方案2)

COCO 探针里 C/D 为 null 是基准假象(GDINO 在 COCO 已极准/近完美校准, 无 headroom)。
DIOR 是 GDINO 的难域(域 gap 大, 检测弱, 校准 headroom 大), 在此重测 C/D 才公平。

复用 phaseB_aux_probe 的 V4Net / encode_full / roc_auc / logistic_fit / ece / tp_match。
A: z_k 在 DIOR 全图上读合成降级档 k(泛化性)。
B: DIOR-clean vs COCO-clean 域 OOD(确认在 DIOR 侧也成立)。
C: DIOR 降级集上 AUC(score) vs AUC(score+w·z_k) 风险排序。
D: DIOR 降级集上 baseline ECE vs z_k 分箱温度缩放 ECE。
环境: GroundingDINO/venv (CPU)。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import numpy as np
import torch

import phaseB_aux_probe as P
import unified_eval as U
import unified_space_v2 as US2

DEVICE = 'cpu'
SAVE = U.SAVE_DIR / 'phaseB_aux_probe_dior.json'
DIOR_CLS = ['airplane', 'ship', 'vehicle']   # DIOR 难域主类 (rerank SRC_BANK)
COCO_CLS = [c for c, _ in P.COCO_MAP]
N_PER = 30
K_CHOICES = P.K_CHOICES


def main():
    print('=' * 70)
    print('B\' 全套 A/B/C/D — 真实 DIOR 硬域 (方案2)')
    print('=' * 70)
    net = P.V4Net().to(DEVICE)
    net.load_state_dict(torch.load(P.V4_PATH, map_location=DEVICE))
    net.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    # DIOR 图像 + GT
    dior_imgs, dior_gts = [], []
    for cls in DIOR_CLS:
        im, g = U.load_dior(cls, n=N_PER)
        dior_imgs += im
        dior_gts += g
    print(f'  DIOR 图: {len(dior_imgs)} ({len(DIOR_CLS)}类×{N_PER})')

    # A: DIOR 全图读合成降级 k=1..4
    print('  [编码] DIOR 全图 k=1..4 ...', flush=True)
    rec_zk, rec_zc = [], []
    for im in dior_imgs:
        for k in K_CHOICES:
            rec_zk.append(k)
            rec_zc.append(US2.degrade_res(im, k))
    Zc_d, Zk_d = P.encode_full(net, rec_zc, bs=16)
    rec_k = np.array(rec_zk)
    r = np.corrcoef(Zk_d, rec_k)[0, 1]
    q = np.quantile(Zk_d, [0.25, 0.5, 0.75])
    pred_k = np.digitize(Zk_d, q) + 1
    acc = float((pred_k == rec_k).mean())
    zc_norm = np.linalg.norm(Zc_d, axis=1)
    r_zc = np.corrcoef(zc_norm, rec_k)[0, 1]
    print(f'  [A DIOR条件识别] R(z_k,k)={r:+.3f}  4-way acc={acc:.3f}(随机0.25)  R(|z_c|,k)={r_zc:+.3f}')

    # B: DIOR-clean vs COCO-clean 域 OOD
    coco_imgs = []
    for cls in COCO_CLS:
        im, _ = U.load_coco(cls, n=15)
        coco_imgs += im
    coco_clean = [US2.degrade_res(im, 1) for im in coco_imgs]
    Zc_c, Zk_c = P.encode_full(net, coco_clean, bs=16)
    y_coco = np.zeros(len(Zk_c))
    y_dior = np.ones(len(Zk_d))
    auc_zk = P.roc_auc(np.concatenate([y_coco, y_dior]), np.concatenate([Zk_c, Zk_d]))
    Xa = np.concatenate([Zk_c, Zk_d]).reshape(-1, 1)
    Xb = np.concatenate([Zc_c, Zc_d], 0)
    Xall = np.hstack([Xa, Xb])
    yall = np.concatenate([y_coco, y_dior])
    n = len(yall)
    tr = np.random.choice(n, n // 2, replace=False)
    te = np.array([i for i in range(n) if i not in tr])
    w, b, mu, sd = P.logistic_fit(Xall[tr], yall[tr])
    auc_zkzc = P.roc_auc(yall[te], P.logistic_pred(w, b, mu, sd, Xall[te]))
    print(f'  [B 域OOD] AUC(z_k)={auc_zk:.3f}  AUC(z_k+z_c)={auc_zkzc:.3f} (随机0.5)')

    # C+D: GDINO 在 DIOR 降级集 (k=1..4, 含 clean) 上, z_k 有变化
    print('  [GDINO] DIOR k=1..4 检测 ...', flush=True)
    all_scores, all_tp, all_zk = [], [], []
    for cls in DIOR_CLS:
        imgs, gts = U.load_dior(cls, n=N_PER)
        for k in K_CHOICES:
            deg = [US2.degrade_res(im, k) for im in imgs]
            Zc_dd, Zk_dd = P.encode_full(net, deg, bs=16)
            preds = U.gdino_detect(deg, f'a photo of {cls}', 0.0)
            for (boxes, scores), gt, zk1 in zip(preds, gts, Zk_dd):
                sc, tp = P.tp_match(np.asarray(boxes), np.asarray(scores), gt)
                all_scores += sc
                all_tp += tp
                all_zk += [float(zk1)] * len(sc)
    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)
    all_zk = np.array(all_zk)
    print(f'    DIOR 检测: {len(all_scores)} 框, TP率={all_tp.mean():.3f}', flush=True)

    base_auc = P.roc_auc(all_tp, all_scores)
    n = len(all_tp)
    tr = np.random.choice(n, n // 2, replace=False)
    te = np.array([i for i in range(n) if i not in tr])
    best_auc, best_w = base_auc, 0.0
    for ww in np.linspace(-3, 3, 25):
        comb = all_scores[tr] + ww * all_zk[tr]
        a = P.roc_auc(all_tp[tr], comb)
        if a > best_auc:
            best_auc, best_w = a, ww
    test_auc = P.roc_auc(all_tp[te], all_scores[te] + best_w * all_zk[te])
    print(f'  [C DIOR样本筛选] AUC(score)={base_auc:.3f}  AUC(score+w·z_k, w={best_w:+.2f})={test_auc:.3f}')

    base_ece = P.ece(all_scores, all_tp)
    qb = np.quantile(all_zk, [0.25, 0.5, 0.75])
    binid = np.clip(np.digitize(all_zk, qb), 0, 3)
    new_scores = all_scores.copy()
    for bb in range(4):
        mt = binid[tr] == bb
        if mt.sum() < 5:
            continue
        s = all_scores[tr][mt]
        y = all_tp[tr][mt]
        best_t, best_ce = 1.0, 1e9
        for t in np.linspace(0.3, 3.0, 28):
            ce = P.ece(1.0 / (1.0 + np.exp(-t * (s - s.mean()))), y)
            if ce < best_ce:
                best_t, best_ce = t, ce
        me = binid[te] == bb
        if me.sum() == 0:
            continue
        st = all_scores[te][me]
        new_scores[te][me] = 1.0 / (1.0 + np.exp(-best_t * (st - st.mean())))
    cal_ece = P.ece(new_scores, all_tp)
    print(f'  [D DIOR ECE校准] baseline ECE={base_ece:.3f}  z_k分箱温度缩放 ECE={cal_ece:.3f}')

    out = {
        'domain': 'DIOR (hard)', 'n_dior': len(dior_imgs), 'n_coco_b': len(coco_imgs),
        'A_condition_dior': {'pearson_r_zk_k': round(float(r), 3),
                             'acc_4way': round(acc, 3), 'random': 0.25,
                             'pearson_r_abszc_k': round(float(r_zc), 3)},
        'B_ood': {'auc_zk': round(float(auc_zk), 3), 'auc_zk_zc': round(float(auc_zkzc), 3), 'random': 0.5},
        'C_sample_selection': {'auc_score': round(float(base_auc), 3),
                               'auc_score_wzk': round(float(test_auc), 3), 'w': round(float(best_w), 2)},
        'D_ece': {'baseline': round(float(base_ece), 3), 'zk_temp_scaled': round(float(cal_ece), 3)},
        'note': 'DIOR 硬域重测: C/D 在 COCO 因无 headroom 为 null; DIOR 弱检测器应有 headroom, 此处定夺校准/风险支柱',
    }
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果 -> {SAVE}')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
