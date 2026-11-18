# -*- coding: utf-8 -*-
"""phaseB_aux_probe.py — B' 辅助层小样本先验 (元批准: 有效再全量)

目的: 验证"信息统一空间"表示在 intended 分布(全图)上是否携带可用于辅助任务的信号,
      而非"检测 AP 增益"(A/B 路径已证伪)。

四个辅助任务(对齐元收口重定位: 条件识别/域外检测/置信度校准/样本筛选):
  A. 条件识别: 全图 z_k 能否读出分辨率降级档 k (k=1..4)。acc / PearsonR vs 随机0.25。
  B. 域 OOD: COCO-clean 全图 vs DIOR 全图, z_k/z_c 能否分离 (AUC vs 0.5)。
  C. 样本筛选/风险: 降级图上 GDINO 检测, 用 (score + w*z_k) 预测 TP 的 AUC
     是否高于只用 score (AUC)。z_k 在降级集上有变化才能测。
  D. 置信度校准(ECE): 降级集上 baseline ECE vs 按 z_k 分箱的温度缩放 ECE。

复用: v4 V4Net/encode (与 acspc_unified_rerank 同构), 权重 unified_space_v4.pt。
环境: GroundingDINO/venv (CPU, 含 groundingdino+dino2)。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms

import unified_eval as U
import unified_space_v2 as US2

DEVICE = 'cpu'
SAVE = U.SAVE_DIR / 'phaseB_aux_probe.json'
V4_PATH = U.SAVE_DIR / 'unified_space_v4.pt'
K_CHOICES = [1, 2, 3, 4]
COCO_MAP = [('airplane', 'airplane'), ('boat', 'ship'), ('car', 'vehicle'),
            ('truck', 'vehicle'), ('bus', 'vehicle'), ('motorcycle', 'vehicle')]
N_PER = 30


# ───────── v4 网络 (与 rerank 同构) ─────────
class ContentEncoder(nn.Module):
    def __init__(self, content_ch=64, freeze=True):
        super().__init__()
        self.dino, self.pre = U.get_dinov2()
        if freeze:
            for p in self.dino.parameters():
                p.requires_grad = False
        self.dino.eval()
        self.proj = nn.Conv2d(768, content_ch, 1)

    def forward(self, x_pre):
        with torch.no_grad():
            h = self.dino(x_pre).hidden_states[11][:, 1:, :]
        B = h.shape[0]
        h = h.reshape(B, 16, 16, 768).permute(0, 3, 1, 2)
        return self.proj(h)


class CondEncoder(nn.Module):
    def __init__(self, cond_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, cond_dim))

    def forward(self, x):
        return self.net(x)


class V4Net(nn.Module):
    def __init__(self, content_ch=64, n_class=20):
        super().__init__()
        self.content = ContentEncoder(content_ch)
        self.cond = CondEncoder(1)
        self.gc = nn.Linear(content_ch, n_class)

    def zc_pool(self, zc):
        return F.adaptive_avg_pool2d(zc, 1).flatten(1)

    def forward(self, xc_pre, xa):
        zc = self.content(xc_pre)
        zk = self.cond(xa)
        return zc, zk


def encode_full(net, pils, bs=16):
    """全图编码 (intended 分布: xc=224 dino_pre, xa=384 ToTensor)。返回 (Zc_pooled, zk)。"""
    net.eval()
    Zc, Zk = [], []
    for i in range(0, len(pils), bs):
        batch = pils[i:i + bs]
        xc = torch.stack([net.content.pre(b.resize((224, 224))) for b in batch]).to(DEVICE)
        xa = torch.stack([transforms.ToTensor()(b.resize((384, 384))) for b in batch]).to(DEVICE)
        with torch.no_grad():
            zc, zk = net(xc, xa)
        Zc.append(net.zc_pool(zc).cpu().numpy())
        Zk.append(zk.cpu().numpy().flatten())
        del xc, xa, zc, zk
    return np.vstack(Zc), np.concatenate(Zk)


# ───────── numpy 度量 ─────────
def roc_auc(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    order = np.argsort(s, kind='mergesort')
    ranks = np.arange(1, len(s) + 1, dtype=float)
    pos = ranks[y[order] == 1]
    n_pos = len(pos)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = pos.sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def logistic_fit(X, y, n_iter=3000, lr=0.05, lam=1e-3):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    Xz = (X - mu) / sd
    N, d = Xz.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        z = Xz @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (Xz.T @ g / N + lam * w)
        b -= lr * g.mean()
    return w, b, mu, sd


def logistic_pred(w, b, mu, sd, X):
    Xz = (np.asarray(X, float) - mu) / sd
    return 1.0 / (1.0 + np.exp(-(Xz @ w + b)))


def ece(scores, tp, n_bin=10):
    scores = np.asarray(scores, float)
    tp = np.asarray(tp, int)
    bins = np.linspace(0, 1, n_bin + 1)
    idx = np.clip(np.digitize(scores, bins) - 1, 0, n_bin - 1)
    e = 0.0
    for b in range(n_bin):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = scores[m].mean()
        acc = tp[m].mean()
        e += abs(conf - acc) * m.sum() / len(tp)
    return float(e)


def tp_match(boxes, scores, gts, iou_th=0.5):
    """逐图贪心匹配, 返回每检测 (score, tp)。boxes/scores numpy; gts list of [x1,y1,x2,y2]。"""
    if len(boxes) == 0:
        return [], []
    ev = []
    matched = [False] * len(gts)
    for i in np.argsort(-np.asarray(scores)):
        bi = boxes[i]
        best, bj = 0.0, -1
        for j, gj in enumerate(gts):
            if not matched[j]:
                v = U.iou(bi, gj)
                if v > best:
                    best, bj = v, j
        if best >= iou_th and bj >= 0:
            ev.append((float(scores[i]), 1))
            matched[bj] = True
        else:
            ev.append((float(scores[i]), 0))
    return [e[0] for e in ev], [e[1] for e in ev]


# ───────── 主流程 ─────────
def main():
    print('=' * 70)
    print('B\' 辅助层小样本先验 — 信息统一空间表示能否支撑辅助任务')
    print('=' * 70)
    net = V4Net().to(DEVICE)
    net.load_state_dict(torch.load(V4_PATH, map_location=DEVICE))
    net.eval()
    torch.set_num_threads(max(1, os.cpu_count() or 4))

    # 收集 COCO 6 类各 n=30 清晰图
    clean_imgs, clean_cls = [], []
    for coco_cls, _ in COCO_MAP:
        imgs, _ = U.load_coco(coco_cls, n=N_PER)
        clean_imgs += imgs
        clean_cls += [coco_cls] * len(imgs)
    print(f'  COCO 清晰图: {len(clean_imgs)} ({len(COCO_MAP)}类×{N_PER})')

    # A+B: 每图生成 k=1..4 全图, 编码
    print('  [编码] COCO 全图 k=1..4 ...', flush=True)
    rec_zk, rec_k, rec_zc = [], [], []
    for im in clean_imgs:
        for k in K_CHOICES:
            d = US2.degrade_res(im, k)
            rec_zk.append(k)
            rec_zc.append(d)
    Zc_coco, Zk_coco = encode_full(net, rec_zc, bs=16)
    rec_k = np.array(rec_zk)
    print(f'    编码 {len(rec_zk)} 张', flush=True)

    # A. 条件识别: z_k -> k
    r = np.corrcoef(Zk_coco, rec_k)[0, 1]
    # 4-way: 按 z_k 分位映射到 k
    q = np.quantile(Zk_coco, [0.25, 0.5, 0.75])
    pred_k = np.digitize(Zk_coco, q) + 1
    acc = float((pred_k == rec_k).mean())
    # z_c 控制: z_c 池化后取模长当标量(内容不应编码条件)
    zc_norm = np.linalg.norm(Zc_coco, axis=1)
    r_zc = np.corrcoef(zc_norm, rec_k)[0, 1]
    print(f'  [A 条件识别] PearsonR(z_k,k)={r:+.3f}  4-way acc={acc:.3f}(随机0.25)  '
          f'R(|z_c|,k)={r_zc:+.3f}')

    # B. 域 OOD: COCO-clean(k=1) vs DIOR 全图
    dior_paths = sorted(__import__('glob').glob(os.path.join(U.DIOR_IMG_DIR, '*.jpg')))[:len(clean_imgs)]
    dior_imgs = []
    for p in dior_paths:
        try:
            dior_imgs.append(Image.open(p).convert('RGB'))
        except Exception:
            pass
    Zc_dior, Zk_dior = encode_full(net, dior_imgs, bs=16)
    print(f'  [编码] DIOR 全图 {len(dior_imgs)} 张', flush=True)
    # COCO-clean = k==1
    m1 = rec_k == 1
    X_coco_clean_zk = Zk_coco[m1]
    X_coco_clean_zc = Zc_coco[m1]
    y_coco = np.zeros(len(X_coco_clean_zk))
    y_dior = np.ones(len(Zk_dior))
    # 只用 z_k
    auc_zk = roc_auc(np.concatenate([y_coco, y_dior]),
                    np.concatenate([X_coco_clean_zk, Zk_dior]))
    # z_k + z_c
    Xa = np.concatenate([X_coco_clean_zk, Zk_dior]).reshape(-1, 1)
    Xb = np.concatenate([X_coco_clean_zc, Zc_dior], 0)
    Xall = np.hstack([Xa, Xb])
    yall = np.concatenate([y_coco, y_dior])
    n = len(yall)
    tr = np.random.choice(n, n // 2, replace=False)
    te = np.array([i for i in range(n) if i not in tr])
    w, b, mu, sd = logistic_fit(Xall[tr], yall[tr])
    auc_zkzc = roc_auc(yall[te], logistic_pred(w, b, mu, sd, Xall[te]))
    print(f'  [B 域OOD] AUC(z_k)={auc_zk:.3f}  AUC(z_k+z_c)={auc_zkzc:.3f} (随机0.5)')

    # C+D: GDINO 在降级集 (k=2,3,4) 上, z_k 有变化
    print('  [GDINO] 降级集 k=2,3,4 检测 ...', flush=True)
    all_scores, all_tp, all_zk = [], [], []
    for ci, (coco_cls, _) in enumerate(COCO_MAP):
        imgs, gts = U.load_coco(coco_cls, n=N_PER)
        for k in [2, 3, 4]:
            deg = [US2.degrade_res(im, k) for im in imgs]
            # 对应全图 z_k (用已编码的 rec set: 同顺序)
            # 这里重新编码降级全图以得 z_k
            Zc_d, Zk_d = encode_full(net, deg, bs=16)
            preds = U.gdino_detect(deg, f'a photo of {coco_cls}', 0.0)
            for (boxes, scores), gt, zk1 in zip(preds, gts, Zk_d):
                sc, tp = tp_match(np.asarray(boxes), np.asarray(scores), gt)
                all_scores += sc
                all_tp += tp
                all_zk += [float(zk1)] * len(sc)
    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)
    all_zk = np.array(all_zk)
    print(f'    降级集检测: {len(all_scores)} 框, TP率={all_tp.mean():.3f}', flush=True)

    # C. 样本筛选: AUC(score) vs AUC(score + w*z_k)
    base_auc = roc_auc(all_tp, all_scores)
    # w 网格在 held-out 上选
    n = len(all_tp)
    tr = np.random.choice(n, n // 2, replace=False)
    te = np.array([i for i in range(n) if i not in tr])
    best_auc, best_w = base_auc, 0.0
    for w in np.linspace(-3, 3, 25):
        comb = all_scores[tr] + w * all_zk[tr]
        a = roc_auc(all_tp[tr], comb)
        if a > best_auc:
            best_auc, best_w = a, w
    test_auc = roc_auc(all_tp[te], all_scores[te] + best_w * all_zk[te])
    print(f'  [C 样本筛选] AUC(score)={base_auc:.3f}  AUC(score+w·z_k, w={best_w:+.2f})={test_auc:.3f}')

    # D. ECE: baseline vs z_k 分箱温度缩放
    base_ece = ece(all_scores, all_tp)
    # 按 z_k 四分位分箱, 箱内拟合温度(最小化 CE), 应用到同箱测试
    qb = np.quantile(all_zk, [0.25, 0.5, 0.75])
    binid = np.clip(np.digitize(all_zk, qb), 0, 3)
    new_scores = all_scores.copy()
    for b in range(4):
        mt = binid[tr] == b
        if mt.sum() < 5:
            continue
        s = all_scores[tr][mt]
        y = all_tp[tr][mt]
        # 粗网格温度
        best_t, best_ce = 1.0, 1e9
        for t in np.linspace(0.3, 3.0, 28):
            ce = ece(1.0 / (1.0 + np.exp(-t * (s - s.mean()))), y)
            if ce < best_ce:
                best_t, best_ce = t, ce
        me = binid[te] == b
        if me.sum() == 0:
            continue
        st = all_scores[te][me]
        new_scores[te][me] = 1.0 / (1.0 + np.exp(-best_t * (st - st.mean())))
    cal_ece = ece(new_scores, all_tp)
    print(f'  [D ECE校准] baseline ECE={base_ece:.3f}  z_k分箱温度缩放 ECE={cal_ece:.3f}')

    out = {
        'n_coco_clean': len(clean_imgs), 'n_dior': len(dior_imgs),
        'A_condition': {'pearson_r_zk_k': round(float(r), 3),
                        'acc_4way': round(acc, 3), 'random': 0.25,
                        'pearson_r_abszc_k': round(float(r_zc), 3)},
        'B_ood': {'auc_zk': round(auc_zk, 3), 'auc_zk_zc': round(auc_zkzc, 3), 'random': 0.5},
        'C_sample_selection': {'auc_score': round(float(base_auc), 3),
                               'auc_score_wzk': round(float(test_auc), 3), 'w': round(float(best_w), 2)},
        'D_ece': {'baseline': round(base_ece, 3), 'zk_temp_scaled': round(cal_ece, 3)},
        'verdict_note': 'A: z_k 在全图 intended 分布能否读条件; B: 域OOD; C/D: 降级集上 z_k 能否改善风险/校准',
    }
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果 -> {SAVE}')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
