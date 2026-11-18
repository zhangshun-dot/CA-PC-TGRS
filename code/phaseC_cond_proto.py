# -*- coding: utf-8 -*-
"""phaseC_cond_proto.py — 条件感知原型对齐小原型 (验证"潜力"是否真实)

假设: 统一空间学到的 z_k(全局条件) 若按 INTENDED 方式用——在原型层对 COCO 原型做
条件自适应(FiLM 偏移), 而非 post-hoc rerank——应能改善跨域(DIOR)区域匹配。

设计(小样本先验证, 不碰全检测AP):
  1. 用 DINOv2 区域特征建 COCO 视觉原型 P_c (6类, 固定).
  2. DIOR 区域(DINOv2 crop) + 每图 z_k (v4 CondEncoder).
  3. baseline: argmax_c cos(f, P_c).   adaptive: P_c'(zk)=normalize(P_c + zk*w_c), w_c 训于 DIOR train.
  4. 比 baseline vs adaptive 在 DIOR test 的 6-way 区域分类准确率.
若 adaptive 显著 > baseline → 表示潜力在原型层兑现 → 进完整检测头; 否则 null.

环境: GroundingDINO/venv (CPU). 冻 backbone, z_k+F护膜极轻.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import unified_eval as U
import phaseB_aux_probe as P

DEVICE = 'cpu'
V4_PATH = U.SAVE_DIR / 'unified_space_v4.pt'
COCO_CLS = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
DIOR_CLS = ['airplane', 'ship', 'vehicle']
DIOR2COCO = {'airplane': 'airplane', 'ship': 'boat', 'vehicle': 'car'}  # vehicle→car 代理
N_PER = 30
SEED = 42


def build_coco_prototypes():
    """每 COCO 类: 干净图 GT crop 的 DINOv2 特征均值 → 归一化原型。"""
    protos = {}
    for c in COCO_CLS:
        imgs, gts = U.load_coco(c, n=N_PER)
        fs = []
        for im, boxes in zip(imgs, gts):
            for b in boxes:
                fs.append(U.encode_dinov2(im.crop(tuple(map(int, b)))))
        fs = torch.stack(fs)  # (M,768)
        protos[c] = fs.mean(0) / fs.mean(0).norm()
        print(f'  P[{c}] from {len(fs)} crops')
    return protos


def get_dior_regions():
    """DIOR 区域: f(DINOv2) + 每图 z_k + 映射COCO标签; 按类 2/3 train 1/3 test。"""
    rng = np.random.default_rng(SEED)
    regions = {'train': [], 'test': []}
    for dc in DIOR_CLS:
        imgs, gts = U.load_dior(dc, n=N_PER)
        zc, zk = P.encode_full(P.NET, imgs, bs=16)   # zk: (len(imgs),)
        idx = np.arange(len(imgs))
        rng.shuffle(idx)
        n_tr = int(0.67 * len(imgs))
        tr_set, te_set = set(idx[:n_tr].tolist()), set(idx[n_tr:].tolist())
        for i, (im, boxes) in enumerate(zip(imgs, gts)):
            split = 'train' if i in tr_set else 'test'
            for b in boxes:
                f = U.encode_dinov2(im.crop(tuple(map(int, b))))
                regions[split].append({
                    'f': f.cpu().numpy(), 'zk': float(zk[i]),
                    'label': DIOR2COCO[dc], 'dior_cls': dc})
    return regions


def classify(regions, protos, W):
    """W: dict c->(768,) offset; None=固定原型. 返回 (acc, per_class_acc)。"""
    Pmat = torch.stack([protos[c] for c in COCO_CLS])  # (6,768)
    correct, total = 0, 0
    pc_correct, pc_total = {c: 0 for c in DIOR_CLS}, {c: 0 for c in DIOR_CLS}
    for r in regions:
        f = torch.tensor(r['f'])
        if W is None:
            Pc = Pmat
        else:
            offs = torch.stack([torch.tensor(W[c]) for c in COCO_CLS])  # (6,768)
            Pc = torch.stack([F.normalize(protos[c] + r['zk'] * torch.tensor(W[c]), dim=0)
                              for c in COCO_CLS])
        logits = f @ Pc.T
        pred = COCO_CLS[int(logits.argmax())]
        total += 1
        pc_total[r['dior_cls']] += 1
        if pred == r['label']:
            correct += 1
            pc_correct[r['dior_cls']] += 1
    acc = correct / total if total else 0
    pca = {c: (pc_correct[c] / pc_total[c] if pc_total[c] else 0) for c in DIOR_CLS}
    return acc, pca


def train_offsets(regions_tr, protos, n_iter=800, lr=0.05):
    W = {c: torch.zeros(768, requires_grad=True) for c in COCO_CLS}
    opt = torch.optim.SGD([W[c] for c in COCO_CLS], lr=lr)
    Pmat = {c: protos[c] for c in COCO_CLS}
    for _ in range(n_iter):
        opt.zero_grad()
        loss = 0.0
        for r in regions_tr:
            f = torch.tensor(r['f'])
            logits = []
            for c in COCO_CLS:
                Pc = F.normalize(Pmat[c] + r['zk'] * W[c], dim=0)
                logits.append(f @ Pc)
            logits = torch.stack(logits)
            t = COCO_CLS.index(r['label'])
            loss += F.cross_entropy(logits.unsqueeze(0), torch.tensor([t]))
        loss.backward()
        opt.step()
    return {c: W[c].detach().numpy() for c in COCO_CLS}


def main():
    global NET
    print('=' * 70)
    print('C 条件感知原型对齐 — 潜力验证 (小样本, DIOR 跨域)')
    print('=' * 70)
    NET = P.V4Net().to(DEVICE)
    NET.load_state_dict(torch.load(V4_PATH, map_location=DEVICE))
    NET.eval()

    print('[1] COCO 视觉原型', flush=True)
    protos = build_coco_prototypes()

    print('[2] DIOR 区域 + z_k', flush=True)
    regions = get_dior_regions()
    print(f'  train={len(regions["train"])} test={len(regions["test"])}')

    print('[3] baseline(固定原型) vs adaptive(zk-FiLM)', flush=True)
    base_acc, base_pca = classify(regions['test'], protos, None)
    W = train_offsets(regions['train'], protos)
    ada_acc, ada_pca = classify(regions['test'], protos, W)

    out = {
        'baseline_acc': base_acc, 'adaptive_acc': ada_acc,
        'delta': ada_acc - base_acc,
        'baseline_per_class': base_pca, 'adaptive_per_class': ada_pca,
        'n_train': len(regions['train']), 'n_test': len(regions['test']),
        'note': 'zk 在原型层做 FiLM 偏移; 正 delta=潜力兑现, 进完整检测头; ~0/null=收口',
    }
    print(f'\n  baseline={base_acc:.3f}  adaptive={ada_acc:.3f}  Δ={ada_acc-base_acc:+.3f}')
    print(f'  per_class baseline={base_pca}')
    print(f'  per_class adaptive={ada_pca}')
    with open(U.SAVE_DIR / 'phaseC_cond_proto.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('[完成] -> 实验数据/phaseC_cond_proto.json')


if __name__ == '__main__':
    main()
