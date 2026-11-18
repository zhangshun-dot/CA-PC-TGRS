# -*- coding: utf-8 -*-
"""tge_probe_adapter.py — CLIP-Adapter 对照 (NeurIPS22): 源域 few-shot 特征适配器 → 目标域零样本应用

与 CoOp 互补的训练型对照: CoOp 学文本提示, CLIP-Adapter 学视觉特征适配器。
  - 训练: DIOR 20 类 GT crop few-shot (K=16/类) 训残差适配器 (MLP 512->512->512 + 残差缩放 λ)
  - 应用: COCO 框 crop 特征过适配器 → 与固定目标模板文本打分 → sqrt(gd*vis) 融合
  - 协议与主表完全一致 (N=100, 后 90 图测试, pre-filter 0.03, 逐图 AP50, 同一框池)

行:
  baseline    — GDINO 原始分数
  adapter_fs  — 源域 K=16 训练适配器后打分 (跨域应用)
图像特征全部预计算 (只训适配器参数)。输出: 实验数据/tge_probe_adapter.json + 控制台表
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_adapter_full.json'

N_COCO = 100000                # 全量: COCO val2017 全部含 GT 图 (主流方法统一协议对比)
K = 10
K_SHOT = 16
EPOCHS = 20
LR = 1e-3
BATCH = 32
SEED = 42
SRC_TMPL = 'an overhead image of {}'
TGT_TMPL = 'a photo of {}'

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


class Adapter(nn.Module):
    """残差特征适配器 (CLIP-Adapter): f' = f + λ·MLP(f), λ 可学习。"""

    def __init__(self, dim=512, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.lam = nn.Parameter(torch.tensor(0.2))

    def forward(self, f):
        return f + self.lam * self.mlp(f)


def dior_train_crops():
    """DIOR 每类 GT crop (few-shot K=16, seed 固定), 返回 (crops, labels, class_names)。"""
    rng = np.random.default_rng(SEED)
    names = list(U.DIOR_ID2NAME.values())
    crops, labels = [], []
    for ci, cls in enumerate(names):
        images, gts = U.load_dior(cls)
        cl = []
        for img, g in zip(images, gts):
            for box in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 > x1 + 10 and y2 > y1 + 10:
                    cl.append(img.crop((x1, y1, x2, y2)))
        if len(cl) > K_SHOT:
            idx = rng.choice(len(cl), K_SHOT, replace=False)
            cl = [cl[i] for i in idx]
        crops.extend(cl)
        labels.extend([ci] * len(cl))
        print('    %-20s %d crops' % (cls, len(cl)))
    return crops, labels, names


def eval_adapted(images, gts, dets, adapter, tgt_text):
    """框 crop 特征过适配器 → 与目标模板打分 → sqrt(gd*vis) → 逐图 AP50。"""
    crops, boxes_all, gd_all, img_ids = [], [], [], []
    for ii, ((boxes, scores), img) in enumerate(zip(dets, images)):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        for box, sc in zip(boxes[keep], scores[keep]):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            boxes_all.append([x1, y1, x2, y2])
            gd_all.append(float(sc))
            img_ids.append(ii)
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    base = U.compute_ap50_img(U.split_by_img(boxes_all, gd, img_ids, len(images)), gts)
    if not crops:
        return base, base
    F = T.encode_clip_batch(crops).to(U.DEVICE)
    with torch.no_grad():
        F_a = adapter(F)
    F_a = F_a / F_a.norm(dim=-1, keepdim=True)
    vis = np.maximum(0.0, (F_a.cpu().numpy() @ tgt_text.cpu().numpy()))
    fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
    ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    return base, ap


def main():
    device = U.DEVICE
    clip_m, _ = U.get_clip()

    # ── 1) DIOR few-shot 训练数据 (GT crop, 图像特征预计算) ──
    print('[1/3] DIOR 训练 crop (K=%d/类) ...' % K_SHOT)
    crops, labels, cls_names = dior_train_crops()
    F_train = T.encode_clip_batch(crops).to(device)          # (N, 512)
    F_n = F_train / F_train.norm(dim=-1, keepdim=True)
    labels = torch.tensor(labels, device=device)

    # 源域文本特征 (固定模板, CLIP-Adapter 不训文本); text_emb 返回 (1,512), 取 [0]
    # detach: jit 模型参数 requires_grad=True, 跨 batch 复用会触发二次 backward
    src_text = torch.stack([T.text_emb([SRC_TMPL.format(c)])[0].cpu() for c in cls_names]).to(device).detach()

    # ── 2) 适配器训练 ──
    print('[2/3] Adapter 训练 (epochs=%d, lr=%.1e) ...' % (EPOCHS, LR))
    adapter = Adapter().to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=LR, weight_decay=1e-4)
    scale = clip_m.logit_scale.exp().float().detach()   # detach: 叶子张量跨 batch 复用
    for ep in range(EPOCHS):
        g = torch.Generator(device=device)
        g.manual_seed(SEED + ep)
        perm = torch.randperm(len(F_n), device=device, generator=g)
        tot_loss, n_b = 0.0, 0
        for i in range(0, len(F_n), BATCH):
            idx = perm[i:i + BATCH]
            fb, lb = F_n[idx], labels[idx]
            fa = adapter(fb)
            fa = fa / fa.norm(dim=-1, keepdim=True)
            logits = fa @ src_text.T * scale
            loss = nn.functional.cross_entropy(logits, lb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(idx)
            n_b += len(idx)
        print('    epoch %2d  loss=%.4f  λ=%.3f' % (ep + 1, tot_loss / max(n_b, 1),
                                                     float(adapter.lam)))

    # ── 3) 应用: COCO 6 类 (N=100, 与主表同协议) ──
    print('[3/3] COCO 6 类应用 (N=%d, 逐图 AP50) ...' % N_COCO)
    tgt_text = {}
    for name, coco_cls, src_cls in TARGETS:
        tgt_text[name] = T.text_emb([TGT_TMPL.format(coco_cls)])[0].cpu()

    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = pickle.load(open(DATA / f'tge_v2_dets_COCO-{name}.pkl', 'rb'))  # 复用 A-CSPC 全量检测缓存
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]
        base, ap_fs = eval_adapted(test_imgs, test_gts, test_dets, adapter, tgt_text[name])
        rows[name] = {'baseline': round(base, 4), 'adapter_fs': round(ap_fs, 4),
                      'fs_gain': round(ap_fs - base, 4)}
        print('  %-16s base=%.4f  adapter_fs=%.4f  [%+.4f]'
              % (name, base, ap_fs, ap_fs - base))

    mean_row = {k: round(float(np.mean([rows[n][k] for n in rows])), 4)
                for k in ['baseline', 'adapter_fs', 'fs_gain']}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_coco': N_COCO, 'k_shot': K_SHOT, 'epochs': EPOCHS,
                              'lr': LR, 'seed': SEED,
                              'protocol': 'CLIP-Adapter (NeurIPS22): DIOR K=16 GT-crop 训练残差'
                                          '适配器, COCO 框 crop 打分, 逐图 AP50, 后90图测试'},
                   'classes': rows, 'mean': mean_row}, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 80)
    print('%-16s %9s %9s %9s' % ('class', 'baseline', 'adapter_fs', 'fs增益'))
    print('-' * 80)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-16s %9.4f %9.4f %+9.4f' % (name, r['baseline'], r['adapter_fs'],
                                             r['fs_gain']))
    print('-' * 80)
    print('%-16s %9.4f %9.4f %+9.4f   <- 6-class mean' %
          ('mean6', mean_row['baseline'], mean_row['adapter_fs'], mean_row['fs_gain']))
    print('=' * 80)
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
