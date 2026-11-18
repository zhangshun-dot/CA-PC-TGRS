# -*- coding: utf-8 -*-
"""tge_probe_negprompt.py — NegPrompt 式负提示对照 (CVPR24 思想, 训练型负提示参照栏)

回答评审问题: "负原型是不是只是可学习负提示的替代品?"
  - 我们的负原型: 源域 DIOR 近邻类均值 (零样本, 无训练)
  - NegPrompt 式: 源域 K=16 训练 M=3 个可学习负文本提示 (与正文本同 CLIP 空间)
训练: DIOR GT crop few-shot, 正文本固定 (CLIP 模板), 负提示可学习 (SOT+8ctx+EOT 无类名);
      CE: logits = [cos(F, pos_c), max_m cos(F, neg_m)]
应用: COCO 框 crop → vis = max(0, cos(F, pos_y) − λ·max_m cos(F, neg_m)) → sqrt(gd*vis) → AP50
行: baseline / negprompt_fs (训练负提示) / (A-CSPC 参考: 读 tge_probe_tge_cmp.json)
协议与主表一致 (N=100, 逐图 AP50)。输出: 实验数据/tge_probe_negprompt.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import clip
import unified_eval as U
import tge_probe as T

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = DATA / 'tge_probe_negprompt_full.json'

N_COCO = 100000                # 全量: COCO val2017 全部含 GT 图 (主流方法统一协议对比)
K = 10
K_SHOT = 16
M_NEG = 3              # 负提示数量 (NegPrompt 风格)
N_CTX = 8
EPOCHS = 10
LR = 2e-3
BATCH = 32
SEED = 42
LAM = 0.5              # 负提示抑制权重 (与主方法 λ 一致, 非挑参)
SOT_ID, EOT_ID = 49406, 49407
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


def build_neg_text(model, ctx_neg, device):
    """负提示文本特征: [SOT] + ctx(8) + [EOT] + pad → CLIP 文本编码 → (M,512) 归一化。
    ctx_neg: (M, N_CTX, 512) 可学习。"""
    L = 77
    with torch.no_grad():
        full_ids = [SOT_ID] + [0] * N_CTX + [EOT_ID] + [0] * (L - N_CTX - 2)
        full_tok = torch.tensor([full_ids], device=device)
        emb = model.token_embedding(full_tok).float()          # (1, 77, 512)
        pos_emb = model.positional_embedding[:L].float()
    emb = emb.clone().repeat(M_NEG, 1, 1)
    emb[:, 1:1 + N_CTX] = ctx_neg
    x = emb + pos_emb
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).float()
    pooled = x[:, 1 + N_CTX, :] @ model.text_projection.float()   # EOT 位置
    return pooled / pooled.norm(dim=-1, keepdim=True)


def dior_train_crops():
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
    return crops, labels, names


def eval_class_neg(images, gts, dets, pos_w, neg_w, name):
    """vis = max(0, F@pos − λ·max_m(F@neg_m)) → sqrt(gd*vis) → 逐图 AP50。"""
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
    if crops:
        F = T.encode_clip_batch(crops).cpu().numpy()
        pos_s = F @ pos_w
        neg_s = (F @ neg_w.T).max(axis=1)
        vis = np.maximum(0.0, pos_s - LAM * neg_s)
        fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
        ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    else:
        ap = base
    return base, ap


def main():
    device = U.DEVICE
    import clip as clip_mod
    clip_m, _ = clip_mod.load(r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt',
                              device=device, jit=False)
    clip_m = clip_m.float().eval()

    # ── 1) 训练数据 ──
    print('[1/3] DIOR 训练 crop (K=%d/类) ...' % K_SHOT)
    crops, labels, cls_names = dior_train_crops()
    F_train = T.encode_clip_batch(crops).to(device)
    F_n = F_train / F_train.norm(dim=-1, keepdim=True)
    labels = torch.tensor(labels, device=device)

    # 正文本 (固定 CLIP 模板, 源域)
    pos_text = torch.stack([T.text_emb([SRC_TMPL.format(c)])[0].cpu()
                            for c in cls_names]).to(device).detach()

    # ── 2) 训练负提示 (M 个共享) ──
    print('[2/3] 负提示训练 (M=%d, ctx=%d, epochs=%d) ...' % (M_NEG, N_CTX, EPOCHS))
    with torch.no_grad():
        prefix_tok = clip.tokenize(['a photo of a ']).to(device)
        prefix_emb = clip_m.token_embedding(prefix_tok).float()
    ctx_neg = nn.Parameter(prefix_emb[0, 1:N_CTX + 1, :].clone().repeat(M_NEG, 1, 1))
    opt = torch.optim.Adam([ctx_neg], lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scale = clip_m.logit_scale.exp().float().detach()

    for ep in range(EPOCHS):
        g = torch.Generator(device=device)
        g.manual_seed(SEED + ep)
        perm = torch.randperm(len(F_n), device=device, generator=g)
        tot_loss, n_b = 0.0, 0
        for i in range(0, len(F_n), BATCH):
            idx = perm[i:i + BATCH]
            fb, lb = F_n[idx], labels[idx]
            neg_t = build_neg_text(clip_m, ctx_neg, device)      # (M, 512)
            pos_s = fb @ pos_text.T                              # (B, 20)
            neg_s = (fb @ neg_t.T).max(dim=1).values             # (B,)
            logits_pos = pos_s.gather(1, lb.unsqueeze(1)).squeeze(1)
            logits = torch.stack([logits_pos, neg_s], dim=1)     # (B, 2) 正 vs 负
            loss = nn.functional.cross_entropy(logits * scale, torch.zeros_like(lb, dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(idx)
            n_b += len(idx)
        sched.step()
        print('    epoch %2d  loss=%.4f' % (ep + 1, tot_loss / max(n_b, 1)))

    # ── 3) 应用: COCO 6 类 ──
    print('[3/3] COCO 6 类应用 (N=%d, 逐图 AP50) ...' % N_COCO)
    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = pickle.load(open(DATA / f'tge_v2_dets_COCO-{name}.pkl', 'rb'))  # 复用 A-CSPC 全量检测缓存
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]
        pos_w = T.text_emb([TGT_TMPL.format(coco_cls)])[0].cpu().numpy()
        with torch.no_grad():
            neg_w = build_neg_text(clip_m, ctx_neg, device).cpu().numpy()
        base, ap = eval_class_neg(test_imgs, test_gts, test_dets, pos_w, neg_w, name)
        rows[name] = {'baseline': round(base, 4), 'negprompt_fs': round(ap, 4),
                      'fs_gain': round(ap - base, 4)}
        print('  %-16s base=%.4f  negprompt_fs=%.4f  [%+.4f]'
              % (name, base, ap, ap - base))

    mean_row = {k: round(float(np.mean([rows[n][k] for n in rows])), 4)
                for k in ['baseline', 'negprompt_fs', 'fs_gain']}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_coco': N_COCO, 'k_shot': K_SHOT, 'm_neg': M_NEG,
                              'n_ctx': N_CTX, 'epochs': EPOCHS, 'lam': LAM, 'seed': SEED,
                              'protocol': 'NegPrompt 式训练负提示 (CVPR24 思想), DIOR K=16 训练, '
                                          'COCO 框 crop 打分, N=100 逐图 AP50'},
                   'classes': rows, 'mean': mean_row}, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 76)
    print('%-16s %9s %14s %9s' % ('class', 'baseline', 'negprompt_fs', 'fs增益'))
    print('-' * 76)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-16s %9.4f %14.4f %+9.4f' % (name, r['baseline'], r['negprompt_fs'],
                                              r['fs_gain']))
    print('-' * 76)
    print('%-16s %9.4f %14.4f %+9.4f   <- mean (A-CSPC 参考 0.4461/+0.0652)' %
          ('mean', mean_row['baseline'], mean_row['negprompt_fs'], mean_row['fs_gain']))
    print('=' * 76)
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
