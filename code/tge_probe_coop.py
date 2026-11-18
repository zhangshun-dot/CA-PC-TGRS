# -*- coding: utf-8 -*-
"""tge_probe_coop.py — CoOp 对照 (CVPR22 提示学习): 源域 few-shot 训练 → 目标域零样本应用

文献对比补强: 我们的设定 (零样本 training-free 重打分) 下, 可比的有监督对照
是"源域训练、目标域零样本应用"的训练型方法。CoOp 是最经典的代表:
  - 训练: DIOR 20 类 GT crop few-shot (K=16/类) 学 8 个共享 context token (8x512 参数)
  - 应用: 学习提示 + 目标类名 → CLIP 文本特征, 对 COCO 框 crop 打分 → sqrt(gd*vis) 融合
  - 协议与主表完全一致 (N=100, 后 90 图测试, pre-filter 0.03, 逐图 AP50, 同一框池)

行:
  baseline      — GDINO 原始分数
  coop_zs       — 固定前缀 "a photo of " 提示 (CoOp 零样本, 无训练)
  coop_fs       — 源域 few-shot 训练后的学习提示 (跨域应用)

CoOp prompt 构造 (官方口径): [SOT] + ctx(N_CTX 可学习) + 类名 token + [EOT],
类名 token 从 tokenize(cls_name) 提取 (去 SOT/EOT), 变长序列走 CLIP 文本 transformer,
EOT 位置池化。图像特征全部预计算 (CLIP 图像编码器不反传)。
输出: 实验数据/tge_probe_coop.json + 控制台表
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
OUT = DATA / 'tge_probe_coop_full.json'

N_COCO = 100000                # 全量: COCO val2017 全部含 GT 图 (主流方法统一协议对比)
K = 10                       # 前 K 图 (与主表协议一致, 此处仅切后 90 图测试)
N_CTX = 8                    # CoOp context token 数 (标准)
K_SHOT = 16                  # 每类训练 crop 数 (few-shot)
EPOCHS = 10
LR = 2e-3
WD = 1e-4
BATCH = 32
SEED = 42
PREFIX = 'a photo of a '     # context 初始化前缀 (CoOp 标准)
SOT_ID, EOT_ID = 49406, 49407
TGT_TMPL = 'a photo of {}'

TARGETS = [
    ('COCO-airplane',   'airplane',   'airplane'),
    ('COCO-boat',       'boat',       'ship'),
    ('COCO-car',        'car',        'vehicle'),
    ('COCO-truck',      'truck',      'vehicle'),
    ('COCO-bus',        'bus',        'vehicle'),
    ('COCO-motorcycle', 'motorcycle', 'vehicle'),
]


def build_class_text(model, cls_name, ctx, device):
    """CoOp prompt: SOT + ctx(N_CTX) + 类名 + EOT + pad → CLIP 文本编码 → (512,) 归一化。

    CLIP 文本 transformer 内置 77x77 固定 causal mask → 序列必须 pad 到 77
    (EOT 后补零 token, causal mask 保证 pad 位不影响 EOT 前输出);
    池化取 EOT 位置 (1+N_CTX+len(name_ids)), 不是末尾。ctx 为 nn.Parameter。
    """
    L = 77
    with torch.no_grad():
        tok = clip.tokenize([cls_name]).to(device)
        ids = tok[0].tolist()
        eot = ids.index(EOT_ID) if EOT_ID in ids else len(ids) - 1
        name_ids = [i for i in ids[1:eot] if i != SOT_ID]
        n_pad = L - N_CTX - len(name_ids) - 2
        assert n_pad >= 0, 'class name too long for context'
        full_ids = [SOT_ID] + [0] * N_CTX + name_ids + [EOT_ID] + [0] * n_pad
        full_tok = torch.tensor([full_ids], device=device)
        emb = model.token_embedding(full_tok).float()       # (1, 77, 512)
        pos = model.positional_embedding[:L].float()        # (77, 512)
        eot_idx = 1 + N_CTX + len(name_ids)
    emb = emb.clone()
    emb[0, 1:1 + N_CTX] = ctx
    x = emb + pos
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).float()
    pooled = x[0, eot_idx] @ model.text_projection.float()  # EOT 位置池化
    return pooled / pooled.norm()


def clip_text_zs(model, text):
    """零样本固定前缀文本特征 (encode_text 直接打分)。"""
    with torch.no_grad():
        tok = clip.tokenize([text]).to(U.DEVICE)
        f = model.encode_text(tok).float()
    return (f / f.norm(dim=-1, keepdim=True))[0]


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


def eval_class_vis(images, gts, dets, w_vec, name):
    """给定文本权重 (512,) 对框 crop 打分 → sqrt(gd*vis) → 逐图 AP50。"""
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
    F = T.encode_clip_batch(crops).cpu().numpy()
    w = w_vec.cpu().numpy()
    vis = np.maximum(0.0, F @ w)
    fused = np.sqrt(gd * np.clip(vis, 1e-6, 1.0))
    ap = U.compute_ap50_img(U.split_by_img(boxes_all, fused, img_ids, len(images)), gts)
    return base, ap


def main():
    # jit=False: 直接访问 transformer/token_embedding 需要非脚本化模型;
    # 且 jit 版内部权重为 Half, 与 float 输入冲突 (RuntimeError: Float vs Half)
    import clip as clip_mod
    clip_m, clip_pre = clip_mod.load(r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt',
                                     device=U.DEVICE, jit=False)
    clip_m = clip_m.float().eval()
    device = U.DEVICE

    # ── 1) DIOR few-shot 训练数据 (GT crop, 图像特征预计算) ──
    print('[1/4] DIOR 训练 crop (K=%d/类) ...' % K_SHOT)
    crops, labels, cls_names = dior_train_crops()
    F_train = T.encode_clip_batch(crops)                     # (N, 512) 预计算
    labels = torch.tensor(labels, device=device)
    F_n = F_train / F_train.norm(dim=-1, keepdim=True)

    # ── 2) CoOp 训练: 可学习 context (8x512), 从 "a photo of a " 初始化 ──
    print('[2/4] CoOp 训练 (ctx=%d, epochs=%d, lr=%.1e) ...' % (N_CTX, EPOCHS, LR))
    with torch.no_grad():
        prefix_tok = clip.tokenize([PREFIX]).to(device)
        prefix_emb = clip_m.token_embedding(prefix_tok).float()
    ctx = nn.Parameter(prefix_emb[0, 1:N_CTX + 1, :].clone())
    opt = torch.optim.Adam([ctx], lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(EPOCHS):
        g = torch.Generator(device=device)
        g.manual_seed(SEED + ep)
        perm = torch.randperm(len(F_n), device=device, generator=g)
        tot_loss, n_b = 0.0, 0
        for i in range(0, len(F_n), BATCH):
            idx = perm[i:i + BATCH]
            fb, lb = F_n[idx], labels[idx]
            text_feats = torch.stack([build_class_text(clip_m, c, ctx, device)
                                      for c in cls_names])
            logits = fb @ text_feats.T * clip_m.logit_scale.exp().float()
            loss = nn.functional.cross_entropy(logits, lb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss) * len(idx)
            n_b += len(idx)
        sched.step()
        print('    epoch %2d  loss=%.4f' % (ep + 1, tot_loss / max(n_b, 1)))

    # ── 3) 应用: COCO 6 类框 crop 打分 (N=100, 与主表同协议) ──
    print('[3/4] COCO 6 类应用 (N=%d, 逐图 AP50) ...' % N_COCO)
    zs_text = {}
    fs_text = {}
    for name, coco_cls, src_cls in TARGETS:
        zs_text[name] = clip_text_zs(clip_m, PREFIX + coco_cls)
        with torch.no_grad():
            fs_text[name] = build_class_text(clip_m, coco_cls, ctx, device)

    rows = {}
    for name, coco_cls, src_cls in TARGETS:
        images, gts = U.load_coco(coco_cls, n=N_COCO)
        dets = pickle.load(open(DATA / f'tge_v2_dets_COCO-{name}.pkl', 'rb'))  # 复用 A-CSPC 全量检测缓存
        test_imgs, test_gts, test_dets = images[K:], gts[K:], dets[K:]
        base, ap_zs = eval_class_vis(test_imgs, test_gts, test_dets, zs_text[name], name)
        _, ap_fs = eval_class_vis(test_imgs, test_gts, test_dets, fs_text[name], name)
        rows[name] = {'baseline': round(base, 4), 'coop_zs': round(ap_zs, 4),
                      'coop_fs': round(ap_fs, 4),
                      'zs_gain': round(ap_zs - base, 4), 'fs_gain': round(ap_fs - base, 4)}
        print('  %-16s base=%.4f  coop_zs=%.4f  coop_fs=%.4f  [zs %+.4f | fs %+.4f]'
              % (name, base, ap_zs, ap_fs, ap_zs - base, ap_fs - base))

    # ── 4) 汇总 ──
    mean_row = {k: round(float(np.mean([rows[n][k] for n in rows])), 4)
                for k in ['baseline', 'coop_zs', 'coop_fs', 'zs_gain', 'fs_gain']}
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'config': {'n_coco': N_COCO, 'n_ctx': N_CTX, 'k_shot': K_SHOT,
                              'epochs': EPOCHS, 'lr': LR, 'seed': SEED,
                              'protocol': 'CoOp (CVPR22): DIOR K=16 GT-crop 训练共享 context, '
                                          'COCO 框 crop 打分, 逐图 AP50, 后90图测试'},
                   'classes': rows, 'mean': mean_row}, f, indent=2, ensure_ascii=False)

    print()
    print('=' * 100)
    print('%-16s %9s %9s %9s %9s %9s' %
          ('class', 'baseline', 'coop_zs', 'coop_fs', 'zs增益', 'fs增益'))
    print('-' * 100)
    for name in [t[0] for t in TARGETS]:
        r = rows[name]
        print('%-16s %9.4f %9.4f %9.4f %+9.4f %+9.4f' %
              (name, r['baseline'], r['coop_zs'], r['coop_fs'],
               r['zs_gain'], r['fs_gain']))
    print('-' * 100)
    print('%-16s %9.4f %9.4f %9.4f %+9.4f %+9.4f   <- 6-class mean' %
          ('mean6', mean_row['baseline'], mean_row['coop_zs'], mean_row['coop_fs'],
           mean_row['zs_gain'], mean_row['fs_gain']))
    print('=' * 100)
    print('→ %s' % OUT)


if __name__ == '__main__':
    main()
