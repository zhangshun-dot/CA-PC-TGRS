# -*- coding: utf-8 -*-
"""multi_prototype.py — 多原型类内聚类，攻难类

背景: 诊断确认难类(COCO-boat)单一均值原型分离度 AUC=0.489≈随机(H2 原型覆盖不足)。
本脚本验证: 类内 k-means 多原型能否拉起难类。

关键判别逻辑(先算数据再定方法):
  域内自检 — 对 DIOR 源域 exemplar 做 leave-one-out:
    single: 每个样本到「其余样本均值原型」的余弦
    multi : 每个样本到「最近子原型中心」的余弦
    · multi 显著 > single → 类内确有可分子结构, 多原型有意义
    · multi ≈ single        → 类内同质, 难类弱是「跨域 gap」, 多原型无效

三部分:
  1. 重提取 DIOR 源域原始 exemplar 特征 (DINOv2 L11 CLS, 每图最大 bbox)
  2. 域内自检 (single vs multi)
  3. k-means 子原型 + 跨域评估 multi-proto AP50 vs CDVM (bank 单一原型)
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'

import io, glob, sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from PIL import Image

sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import unified_eval as U

PARQUET_PATH = U.DIOR_PARQUET
DIOR_IMG_DIR = U.DIOR_IMG_DIR
K = 3  # 子原型数

# (name, dataset, cls, prefix, dior_cls)
TARGETS = [
    ('COCO-boat',     'coco', 'boat',    'a photo of',           'ship'),     # 跨域难类
    ('DIOR-vehicle',  'dior', 'vehicle', 'an overhead image of', 'vehicle'),  # 同域易类(对照)
]


# ══════════════════════════ 1. 提取原始特征 ══════════════════════════

def extract_raw_feats(dior_cls):
    """从 parquet 提取该类所有训练 crop 的 DINOv2 L11 特征。返回 [N,768] 归一化。"""
    df = pd.read_parquet(PARQUET_PATH)
    test_ids = set()
    for f in glob.glob(os.path.join(DIOR_IMG_DIR, '*.jpg')):
        for part in reversed(Path(f).stem.split('_')):
            if part.isdigit():
                test_ids.add(int(part))
                break
    name2id = {v: k for k, v in U.DIOR_ID2NAME.items()}
    target = name2id[dior_cls]

    # 每图取最大 bbox (与 bank 构建一致)
    img_best = {}
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        if img_id in test_ids:
            continue
        objs = row['objects']
        for i, cat in enumerate(objs['category']):
            if int(cat) == target:
                area = int(objs['area'][i])
                if img_id not in img_best or area > img_best[img_id][0]:
                    img_best[img_id] = (area, row, i)

    feats = []
    for img_id, (_, row, i) in img_best.items():
        try:
            pil = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
            x, y, w, h = [int(v) for v in row['objects']['bbox'][i]]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(pil.width, x + w), min(pil.height, y + h)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            f = U.encode_dinov2(pil.crop((x1, y1, x2, y2)), U.VEP_LAYER)
            feats.append(f.cpu())
        except Exception:
            continue
    return torch.stack(feats)  # [N,768] 已归一化


# ══════════════════════════ 2. 球形 k-means ══════════════════════════

def kmeans(F, k, iters=30, seed=0):
    """余弦度量球形 k-means (对归一化特征 = 常规 k-means)。k-means++ 初始化。"""
    torch.manual_seed(seed)
    n = F.shape[0]
    k = min(k, n)
    idx = [int(torch.randint(n, (1,)).item())]
    for _ in range(1, k):
        d = torch.stack([1.0 - (F @ F[i]).clamp(-1, 1) for i in idx])  # (S, n) 余弦距离
        dmin = d.min(dim=0).values
        p = (dmin + 1e-12) / (dmin.sum() + 1e-12)
        idx.append(int(torch.multinomial(p, 1).item()))
    centers = F[idx].clone()  # [k,768]
    for _ in range(iters):
        assign = (F @ centers.T).argmax(dim=1)  # 最近中心 = 最大余弦
        new = centers.clone()
        for j in range(k):
            m = assign == j
            if m.sum() > 0:
                new[j] = F[m].mean(dim=0)
                new[j] = new[j] / new[j].norm()
        if (new - centers).abs().max() < 1e-6:
            break
        centers = new
    return centers


# ══════════════════════════ 3. 域内自检 ══════════════════════════

def domain_selfcheck(F, centers):
    """leave-one-out: single(到其余均值) vs multi(到最近中心) 的余弦分布。"""
    single = []
    for i in range(len(F)):
        rest = torch.cat([F[:i], F[i + 1:]])
        p = rest.mean(dim=0)
        p = p / p.norm()
        single.append(float((F[i] @ p).item()))
    multi = (F @ centers.T).max(dim=1).values.numpy()
    return np.array(single), multi


# ══════════════════════════ 4. 多原型重打分 ══════════════════════════

def _neg_proto(dior_cls):
    """与 CDVM 一致: 与单一均值原型最接近的 top-3 其他类原型平均。"""
    bank = U.get_bank()
    pos = bank[dior_cls][U.VEP_LAYER]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][U.VEP_LAYER]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def multi_proto_rerank(images, dets, gts, centers, neg_proto=None):
    """候选框对 k 个子原型取 max 余弦, fused=sqrt(gd*vis)。同候选集, 只改排序。"""
    centers = centers.to(U.DEVICE)
    if neg_proto is not None:
        neg_proto = neg_proto.to(U.DEVICE)
    all_b, all_s = [], []
    for (boxes, scores), img in zip(dets, images):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            continue
        boxes_f, scores_f = boxes[keep], scores[keep]
        vis = []
        for box in boxes_f:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                vis.append(0.0)
                continue
            f = U.encode_dinov2(img.crop((x1, y1, x2, y2)), U.VEP_LAYER)
            pos = float((f @ centers.T).max().item())
            if neg_proto is not None:
                neg = float((f @ neg_proto).item())
                vis.append(max(0.0, pos - U.NEG_LAM * neg))
            else:
                vis.append(max(0.0, pos))
        gd = np.clip(scores_f, 1e-6, 1.0)
        vs = np.clip(np.array(vis), 1e-6, 1.0)
        fused = np.sqrt(gd * vs)
        all_b.extend(boxes_f.tolist())
        all_s.extend(fused.tolist())
    gt_flat = [g for img_gts in gts for g in img_gts]
    return U.compute_ap50(all_b, all_s, gt_flat)


# ══════════════════════════ 主流程 ══════════════════════════

def main():
    for name, dataset, cls, prefix, dior_cls in TARGETS:
        print('\n' + '=' * 70)
        print(f'[{name}]  ← DIOR {dior_cls} 原型')

        # 1. 原始特征
        F = extract_raw_feats(dior_cls)
        N = F.shape[0]
        print(f'  原始 exemplar: {N} 个 (DINOv2 L11)')

        # 2. 聚类
        centers = kmeans(F, K)
        sizes = (F @ centers.T).argmax(dim=1).bincount(minlength=K).tolist()
        print(f'  k={K} 子原型, 簇大小 = {sizes}')

        # 3. 域内自检
        single, multi = domain_selfcheck(F, centers)
        print(f'  域内自检  single(到其余均值) 中位={np.median(single):.4f}')
        print(f'            multi (到最近子原型) 中位={np.median(multi):.4f}')

        # 4. 跨域评估
        print(f'  加载 {dataset} 测试集...')
        images, gts = U.load_coco(cls) if dataset == 'coco' else U.load_dior(cls)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        # CDVM (bank 单一均值原型 + 负原型) — 对照
        ap_cdvm = U.rerank_visual(images, dets, gts, dior_cls, use_neg=True)
        print(f'  CDVM (bank 单一原型)      AP50 = {ap_cdvm:.4f}')

        neg = _neg_proto(dior_cls)
        ap_multi = multi_proto_rerank(images, dets, gts, centers, neg)
        ap_multi_raw = multi_proto_rerank(images, dets, gts, centers, None)
        print(f'  MPVM k={K} (多原型)        AP50 = {ap_multi:.4f}')
        print(f'  MPVM k={K} (无负原型)      AP50 = {ap_multi_raw:.4f}')

    print('\n✅ 多原型诊断完成')


if __name__ == '__main__':
    main()
