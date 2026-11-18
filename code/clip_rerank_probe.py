# -*- coding: utf-8 -*-
"""clip_rerank_probe.py — CLIP 视觉空间重打分探测

回答关键问题: boat 的 0.53 跨域 gap 是否来自「编码器错配」(DINOv2 是外来编码器,
GDINO 内部是 CLIP/BERT 对齐)。用 CLIP 视觉空间重打分, 对比 DINOv2 L11/L12:
  · CLIP AUC 明显 > DINOv2 L11(0.49) → boat 有救, 问题在选错编码器
  · CLIP AUC ≈ 0.5             → boat 真无解, 接受 negative result
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

CLIP_BANK = r'C:\Users\张三\Desktop\实验数据\exemplar_bank_k5.pt'

TARGETS = [
    ('COCO-boat',    'coco', 'boat',    'a photo of',           'ship'),
    ('COCO-car',     'coco', 'car',     'a photo of',           'vehicle'),
    ('DIOR-vehicle', 'dior', 'vehicle', 'an overhead image of', 'vehicle'),
]


def auc_sep(vis, lab):
    vis = np.array(vis)
    lab = np.array(lab)
    pos = vis[lab == 1]
    neg = vis[lab == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(vis)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(vis))
    return (rank[lab == 1].mean() - (len(pos) - 1) / 2) / len(neg)


def main():
    clip_m, clip_pre = U.get_clip()
    cbank = torch.load(CLIP_BANK, map_location='cpu')

    for name, dataset, cls, prefix, dior_cls in TARGETS:
        pos = cbank[dior_cls]['prototype'].to(U.DEVICE).float()
        others = torch.stack([cbank[c]['prototype'] for c in cbank if c != dior_cls]).to(U.DEVICE).float()
        neg = others[(others @ pos).topk(3).indices].mean(dim=0)
        neg = neg / neg.norm()

        images, gts = U.load_coco(cls) if dataset == 'coco' else U.load_dior(cls)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        boxes_all, gd_all, vis_all, lab_all = [], [], [], []
        for (boxes, scores), img, g in zip(dets, images, gts):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                t = clip_pre(crop).unsqueeze(0).to(U.DEVICE)
                with torch.no_grad():
                    f = clip_m.encode_image(t).float()
                f = f / f.norm(dim=-1, keepdim=True)
                p = float((f @ pos).item())
                n = float((f @ neg).item())
                vis = max(0.0, p - U.NEG_LAM * n)
                lab = 1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in g) else 0
                boxes_all.append([x1, y1, x2, y2])
                gd_all.append(float(sc))
                vis_all.append(vis)
                lab_all.append(lab)

        auc = auc_sep(vis_all, lab_all)
        gd = np.clip(np.array(gd_all), 1e-6, 1.0)
        vs = np.clip(np.array(vis_all), 1e-6, 1.0)
        gt_flat = [g for gs in gts for g in gs]
        ap = U.compute_ap50(boxes_all, np.sqrt(gd * vs), gt_flat)
        print(f'\n[{name}]  (CLIP 512维, {len(boxes_all)} 框)')
        print(f'  AUC  = {auc:.4f}')
        print(f'  AP50 = {ap:.4f}')

    print('\n✅ CLIP 探测完成')


if __name__ == '__main__':
    main()
