# -*- coding: utf-8 -*-
"""cross_domain_gap_scan.py — 定位跨域 gap 是否 DINOv2 层特异

难类病根已确认 = 跨域 gap（DIOR 俯视船 vs COCO 侧视船，余弦差 0.53）。
本脚本扫 DINOv2 12 层下, 难类候选框的分离度 AUC:
  · 某层 AUC 显著高(>0.6) → gap 是 L11 层特异, 换层即救
  · 各层 AUC 都 ≈0.5      → gap 语义级, 视觉空间无解, 需文本锚定/few-shot
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

TARGETS = [
    ('COCO-boat',    'coco', 'boat',    'a photo of',           'ship'),     # 跨域难类
    ('DIOR-vehicle', 'dior', 'vehicle', 'an overhead image of', 'vehicle'),  # 同域易类(对照)
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


def encode_all_layers(crop):
    """一次 forward 返回 12 层 CLS 特征(未归一化)。"""
    dino, pre = U.get_dinov2()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = dino(t)
    return [o.hidden_states[l + 1][0, 0, :] for l in range(12)]


def main():
    for name, dataset, cls, prefix, dior_cls in TARGETS:
        print('\n' + '=' * 70)
        print(f'[{name}]  ← DIOR {dior_cls} 原型')
        images, gts = U.load_coco(cls) if dataset == 'coco' else U.load_dior(cls)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)
        bank = U.get_bank()
        protos = {l: bank[dior_cls][l]['prototype'].to(U.DEVICE) for l in range(12)}

        layer_vis = {l: [] for l in range(12)}
        layer_lab = {l: [] for l in range(12)}
        for (boxes, scores), img, g in zip(dets, images, gts):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box in boxes[keep]:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                feats = encode_all_layers(img.crop((x1, y1, x2, y2)))
                lab = 1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in g) else 0
                for l in range(12):
                    f = feats[l] / feats[l].norm()
                    vis = float((f @ protos[l]).item())
                    layer_vis[l].append(vis)
                    layer_lab[l].append(lab)

        print(f'  {"层":>4s} {"AUC":>8s}  难类看 AUC 能否 >0.6')
        for l in range(12):
            auc = auc_sep(layer_vis[l], layer_lab[l])
            bar = '█' * int(max(0.0, (auc - 0.4) / 0.6 * 40)) if auc == auc else ''
            print(f'  L{l+1:>2d}  {auc:>8.4f}  {bar}')

    print('\n✅ 层扫描完成')


if __name__ == '__main__':
    main()
