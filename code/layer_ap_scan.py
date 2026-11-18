# -*- coding: utf-8 -*-
"""layer_ap_scan.py — 逐层 CDVM 重打分的 AP50，定位难类最佳层

层扫描(cross_domain_gap_scan)已发现: 难类(COCO-boat)在浅层 L2 的分离度 AUC=0.62,
远高于我们一直用的 L11(0.45)。本脚本直接把「每层 CDVM 重打分」的 AP50 测出来,
看难类换浅层能提升多少。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

# 全 6 类: 补全强类, 做 per-class 层曲线
TARGETS = [(n, d, c, p, dc) for (n, d, c, dom, p, dc) in U.EVALS]


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def encode_all_layers(crop):
    dino, pre = U.get_dinov2()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = dino(t)
    return [o.hidden_states[l + 1][0, 0, :] for l in range(12)]


def main():
    bank = U.get_bank()
    for name, dataset, cls, prefix, dior_cls in TARGETS:
        print('\n' + '=' * 70)
        print(f'[{name}]  ← DIOR {dior_cls} 原型')
        images, gts = U.load_coco(cls) if dataset == 'coco' else U.load_dior(cls)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        boxes_all, gd_all, feats_all = [], [], []
        for (boxes, scores), img in zip(dets, images):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                feats = [f / f.norm() for f in encode_all_layers(img.crop((x1, y1, x2, y2)))]
                boxes_all.append([x1, y1, x2, y2])
                gd_all.append(float(sc))
                feats_all.append(feats)

        gt_flat = [g for img_gts in gts for g in img_gts]
        gd = np.clip(np.array(gd_all), 1e-6, 1.0)
        print(f'  {len(boxes_all)} 候选框, {len(gt_flat)} GT')

        aps = []
        for l in range(12):
            pos = bank[dior_cls][l]['prototype'].to(U.DEVICE)
            neg = get_neg_proto(bank, dior_cls, l)
            vis = []
            for feats in feats_all:
                f = feats[l]
                p = float((f @ pos).item())
                n = float((f @ neg).item())
                vis.append(max(0.0, p - U.NEG_LAM * n))
            vs = np.clip(np.array(vis), 1e-6, 1.0)
            ap = U.compute_ap50(boxes_all, np.sqrt(gd * vs), gt_flat)
            aps.append(ap)
            print(f'  L{l+1:>2d}  AP50 = {ap:.4f}')

        best = int(np.argmax(aps))
        print(f'  → 最佳层 L{best+1} (AP50={aps[best]:.4f}), L11(旧)={aps[10]:.4f}, '
              f'提升 {aps[best]-aps[10]:+.4f}')

    print('\n✅ 逐层 AP50 扫描完成')


if __name__ == '__main__':
    main()
