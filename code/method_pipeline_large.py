# -*- coding: utf-8 -*-
"""method_pipeline_large.py — CDVM 方法主表（大样本版, COCO 100 张/类）

与 method_pipeline.py 差异: COCO 扩到 100 张/类(airplane 97 全用),
DIOR 仍 20 张/类(本地无完整测试集)。

pipeline ablative:
  baseline  — GDINO 原始分数
  VEP-L12   — 正原型重打分(无负), L12
  CDVM-L12  — 正+负原型, L12, λ ∈ {0.3, 0.5, 0.7}
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

LAYER = 11  # L12 (0-indexed)
N_COCO_TARGET = 100
LAMS = [0.3, 0.5, 0.7]


def get_protos(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return pos, neg / neg.norm()


def encode_l12(crop):
    dino, pre = U.get_dinov2()
    t = pre(crop).unsqueeze(0).to(U.DEVICE)
    with torch.no_grad():
        o = dino(t)
    f = o.hidden_states[LAYER + 1][0, 0, :]
    return f / f.norm()


def compute_ap_from_vis(boxes_all, gd_all, vis_all, gt_flat):
    gd = np.clip(np.array(gd_all), 1e-6, 1.0)
    vs = np.clip(np.array(vis_all), 1e-6, 1.0)
    return U.compute_ap50(boxes_all, np.sqrt(gd * vs), gt_flat)


def main():
    bank = U.get_bank()
    summary = {}
    for name, dataset, cls, domain, prefix, dior_cls in U.EVALS:
        print('\n' + '=' * 70)
        print(f'[{name}]')
        images, gts = U.load_coco(cls, n=N_COCO_TARGET) if dataset == 'coco' else U.load_dior(cls)
        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        bb, bs, bgt = U.filter_flatten(dets, gts)
        base = U.compute_ap50(bb, bs, bgt)
        row = {'baseline': base}
        print(f'  baseline          = {base:.4f}')

        pos, neg = get_protos(bank, dior_cls, LAYER)

        boxes_all, gd_all, feats = [], [], []
        for (boxes, scores), img in zip(dets, images):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                feats.append(encode_l12(img.crop((x1, y1, x2, y2))))
                boxes_all.append([x1, y1, x2, y2])
                gd_all.append(float(sc))
        gt_flat = [g for gs in gts for g in gs]

        vis_vep = [max(0.0, float((f @ pos).item())) for f in feats]
        row['vep'] = compute_ap_from_vis(boxes_all, gd_all, vis_vep, gt_flat)
        print(f'  VEP  L12          = {row["vep"]:.4f}')

        for lam in LAMS:
            vis = [max(0.0, float((f @ pos).item()) - lam * float((f @ neg).item()))
                   for f in feats]
            row[f'cdvm_l{lam}'] = compute_ap_from_vis(boxes_all, gd_all, vis, gt_flat)
            print(f'  CDVM L12 λ={lam}    = {row[f"cdvm_l{lam}"]:.4f}')

        summary[name] = row

    cols = ['baseline', 'vep', 'cdvm_l0.3', 'cdvm_l0.5', 'cdvm_l0.7']
    print('\n\n' + '=' * 90)
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols))
    print('-' * 90)
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]:>12.4f}' for c in cols))
    print('-' * 90)
    for label, sel in [('COCO 均值', [n for n in summary if n.startswith('COCO')]),
                       ('DIOR 均值', [n for n in summary if n.startswith('DIOR')]),
                       ('总均值', list(summary))]:
        print(f'{label:<16s}' + ''.join(
            f'{np.mean([summary[n][c] for n in sel]):>12.4f}' for c in cols))

    print('\nΔ vs baseline:')
    print(f'{"Test":<16s}' + ''.join(f'{c:>12s}' for c in cols[1:]))
    for name, r in summary.items():
        print(f'{name:<16s}' + ''.join(f'{r[c]-r["baseline"]:>+12.4f}' for c in cols[1:]))

    print('\n✅ 大样本主表完成')


if __name__ == '__main__':
    main()
