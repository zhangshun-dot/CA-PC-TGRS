# -*- coding: utf-8 -*-
"""diagnose_weak.py — 诊断难类为什么 CDVM 增益弱

两个互斥假设:
  H1 图级偏移: 域漂移是每张图的全局平移 → 框间相对序仍对, 图内相对化能救
  H2 原型覆盖不足: 难类类内多样性大, 单一均值原型对正例框也无判别力 → 需多原型

诊断输出:
  1) 正例框(IoU≥0.5) vs 负例框 的视觉分数分布 (均值/std/分离度AUC)
     - 完全重叠 → H2 (原型无判别力)
     - 正例略高但整体偏低 → H1 (图级偏移)
  2) 图内相对化 (去均值 / softmax) 后的 AP50, 对比原始 CDVM
     - 回升明显 → H1 成立, 该上相对化
     - 无变化 → H2 成立, 该上多原型
"""

import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import unified_eval as U

# 诊断对象: 一个难类 + 一个易类对照
TARGETS = [
    ('COCO-boat',   'coco', 'boat',   'a photo of',           'ship'),     # 难
    ('DIOR-vehicle', 'dior', 'vehicle', 'an overhead image of', 'vehicle'), # 易(对照)
]


def vis_scores_for(images, dets, gts, dior_cls):
    """返回每图每个候选框的 vis 原始分数 + 正/负例标签。"""
    bank = U.get_bank()
    pos_proto = bank[dior_cls][U.VEP_LAYER]['prototype'].to(U.DEVICE)

    # 类间负原型
    others = []
    for cls in bank:
        if cls != dior_cls:
            others.append(bank[cls][U.VEP_LAYER]['prototype'])
    others = torch.stack(others).to(U.DEVICE)
    topk = (others @ pos_proto).topk(3).indices
    neg_proto = others[topk].mean(dim=0)
    neg_proto = neg_proto / neg_proto.norm()

    per_image = []  # [(vis_list, label_list, gd_list)]
    for (boxes, scores), img, g in zip(dets, images, gts):
        keep = scores > U.PRE_FILTER
        if not keep.any():
            per_image.append(([], [], []))
            continue
        boxes_f, scores_f = boxes[keep], scores[keep]
        vis_list, label_list, gd_list = [], [], []
        for box in boxes_f:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                vis_list.append(0.0); label_list.append(0); gd_list.append(0.0)
                continue
            f = U.encode_dinov2(img.crop((x1, y1, x2, y2)), U.VEP_LAYER)
            pos = float((f @ pos_proto).item())
            neg = float((f @ neg_proto).item())
            vis = max(0.0, pos - U.NEG_LAM * neg)
            vis_list.append(vis)
            # 正例 = 与任一 GT IoU>=0.5
            lab = 0
            for gt in g:
                if U.iou([x1, y1, x2, y2], gt) >= 0.5:
                    lab = 1
                    break
            label_list.append(lab)
            gd_list.append(float(scores_f[len(gd_list)]))
        per_image.append((vis_list, label_list, gd_list))
    return per_image, pos_proto, neg_proto


def auc_sep(vis_all, lab_all):
    """vis 分数区分正负例的 AUC (1=完美, 0.5=随机)。"""
    vis_all = np.array(vis_all); lab_all = np.array(lab_all)
    pos = vis_all[lab_all == 1]; neg = vis_all[lab_all == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan'), len(pos), len(neg)
    # 简单 rank AUC
    n_pos, n_neg = len(pos), len(neg)
    ranks = (vis_all[:, None] > vis_all[None, :]).sum(axis=1)  # 每个样本胜过多少个
    # 更简单: 用正例的平均 rank
    order = np.argsort(vis_all)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(vis_all))
    auc = (rank[lab_all == 1].mean() - (n_pos - 1) / 2) / n_neg
    return auc, n_pos, n_neg


def ap_with_relative(images, dets, gts, dior_cls, mode):
    """mode: 'raw' | 'demean' | 'softmax'。图内相对化的 CDVM AP50。"""
    bank = U.get_bank()
    pos_proto = bank[dior_cls][U.VEP_LAYER]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][U.VEP_LAYER]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg_proto = others[(others @ pos_proto).topk(3).indices].mean(dim=0)
    neg_proto = neg_proto / neg_proto.norm()

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
                vis.append(0.0); continue
            f = U.encode_dinov2(img.crop((x1, y1, x2, y2)), U.VEP_LAYER)
            vis.append(max(0.0, float((f @ pos_proto).item()) - U.NEG_LAM * float((f @ neg_proto).item())))
        vis = np.array(vis)
        if mode == 'demean':
            vis = vis - vis.mean()
        elif mode == 'softmax':
            vis = np.exp((vis - vis.max()) / 0.1)
            vis = vis / vis.sum()
        gd = np.clip(scores_f, 1e-6, 1.0)
        fused = np.sqrt(gd * np.clip(vis, 1e-6, None))
        all_b.extend(boxes_f.tolist())
        all_s.extend(fused.tolist())
    gt_flat = [g for img_gts in gts for g in img_gts]
    return U.compute_ap50(all_b, all_s, gt_flat)


def main():
    for name, dataset, cls, prefix, dior_cls in TARGETS:
        print('\n' + '=' * 70)
        print(f'[{name}]')
        if dataset == 'coco':
            images, gts = U.load_coco(cls)
        else:
            images, gts = U.load_dior(cls)
        print(f'  {len(images)} 图, {sum(len(g) for g in gts)} GT')

        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)
        per_image, _, _ = vis_scores_for(images, dets, gts, dior_cls)

        vis_all = [v for (v, l, g) in per_image for v in v]
        lab_all = [l for (v, l, g) in per_image for l in l]
        vis_all = np.array(vis_all); lab_all = np.array(lab_all)
        pos = vis_all[lab_all == 1]; neg = vis_all[lab_all == 0]
        auc, np_, nn = auc_sep(vis_all, lab_all)
        print(f'  正例框 {np_} 个: vis 均值={pos.mean():.4f} std={pos.std():.4f}')
        print(f'  负例框 {nn} 个: vis 均值={neg.mean():.4f} std={neg.std():.4f}')
        print(f'  分离度 AUC = {auc:.4f} (1=完美, 0.5=随机)')

        for mode in ['raw', 'demean', 'softmax']:
            ap = ap_with_relative(images, dets, gts, dior_cls, mode)
            print(f'  CDVM[{mode:<7s}] AP50 = {ap:.4f}')

    print('\n诊断完成')


if __name__ == '__main__':
    main()
