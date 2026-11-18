# -*- coding: utf-8 -*-
"""pseudo_label_quality.py — 伪标签质量探测（无标注目标域适配方向1的前置验证）

问题: 方向1（无标注目标域区域原型自适应）用"高置信伪标签"更新目标域原型,
      但 boat 这类形态鸿沟类的分离度 AUC≈0.5, 伪标签正确率是否也≈随机?
      若≈随机, EMA 平均的原型 = 0.5·真船 + 0.5·噪声, 方向1在形态鸿沟类上
      退化成"垃圾进垃圾出", 救不回 boat。

三层度量（跨域 3 类 COCO-airplane/boat/car + 同域对照 DIOR-vehicle）:
  A. 判别力:   vis 分数分离度 AUC（1=完美, 0.5=随机）—— 复现 diagnose_weak 并扩到 L12
  B. 正确率:   precision@k —— 按分数( gd / vis / fused )取全局 top-k 框,
               命中任一 GT(IoU≥0.5) 的比例。few-shot 真标注正确率恒为 1.0,
               伪标签正确率 P 直接决定原型里真信号占比。
  C. 原型质量: 伪标签原型 vs 真 GT 原型 的 cosine
     - cos_src:    源域原型 vs 真 GT（boat 已知≈0.49）
     - cos_pseudo: top-k vis 分数框原型 vs 真 GT（> cos_src 才算有增益）
     - cos_all:    全部候选框原型 vs 真 GT（= 随机伪标签的期望下界）

判据: 若 boat 的 precision@10 ≈ 0.5 且 cos_pseudo ≈ cos_all,
      则无标注伪标签在形态鸿沟类上不可靠 → 方向1需另找"独立于源域原型的
      目标域内部判别信号"（聚类簇结构 / 几何先验 / 分布几何）。

协议与最终方法一致: dinov2-base L12 (layer_idx=11) + CDVM(负原型相对化)。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import unified_eval as U

LAYER = 11           # dinov2-base L12（最终方法层, bank 索引 0-11）
N_COCO = 100         # 与 few_shot_calib 一致
KS = [1, 5, 10, 20, 50, 100]

TARGETS = [
    ('COCO-airplane', 'coco', 'airplane', 'a photo of',           'airplane'),
    ('COCO-boat',     'coco', 'boat',     'a photo of',           'ship'),
    ('COCO-car',      'coco', 'car',      'a photo of',           'vehicle'),
    ('DIOR-vehicle',  'dior', 'vehicle',  'an overhead image of', 'vehicle'),
]


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def gt_proto(images, gts):
    """真 GT 框的 L12 特征均值 → 目标域真实原型。"""
    feats = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            feats.append(U.encode_dinov2(img.crop((x1, y1, x2, y2)), LAYER))
    if not feats:
        return None
    p = torch.stack(feats).mean(dim=0)
    return p / p.norm()


def rank_auc(vis_all, lab_all):
    """vis 分数区分正负例的 AUC（与 diagnose_weak 同口径）。"""
    vis_all = np.asarray(vis_all); lab_all = np.asarray(lab_all)
    pos = vis_all[lab_all == 1]; neg = vis_all[lab_all == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan'), len(pos), len(neg)
    n_pos, n_neg = len(pos), len(neg)
    order = np.argsort(vis_all)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(vis_all))
    auc = (rank[lab_all == 1].mean() - (n_pos - 1) / 2) / n_neg
    return auc, n_pos, n_neg


def precision_at_k(order_idx, labels, k):
    top = order_idx[:k]
    return float(labels[top].mean())


def proto_from_rows(feats_list, idxs):
    if len(idxs) == 0:
        return None
    p = torch.stack([feats_list[i] for i in idxs]).mean(dim=0)
    return p / p.norm()


def main():
    bank = U.get_bank()
    report = {}

    for name, dataset, cls, prefix, dior_cls in TARGETS:
        print('\n' + '=' * 72)
        print(f'[{name}]')
        if dataset == 'coco':
            images, gts = U.load_coco(cls, n=N_COCO)
        else:
            images, gts = U.load_dior(cls)
        n_gt = sum(len(g) for g in gts)
        print(f'  {len(images)} 图, {n_gt} GT 框')

        # 源域原型 + 负原型 + 真目标域原型
        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)
        P_tgt = gt_proto(images, gts)

        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        # 逐框: 编码 + 三种分数 + 正负例标签
        gd_all, vis_all, labels = [], [], []
        feats_list = []
        for (boxes, scores), img, g in zip(dets, images, gts):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                f = U.encode_dinov2(img.crop((x1, y1, x2, y2)), LAYER)
                vis = max(0.0, float((f @ P_src).item()) - U.NEG_LAM * float((f @ neg).item()))
                lab = 1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in g) else 0
                feats_list.append(f)
                gd_all.append(float(sc))
                vis_all.append(vis)
                labels.append(lab)

        gd_all = np.array(gd_all)
        vis_all = np.array(vis_all)
        labels = np.array(labels)
        fused_all = np.sqrt(np.clip(gd_all, 1e-6, 1.0) * np.clip(vis_all, 1e-6, 1.0))

        n_box = len(labels)
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
        print(f'  候选框 {n_box}（正 {n_pos} / 负 {n_neg}，正例占比 {n_pos/max(1,n_box):.3f}）')

        # A. 判别力 AUC
        auc, np_, nn_ = rank_auc(vis_all, labels)
        print(f'  [A] vis 分离度 AUC = {auc:.4f}（{np_}正/{nn_}负, 0.5=随机）')

        # B. precision@k（三种分数源）
        print(f'  [B] precision@k（top-k 框命中 GT 的比例）:')
        print(f'      {"k":>6s}' + ''.join(f'{k:>8d}' for k in KS))
        for sname, sarr in [('gd', gd_all), ('vis', vis_all), ('fused', fused_all)]:
            order = np.argsort(-sarr)
            line = '      {:>6s}'.format(sname)
            for k in KS:
                kk = min(k, n_box)
                line += f'{precision_at_k(order, labels, kk):>8.3f}'
            print(line)
        # few-shot 真标注对照（正确率恒 1.0）
        print(f'      {"few-shot(真标注)":<14s}正确率恒为 1.0（K=10 已够, 见 few_shot_calib）')

        # C. 原型质量
        print(f'  [C] 原型 vs 真 GT 原型 的 cosine:')
        if P_tgt is not None:
            cos_src = float((P_src @ P_tgt).item())
            print(f'      cos_src(源域原型)   = {cos_src:.4f}')
            all_idx = list(range(n_box))
            cos_all = float((proto_from_rows(feats_list, all_idx) @ P_tgt).item()) if n_box else float('nan')
            print(f'      cos_all(全体框=随机期望) = {cos_all:.4f}')
            for k in [10, 20, 50]:
                kk = min(k, n_box)
                order = np.argsort(-vis_all)
                p_pseudo = proto_from_rows(feats_list, order[:kk])
                cos_pseudo = float((p_pseudo @ P_tgt).item()) if p_pseudo is not None else float('nan')
                flag = ' ↑有增益' if cos_pseudo > cos_src else ''
                print(f'      cos_pseudo(top-{k} vis) = {cos_pseudo:.4f}{flag}')
        else:
            print('      真 GT 原型缺失, 跳过')

        report[name] = {
            'n_box': n_box, 'n_pos': n_pos, 'n_neg': n_neg,
            'auc': round(auc, 4),
            'precision_at_k': {s: {str(k): round(precision_at_k(
                np.argsort(-arr), labels, min(k, n_box)), 4) for k in KS}
                for s, arr in [('gd', gd_all), ('vis', vis_all), ('fused', fused_all)]},
        }

    out = SAVE = Path(r'C:\Users\张三\Desktop\实验数据') / 'pseudo_label_quality.json'
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {SAVE}')
    print('✅ 伪标签质量探测完成')


if __name__ == '__main__':
    main()
