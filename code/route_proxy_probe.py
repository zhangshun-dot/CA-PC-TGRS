# -*- coding: utf-8 -*-
"""route_proxy_probe.py — 无标签路由代理指标探测（P10 后续）

问题: P10 的"分治"依据是目标域分离度 AUC, 而 AUC 需要目标域正负真值标签
      → 部署时不可用, 是 oracle routing。
本脚本测最值得先试的无标签判别力代理, 看它们能否【不看 AUC 真值】复现路由:
      boat→全框, car→top-20, airplane→全框。

候选无标签指标（全部只用候选框特征 + vis 分数, 不用任何 GT）:
  1. R_c       = cos(π_top20, π_all)        top-k 原型 vs 全框原型的 cosine
                假设: AUC≈0.5 → vis 无序 → top 是随机子集 → R_c 高(≈1);
                      AUC 高  → vis 有序 → top 是精选真样本 → R_c 低(<1)
                路由方向: R_c 高→用全框; R_c 低→用 top-k
  2. head_ratio = mean(top-20 vis) / mean(全 vis)   头部分数集中度
                假设: AUC 高 → 头部显著高于全体 → head_ratio 大
  3. stability  = mean pairwise cos(π_top10, π_top20, π_top50)   跨 k 子集稳定性
                假设: top-k 稳定(高) → 门控可信; 不稳定(低) → 门控噪声

对照: 真值 AUC（oracle, 仅用于验证指标能否复现它的排序, 非部署输入）

结论判据: 若 R_c / head_ratio 与 AUC 单调负/正相关, 且能区分 boat(全框)
          vs car(top-k), 则存在可行的无标签代理 → 可升贡献方向。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import unified_eval as U

LAYER = 11
N_COCO = 100
KS = [10, 20, 50]

TARGETS = [
    ('COCO-airplane', 'coco', 'airplane', 'a photo of',           'airplane'),
    ('COCO-boat',     'coco', 'boat',     'a photo of',           'ship'),
    ('COCO-car',      'coco', 'car',      'a photo of',           'vehicle'),
    ('DIOR-airplane', 'dior', 'airplane', 'an overhead image of', 'airplane'),
    ('DIOR-ship',     'dior', 'ship',     'an overhead image of', 'ship'),
    ('DIOR-vehicle',  'dior', 'vehicle',  'an overhead image of', 'vehicle'),
]


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def rank_auc(vis_all, lab_all):
    vis_all = np.asarray(vis_all); lab_all = np.asarray(lab_all)
    pos = vis_all[lab_all == 1]; neg = vis_all[lab_all == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    n_pos, n_neg = len(pos), len(neg)
    order = np.argsort(vis_all)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(vis_all))
    return (rank[lab_all == 1].mean() - (n_pos - 1) / 2) / n_neg


def proto(feats, idxs):
    idxs = [i for i in idxs if feats[i] is not None]
    if not idxs:
        return None
    p = torch.stack([feats[i] for i in idxs]).mean(dim=0)
    return p / p.norm()


def main():
    bank = U.get_bank()
    rows = {}

    for name, dataset, cls, prefix, dior_cls in TARGETS:
        print('\n' + '=' * 72)
        print(f'[{name}]')
        if dataset == 'coco':
            images, gts = U.load_coco(cls, n=N_COCO)
        else:
            images, gts = U.load_dior(cls)

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)

        dets = U.gdino_detect(images, f'{prefix} {cls}', 0.0)

        feats, vis_all, labels = [], [], []
        for (boxes, scores), img, g in zip(dets, images, gts):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            for box, sc in zip(boxes[keep], scores[keep]):
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    feats.append(None); vis_all.append(0.0); labels.append(0); continue
                f = U.encode_dinov2(img.crop((x1, y1, x2, y2)), LAYER)
                v = max(0.0, float((f @ P_src).item()) - U.NEG_LAM * float((f @ neg).item()))
                lab = 1 if any(U.iou([x1, y1, x2, y2], gt) >= 0.5 for gt in g) else 0
                feats.append(f); vis_all.append(v); labels.append(lab)

        vis_all = np.array(vis_all)
        labels = np.array(labels)
        n = len(vis_all)
        all_idx = list(range(n))
        order = np.argsort(-vis_all)

        # oracle 对照: 真值 AUC
        auc = rank_auc(vis_all, labels)

        # 无标签指标
        P_all = proto(feats, all_idx)
        protos = {}
        for k in KS:
            protos[k] = proto(feats, order[:k])

        R_c = float((protos[20] @ P_all).item()) if (protos[20] is not None and P_all is not None) else float('nan')

        head_ratio = float(vis_all[order[:20]].mean() / max(vis_all.mean(), 1e-8))

        cos_vals = []
        pks = [protos[k] for k in KS]
        for i in range(len(pks)):
            for j in range(i + 1, len(pks)):
                if pks[i] is not None and pks[j] is not None:
                    cos_vals.append(float((pks[i] @ pks[j]).item()))
        stability = float(np.mean(cos_vals)) if cos_vals else float('nan')

        print(f'  AUC(oracle)={auc:.4f}  R_c={R_c:.4f}  head_ratio={head_ratio:.3f}  stability={stability:.4f}')
        rows[name] = {'auc': round(auc, 4), 'R_c': round(R_c, 4),
                      'head_ratio': round(head_ratio, 3), 'stability': round(stability, 4)}

    print('\n\n' + '=' * 72)
    print('汇总（oracle 路由 vs 无标签指标）:')
    print(f'{"类":<16s}{"AUC(真值)":>12s}{"R_c":>10s}{"head_ratio":>12s}{"stability":>12s}   oracle路由')
    oracle_route = {
        'COCO-airplane': '全框', 'COCO-boat': '全框', 'COCO-car': 'top-20',
        'DIOR-airplane': '-', 'DIOR-ship': '-', 'DIOR-vehicle': '-',
    }
    for name, r in rows.items():
        print(f'{name:<16s}{r["auc"]:>12.4f}{r["R_c"]:>10.4f}{r["head_ratio"]:>12.3f}'
              f'{r["stability"]:>12.4f}   {oracle_route[name]}')

    SAVE = Path(r'C:\Users\张三\Desktop\实验数据') / 'route_proxy_probe.json'
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {SAVE}')
    print('✅ 无标签路由代理探测完成')


if __name__ == '__main__':
    main()
