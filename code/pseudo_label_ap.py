# -*- coding: utf-8 -*-
"""pseudo_label_ap.py — 伪标签原型的下游 AP 验证（C 层, 最终答案）

A/B 层（pseudo_label_quality.py）发现: boat 的 vis 分离度 AUC≈0.48(≈随机),
precision@k 低(0.2~0.6), 但伪标签原型 vs 真 GT 的 cosine 仍远超源域原型
(cos 0.79~0.81 vs 0.52)。这提示: 形态鸿沟的修复可能不依赖"伪标签正确率",
而依赖"伪标签框在目标域视觉空间"。

但 cosine 高 ≠ AP 升。CDVM 的判别力取决于"原型对正例 vs 负例 的 gap",
而不是"原型离真 GT 的 cosine"。若伪标签原型(含大量负例)把正负例一起拉高,
区分度没变, AP 不升反降。

本脚本直接测 AP, 回答: 无标注伪标签原型 能否替代 few-shot 真标注, 修复形态鸿沟类?

协议(与 few_shot_calib.py 一致):
  - 跨域 COCO 每类 n=100, 前 K=10 张作 few-shot 校准集, 后 90 张作测试集
  - 全部 AP 在【同一测试集】评估, 只改"原型来源"
  - 原型来源:
      P_src   = 源域 DIOR bank 原型          (零样本基线)
      P_tgt   = few-shot 集 真 GT 框均值      (有监督上界)
      P_all   = 测试集 全体候选框 特征均值    (无标注 transductive, 最简单)
      P_top20 = 测试集 top-20 vis 框 特征均值 (无标注 + 门控)
  - 对每种原型扫 α∈{1, 0.5, 0}: P_cal = normalize(α·P_src + (1-α)·P_proto)
  - CDVM 负原型抑制 NEG_LAM=0.5, dinov2-base L12
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import torch
import unified_eval as U

LAYER = 11
K = 10
N_COCO = 100
ALPHAS = [1.0, 0.5, 0.0]

TARGETS = [
    ('COCO-airplane', 'airplane', 'airplane', 'a photo of'),
    ('COCO-boat',     'boat',     'ship',                         'a photo of'),
    ('COCO-car',      'car',      'vehicle',                      'a photo of'),
]


def get_neg_proto(bank, dior_cls, layer):
    pos = bank[dior_cls][layer]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][layer]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return neg / neg.norm()


def encode(crop):
    return U.encode_dinov2(crop, LAYER)


def feats_from_boxes(img, boxes):
    """从一张图的候选框提取 L12 特征。"""
    feats = []
    for box in boxes:
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        x2, y2 = min(img.width, x2), min(img.height, y2)
        if x2 <= x1 + 10 or y2 <= y1 + 10:
            feats.append(None)
        else:
            feats.append(encode(img.crop((x1, y1, x2, y2))))
    return feats


def norm(p):
    return p / p.norm()


def cdvm_ap(boxes_flat, feats_flat, gd_flat, P_cal, neg, gt_flat):
    """用 P_cal 做 CDVM 重打分, 算数据集级 AP50。"""
    fused = []
    for f, gd in zip(feats_flat, gd_flat):
        if f is None:
            fused.append(0.0)
        else:
            vis = max(0.0, float((f @ P_cal).item()) - U.NEG_LAM * float((f @ neg).item()))
            fused.append(float(np.sqrt(np.clip(gd, 1e-6, 1.0) * np.clip(vis, 1e-6, 1.0))))
    return U.compute_ap50(boxes_flat, fused, gt_flat)


def main():
    bank = U.get_bank()
    report = {}

    for name, cls, dior_cls, prefix in TARGETS:
        print('\n' + '=' * 72)
        print(f'[{name}]')
        images, gts = U.load_coco(cls, n=N_COCO)
        fs_imgs, fs_gts = images[:K], gts[:K]
        test_imgs, test_gts = images[K:], gts[K:]
        print(f'  few-shot {len(fs_imgs)} 图, 测试 {len(test_imgs)} 图')

        P_src = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
        neg = get_neg_proto(bank, dior_cls, LAYER)

        # few-shot 真 GT 原型（上界）
        fs_feats = []
        for img, g in zip(fs_imgs, fs_gts):
            for gt in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                fs_feats.append(encode(img.crop((x1, y1, x2, y2))))
        P_tgt = norm(torch.stack(fs_feats).mean(dim=0)) if fs_feats else None
        print(f'  源域-目标域原型 cosine = {float(P_src @ P_tgt):.4f}'
              if P_tgt is not None else '  P_tgt 缺失')

        # 测试集检测 + 编码（复用同一候选框集，只改原型）
        dets = U.gdino_detect(test_imgs, f'{prefix} {cls}', 0.0)
        boxes_flat, feats_flat, gd_flat, vis_flat = [], [], [], []
        for (boxes, scores), img in zip(dets, test_imgs):
            keep = scores > U.PRE_FILTER
            if not keep.any():
                continue
            bf = boxes[keep]
            sf = scores[keep]
            ff = feats_from_boxes(img, bf)
            for box, sc, f in zip(bf, sf, ff):
                boxes_flat.append([max(0, int(v)) for v in box])
                gd_flat.append(float(sc))
                feats_flat.append(f)
                if f is None:
                    vis_flat.append(0.0)
                else:
                    v = max(0.0, float((f @ P_src).item()) - U.NEG_LAM * float((f @ neg).item()))
                    vis_flat.append(v)
        gt_flat = [g for gs in test_gts for g in gs]

        # 伪标签原型: P_all（全体候选框） / P_top20（top-20 vis）
        valid_idx = [i for i, f in enumerate(feats_flat) if f is not None]
        P_all = norm(torch.stack([feats_flat[i] for i in valid_idx]).mean(dim=0)) if valid_idx else None
        order = np.argsort(-np.array(vis_flat))
        top20 = [i for i in order[:20] if feats_flat[i] is not None]
        P_top20 = norm(torch.stack([feats_flat[i] for i in top20]).mean(dim=0)) if top20 else None

        # baseline = source-only（α=1）
        base = cdvm_ap(boxes_flat, feats_flat, gd_flat, P_src, neg, gt_flat)
        print(f'  [基线] source-only (α=1)     AP50 = {base:.4f}')

        row = {'source_only': base}
        for pname, P_proto in [('few-shot真GT', P_tgt), ('伪标签全框', P_all), ('伪标签top20', P_top20)]:
            if P_proto is None:
                continue
            line = f'  [{pname}]'
            for alpha in [0.5, 0.0]:
                P_cal = norm(alpha * P_src + (1 - alpha) * P_proto)
                ap = cdvm_ap(boxes_flat, feats_flat, gd_flat, P_cal, neg, gt_flat)
                key = f'{pname}_a{alpha}'
                row[key] = ap
                line += f'  α={alpha}: {ap:.4f}'
            print(line)
        report[name] = row

    print('\n\n' + '=' * 72)
    print('汇总（Δ vs source-only）:')
    for name, r in report.items():
        b = r['source_only']
        print(f'  {name:<14s} src={b:.4f}' + ''.join(
            f'  {k}={v:.4f}({v-b:+.4f})' for k, v in r.items() if k != 'source_only'))

    SAVE = Path(r'C:\Users\张三\Desktop\实验数据') / 'pseudo_label_ap.json'
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {SAVE}')
    print('✅ 伪标签原型 AP 验证完成')


if __name__ == '__main__':
    main()
