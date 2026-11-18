# -*- coding: utf-8 -*-
"""single_scenario_weight_probe.py — 降维验证: 在特定场景确定因子权重 → 看 mAP 是否涨

严格复用 unified_eval.py 统一协议:
  - 候选框集 = GDINO th=0.0 → pre-filter=0.03, baseline 与重打分类同一候选集
  - 两套因子(分子): gd=GDINO 置信度, vis=DINOv2 L11 正原型余弦
  - 融合 = 广义几何平均 fused = gd^α · vis^(1-α), α∈[0,0.25,0.5,0.75,0.9,1]
      α=1 → 纯 gd (baseline 排序); α=0.5 → 写死 sqrt(gd·vis)
  - 指标 = 论文正确口径 compute_ap50_img (逐图匹配 11-point AP50)
  - 每场景报告 baseline / 各α / 最优α / Δ最优 / Δ写死0.5

性能: 每场景裁剪框只编码一次 vis, α 扫描复用(不再重复 DINOv2 前向)。
"""
import os, sys, json
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from unified_eval import (get_gdino, get_dinov2, get_bank, gdino_detect, encode_dinov2,
                          compute_ap50_img, load_coco, load_dior, PRE_FILTER, VEP_LAYER,
                          EVALS, DEVICE, SAVE_DIR)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


def prep_per_img(images, base_dets, dior_cls):
    """每场景只编码一次: 返回逐图 (boxes_f, gd_clip, vis) 列表, 已 pre-filter。"""
    bank = get_bank()
    pos_proto = bank[dior_cls][VEP_LAYER]['prototype'].to(DEVICE)
    out = []
    for (boxes, scores), img in zip(base_dets, images):
        keep = scores > PRE_FILTER
        if not keep.any():
            out.append((np.empty((0, 4)), np.empty((0,)), np.empty((0,))))
            continue
        boxes_f = boxes[keep]
        scores_f = scores[keep]
        vis = []
        for box in boxes_f:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                vis.append(0.0)
                continue
            f = encode_dinov2(img.crop((x1, y1, x2, y2)), VEP_LAYER)
            vis.append(max(0.0, float((f @ pos_proto).item())))
        gd = np.clip(scores_f, 1e-6, 1.0)
        vv = np.clip(np.array(vis), 1e-6, 1.0)
        out.append((boxes_f, gd, vv))
    return out


def ap_at_alpha(prep, alpha, gts):
    pi = []
    for (boxes_f, gd, vv) in prep:
        if len(boxes_f) == 0:
            pi.append((boxes_f, np.empty((0,))))
            continue
        if alpha <= 0.0:
            fused = vv
        elif alpha >= 1.0:
            fused = gd
        else:
            fused = gd ** alpha * vv ** (1 - alpha)
        pi.append((boxes_f, fused))
    return compute_ap50_img(pi, gts)


def main():
    results = {}
    for name, dataset, cls_name, domain, prefix, dior_cls in EVALS:
        print(f'\n[{"="*12}] {name}', flush=True)
        if dataset == 'coco':
            images, gts = load_coco(cls_name)
        else:
            images, gts = load_dior(cls_name)
        if len(images) < 5:
            print('  图太少, 跳过', flush=True)
            continue
        base_dets = gdino_detect(images, f'{prefix} {cls_name}', 0.0)
        prep = prep_per_img(images, base_dets, dior_cls)
        base_ap = ap_at_alpha(prep, 1.0, gts)  # α=1 → 纯 gd = baseline
        row = {'baseline': float(base_ap), 'n_img': len(images),
               'n_gt': int(sum(len(g) for g in gts))}
        for a in ALPHAS:
            row[f'alpha_{a}'] = float(ap_at_alpha(prep, a, gts))
        best_a = max(ALPHAS, key=lambda a: row[f'alpha_{a}'])
        row['best_alpha'] = best_a
        row['best_ap'] = row[f'alpha_{best_a}']
        row['delta_best'] = round(row['best_ap'] - row['baseline'], 4)
        row['delta_fixed05'] = round(row['alpha_0.5'] - row['baseline'], 4)
        results[name] = row
        print(f'  baseline={base_ap:.4f}  最优α={best_a} 最优AP={row["best_ap"]:.4f} '
              f'Δ最优={row["delta_best"]:+.4f}  Δ写死0.5={row["delta_fixed05"]:+.4f}', flush=True)
    with open(SAVE_DIR / 'weight_probe_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果 → {SAVE_DIR / "weight_probe_results.json"}', flush=True)


if __name__ == '__main__':
    main()
