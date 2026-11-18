# -*- coding: utf-8 -*-
"""M3 复核: 主协议下池覆盖 R_max 的两种聚合口径 (纯 CPU, 只读 table6 缓存)。

背景: 审稿意见指出论文 R_max=0.74 < R@100=0.8049 看似矛盾 (同协议应 R@100<=R_max)。
核查结论(见 tge_recall_gdino_n500_perimg.py):
  - 论文 R_max=0.74 的来源是 per-GT 汇总口径 (hit/n_gt, 逐图匹配), N=500 探针;
  - 表VII R@100=0.8049 是逐图宏口径 (recall_k_results.json, run_recall_k.py 的
    recall_at_k "averaged over images, same view as the AP protocol"), 主协议全集-10。
两者聚合方式不同 (+图像集轻微不同), 不是同一口径的矛盾。

本脚本在主协议缓存 (outputs/cache/table6_*.pkl, per_img[10:] 即评测集) 上重算:
  pooled   : sum(hit)/sum(gt)            逐类, 再六类宏 -> 与 Eq.bound 的 R_max 定义一致
  macro    : mean_over_images(hit/gt)    逐类, 再六类宏 -> 与表VII R@K 同视图
  池规模    : 每图池大小均值/最大值 (验证 R@100 是否已饱和)
输出: tge_recall_mainprotocol_recompute.json
"""
import json
import pickle
from pathlib import Path

import numpy as np

CACHE = Path(r"C:\Users\张三\Desktop\A-CSPC\outputs\cache")
OUT = Path(r"C:\Users\张三\Desktop\实验数据\tge_recall_mainprotocol_recompute.json")
CLASSES = ["airplane", "boat", "car", "truck", "bus", "motorcycle"]
DETS = ["gdino-tiny", "gdino-base", "owlv2"]
K_CALIB = 10


def iou_max(boxes, gts):
    """boxes:(N,4) gts:(M,4) -> (M,) 每个 GT 与任一 box 的最大 IoU (逐图匹配)。"""
    if len(boxes) == 0 or len(gts) == 0:
        return np.zeros(len(gts))
    b = np.asarray(boxes, dtype=np.float64)
    g = np.asarray(gts, dtype=np.float64)
    xx1 = np.maximum(b[:, None, 0], g[None, :, 0])
    yy1 = np.maximum(b[:, None, 1], g[None, :, 1])
    xx2 = np.minimum(b[:, None, 2], g[None, :, 2])
    yy2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    iou = inter / (ab[:, None] + ag[None, :] - inter + 1e-8)
    return iou.max(axis=0)


def main():
    out = {"config": {
        "protocol": "main protocol: all val2017 images with non-crowd ann of the class, "
                    "first k_calib=10 held out (same eval view as run_table6/run_recall_k)",
        "iou": 0.5, "pre_filter": 0.03, "small_box_drop": "<=10px (same as detect_and_encode)",
        "pooled_def": "sum over images of covered GT / sum over images of GT (per-object)",
        "macro_def": "mean over images (with >=1 GT) of covered-GT fraction (per-image, "
                     "same view as the AP protocol and Table VII Recall@K)",
    }, "detectors": {}}
    for det in DETS:
        rows = {}
        for cls in CLASSES:
            p = CACHE / f"table6_{det}_COCO-{cls}.pkl"
            blob = pickle.load(open(p, "rb"))
            per_img = blob["per_img"][K_CALIB:]      # 评测集, 同 evaluate_class
            hit_tot = gt_tot = 0
            per_img_cov, pool_sizes = [], []
            for rec in per_img:
                gt = rec["gt"]
                if len(gt) == 0:
                    continue
                m = iou_max(rec["box"], gt)
                h = int((m >= 0.5).sum())
                hit_tot += h
                gt_tot += len(gt)
                per_img_cov.append(h / len(gt))
                pool_sizes.append(len(rec["box"]))
            rows[cls] = {
                "n_images": len(per_img_cov),
                "n_gt": gt_tot,
                "pooled": round(hit_tot / gt_tot, 4),
                "macro": round(float(np.mean(per_img_cov)), 4),
                "pool_mean": round(float(np.mean(pool_sizes)), 2),
                "pool_max": int(np.max(pool_sizes)),
            }
        macro_of = lambda key: round(float(np.mean([rows[c][key] for c in CLASSES])), 4)
        out["detectors"][det] = {
            "classes": rows,
            "six_class_macro_pooled": macro_of("pooled"),
            "six_class_macro_macro": macro_of("macro"),
        }
        print(f"== {det}")
        for c in CLASSES:
            r = rows[c]
            print(f"   {c:<11} n={r['n_images']:>3} gt={r['n_gt']:>4} "
                  f"pooled={r['pooled']:.4f} macro={r['macro']:.4f} "
                  f"pool(mean/max)={r['pool_mean']:.1f}/{r['pool_max']}")
        print(f"   SIX-CLASS  pooled={out['detectors'][det]['six_class_macro_pooled']:.4f} "
              f"macro={out['detectors'][det]['six_class_macro_macro']:.4f}")
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
