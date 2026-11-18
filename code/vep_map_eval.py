# -*- coding: utf-8 -*-
"""VEP mAP 评估 — 用 IoU 匹配 GT 框，测 VEP 对定位精度的影响。

与 DR 对比: DR 只问"有没有框"，mAP 问"框对不对"。

Usage:
    python vep_map_eval.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import io, json, glob, time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
DIOR_TEST_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
PARQUET_PATH = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5.pt'
IOU_THRESH = 0.5  # AP50

DIOR_ID2NAME = {
    0: 'airplane', 1: 'airport', 2: 'baseball field', 3: 'basketball court',
    4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'expressway service area',
    8: 'expressway toll station', 9: 'golf course', 10: 'ground track field',
    11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
    15: 'storage tank', 16: 'tennis court', 17: 'train station',
    18: 'vehicle', 19: 'windmill',
}
DIOR_CLASSES = list(DIOR_ID2NAME.values())
WEAK_CLASSES = ['tennis court', 'ship', 'harbor', 'windmill']

# ── 模型加载 ──
_gdino_proc, _gdino_model = None, None
_clip_model, _clip_preprocess = None, None


def get_gdino():
    global _gdino_proc, _gdino_model
    if _gdino_model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
        _gdino_proc = AutoProcessor.from_pretrained(gd_cache)
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gd_cache, dtype=torch.float32
        ).eval().to(DEVICE)
    return _gdino_proc, _gdino_model


def get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import clip as clip_module
        clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
        _clip_model, _clip_preprocess = clip_module.load(clip_path, device=DEVICE)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


# ── GT 加载 ──
def load_parquet_gt():
    """加载 parquet, 建 image_id → {class_name → [xyxy_boxes]} 索引。"""
    df = pd.read_parquet(PARQUET_PATH)
    gt_index = defaultdict(lambda: defaultdict(list))  # image_id → class → [boxes_xyxy]

    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        objs = row['objects']
        for i, cat in enumerate(objs['category']):
            cls_name = DIOR_ID2NAME[int(cat)]
            x, y, w, h = [int(v) for v in objs['bbox'][i]]
            gt_index[img_id][cls_name].append([x, y, x + w, y + h])  # xywh → xyxy

    return dict(gt_index), df


def get_gt_boxes(image_id, class_name, gt_index):
    """获取某张图某个类的 GT boxes (xyxy)。"""
    return gt_index.get(image_id, {}).get(class_name, [])


# ── IoU ──
def compute_iou(box1, box2):
    """box1, box2: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """计算 Average Precision (AP@IoU)。

    Args:
        pred_boxes: List[[x1,y1,x2,y2], ...]
        pred_scores: List[float]
        gt_boxes: List[[x1,y1,x2,y2], ...]

    Returns:
        float: AP
    """
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    if len(pred_boxes) == 0:
        return 0.0

    # 按置信度排序
    sorted_idx = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_idx]

    gt_matched = [False] * len(gt_boxes)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, pb in enumerate(pred_boxes):
        best_iou = 0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if not gt_matched[j]:
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp[i] = 1
            gt_matched[best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / len(gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)

    # 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += float(np.max(precisions[mask]))
    return ap / 11.0


# ── 检测 + 重打分 (mAP 版) ──
def gdino_detect_with_scores(images, prompt):
    """GDINO 检测 → 返回每图的所有 boxes + scores + 图像尺寸。"""
    gd_p, gd_m = get_gdino()
    results = []
    for img in images:
        inputs = gd_p(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_m(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_p.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        results.append({
            'boxes': r['boxes'].cpu(),
            'scores': r['scores'].cpu(),
            'img_size': img.size,  # (W, H)
        })
    return results


def vep_rerank_for_map(gdino_results, images, class_name, bank, alpha=0.5,
                       prefilter_thresh=0.03):
    """VEP 重打分 → 返回 (boxes_xyxy, fused_scores) 列表。

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: [(boxes_Nx4, scores_N), ...]
    """
    proto = bank[class_name]['prototype'].to(DEVICE)
    clip_m, clip_prep = get_clip()

    all_preds = []

    for gd_r, img in zip(gdino_results, images):
        boxes = gd_r['boxes']
        scores = gd_r['scores']

        if len(boxes) == 0:
            all_preds.append((np.empty((0, 4)), np.empty(0)))
            continue

        keep = scores > prefilter_thresh
        boxes_f = boxes[keep]
        scores_f = scores[keep]

        if len(boxes_f) == 0:
            all_preds.append((np.empty((0, 4)), np.empty(0)))
            continue

        visual_sims = []
        for box in boxes_f:
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 5 or y2 <= y1 + 5:
                visual_sims.append(0.0)
                continue
            crop = img.crop((x1, y1, x2, y2))
            img_t = clip_prep(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = clip_m.encode_image(img_t)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = max(0.0, float((feat @ proto).item()))
            visual_sims.append(sim)

        gd = np.clip(scores_f.numpy(), 1e-6, 1.0)
        vs = np.clip(np.array(visual_sims), 1e-6, 1.0)
        fused = (gd ** alpha) * (vs ** (1.0 - alpha))

        all_preds.append((boxes_f.numpy(), fused))

    return all_preds


def gdino_baseline_for_map(images, prompt):
    """Baseline GDINO (th=0.25) → 返回 (boxes, scores)。"""
    gd_p, gd_m = get_gdino()
    results = []
    for img in images:
        inputs = gd_p(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_m(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_p.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=ts)[0]
        results.append((r['boxes'].cpu().numpy(), r['scores'].cpu().numpy()))
    return results


# ── 主评估 ──
def evaluate_map(class_name, bank, gt_index, df, alpha=0.5):
    """对单个类计算 AP50 (baseline vs VEP)。

    Returns:
        Dict: {baseline_ap, vep_ap, n_gt_boxes, n_images, ...}
    """
    paths = sorted(glob.glob(
        os.path.join(DIOR_TEST_DIR, f"{class_name.replace(' ','_')}_*.jpg")
    ))
    if not paths:
        paths = sorted(glob.glob(
            os.path.join(DIOR_TEST_DIR, f"{class_name}_*.jpg")
        ))

    # 提取 image_id
    img_entries = []  # [(path, image_id)]
    for p in paths:
        parts = Path(p).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                img_entries.append((p, int(part)))
                break

    images = [Image.open(p).convert('RGB') for p, _ in img_entries]
    prompt = f"an overhead image of {class_name}"

    # Baseline
    baseline_preds = gdino_baseline_for_map(images, prompt)

    # VEP
    raw = gdino_detect_with_scores(images, prompt)
    vep_preds = vep_rerank_for_map(raw, images, class_name, bank, alpha=alpha)

    # 聚合所有图像的 pred + GT
    all_baseline_boxes, all_baseline_scores = [], []
    all_vep_boxes, all_vep_scores = [], []
    all_gt_boxes = []
    n_gt_total = 0

    for i, (p, img_id) in enumerate(img_entries):
        gt = get_gt_boxes(img_id, class_name, gt_index)
        n_gt_total += len(gt)

        bb, bs = baseline_preds[i]
        vb, vs = vep_preds[i]

        all_baseline_boxes.extend(bb.tolist() if hasattr(bb, 'tolist') else bb)
        all_baseline_scores.extend(bs.tolist() if hasattr(bs, 'tolist') else bs)
        all_vep_boxes.extend(vb.tolist() if hasattr(vb, 'tolist') else vb)
        all_vep_scores.extend(vs.tolist() if hasattr(vs, 'tolist') else vs)
        all_gt_boxes.extend(gt)

    ap_baseline = compute_ap(all_baseline_boxes, all_baseline_scores, all_gt_boxes, IOU_THRESH)
    ap_vep = compute_ap(all_vep_boxes, all_vep_scores, all_gt_boxes, IOU_THRESH)

    return {
        'class': class_name,
        'ap_baseline': ap_baseline,
        'ap_vep': ap_vep,
        'delta_ap': ap_vep - ap_baseline,
        'n_images': len(images),
        'n_gt_boxes': n_gt_total,
        'n_baseline_preds': len(all_baseline_boxes),
        'n_vep_preds': len(all_vep_boxes),
    }


def main():
    print("=" * 65)
    print("VEP mAP 评估 (AP50 @ IoU=0.5)")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    # 预热
    print("\n加载模型...")
    get_clip()
    get_gdino()

    # 加载数据
    print("加载 GT 标注...")
    gt_index, df = load_parquet_gt()
    print(f"  GT classes: {len(set().union(*[{c for c in v} for v in gt_index.values()]))}")

    # 加载 exemplar bank
    print(f"加载 Exemplar 库: {BANK_PATH}")
    bank = torch.load(BANK_PATH, map_location='cpu')

    # 评估
    ALPHA = 0.5
    results = {}
    print(f"\n{'='*60}")
    print(f"全 20 类 AP50 评估 (alpha={ALPHA})")
    print(f"{'='*60}")

    for i, cls_name in enumerate(DIOR_CLASSES):
        if cls_name not in bank:
            print(f"[{i+1:2d}/20] {cls_name}: 跳过 (无 exemplar)")
            continue
        r = evaluate_map(cls_name, bank, gt_index, df, alpha=ALPHA)
        results[cls_name] = r
        marker = "✅" if r['delta_ap'] > 0.01 else ("⚠️" if r['delta_ap'] < -0.01 else "  ")
        print(f"[{i+1:2d}/20] {cls_name:30s} | "
              f"AP50: {r['ap_baseline']:.3f} → {r['ap_vep']:.3f} "
              f"(Δ={r['delta_ap']:+.3f}) {marker} | "
              f"GT={r['n_gt_boxes']:3d} Pred: {r['n_baseline_preds']:4d}/{r['n_vep_preds']:4d}")

    # 汇总
    n = len(results)
    mAP_base = sum(r['ap_baseline'] for r in results.values()) / n
    mAP_vep = sum(r['ap_vep'] for r in results.values()) / n
    mAP_weak_base = sum(results[c]['ap_baseline'] for c in WEAK_CLASSES if c in results) / 4
    mAP_weak_vep = sum(results[c]['ap_vep'] for c in WEAK_CLASSES if c in results) / 4

    print(f"\n{'='*60}")
    print(f"mAP 汇总")
    print(f"{'='*60}")
    print(f"  全 20 类 mAP50:    {mAP_base:.4f} → {mAP_vep:.4f} (Δ={mAP_vep-mAP_base:+.4f})")
    print(f"  弱类 mAP50:        {mAP_weak_base:.4f} → {mAP_weak_vep:.4f} (Δ={mAP_weak_vep-mAP_weak_base:+.4f})")

    # 对比 DR
    print(f"\n  对比 DR (之前的):  0.835 → 1.000 (+16.5%)")
    print(f"  DR vs mAP: DR 涨了但 mAP 变化才是真实精度")

    # 保存
    output = {
        'config': {'alpha': ALPHA, 'iou_thresh': IOU_THRESH, 'K': 5},
        'per_class': results,
        'summary': {
            'mAP_baseline': mAP_base,
            'mAP_vep': mAP_vep,
            'mAP_delta': mAP_vep - mAP_base,
            'mAP_weak_baseline': mAP_weak_base,
            'mAP_weak_vep': mAP_weak_vep,
            'mAP_weak_delta': mAP_weak_vep - mAP_weak_base,
        },
    }

    save_path = SAVE_DIR / 'vep_map_results.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存至: {save_path}")
    print("✅ mAP 评估完成")


if __name__ == '__main__':
    main()
