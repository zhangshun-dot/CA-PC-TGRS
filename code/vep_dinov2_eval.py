# -*- coding: utf-8 -*-
"""VEP + DINOv2 ViT-B/14 — 替换 CLIP ViT-B/32

Patch粒度: 14px vs 32px → 16×16 vs 7×7 grid → 5.2× 空间精度
参数量:    87M vs 88M → 几乎一样

DINOv2 是纯视觉模型 (无文本编码器)，但因为 VEP 做的是 visual→visual
对比 (crop vs exemplar)，不需要跨模态对齐——可能比 CLIP 更适合。

Usage:
    python vep_dinov2_eval.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import io, json, glob, time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
DIOR_TEST_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
PARQUET_PATH = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
DINOV2_PATH = r'C:\Users\张三\.cache\modelscope\tiansz\dinov2-base'
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5_dinov2.pt'
IOU_THRESH = 0.5

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

# ── 模型 ──
_gdino_proc, _gdino_model = None, None
_dinov2_model, _dinov2_transform = None, None


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


def get_dinov2():
    """DINOv2 ViT-B/14: 14px patch, 16×16 grid, 768-d features."""
    global _dinov2_model, _dinov2_transform
    if _dinov2_model is None:
        from transformers import AutoModel
        _dinov2_model = AutoModel.from_pretrained(DINOV2_PATH).eval().to(DEVICE)
        # DINOv2 使用 ImageNet 标准归一化
        _dinov2_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _dinov2_model, _dinov2_transform


def dino_encode(crop_pil):
    """DINOv2 编码 → 归一化 CLS token [768]"""
    model, transform = get_dinov2()
    img_t = transform(crop_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = model(img_t)
    feat = o.pooler_output[0]  # [768] CLS token
    return feat / feat.norm(dim=-1, keepdim=True)


# ── Exemplar 库构建 ──
def get_test_image_ids():
    """从测试文件名提取 image_id 集合。"""
    ids = set()
    for f in os.listdir(DIOR_TEST_DIR):
        if f.endswith('.jpg'):
            parts = Path(f).stem.split('_')
            for part in reversed(parts):
                if part.isdigit():
                    ids.add(int(part))
                    break
    return ids


def build_exemplar_bank_dinov2(k=5):
    """用 DINOv2 为每类构建 K=5 个 exemplar 的视觉原型。

    复用 parquet 中的 image bytes (无需外部图像文件)。

    Returns:
        bank: {class_name: {'prototype': Tensor[768], 'n_available': int, ...}}
    """
    print("=" * 60)
    print(f"构建 DINOv2 Exemplar 库 (K={k})")
    print("=" * 60)

    df = pd.read_parquet(PARQUET_PATH)
    test_ids = get_test_image_ids()
    print(f"  Test IDs: {len(test_ids)}, Parquet rows: {len(df)}")

    # 排除 test ID
    train_rows_by_class = defaultdict(list)
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        if img_id in test_ids:
            continue
        cats = row['objects']['category']
        for i, cat in enumerate(cats):
            cls_name = DIOR_ID2NAME[int(cat)]
            train_rows_by_class[cls_name].append((img_id, row, i))

    print(f"  训练类别: {len(train_rows_by_class)}")

    # 预热 DINOv2
    model, transform = get_dinov2()

    bank = {}
    for cls_name in DIOR_CLASSES:
        rows = train_rows_by_class.get(cls_name, [])
        if len(rows) == 0:
            print(f"  {cls_name}: 0 exemplar (skip)")
            continue

        # 每图取 1 个最大 bbox → 去重
        img_best = {}  # img_id → (row, best_bbox_index)
        for img_id, row, cat_idx in rows:
            bbox = row['objects']['bbox'][cat_idx]
            area = bbox[2] * bbox[3]
            if img_id not in img_best or area > img_best[img_id][2]:
                img_best[img_id] = (row, cat_idx, area)

        # 编码所有候选 crop
        features = []
        crops_info = []
        for img_id, (row, cat_idx, area) in img_best.items():
            try:
                img_bytes = row['image']['bytes']
                pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                x, y, w, h = [int(v) for v in row['objects']['bbox'][cat_idx]]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(pil_img.width, x + w), min(pil_img.height, y + h)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue
                crop = pil_img.crop((x1, y1, x2, y2))
                img_t = transform(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    o = model(img_t)
                feat = o.pooler_output[0]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                features.append(feat.cpu())
                crops_info.append((crop, img_id))
            except Exception:
                continue

        if len(features) < k:
            print(f"  {cls_name}: only {len(features)} valid crops (need {k})")
            if len(features) == 0:
                continue
            k_use = len(features)
        else:
            k_use = k

        # Maximin 多样化选择
        feats_stacked = torch.stack(features)  # [N, 768]
        selected = [0]  # 从第一个开始
        for _ in range(1, k_use):
            # 每个未选 crop 到已选集合的最小距离
            dists = []
            for j in range(len(feats_stacked)):
                if j in selected:
                    dists.append(-1.0)
                else:
                    min_d = min(
                        float(1.0 - (feats_stacked[j] @ feats_stacked[s]).item())
                        for s in selected
                    )
                    dists.append(min_d)
            best_j = max(range(len(dists)), key=lambda j: dists[j])
            selected.append(best_j)

        exemplar_feats = feats_stacked[selected]  # [K, 768]
        prototype = exemplar_feats.mean(dim=0)
        prototype = prototype / prototype.norm(dim=-1, keepdim=True)

        bank[cls_name] = {
            'prototype': prototype,
            'exemplar_features': exemplar_feats,
            'image_ids': [crops_info[j][1] for j in selected],
            'n_available': len(features),
        }

        print(f"  {cls_name}: {k_use}/{len(features)} exemplars, "
              f"proto_norm={prototype.norm().item():.3f}")

    # 保存
    torch.save(bank, BANK_PATH)
    print(f"\n保存至: {BANK_PATH}")
    print(f"类别数: {len(bank)}")

    # 打印 prototype 统计
    sims = []
    for c, info in bank.items():
        ef = info['exemplar_features']
        if len(ef) > 1:
            sim_matrix = ef @ ef.T
            for i in range(len(ef)):
                for j in range(i + 1, len(ef)):
                    sims.append(float(sim_matrix[i, j]))
    print(f"Exemplar 间相似度: mean={np.mean(sims):.3f}, std={np.std(sims):.3f}")

    return bank

    # ... (如果找到图像，继续构建 bank)


# ── GT 加载 ──
def load_parquet_gt():
    df = pd.read_parquet(PARQUET_PATH)
    gt_index = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        for i, cat in enumerate(row['objects']['category']):
            cls_name = DIOR_ID2NAME[int(cat)]
            x, y, w, h = [int(v) for v in row['objects']['bbox'][i]]
            gt_index[img_id][cls_name].append([x, y, x + w, y + h])
    return dict(gt_index), df


def get_gt_boxes(image_id, class_name, gt_index):
    return gt_index.get(image_id, {}).get(class_name, [])


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    if len(pred_boxes) == 0:
        return 0.0
    sorted_idx = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_idx]
    gt_matched = [False] * len(gt_boxes)
    tp, fp = np.zeros(len(pred_boxes)), np.zeros(len(pred_boxes))
    for i, pb in enumerate(pred_boxes):
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if not gt_matched[j]:
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp[i] = 1
            gt_matched[best_j] = True
        else:
            fp[i] = 1
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cum / len(gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += float(np.max(precisions[mask]))
    return ap / 11.0


# ── GDINO ──
def gdino_detect_all(images, prompt):
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
        results.append({'boxes': r['boxes'].cpu(), 'scores': r['scores'].cpu()})
    return results


def gdino_baseline(images, prompt):
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


# ── VEP + DINOv2 ──
def vep_dinov2_rerank(gdino_results, images, class_name, bank,
                       alpha=0.5, prefilter_thresh=0.03):
    """DINOv2 visual similarity 重打分。"""
    proto = bank[class_name]['prototype'].to(DEVICE)
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
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                visual_sims.append(0.0)
                continue
            crop = img.crop((x1, y1, x2, y2))
            feat = dino_encode(crop)
            sim = max(0.0, float((feat @ proto).item()))
            visual_sims.append(sim)

        gd = np.clip(scores_f.numpy(), 1e-6, 1.0)
        vs = np.clip(np.array(visual_sims), 1e-6, 1.0)
        fused = (gd ** alpha) * (vs ** (1.0 - alpha))
        all_preds.append((boxes_f.numpy(), fused))

    return all_preds


# ── 评估 ──
def evaluate_class_dinov2(class_name, bank, gt_index, alpha=0.5):
    paths = sorted(glob.glob(
        os.path.join(DIOR_TEST_DIR, f"{class_name.replace(' ','_')}_*.jpg")
    ))
    if not paths:
        paths = sorted(glob.glob(os.path.join(DIOR_TEST_DIR, f"{class_name}_*.jpg")))

    img_entries = []
    for p in paths:
        parts = Path(p).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                img_entries.append((p, int(part)))
                break

    images = [Image.open(p).convert('RGB') for p, _ in img_entries]
    prompt = f"an overhead image of {class_name}"

    bl = gdino_baseline(images, prompt)
    raw = gdino_detect_all(images, prompt)
    vp = vep_dinov2_rerank(raw, images, class_name, bank, alpha=alpha)

    all_bl_b, all_bl_s, all_vp_b, all_vp_s, all_gt = [], [], [], [], []
    n_gt = 0
    for i, (_, img_id) in enumerate(img_entries):
        gt = get_gt_boxes(img_id, class_name, gt_index)
        n_gt += len(gt)
        all_gt.extend(gt)
        bb, bs = bl[i]
        all_bl_b.extend(bb.tolist() if hasattr(bb, 'tolist') else bb)
        all_bl_s.extend(bs.tolist() if hasattr(bs, 'tolist') else bs)
        vb, vs = vp[i]
        all_vp_b.extend(vb.tolist() if hasattr(vb, 'tolist') else vb)
        all_vp_s.extend(vs.tolist() if hasattr(vs, 'tolist') else vs)

    ap_bl = compute_ap(all_bl_b, all_bl_s, all_gt)
    ap_vp = compute_ap(all_vp_b, all_vp_s, all_gt)

    return {
        'class': class_name, 'ap_baseline': ap_bl, 'ap_vep': ap_vp,
        'delta_ap': ap_vp - ap_bl, 'n_images': len(images), 'n_gt_boxes': n_gt,
        'n_bl_preds': len(all_bl_b), 'n_vep_preds': len(all_vp_b),
    }


# ── 主函数 ──
def main():
    print("=" * 65)
    print("VEP + DINOv2 ViT-B/14 评估")
    print("=" * 65)

    # 检查 exemplar bank 是否存在
    if not BANK_PATH.exists():
        print(f"\n❌ Exemplar bank 不存在: {BANK_PATH}")
        print("需要先用 DIOR 训练图像构建。")
        print("尝试构建...")
        bank = build_exemplar_bank_dinov2(k=5)
        if bank is None:
            print("\n⚠️ 无法构建 exemplar bank，需要 DIOR 训练图像文件。")
            print("训练图像应放在以下目录之一:")
            print("  C:\\Users\\张三\\modeling\\dior_data\\train\\")
            print("  C:\\Users\\张三\\modeling\\dior_data\\JPEGImages\\")
            return
    else:
        print(f"加载 Exemplar 库: {BANK_PATH}")
        bank = torch.load(BANK_PATH, map_location='cpu')
        print(f"  类别: {len(bank)}")

    print("加载 GT 标注...")
    gt_index, _ = load_parquet_gt()

    # 快速对比: tennis court
    print(f"\n{'='*65}")
    print("🎾 Tennis Court 对比")
    print(f"{'='*65}")

    if 'tennis court' in bank:
        r = evaluate_class_dinov2('tennis court', bank, gt_index, alpha=0.5)
        print(f"  GDINO baseline: {r['ap_baseline']:.4f}")
        print(f"  VEP+DINOv2:     {r['ap_vep']:.4f}")
        print(f"  Δ:              {r['delta_ap']:+.4f}")
        print(f"  (对比: VEP+CLIP B/32  tennis AP50 = 0.0148)")
    else:
        print("  tennis court 不在 exemplar bank 中")

    # 全 20 类
    results = {}
    for i, cls_name in enumerate(DIOR_CLASSES):
        if cls_name not in bank:
            continue
        r = evaluate_class_dinov2(cls_name, bank, gt_index, alpha=0.5)
        results[cls_name] = r
        marker = "✅" if r['delta_ap'] > 0.01 else ("⚠️" if r['delta_ap'] < -0.01 else "  ")
        print(f"[{i+1:2d}/20] {cls_name:30s} | "
              f"AP50: {r['ap_baseline']:.3f} → {r['ap_vep']:.3f} "
              f"(Δ={r['delta_ap']:+.3f}) {marker} | GT={r['n_gt_boxes']:3d}")

    n = len(results)
    mAP_base = sum(r['ap_baseline'] for r in results.values()) / n
    mAP_vep = sum(r['ap_vep'] for r in results.values()) / n
    mAP_weak_base = sum(results[c]['ap_baseline'] for c in WEAK_CLASSES if c in results) / 4
    mAP_weak_vep = sum(results[c]['ap_vep'] for c in WEAK_CLASSES if c in results) / 4

    print(f"\n{'='*65}")
    print(f"mAP 对比")
    print(f"{'='*65}")
    print(f"  CLIP  B/32:  mAP 0.1005 → 0.2254 (+125%)")
    print(f"  DINOv2 B/14:  mAP {mAP_base:.4f} → {mAP_vep:.4f} (+{(mAP_vep/mAP_base-1)*100:.1f}%)")
    print(f"  CLIP  弱类:   0.0530 → 0.1314")
    print(f"  DINOv2 弱类:  {mAP_weak_base:.4f} → {mAP_weak_vep:.4f}")

    output = {'config': {'model': 'dinov2_vitb14', 'alpha': 0.5, 'K': 5},
              'per_class': results,
              'summary': {'mAP_baseline': mAP_base, 'mAP_vep': mAP_vep,
                          'mAP_weak_baseline': mAP_weak_base, 'mAP_weak_vep': mAP_weak_vep}}
    save_path = SAVE_DIR / 'vep_dinov2_results.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n保存至: {save_path}")


if __name__ == '__main__':
    main()
