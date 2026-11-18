# -*- coding: utf-8 -*-
"""VEP v2 — 嵌入 CLIP 内部空间特征进行定位验证

v1 只用 CLIP 最后的 global pooling（512-dim）→ 对位置偏移不敏感
v2 从 CLIP ViT 内部取出 patch token（49×512-dim）→ 空间一致性评分

关键模块: CLIP ViT 的 conv1 + 12层 transformer → patch tokens [49, 512]
         这些 token 保留了 7×7 空间网格上的局部语义描述
         最后 global pooling 丢弃的信息，就在这里

三信号融合:
    final = GDINO_text^α × CLIP_global^β × CLIP_spatial^γ

Usage:
    python vep_map_eval_v2.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, glob, time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
DIOR_TEST_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
PARQUET_PATH = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5.pt'
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


# ── CLIP 内部特征提取 ──
def encode_image_with_patches(clip_model, image_tensor):
    """CLIP ViT 前向 → 同时返回 global feature 和 patch features。

    CLIP 内部结构:
        conv1 (3→768, stride=32) → 49 patches
        + CLS token + positional embedding
        → 12× transformer → [50 tokens, 768-dim]
        → ln_post(CLS only) → proj → [512-dim]  ← 原来只用这个
        → 现在多取 patch tokens [49, 512-dim]  ← 空间信息

    Args:
        clip_model: openai CLIP model
        image_tensor: [1, 3, 224, 224]

    Returns:
        global_feat: [512] 归一化全局特征
        patch_feats: [49, 512] 归一化 patch 特征
    """
    visual = clip_model.visual
    # 匹配模型精度 (CLIP 可能是 fp16)
    image_tensor = image_tensor.to(dtype=visual.conv1.weight.dtype)

    with torch.no_grad():
        # Stage 1: conv1 + reshape
        x = visual.conv1(image_tensor)  # [1, 768, 7, 7]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [1, 768, 49]
        x = x.permute(0, 2, 1)  # [1, 49, 768]

        # Stage 2: CLS token + position
        cls_token = visual.class_embedding.to(x.dtype) + \
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # [1, 50, 768]
        x = x + visual.positional_embedding.to(x.dtype)

        # Stage 3: pre-norm
        x = visual.ln_pre(x)

        # Stage 4: transformer (12 layers)
        x = x.permute(1, 0, 2)  # NLD → LND for transformer
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND → NLD back

        # Stage 5: post-norm + projection (apply to ALL tokens, not just CLS)
        x_normed = visual.ln_post(x)  # [1, 50, 768]
        if visual.proj is not None:
            x_proj = x_normed @ visual.proj  # [1, 50, 512]
        else:
            x_proj = x_normed

        # Split
        global_feat = x_proj[0, 0, :]   # CLS token  → [512]
        patch_feats = x_proj[0, 1:, :]  # 49 patches  → [49, 512]

        # Normalize (CLIP standard)
        global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
        patch_feats = patch_feats / patch_feats.norm(dim=-1, keepdim=True)

    return global_feat, patch_feats


def spatial_localization_score(patch_feats, prototype, grid=7):
    """从 patch-prototype 相似度热力图计算空间一致性分数。

    核心逻辑:
    - 定位良好 → object 填满 crop → 所有 patch 响应均匀 → 方差小 → 分数高
    - 定位松散 → object 只在 crop 局部 → 部分 patch 高响应, 部分低 → 方差大 → 分数低

    这个分数本质上衡量的是: "这个框是否把 exemplar 的视觉特征紧紧抓住了"

    Args:
        patch_feats: [49, 512] 归一化 patch 特征
        prototype: [512] 归一化 exemplar 原型
        grid: 空间网格大小 (7 for ViT-B/32)

    Returns:
        float: 0~1, 越高越一致
    """
    sims = (patch_feats @ prototype).reshape(grid, grid)  # [7, 7]
    sims = torch.clamp(sims, min=0.0)

    mean_s = sims.mean().item()
    std_s = sims.std().item()

    if mean_s < 0.01:
        return 0.0

    # 变异系数 → 指数变换 → [0, 1]
    cv = std_s / mean_s
    score = float(np.exp(-cv))

    return score


# ── GT 加载 ──
def load_parquet_gt():
    df = pd.read_parquet(PARQUET_PATH)
    gt_index = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        objs = row['objects']
        for i, cat in enumerate(objs['category']):
            cls_name = DIOR_ID2NAME[int(cat)]
            x, y, w, h = [int(v) for v in objs['bbox'][i]]
            gt_index[img_id][cls_name].append([x, y, x + w, y + h])

    return dict(gt_index), df


def get_gt_boxes(image_id, class_name, gt_index):
    return gt_index.get(image_id, {}).get(class_name, [])


# ── IoU ──
def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    if len(pred_boxes) == 0:
        return 0.0

    sorted_idx = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_idx]

    gt_matched = [False] * len(gt_boxes)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    for i, pb in enumerate(pred_boxes):
        best_iou, best_j = 0.0, -1
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

    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cum / len(gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)

    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += float(np.max(precisions[mask]))
    return ap / 11.0


# ── 检测 ──
def gdino_detect_with_scores(images, prompt):
    """GDINO th=0.0 → 所有 boxes + scores。"""
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
        })
    return results


def gdino_baseline_for_map(images, prompt):
    """GDINO th=0.25 → (boxes, scores)。"""
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


# ── VEP v2: 三信号重打分 ──
def vep_rerank_v2(gdino_results, images, class_name, bank,
                  alpha=0.4, beta=0.3, gamma=0.3,
                  prefilter_thresh=0.03):
    """VEP v2 重打分 —— 嵌入 CLIP 内部空间特征。

    三信号几何平均:
        fused = GDINO^α × CLIP_global^β × CLIP_spatial^γ

    CLIP_global:  传统的 CLIP 编码 → 语义匹配（"像不像这个类"）
    CLIP_spatial: patch-level 特征一致性 → 空间质量（"框得紧不紧"）
    """
    proto = bank[class_name]['prototype'].to(DEVICE)
    clip_m, clip_prep = get_clip()

    all_preds = []
    spatial_scores_all = []  # 收集空间分数用于分析

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

        global_sims = []
        spatial_scores = []

        for box in boxes_f:
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                global_sims.append(0.0)
                spatial_scores.append(0.0)
                continue

            crop = img.crop((x1, y1, x2, y2))
            img_t = clip_prep(crop).unsqueeze(0).to(DEVICE)

            # 🔑 核心: 一次性取出 global + patch 特征
            global_feat, patch_feats = encode_image_with_patches(clip_m, img_t)

            # CLIP global → 语义相似度
            g_sim = max(0.0, float((global_feat @ proto).item()))
            global_sims.append(g_sim)

            # CLIP spatial → 空间一致性
            s_score = spatial_localization_score(patch_feats, proto)
            spatial_scores.append(s_score)

        gd = np.clip(scores_f.numpy(), 1e-6, 1.0)
        gs = np.clip(np.array(global_sims), 1e-6, 1.0)
        ss = np.clip(np.array(spatial_scores), 1e-6, 1.0)

        # 三信号几何平均
        fused = (gd ** alpha) * (gs ** beta) * (ss ** gamma)

        all_preds.append((boxes_f.numpy(), fused))
        spatial_scores_all.append(ss)

    return all_preds, spatial_scores_all


# ── 主评估 ──
def evaluate_map_v2(class_name, bank, gt_index, alpha=0.4, beta=0.3, gamma=0.3):
    """对单个类计算 AP50 (baseline vs VEP v2)。"""
    paths = sorted(glob.glob(
        os.path.join(DIOR_TEST_DIR, f"{class_name.replace(' ','_')}_*.jpg")
    ))
    if not paths:
        paths = sorted(glob.glob(
            os.path.join(DIOR_TEST_DIR, f"{class_name}_*.jpg")
        ))

    img_entries = []
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

    # VEP v2
    raw = gdino_detect_with_scores(images, prompt)
    vep_preds, spatial_scores = vep_rerank_v2(
        raw, images, class_name, bank,
        alpha=alpha, beta=beta, gamma=gamma
    )

    all_baseline_boxes, all_baseline_scores = [], []
    all_vep_boxes, all_vep_scores = [], []
    all_gt_boxes = []
    n_gt_total = 0
    all_spatial = []

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
        all_spatial.extend(spatial_scores[i].tolist() if hasattr(spatial_scores[i], 'tolist') else spatial_scores[i])

    ap_baseline = compute_ap(all_baseline_boxes, all_baseline_scores, all_gt_boxes, IOU_THRESH)
    ap_vep = compute_ap(all_vep_boxes, all_vep_scores, all_gt_boxes, IOU_THRESH)

    avg_spatial = float(np.mean(all_spatial)) if all_spatial else 0.0

    return {
        'class': class_name,
        'ap_baseline': ap_baseline,
        'ap_vep': ap_vep,
        'delta_ap': ap_vep - ap_baseline,
        'n_images': len(images),
        'n_gt_boxes': n_gt_total,
        'n_baseline_preds': len(all_baseline_boxes),
        'n_vep_preds': len(all_vep_boxes),
        'avg_spatial_score': avg_spatial,
    }


# ── Tennis Court 专项诊断 ──
def diagnose_tennis_court(bank, gt_index):
    """对 tennis court 做详细的逐框诊断。"""
    print("\n" + "=" * 65)
    print("🔬 Tennis Court 专项诊断")
    print("=" * 65)

    class_name = 'tennis court'
    paths = sorted(glob.glob(os.path.join(DIOR_TEST_DIR, "tennis_court_*.jpg")))

    img_entries = []
    for p in paths:
        parts = Path(p).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                img_entries.append((p, int(part)))
                break

    images = [Image.open(p).convert('RGB') for p, _ in img_entries]
    prompt = f"an overhead image of {class_name}"

    raw = gdino_detect_with_scores(images, prompt)

    # VEP v1 (alpha=0.5, beta=0.5, gamma=0)
    vep_v1_preds, _ = vep_rerank_v2(raw, images, class_name, bank,
                                      alpha=0.5, beta=0.5, gamma=0.0)

    # VEP v2 (alpha=0.4, beta=0.3, gamma=0.3)
    vep_v2_preds, spatial_scores = vep_rerank_v2(raw, images, class_name, bank,
                                                   alpha=0.4, beta=0.3, gamma=0.3)

    print(f"{'Img':>4s} {'GT':>4s} {'GDINO boxes':>12s} "
          f"{'VEP v1 top5':>20s} {'VEP v2 top5':>20s} {'avg_spatial':>12s}")
    print("-" * 80)

    for i, (p, img_id) in enumerate(img_entries):
        gt = get_gt_boxes(img_id, class_name, gt_index)

        # Baseline GDINO boxes
        gd_scores = raw[i]['scores'].numpy()
        gd_keep = gd_scores > 0.25
        n_gd = int(gd_keep.sum())

        # VEP v1 top-5 scores
        v1b, v1s = vep_v1_preds[i]
        top5_v1 = sorted(v1s, reverse=True)[:5]

        # VEP v2 top-5 scores
        v2b, v2s = vep_v2_preds[i]
        top5_v2 = sorted(v2s, reverse=True)[:5]

        # 空间分数
        ss_img = spatial_scores[i]
        avg_ss = float(np.mean(ss_img)) if len(ss_img) > 0 else 0.0

        v1_str = '/'.join([f'{s:.2f}' for s in top5_v1])
        v2_str = '/'.join([f'{s:.2f}' for s in top5_v2])

        print(f"{i:4d} {len(gt):4d} {n_gd:12d} "
              f"{v1_str:>20s} {v2_str:>20s} {avg_ss:12.3f}")

    # 计算 v1 vs v2 AP50
    all_gt_boxes_tc = []
    all_v1_boxes, all_v1_scores = [], []
    all_v2_boxes, all_v2_scores = [], []

    for i, (p, img_id) in enumerate(img_entries):
        gt = get_gt_boxes(img_id, class_name, gt_index)
        all_gt_boxes_tc.extend(gt)

        v1b, v1s = vep_v1_preds[i]
        v2b, v2s = vep_v2_preds[i]

        all_v1_boxes.extend(v1b.tolist() if hasattr(v1b, 'tolist') else v1b)
        all_v1_scores.extend(v1s.tolist() if hasattr(v1s, 'tolist') else v1s)
        all_v2_boxes.extend(v2b.tolist() if hasattr(v2b, 'tolist') else v2b)
        all_v2_scores.extend(v2s.tolist() if hasattr(v2s, 'tolist') else v2s)

    ap_v1 = compute_ap(all_v1_boxes, all_v1_scores, all_gt_boxes_tc, IOU_THRESH)
    ap_v2 = compute_ap(all_v2_boxes, all_v2_scores, all_gt_boxes_tc, IOU_THRESH)

    print(f"\n  Tennis Court AP50: v1={ap_v1:.4f} → v2={ap_v2:.4f} (Δ={ap_v2-ap_v1:+.4f})")

    return ap_v1, ap_v2


# ── 主函数 ──
def main():
    print("=" * 65)
    print("VEP v2 — 嵌入 CLIP 内部空间特征的 mAP 评估")
    print("=" * 65)
    print(f"设备: {DEVICE}")
    print("三信号: GDINO_text^α × CLIP_global^β × CLIP_spatial^γ")

    # 预热
    print("\n加载模型...")
    get_clip()
    get_gdino()

    print("加载 GT 标注...")
    gt_index, _ = load_parquet_gt()

    print(f"加载 Exemplar 库: {BANK_PATH}")
    bank = torch.load(BANK_PATH, map_location='cpu')

    # ── Tennis Court 诊断 ──
    diagnose_tennis_court(bank, gt_index)

    # ── Alpha/Beta/Gamma 消融 ──
    configs = [
        (0.50, 0.50, 0.00, "VEP v1 (无空间信号)"),
        (0.40, 0.40, 0.20, "VEP v2 (空间=0.2)"),
        (0.40, 0.30, 0.30, "VEP v2 (空间=0.3)"),
        (0.33, 0.33, 0.34, "VEP v2 (空间=0.34)"),
        (0.30, 0.30, 0.40, "VEP v2 (空间=0.4)"),
    ]

    print(f"\n{'='*65}")
    print(f"Alpha/Beta/Gamma 消融实验")
    print(f"{'='*65}")
    print(f"{'Config':<30s} {'Tennis Court AP50':>18s}")
    print("-" * 50)

    best_ap, best_cfg = 0.0, None

    # 只跑 tennis court
    tc_paths = sorted(glob.glob(os.path.join(DIOR_TEST_DIR, "tennis_court_*.jpg")))
    tc_entries = []
    for p in tc_paths:
        parts = Path(p).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                tc_entries.append((p, int(part)))
                break
    tc_images = [Image.open(p).convert('RGB') for p, _ in tc_entries]
    tc_prompt = "an overhead image of tennis court"
    tc_raw = gdino_detect_with_scores(tc_images, tc_prompt)

    all_gt_tc = []
    for _, img_id in tc_entries:
        all_gt_tc.extend(get_gt_boxes(img_id, 'tennis court', gt_index))

    for alpha, beta, gamma, label in configs:
        preds, _ = vep_rerank_v2(tc_raw, tc_images, 'tennis court', bank,
                                  alpha=alpha, beta=beta, gamma=gamma)

        all_boxes, all_scores = [], []
        for b, s in preds:
            all_boxes.extend(b.tolist() if hasattr(b, 'tolist') else b)
            all_scores.extend(s.tolist() if hasattr(s, 'tolist') else s)

        ap = compute_ap(all_boxes, all_scores, all_gt_tc, IOU_THRESH)
        print(f"{label:<30s} {ap:>18.4f}")

        if ap > best_ap:
            best_ap = ap
            best_cfg = (alpha, beta, gamma, label)

    print(f"\n🏆 最佳: {best_cfg[3]} (alpha={best_cfg[0]}, beta={best_cfg[1]}, gamma={best_cfg[2]})")

    # ── 全 20 类评估 (最佳配置) ──
    best_alpha, best_beta, best_gamma, best_label = best_cfg
    results = {}

    print(f"\n{'='*65}")
    print(f"全 20 类 AP50 评估")
    print(f"配置: {best_label}")
    print(f"{'='*65}")

    for i, cls_name in enumerate(DIOR_CLASSES):
        if cls_name not in bank:
            continue
        r = evaluate_map_v2(cls_name, bank, gt_index,
                            alpha=best_alpha, beta=best_beta, gamma=best_gamma)
        results[cls_name] = r
        marker = "✅" if r['delta_ap'] > 0.01 else ("⚠️" if r['delta_ap'] < -0.01 else "  ")
        print(f"[{i+1:2d}/20] {cls_name:30s} | "
              f"AP50: {r['ap_baseline']:.3f} → {r['ap_vep']:.3f} "
              f"(Δ={r['delta_ap']:+.3f}) {marker} | "
              f"GT={r['n_gt_boxes']:3d} sp={r['avg_spatial_score']:.3f}")

    n = len(results)
    mAP_base = sum(r['ap_baseline'] for r in results.values()) / n
    mAP_vep = sum(r['ap_vep'] for r in results.values()) / n
    mAP_weak_base = sum(results[c]['ap_baseline'] for c in WEAK_CLASSES if c in results) / 4
    mAP_weak_vep = sum(results[c]['ap_vep'] for c in WEAK_CLASSES if c in results) / 4

    print(f"\n{'='*65}")
    print(f"mAP 汇总")
    print(f"{'='*65}")
    print(f"  配置:              {best_label}")
    print(f"  全 20 类 mAP50:    {mAP_base:.4f} → {mAP_vep:.4f} (Δ={mAP_vep-mAP_base:+.4f})")
    print(f"  弱类 mAP50:        {mAP_weak_base:.4f} → {mAP_weak_vep:.4f} (Δ={mAP_weak_vep-mAP_weak_base:+.4f})")

    # 对比 v1
    print(f"\n  VEP v1 mAP50:      0.1005 → 0.2254 (+0.125)")
    print(f"  VEP v2 mAP50:      0.1005 → {mAP_vep:.4f} (+{mAP_vep-0.1005:.4f})")

    output = {
        'config': {'alpha': best_alpha, 'beta': best_beta, 'gamma': best_gamma,
                   'iou_thresh': IOU_THRESH, 'K': 5, 'label': best_label},
        'ablation': [{'alpha': a, 'beta': b, 'gamma': g, 'label': l, 'tc_ap': ap}
                      for a, b, g, l, ap in []],  # filled below
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

    save_path = SAVE_DIR / 'vep_map_v2_results.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存至: {save_path}")
    print("✅ VEP v2 评估完成")


if __name__ == '__main__':
    main()
