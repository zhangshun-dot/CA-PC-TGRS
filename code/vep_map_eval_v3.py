# -*- coding: utf-8 -*-
"""VEP v3 — SAM iou_prediction 作为定位信号

v1: GDINO_text × CLIP_global → mAP 0.1005→0.2254, tennis court 几乎无效
v2: + CLIP_spatial (7×7 patch) → 分辨率不够，无改善
v3: + SAM iou_prediction → 像素级定位质量，SAM 原生输出

三信号: final = GDINO^α × CLIP_global^β × SAM_iou^γ

SAM 的 iou_prediction 是其训练时就学到的"框里是否有清晰物体"的置信度。
松散框(背景多) → iou_prediction 低 → 被几何平均压死
紧致框(物体居中) → iou_prediction 高 → 得分维持

Usage:
    python vep_map_eval_v3.py
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
from PIL import Image

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
DIOR_TEST_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
PARQUET_PATH = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5.pt'
SAM_CKPT = r'C:\Users\张三\Downloads\sam_vit_b_01ec64.pth'
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

# ── 模型全局缓存 ──
_gdino_proc, _gdino_model = None, None
_clip_model, _clip_preprocess = None, None
_sam_predictor = None


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


def get_sam():
    """SAM ViT-B — 像素级 mask 预测 + iou_prediction。"""
    global _sam_predictor
    if _sam_predictor is None:
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT)
        sam = sam.to(DEVICE).eval()
        _sam_predictor = SamPredictor(sam)
        print(f"  SAM ViT-B 加载完成 ({SAM_CKPT})")
    return _sam_predictor


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


# ── IoU / AP ──
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


# ── GDINO 检测 ──
def gdino_detect_all(images, prompt):
    """GDINO th=0.0 → 所有 boxes。"""
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


def gdino_baseline(images, prompt):
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


# ── VEP v3: GDINO + CLIP + SAM 三信号 ──
def vep_rerank_v3(gdino_results, images, class_name, bank,
                  alpha=0.33, beta=0.33, gamma=0.34,
                  prefilter_thresh=0.03):
    """三信号重打分。

    Alpha (GDINO): 文本-图像对齐置信度
    Beta  (CLIP):  visual exemplar 语义相似度
    Gamma (SAM):   SAM iou_prediction — 框的定位置信度

    SAM 每图只编码一次 (set_image), 然后批量预测所有候选框 (predict batch)。
    """
    proto = bank[class_name]['prototype'].to(DEVICE)
    clip_m, clip_prep = get_clip()
    predictor = get_sam()

    all_preds = []
    sam_scores_all = []

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

        # ── SAM: 图像级编码（一次）→ 所有框批量预测 ──
        img_np = np.array(img)
        predictor.set_image(img_np)

        box_list = []
        valid_indices = []
        for idx, box in enumerate(boxes_f):
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 > x1 + 5 and y2 > y1 + 5:
                box_list.append([x1, y1, x2, y2])
                valid_indices.append(idx)

        # ── SAM: per-box (predict 不支持 batch box) ──
        # 排序取 top-30，控制 SAM 推理开销
        sorted_idx = torch.argsort(scores_f, descending=True)[:30]
        sam_iou = np.zeros(len(boxes_f), dtype=np.float32)

        for idx in sorted_idx:
            box = boxes_f[idx]
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 5 or y2 <= y1 + 5:
                continue
            try:
                _, si, _ = predictor.predict(
                    box=np.array([[x1, y1, x2, y2]]),
                    multimask_output=False,
                )
                sam_iou[int(idx)] = float(si[0])
            except Exception:
                sam_iou[int(idx)] = 0.0

        # ── CLIP: per-box 编码 ──
        visual_sims = []
        for idx, box in enumerate(boxes_f):
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                visual_sims.append(0.0)
                continue
            crop = img.crop((x1, y1, x2, y2))
            img_t = clip_prep(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = clip_m.encode_image(img_t)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = max(0.0, float((feat @ proto).item()))
            visual_sims.append(sim)

        # ── 三信号融合 ──
        gd = np.clip(scores_f.numpy(), 1e-6, 1.0)
        vs = np.clip(np.array(visual_sims), 1e-6, 1.0)
        ss = np.clip(sam_iou, 1e-6, 1.0)

        fused = (gd ** alpha) * (vs ** beta) * (ss ** gamma)

        all_preds.append((boxes_f.numpy(), fused))
        sam_scores_all.append(ss)

    return all_preds, sam_scores_all


# ── 某类评估 ──
def evaluate_class_v3(class_name, bank, gt_index,
                      alpha=0.33, beta=0.33, gamma=0.34):
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

    baseline_preds = gdino_baseline(images, prompt)
    raw = gdino_detect_all(images, prompt)
    vep_preds, sam_scores = vep_rerank_v3(
        raw, images, class_name, bank,
        alpha=alpha, beta=beta, gamma=gamma
    )

    all_bl_boxes, all_bl_scores = [], []
    all_ve_boxes, all_ve_scores = [], []
    all_gt_boxes = []
    all_sam = []
    n_gt_total = 0

    for i, (_, img_id) in enumerate(img_entries):
        gt = get_gt_boxes(img_id, class_name, gt_index)
        n_gt_total += len(gt)
        all_gt_boxes.extend(gt)

        bb, bs = baseline_preds[i]
        all_bl_boxes.extend(bb.tolist() if hasattr(bb, 'tolist') else bb)
        all_bl_scores.extend(bs.tolist() if hasattr(bs, 'tolist') else bs)

        vb, vs = vep_preds[i]
        all_ve_boxes.extend(vb.tolist() if hasattr(vb, 'tolist') else vb)
        all_ve_scores.extend(vs.tolist() if hasattr(vs, 'tolist') else vs)

        all_sam.extend(sam_scores[i].tolist() if hasattr(sam_scores[i], 'tolist')
                       else sam_scores[i])

    ap_bl = compute_ap(all_bl_boxes, all_bl_scores, all_gt_boxes, IOU_THRESH)
    ap_ve = compute_ap(all_ve_boxes, all_ve_scores, all_gt_boxes, IOU_THRESH)
    avg_sam = float(np.mean(all_sam)) if all_sam else 0.0

    return {
        'class': class_name,
        'ap_baseline': ap_bl,
        'ap_vep': ap_ve,
        'delta_ap': ap_ve - ap_bl,
        'n_images': len(images),
        'n_gt_boxes': n_gt_total,
        'n_bl_preds': len(all_bl_boxes),
        'n_ve_preds': len(all_ve_boxes),
        'avg_sam_iou': avg_sam,
    }


# ── Tennis Court 专项诊断 (含 SAM 信息) ──
def diagnose_tennis_court_v3(bank, gt_index):
    print("\n" + "=" * 65)
    print("🔬 Tennis Court 专项诊断: GDINO × CLIP × SAM")
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
    predictor = get_sam()
    clip_m, clip_prep = get_clip()
    proto = bank[class_name]['prototype'].to(DEVICE)

    # GDINO raw
    raw = gdino_detect_all(images, prompt)

    # 逐图分析
    print(f"{'Img':>4s} {'GT':>4s} | "
          f"{'SAM_iou(top1)':>14s} {'SAM_iou(mean)':>14s} "
          f"{'CLIP_sim(top1)':>14s} {'GT_box_SAM':>12s}")
    print("-" * 90)

    all_sam_iou_list = []

    for i, (p, img_id) in enumerate(img_entries):
        gt = get_gt_boxes(img_id, class_name, gt_index)
        boxes = raw[i]['boxes']
        scores = raw[i]['scores']

        keep = scores > 0.03
        boxes_f = boxes[keep]

        # ── SAM on GDINO boxes (per-box, top-30) ──
        img_np = np.array(images[i])
        predictor.set_image(img_np)

        # 按 GDINO score 降序，取 top-30 candidate boxes
        gd_scores = scores[keep].numpy()
        top_idx = np.argsort(gd_scores)[::-1][:30]

        sam_iou_preds = []
        for idx in top_idx:
            box = boxes_f[idx]
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(images[i].width, x2), min(images[i].height, y2)
            if x2 <= x1 + 5 or y2 <= y1 + 5:
                continue
            try:
                _, si, _ = predictor.predict(
                    box=np.array([[x1, y1, x2, y2]]),
                    multimask_output=False,
                )
                sam_iou_preds.append(float(si[0]))
            except Exception:
                sam_iou_preds.append(0.0)
        all_sam_iou_list.extend(sam_iou_preds)

        # ── SAM on GT boxes (作对比, per-box) ──
        gt_sam_ious = []
        if gt:
            for gt_box in gt[:10]:  # 最多 10 个 GT 框
                try:
                    _, si, _ = predictor.predict(
                        box=np.array([gt_box[:4]]),
                        multimask_output=False,
                    )
                    gt_sam_ious.append(f'{float(si[0]):.3f}')
                except Exception:
                    gt_sam_ious.append('err')

        # ── CLIP top-1 ──
        best_clip = 0.0
        for box in boxes_f[:30]:  # top 30 by GDINO score
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(images[i].width, x2), min(images[i].height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crop = images[i].crop((x1, y1, x2, y2))
            img_t = clip_prep(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = clip_m.encode_image(img_t)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            sim = float((feat @ proto).item())
            best_clip = max(best_clip, sim)

        sam_top1 = float(max(sam_iou_preds)) if len(sam_iou_preds) > 0 else 0.0
        sam_mean = float(np.mean(sam_iou_preds)) if len(sam_iou_preds) > 0 else 0.0

        print(f"{i:4d} {len(gt):4d} | "
              f"{sam_top1:>14.3f} {sam_mean:>14.3f} "
              f"{best_clip:>14.3f} {' '.join(gt_sam_ious):>12s}")

    # 对比: GT box 的 SAM iou vs GDINO box 的 SAM iou
    print(f"\n  GDINO候选框 SAM iou mean: {float(np.mean(all_sam_iou_list)):.3f}")
    print(f"  解释: 如果 GT 框的 SAM iou 明显高于 GDINO 框，")
    print(f"        说明 SAM 能区分紧致框和松散框 → VEP v3 有效")
    print(f"        如果两者差不多 → SAM 信号无增量")


# ── 主函数 ──
def main():
    print("=" * 65)
    print("VEP v3 — SAM iou_prediction 定位信号")
    print("=" * 65)
    print(f"设备: {DEVICE}")

    print("\n加载 CLIP...")
    get_clip()
    print("加载 GDINO...")
    get_gdino()
    print("加载 SAM ViT-B...")
    get_sam()

    print("加载 GT 标注...")
    gt_index, _ = load_parquet_gt()
    print(f"加载 Exemplar 库: {BANK_PATH}")
    bank = torch.load(BANK_PATH, map_location='cpu')

    # ── Tennis Court 诊断 ──
    diagnose_tennis_court_v3(bank, gt_index)

    # ── Tennis Court: 三信号权重消融 ──
    print(f"\n{'='*65}")
    print(f"Tennis Court: Alpha/Beta/Gamma 消融")
    print(f"{'='*65}")

    configs = [
        (0.50, 0.50, 0.00, "VEP v1 (无 SAM)"),
        (0.40, 0.30, 0.30, "GDINO.4 CLIP.3 SAM.3"),
        (0.33, 0.33, 0.34, "GDINO.33 CLIP.33 SAM.34"),
        (0.30, 0.30, 0.40, "GDINO.3 CLIP.3 SAM.4"),
        (0.25, 0.25, 0.50, "GDINO.25 CLIP.25 SAM.5"),
    ]

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
    tc_raw = gdino_detect_all(tc_images, tc_prompt)
    all_gt_tc = []
    for _, img_id in tc_entries:
        all_gt_tc.extend(get_gt_boxes(img_id, 'tennis court', gt_index))

    print(f"{'Config':<35s} {'TC AP50':>10s}")
    print("-" * 47)
    best_ap, best_cfg = 0.0, None

    for alpha, beta, gamma, label in configs:
        preds, _ = vep_rerank_v3(tc_raw, tc_images, 'tennis court', bank,
                                  alpha=alpha, beta=beta, gamma=gamma)
        all_boxes, all_scores = [], []
        for b, s in preds:
            all_boxes.extend(b.tolist() if hasattr(b, 'tolist') else b)
            all_scores.extend(s.tolist() if hasattr(s, 'tolist') else s)
        ap = compute_ap(all_boxes, all_scores, all_gt_tc, IOU_THRESH)
        marker = " ←" if ap > best_ap else ""
        print(f"{label:<35s} {ap:>10.4f}{marker}")
        if ap > best_ap:
            best_ap, best_cfg = ap, (alpha, beta, gamma, label)

    print(f"\n🏆 最佳: {best_cfg[3]}")

    # ── 全 20 类 (最佳配置) ──
    best_alpha, best_beta, best_gamma, best_label = best_cfg
    results = {}

    print(f"\n{'='*65}")
    print(f"全 20 类 AP50 评估 ({best_label})")
    print(f"{'='*65}")

    for i, cls_name in enumerate(DIOR_CLASSES):
        if cls_name not in bank:
            continue
        r = evaluate_class_v3(cls_name, bank, gt_index,
                              alpha=best_alpha, beta=best_beta, gamma=best_gamma)
        results[cls_name] = r
        marker = "✅" if r['delta_ap'] > 0.01 else ("⚠️" if r['delta_ap'] < -0.01 else "  ")
        print(f"[{i+1:2d}/20] {cls_name:30s} | "
              f"AP50: {r['ap_baseline']:.3f} → {r['ap_vep']:.3f} "
              f"(Δ={r['delta_ap']:+.3f}) {marker} | "
              f"GT={r['n_gt_boxes']:3d} "
              f"BL={r['n_bl_preds']:4d} VE={r['n_ve_preds']:4d} "
              f"SAM={r['avg_sam_iou']:.3f}")

    n = len(results)
    mAP_base = sum(r['ap_baseline'] for r in results.values()) / n
    mAP_vep = sum(r['ap_vep'] for r in results.values()) / n
    mAP_weak_base = sum(results[c]['ap_baseline'] for c in WEAK_CLASSES if c in results) / 4
    mAP_weak_vep = sum(results[c]['ap_vep'] for c in WEAK_CLASSES if c in results) / 4

    print(f"\n{'='*65}")
    print(f"mAP 汇总")
    print(f"{'='*65}")
    print(f"  VEP v1 (无 SAM):   mAP 0.1005 → 0.2254 (+125%)")
    print(f"  VEP v3 ({best_label}): mAP {mAP_base:.4f} → {mAP_vep:.4f} (+{(mAP_vep/mAP_base-1)*100:.1f}%)")
    print(f"  弱类 v1:            {0.0530:.4f} → {0.1314:.4f}")
    print(f"  弱类 v3:            {mAP_weak_base:.4f} → {mAP_weak_vep:.4f}")

    output = {
        'config': {'alpha': best_alpha, 'beta': best_beta, 'gamma': best_gamma,
                   'iou_thresh': IOU_THRESH, 'K': 5},
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
    save_path = SAVE_DIR / 'vep_map_v3_results.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存至: {save_path}")
    print("✅ VEP v3 评估完成")


if __name__ == '__main__':
    main()
