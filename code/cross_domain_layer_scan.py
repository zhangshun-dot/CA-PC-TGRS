# -*- coding: utf-8 -*-
"""跨域逐层粒度扫描 — DIOR exemplar → COCO 检测

核心实验: DIOR 训练的 exemplar prototype 直接用于 COCO 重打分
→ 验证 φ(class→best_layer) 是否跨域不变

DIOR ↔ COCO 共享类:
  airplane  → airplane  (俯拍 vs 侧拍)
  ship      → boat      (头顶 vs 水面)
  vehicle   → car       (俯拍 vs 街拍)

Usage:
    python cross_domain_layer_scan.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import io, json, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

# COCO
COCO_IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
COCO_ANN_FILE = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'

# DIOR
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5_dinov2_layers.pt'
DINOV2_PATH = r'C:\Users\张三\.cache\modelscope\tiansz\dinov2-base'

# 跨域类别映射
DOMAIN_PAIRS = [
    {'dior_class': 'airplane', 'coco_class': 'airplane', 'prompt': 'a photo of an airplane'},
    {'dior_class': 'ship',      'coco_class': 'boat',     'prompt': 'a photo of a boat'},
    {'dior_class': 'vehicle',   'coco_class': 'car',      'prompt': 'a photo of a car'},
    # 额外尝试: 看看哪些 COCO 类能对上 DIOR prototype
    {'dior_class': 'tennis court', 'coco_class': 'sports ball', 'prompt': 'a photo of a sports ball'},
]
N_IMAGES_PER_CLASS = 20
PRE_FILTER_THRESH = 0.03


# ── 模型 ──
def get_dinov2_with_layers():
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(DINOV2_PATH)
    config.output_hidden_states = True
    model = AutoModel.from_pretrained(DINOV2_PATH, config=config).eval().to(DEVICE)
    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def get_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
    proc = AutoProcessor.from_pretrained(gd_cache)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gd_cache, dtype=torch.float32
    ).eval().to(DEVICE)
    return proc, model


# ── COCO 数据 ──
def load_coco_gt():
    from pycocotools.coco import COCO
    return COCO(COCO_ANN_FILE)


def get_coco_images_and_gt(coco, class_name, n=N_IMAGES_PER_CLASS):
    """加载 COCO 图像 + GT boxes (xyxy)。

    Returns:
        images: List[PIL.Image]
        gt_boxes_per_image: List[List[[x1,y1,x2,y2], ...]]
    """
    cat_ids = coco.getCatIds(catNms=[class_name])
    if not cat_ids:
        return [], []
    cat_id = cat_ids[0]

    img_ids = coco.getImgIds(catIds=[cat_id])
    # 找有足够多 GT 框的图像
    good_ids = []
    for iid in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False)
        if len(ann_ids) >= 1:
            good_ids.append(iid)
        if len(good_ids) >= n * 3:
            break

    images, gt_per_image = [], []
    for iid in good_ids[:n * 3]:
        img_info = coco.loadImgs([iid])[0]
        path = os.path.join(COCO_IMG_DIR, img_info['file_name'])
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            continue

        ann_ids = coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        for a in anns:
            x, y, w, h = a['bbox']  # COCO: xywh
            gt_boxes.append([x, y, x + w, y + h])

        images.append(img)
        gt_per_image.append(gt_boxes)
        if len(images) >= n:
            break

    return images[:n], gt_per_image[:n]


# ── IoU / AP ──
def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)


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


# ── 跨域逐层评估 ──
def cross_domain_layer_eval(pair, bank, model, transform):
    coco = load_coco_gt()
    dior_class = pair['dior_class']
    coco_class = pair['coco_class']

    if dior_class not in bank:
        print(f"  {dior_class} 不在 DIOR bank 中")
        return None

    images, gts = get_coco_images_and_gt(coco, coco_class)
    if len(images) == 0:
        print(f"  COCO 无 {coco_class} 图像")
        return None

    print(f"  加载 {len(images)} 张 COCO {coco_class} 图像, "
          f"GT boxes: {sum(len(g) for g in gts)}")

    prompt = pair['prompt']
    raw = gdino_detect_all(images, prompt)

    # 统计 GDINO baseline
    n_det = sum(1 for r in raw if len(r['boxes']) > 0)
    print(f"  GDINO 检测到图像: {n_det}/{len(images)}")

    layer_aps = {}
    for layer_idx in range(12):
        proto = bank[dior_class][layer_idx]['prototype'].to(DEVICE)

        all_pred_boxes, all_pred_scores = [], []
        all_gt = []
        all_vis_sims = []

        for i in range(len(images)):
            all_gt.extend(gts[i])
            boxes = raw[i]['boxes']
            scores = raw[i]['scores']
            if len(boxes) == 0:
                continue

            keep = scores > PRE_FILTER_THRESH
            boxes_f = boxes[keep]
            scores_f = scores[keep]
            if len(boxes_f) == 0:
                continue

            visual_sims = []
            for box in boxes_f:
                x1, y1, x2, y2 = [int(v.item()) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(images[i].width, x2), min(images[i].height, y2)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    visual_sims.append(0.0)
                    continue
                crop = images[i].crop((x1, y1, x2, y2))

                img_t = transform(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    o = model(img_t)
                hs = o.hidden_states[layer_idx + 1]
                feat = hs[0, 0, :]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                sim = max(0.0, float((feat @ proto).item()))
                visual_sims.append(sim)

            gd = np.clip(scores_f.numpy(), 1e-6, 1.0)
            vs = np.clip(np.array(visual_sims), 1e-6, 1.0)
            fused = np.sqrt(gd * vs)

            all_pred_boxes.extend(boxes_f.numpy().tolist())
            all_pred_scores.extend(fused.tolist())
            all_vis_sims.extend(vs.tolist())

        ap = compute_ap(all_pred_boxes, all_pred_scores, all_gt)
        layer_aps[layer_idx] = ap

    return layer_aps, float(np.mean(all_vis_sims)) if all_vis_sims else 0.0


# ── 主函数 ──
def main():
    print("=" * 60)
    print("跨域逐层粒度扫描")
    print("DIOR exemplar prototypes → COCO detection re-scoring")
    print("=" * 60)

    # 加载
    print(f"\n加载 DIOR exemplar bank: {BANK_PATH}")
    bank = torch.load(BANK_PATH, map_location='cpu')
    print(f"  类别: {list(bank.keys())}")

    print("加载 DINOv2 (output_hidden_states=True)...")
    model, transform = get_dinov2_with_layers()

    all_results = {}
    summary_lines = []

    for pair in DOMAIN_PAIRS:
        dior_cls = pair['dior_class']
        coco_cls = pair['coco_class']
        print(f"\n{'='*60}")
        print(f"🔬 {dior_cls}(DIOR) → {coco_cls}(COCO)")
        print(f"{'='*60}")

        result = cross_domain_layer_eval(pair, bank, model, transform)
        if result is None:
            continue
        layer_aps, avg_vis_sim = result

        print(f"\n  Layer  AP50 (cross-domain)")
        print(f"  {'─'*30}")
        best_layer = max(layer_aps, key=layer_aps.get)
        worst_layer = min(layer_aps, key=layer_aps.get)

        for l in range(12):
            ap = layer_aps[l]
            bar = '█' * int(ap / max(layer_aps[best_layer], 0.001) * 20)
            tag = " ← BEST" if l == best_layer else (" ← WORST" if l == worst_layer else "")
            print(f"  L{l+1:>2d}  {ap:>8.4f}  {bar}{tag}")

        print(f"\n  平均 vis_sim: {avg_vis_sim:.4f}")
        print(f"  最佳层: L{best_layer+1} (AP50={layer_aps[best_layer]:.4f})")
        print(f"  最差层: L{worst_layer+1} (AP50={layer_aps[worst_layer]:.4f})")
        print(f"  跨域 AP 相对 L12: Δ={layer_aps[best_layer]-layer_aps[11]:+.4f}")

        all_results[pair['dior_class']] = {
            'layer_aps': {f'L{l+1}': layer_aps[l] for l in range(12)},
            'best_layer': f'L{best_layer+1}',
            'best_ap': layer_aps[best_layer],
            'l12_ap': layer_aps[11],
            'avg_vis_sim': avg_vis_sim,
        }
        summary_lines.append({
            'dior_class': dior_cls,
            'coco_class': coco_cls,
            'best_layer': f'L{best_layer+1}',
            'best_ap': layer_aps[best_layer],
            'l12_ap': layer_aps[11],
            'avg_vis_sim': avg_vis_sim,
        })

    # 汇总
    print(f"\n{'='*60}")
    print("跨域汇总: DIOR 模型 → COCO 检测")
    print(f"{'='*60}")
    print(f"{'DIOR→COCO':<25s} {'Best Layer':>10s} {'AP50':>8s} {'Avg Sim':>8s}")
    print(f"{'─'*55}")
    for s in summary_lines:
        print(f"{s['dior_class']+'→'+s['coco_class']:<25s} "
              f"{s['best_layer']:>10s} {s['best_ap']:>8.4f} {s['avg_vis_sim']:>8.4f}")

    # 🔑 关键结论
    print(f"\n{'='*60}")
    print("结论: φ(class→best_layer) 是否跨域不变?")
    print(f"{'='*60}")
    for s in summary_lines:
        dior_best = 'L12'  # DIOR 逐层扫描结果
        if s['dior_class'] == 'airplane':
            dior_best = 'L12'
        elif s['dior_class'] == 'ship':
            dior_best = 'L12'
        elif s['dior_class'] == 'vehicle':
            dior_best = 'L12'
        elif s['dior_class'] == 'tennis court':
            dior_best = 'L12'
        match = "✅ 一致" if s['best_layer'] == dior_best else f"❌ 不一致 ({dior_best}→{s['best_layer']})"
        print(f"  {s['dior_class']}: DIOR={dior_best}, COCO={s['best_layer']} → {match}")

    save_path = SAVE_DIR / 'cross_domain_layer_scan.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({'per_class': all_results, 'summary': summary_lines}, f, indent=2, ensure_ascii=False)
    print(f"\n保存至: {save_path}")
    print("✅ 跨域扫描完成")


if __name__ == '__main__':
    main()
