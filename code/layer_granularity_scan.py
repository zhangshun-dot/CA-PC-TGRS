# -*- coding: utf-8 -*-
"""DINOv2 逐层粒度扫描 — 验证"粒度可设计"

DINOv2 ViT-B/14, 12层 transformer → 每层有效粒度不同
→ 逐层测 tennis court AP50 → 看哪层 peak

关键发现:
  Layer 1-3:  局部纹理 (白线、边缘) ← 应为 tennis/basketball 区分最佳层
  Layer 9-12: 全局语义 (运动场大类) ← CLIP B/32 等价，应最差

Usage:
    python layer_granularity_scan.py
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import io, json, glob
from pathlib import Path
from collections import defaultdict

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
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5_dinov2_layers.pt'

DIOR_ID2NAME = {
    0: 'airplane', 1: 'airport', 2: 'baseball field', 3: 'basketball court',
    4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'expressway service area',
    8: 'expressway toll station', 9: 'golf course', 10: 'ground track field',
    11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
    15: 'storage tank', 16: 'tennis court', 17: 'train station',
    18: 'vehicle', 19: 'windmill',
}
DIOR_CLASSES = list(DIOR_ID2NAME.values())


# ── 模型 ──
def get_dinov2_with_layers():
    """返回带所有隐藏层的 DINOv2 + 预处理。"""
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


# ── GT ──
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


# ── 逐层 Exemplar 库 ──
def get_test_ids():
    ids = set()
    for f in glob.glob(os.path.join(DIOR_TEST_DIR, '*.jpg')):
        parts = Path(f).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                ids.add(int(part))
                break
    return ids


def build_layer_exemplar_bank(k=5):
    """一次 forward pass → 12 层各自建 prototype。"""
    print("=" * 60)
    print(f"构建逐层 Exemplar 库 (K={k})")
    print("=" * 60)

    model, transform = get_dinov2_with_layers()
    N_LAYERS = 12

    df = pd.read_parquet(PARQUET_PATH)
    test_ids = get_test_ids()

    # 建训练索引
    train_rows = defaultdict(list)
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        if img_id in test_ids:
            continue
        cats = row['objects']['category']
        for i, cat in enumerate(cats):
            cls_name = DIOR_ID2NAME[int(cat)]
            train_rows[cls_name].append((img_id, row, i))

    print(f"  训练类别: {len(train_rows)}")

    bank = {}  # bank[class_name][layer_idx] = {'prototype': [768], ...}

    for cls_name in DIOR_CLASSES:
        rows = train_rows.get(cls_name, [])
        if len(rows) == 0:
            continue

        # 每图取最大 bbox
        img_best = {}
        for img_id, row, cat_idx in rows:
            bbox = row['objects']['bbox'][cat_idx]
            area = bbox[2] * bbox[3]
            if img_id not in img_best or area > img_best[img_id][2]:
                img_best[img_id] = (row, cat_idx, area)

        # 编码所有 crop → 取所有层
        all_layer_feats = defaultdict(list)  # layer → [feat_768, ...]
        for img_id, (row, cat_idx, _) in img_best.items():
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
                # hidden_states: (embed + 12 layers) = 13 个
                # 只用 layer 1-12 (跳过 embedding 层的 CLS)
                for layer_idx in range(N_LAYERS):
                    hs = o.hidden_states[layer_idx + 1]  # +1 跳过 embedding
                    cls_feat = hs[0, 0, :]  # CLS token
                    all_layer_feats[layer_idx].append(cls_feat.cpu())
            except Exception:
                continue

        if len(all_layer_feats[0]) < k:
            print(f"  {cls_name}: only {len(all_layer_feats[0])} crops (skip)")
            continue

        cls_bank = {}
        for layer_idx in range(N_LAYERS):
            feats = torch.stack(all_layer_feats[layer_idx])  # [N, 768]
            feats = feats / feats.norm(dim=-1, keepdim=True)

            # Maximin 选 K
            selected = [0]
            for _ in range(1, k):
                dists = [-1.0] * len(feats)
                for j in range(len(feats)):
                    if j in selected:
                        continue
                    dists[j] = min(
                        float(1.0 - (feats[j] @ feats[s]).item())
                        for s in selected
                    )
                best_j = max(range(len(dists)), key=lambda j: dists[j])
                selected.append(best_j)

            exemplars = feats[selected]
            proto = exemplars.mean(dim=0)
            proto = proto / proto.norm(dim=-1, keepdim=True)

            cls_bank[layer_idx] = {
                'prototype': proto,
                'n_available': len(feats),
            }

        bank[cls_name] = cls_bank
        print(f"  {cls_name}: {k}/{len(feats)} exemplars × {N_LAYERS} layers")

    torch.save(bank, BANK_PATH)
    print(f"\n保存至: {BANK_PATH}")
    print(f"类别数: {len(bank)}")
    return bank, N_LAYERS


# ── 逐层评估 ──
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


def encode_with_layer(crop_pil, layer_idx, model, transform):
    """用 DINOv2 某层的 CLS token 编码 crop。"""
    img_t = transform(crop_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = model(img_t)
    hs = o.hidden_states[layer_idx + 1]  # +1 跳过 embedding
    feat = hs[0, 0, :]  # CLS token
    return feat / feat.norm(dim=-1, keepdim=True)


def eval_tc_per_layer(class_name, bank, gt_index, model, transform):
    """对 tennis court 逐层测 AP50。"""
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

    raw = gdino_detect_all(images, prompt)

    layer_aps = {}
    for layer_idx in range(12):
        proto = bank[class_name][layer_idx]['prototype'].to(DEVICE)
        all_pred_boxes, all_pred_scores = [], []
        all_gt = []
        for i, (_, img_id) in enumerate(img_entries):
            gt = get_gt_boxes(img_id, class_name, gt_index)
            all_gt.extend(gt)
            boxes = raw[i]['boxes']
            scores = raw[i]['scores']
            if len(boxes) == 0:
                continue
            keep = scores > 0.03
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
                feat = encode_with_layer(crop, layer_idx, model, transform)
                sim = max(0.0, float((feat @ proto).item()))
                visual_sims.append(sim)

            gd = np.clip(scores_f.numpy(), 1e-6, 1.0)
            vs = np.clip(np.array(visual_sims), 1e-6, 1.0)
            fused = np.sqrt(gd * vs)  # alpha=0.5 geometric mean
            all_pred_boxes.extend(boxes_f.numpy().tolist())
            all_pred_scores.extend(fused.tolist())

        ap = compute_ap(all_pred_boxes, all_pred_scores, all_gt)
        layer_aps[layer_idx] = ap

    return layer_aps


# ── 主函数 ──
def main():
    print("=" * 60)
    print("DINOv2 逐层粒度扫描 — Tennis Court")
    print("=" * 60)

    # 加载 / 构建 bank
    if BANK_PATH.exists():
        print(f"\n加载现有 bank: {BANK_PATH}")
        bank = torch.load(BANK_PATH, map_location='cpu')
        N_LAYERS = len(bank[DIOR_CLASSES[0]])
        print(f"  类别: {len(bank)}, 层数: {N_LAYERS}")
    else:
        bank, N_LAYERS = build_layer_exemplar_bank(k=5)

    # 加载 GT + 模型
    print("\n加载 GT 标注...")
    gt_index, _ = load_parquet_gt()

    print("加载 DINOv2 (output_hidden_states=True)...")
    model, transform = get_dinov2_with_layers()

    # 测试类
    test_classes = ['tennis court', 'basketball court', 'ground track field',
                    'airplane', 'ship', 'windmill']

    all_results = {}
    for cls_name in test_classes:
        if cls_name not in bank:
            print(f"  {cls_name} 不在 bank 中, 跳过")
            continue

        print(f"\n{'─'*50}")
        print(f"🔬 {cls_name} 逐层扫描")
        print(f"{'─'*50}")
        layer_aps = eval_tc_per_layer(cls_name, bank, gt_index, model, transform)

        print(f"{'Layer':>6s}  {'AP50':>8s}  {'Bar'}")
        print(f"{'─'*40}")
        max_ap = max(layer_aps.values())
        for l in range(12):
            ap = layer_aps[l]
            bar_len = int(ap / max(max_ap, 0.01) * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            layer_type = "纹理" if l < 4 else ("部件" if l < 8 else "语义")
            print(f"  L{l+1:>2d}   {ap:>8.4f}  {bar} [{layer_type}]")

        # 找最佳层
        best_layer = max(layer_aps, key=layer_aps.get)
        worst_layer = min(layer_aps, key=layer_aps.get)
        print(f"\n  最佳层: L{best_layer+1} (AP50={layer_aps[best_layer]:.4f})")
        print(f"  最差层: L{worst_layer+1} (AP50={layer_aps[worst_layer]:.4f})")
        print(f"  L12 作为 baseline: AP50={layer_aps[11]:.4f}")
        print(f"  L12→L{best_layer+1} 提升: {layer_aps[best_layer]-layer_aps[11]:+.4f}")

        all_results[cls_name] = {
            'layer_aps': {f'L{l+1}': layer_aps[l] for l in range(12)},
            'best_layer': f'L{best_layer+1}',
            'best_ap': layer_aps[best_layer],
            'l12_ap': layer_aps[11],
            'l12_to_best_delta': layer_aps[best_layer] - layer_aps[11],
        }

    # 汇总
    print(f"\n{'='*60}")
    print("汇总: 各类最佳层")
    print(f"{'='*60}")
    print(f"{'Class':<25s} {'Best Layer':>10s} {'AP50':>8s} {'L12 AP':>8s} {'Δ':>8s}")
    print(f"{'─'*60}")
    for cls_name, r in all_results.items():
        print(f"{cls_name:<25s} {r['best_layer']:>10s} "
              f"{r['best_ap']:>8.4f} {r['l12_ap']:>8.4f} {r['l12_to_best_delta']:>+8.4f}")

    # 保存
    save_path = SAVE_DIR / 'layer_granularity_scan.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n保存至: {save_path}")
    print("✅ 逐层扫描完成")


if __name__ == '__main__':
    main()
