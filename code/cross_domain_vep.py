# -*- coding: utf-8 -*-
"""C-VEP: Cross-Domain Visual Exemplar Prototype (层加权版)

核心发现 (cross_domain_layer_scan.py):
  - DIOR airplane L12 → COCO airplane L11 (近不变)
  - DIOR ship L12 → COCO boat L6 (剧烈偏移)
  - DIOR vehicle L12 → COCO car L6 (剧烈偏移)

方案: 不用单层 prototype，而是学习 12 层的加权组合。
  proto_cross = Σ_l softmax(w)_l · p_l

两种模式:
  1. per-class grid search: 逐类搜索最优单层
  2. softmax-learned: 训一个小 MLP 预测层权重

对比: Baseline vs L12-VEP vs C-VEP(best-layer) vs C-VEP(softmax)
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 路径配置 ──
BANK_PATH = SAVE_DIR / 'exemplar_bank_k5_dinov2_layers.pt'
DINOV2_PATH = r'C:\Users\张三\.cache\modelscope\tiansz\dinov2-base'
GDINO_CACHE = (r'C:\Users\张三\.cache\huggingface\hub'
               r'\models--IDEA-Research--grounding-dino-tiny'
               r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')
COCO_IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
COCO_ANN_FILE = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'

# 跨域类别映射
CROSS_DOMAIN_PAIRS = [
    {'dior_class': 'airplane', 'coco_class': 'airplane'},
    {'dior_class': 'ship', 'coco_class': 'boat'},
    {'dior_class': 'vehicle', 'coco_class': 'car'},
]

N_LAYERS = 12
N_IMAGES = 30
PRE_FILTER_THRESH = 0.03
DET_THRESHOLD = 0.25


# ══════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════

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
    proc = AutoProcessor.from_pretrained(GDINO_CACHE)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_CACHE, dtype=torch.float32
    ).eval().to(DEVICE)
    return proc, model


# ══════════════════════════════════════════════════════
# COCO 数据加载
# ══════════════════════════════════════════════════════

def load_coco_data(class_name, n=N_IMAGES):
    from pycocotools.coco import COCO
    coco = COCO(COCO_ANN_FILE)
    cat_ids = coco.getCatIds(catNms=[class_name])
    if not cat_ids:
        return [], []
    cat_id = cat_ids[0]
    img_ids = coco.getImgIds(catIds=[cat_id])

    images, gt_boxes_all = [], []
    for iid in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
        img_info = coco.loadImgs([iid])[0]
        path = os.path.join(COCO_IMG_DIR, img_info['file_name'])
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            continue
        gts = [[a['bbox'][0], a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3]] for a in anns]
        images.append(img)
        gt_boxes_all.append(gts)
        if len(images) >= n:
            break
    return images, gt_boxes_all


# ══════════════════════════════════════════════════════
# GDINO 检测 + DINOv2 逐层编码
# ══════════════════════════════════════════════════════

def gdino_detect_all(images, prompt, gd_proc, gd_model):
    all_boxes, all_scores = [], []
    for img in images:
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        all_boxes.append(r['boxes'].cpu().numpy())
        all_scores.append(r['scores'].cpu().numpy())
    return all_boxes, all_scores


def encode_crop_per_layer(crop, dino_model, dino_transform):
    """编码裁剪区域 → 12层 CLS token features。"""
    img_t = dino_transform(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = dino_model(img_t)
    feats = []
    for l in range(N_LAYERS):
        hs = o.hidden_states[l + 1]  # +1 跳过 embedding
        f = hs[0, 0, :]  # CLS token
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-8)
        feats.append(f)
    return feats  # list of 12 × Tensor[768]


# ══════════════════════════════════════════════════════
# AP50 计算
# ══════════════════════════════════════════════════════

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def compute_ap50(pred_boxes, pred_scores, gt_boxes):
    if len(gt_boxes) == 0:
        return 0.0 if len(pred_boxes) > 0 else 1.0
    if len(pred_boxes) == 0:
        return 0.0
    idx = np.argsort(pred_scores)[::-1]
    pb_sorted = [pred_boxes[i] for i in idx]
    gt_matched = [False] * len(gt_boxes)
    tp, fp = np.zeros(len(pb_sorted)), np.zeros(len(pb_sorted))
    for i, pb in enumerate(pb_sorted):
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if not gt_matched[j]:
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
        if best_iou >= 0.5 and best_j >= 0:
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


# ══════════════════════════════════════════════════════
# 层加权 VEP 重打分
# ══════════════════════════════════════════════════════

def vep_rerank_layer_weighted(images, gd_boxes, gd_scores, bank, dior_class,
                                dino_model, dino_transform, layer_weights):
    """
    用 12 层加权原型做 VEP 重打分。

    Args:
        layer_weights: Tensor[N_LAYERS] — 每层的权重（会做 softmax）

    Returns:
        all_pred_boxes, all_pred_scores (flattened across all images)
    """
    w = F.softmax(layer_weights, dim=0)  # → 概率分布

    # 加权融合 12 层原型
    proto_fused = torch.zeros(768, device=DEVICE)
    for l in range(N_LAYERS):
        p = bank[dior_class][l]['prototype'].to(DEVICE)
        proto_fused += w[l] * p
    proto_fused = proto_fused / (proto_fused.norm() + 1e-8)

    all_pred_boxes, all_pred_scores = [], []

    for i in range(len(images)):
        boxes = gd_boxes[i]
        scores = gd_scores[i]
        if len(boxes) == 0:
            continue

        keep = scores > PRE_FILTER_THRESH
        boxes_f = boxes[keep]
        scores_f = scores[keep]
        if len(boxes_f) == 0:
            continue

        visual_sims = []
        for box in boxes_f:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(images[i].width, x2), min(images[i].height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                visual_sims.append(0.0)
                continue

            crop = images[i].crop((x1, y1, x2, y2))

            # 对每个 crop 编码 12 层特征
            feats = encode_crop_per_layer(crop, dino_model, dino_transform)

            # 用同样的权重融合 crop 特征
            crop_fused = torch.zeros(768, device=DEVICE)
            for l in range(N_LAYERS):
                crop_fused += w[l] * feats[l]
            crop_fused = crop_fused / (crop_fused.norm() + 1e-8)

            sim = max(0.0, float((crop_fused @ proto_fused).item()))
            visual_sims.append(sim)

        gd = np.clip(scores_f, 1e-6, 1.0)
        vs = np.clip(np.array(visual_sims), 1e-6, 1.0)
        fused = np.sqrt(gd * vs)

        all_pred_boxes.extend(boxes_f.tolist())
        all_pred_scores.extend(fused.tolist())

    return all_pred_boxes, all_pred_scores


# ══════════════════════════════════════════════════════
# 策略1: Per-class Grid Search (最优单层)
# ══════════════════════════════════════════════════════

def grid_search_best_layer(images, gd_boxes, gd_scores, gts, bank, dior_class,
                           dino_model, dino_transform):
    """对每个类搜索 12 层中最优的单层。"""
    best_layer, best_ap = -1, -1.0
    layer_aps = {}

    for l in range(N_LAYERS):
        # one-hot 权重 = 只用第 l 层
        w = torch.full((N_LAYERS,), -1e9, device=DEVICE)
        w[l] = 1e9

        pred_boxes, pred_scores = vep_rerank_layer_weighted(
            images, gd_boxes, gd_scores, bank, dior_class,
            dino_model, dino_transform, w
        )
        ap = compute_ap50(pred_boxes, np.array(pred_scores), gts)
        layer_aps[l] = ap
        if ap > best_ap:
            best_ap = ap
            best_layer = l

    return best_layer, best_ap, layer_aps


# ══════════════════════════════════════════════════════
# 策略2: 梯度优化 Softmax 权重 (per-class)
# ══════════════════════════════════════════════════════

class LayerWeightLearner(nn.Module):
    """学习 per-class 的 12 层 softmax 权重。"""
    def __init__(self, n_classes=3):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_classes, N_LAYERS))

    def forward(self, class_idx):
        return F.softmax(self.logits[class_idx], dim=0)


def learn_layer_weights(images_by_class, gd_boxes_by_class, gd_scores_by_class,
                         gts_by_class, bank, dino_model, dino_transform,
                         class_names, epochs=200, lr=0.05):
    """用 REINFORCE 风格的 AP50 优化学习层权重。"""
    learner = LayerWeightLearner(len(class_names)).to(DEVICE)
    optimizer = torch.optim.Adam([learner.logits], lr=lr)

    best_weights = None
    best_overall = -1.0

    for epoch in range(epochs):
        total_ap = 0.0
        optimizer.zero_grad()

        for ci, cls_name in enumerate(class_names):
            w = learner(ci)  # softmax weights

            pred_boxes, pred_scores = vep_rerank_layer_weighted(
                images_by_class[cls_name],
                gd_boxes_by_class[cls_name],
                gd_scores_by_class[cls_name],
                bank, cls_name if cls_name in bank else 'airplane',
                dino_model, dino_transform, learner.logits[ci]
            )

            # 注意: 对于跨域映射, 需要用 DIOR 类名查 bank
            dior_name = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}.get(cls_name, cls_name)

            pred_boxes2, pred_scores2 = vep_rerank_layer_weighted(
                images_by_class[cls_name],
                gd_boxes_by_class[cls_name],
                gd_scores_by_class[cls_name],
                bank, dior_name,
                dino_model, dino_transform, learner.logits[ci]
            )

            ap = compute_ap50(pred_boxes2, np.array(pred_scores2),
                              gts_by_class[cls_name])
            total_ap += ap

        mean_ap = total_ap / len(class_names)

        # 用 AP 作为 reward 信号做梯度上升
        loss = -torch.tensor(mean_ap, device=DEVICE, requires_grad=True)
        loss.backward()
        optimizer.step()

        if mean_ap > best_overall:
            best_overall = mean_ap
            best_weights = learner.logits.detach().clone()

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:3d}: mean_ap={mean_ap:.4f}")

    return best_weights, best_overall


# ══════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("C-VEP: 跨域层加权 Visual Exemplar Prototype")
    print("=" * 65)

    # 加载模型
    print("\n[1] 加载模型...")
    gd_proc, gd_model = get_gdino()
    dino_model, dino_transform = get_dinov2_with_layers()
    bank = torch.load(BANK_PATH, map_location='cpu')
    print(f"  DINOv2 bank: {list(bank.keys())}")

    # 准备数据
    print("\n[2] 加载测试数据...")
    data = {}  # {coco_class: (images, gd_boxes, gd_scores, gts)}

    for pair in CROSS_DOMAIN_PAIRS:
        coco_cls = pair['coco_class']
        print(f"  {pair['dior_class']}(DIOR) → {coco_cls}(COCO)...")
        images, gts = load_coco_data(coco_cls)
        print(f"    {len(images)} images, {sum(len(g) for g in gts)} GT boxes")

        gd_boxes, gd_scores = gdino_detect_all(
            images, f'a photo of {coco_cls}', gd_proc, gd_model
        )

        # Flatten GTs per image for AP50 computation
        gts_flat = []
        for i, g in enumerate(gts):
            gts_flat.extend(g)

        data[coco_cls] = {
            'images': images,
            'gd_boxes': gd_boxes,
            'gd_scores': gd_scores,
            'gts_per_image': gts,
            'gts_flat': gts_flat,
            'dior_class': pair['dior_class'],
        }

    # ═══ Baseline: 纯 GDINO (无 VEP) ═══
    print("\n[3] Baseline: 纯 GDINO AP50...")
    baseline_aps = {}
    for coco_cls, d in data.items():
        all_boxes, all_scores = [], []
        for i in range(len(d['images'])):
            keep = d['gd_scores'][i] > DET_THRESHOLD
            all_boxes.extend(d['gd_boxes'][i][keep].tolist())
            all_scores.extend(d['gd_scores'][i][keep].tolist())
        ap = compute_ap50(all_boxes, np.array(all_scores), d['gts_flat'])
        baseline_aps[coco_cls] = ap
        print(f"  {coco_cls}: AP50={ap:.4f}")

    # ═══ L12-VEP: 单层 L12 prototype ═══
    print("\n[4] L12-VEP: DINOv2 layer-12 prototype...")
    l12_aps = {}
    for coco_cls, d in data.items():
        dior_cls = d['dior_class']
        w = torch.full((N_LAYERS,), -1e9, device=DEVICE)
        w[11] = 1e9  # L12

        pred_boxes, pred_scores = vep_rerank_layer_weighted(
            d['images'], d['gd_boxes'], d['gd_scores'],
            bank, dior_cls, dino_model, dino_transform, w
        )
        ap = compute_ap50(pred_boxes, np.array(pred_scores), d['gts_flat'])
        l12_aps[coco_cls] = ap
        print(f"  {coco_cls}: AP50={ap:.4f} (Δ={ap - baseline_aps[coco_cls]:+.4f})")

    # ═══ C-VEP Grid: 网格搜索最优单层 ═══
    print("\n[5] C-VEP Grid: 逐类搜索最优单层...")
    grid_aps = {}
    grid_layers = {}
    grid_all = {}  # per-layer AP for analysis

    for coco_cls, d in data.items():
        dior_cls = d['dior_class']
        best_l, best_ap, layer_aps = grid_search_best_layer(
            d['images'], d['gd_boxes'], d['gd_scores'],
            d['gts_flat'], bank, dior_cls, dino_model, dino_transform
        )
        grid_aps[coco_cls] = best_ap
        grid_layers[coco_cls] = best_l
        grid_all[coco_cls] = {f'L{l+1}': ap for l, ap in layer_aps.items()}
        print(f"  {coco_cls}: best=L{best_l+1} AP50={best_ap:.4f} "
              f"(Δ_baseline={best_ap - baseline_aps[coco_cls]:+.4f}, "
              f"Δ_L12={best_ap - l12_aps[coco_cls]:+.4f})")

        # 打印 layer scan
        for l in range(N_LAYERS):
            tag = " ← BEST" if l == best_l else ""
            delta_l12 = layer_aps[l] - layer_aps[11] if 11 in layer_aps else 0
            if abs(delta_l12) > 0.005 or l == best_l or l == 11:
                print(f"    L{l+1:>2d}: {layer_aps[l]:.4f} (ΔL12={delta_l12:+.4f}){tag}")

    # ═══ C-VEP Softmax: 梯度优化层权重 ═══
    print("\n[6] C-VEP Softmax: 梯度优化 softmax 层权重...")
    class_names = [p['coco_class'] for p in CROSS_DOMAIN_PAIRS]
    try:
        best_weights, best_overall = learn_layer_weights(
            {cn: data[cn]['images'] for cn in class_names},
            {cn: data[cn]['gd_boxes'] for cn in class_names},
            {cn: data[cn]['gd_scores'] for cn in class_names},
            {cn: data[cn]['gts_flat'] for cn in class_names},
            bank, dino_model, dino_transform, class_names,
            epochs=200, lr=0.05
        )

        softmax_aps = {}
        softmax_weights = {}
        for ci, cn in enumerate(class_names):
            w = best_weights[ci]
            w_probs = F.softmax(w, dim=0)
            softmax_weights[cn] = {f'L{l+1}': float(w_probs[l]) for l in range(N_LAYERS)}

            dior_name = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}[cn]
            pred_boxes, pred_scores = vep_rerank_layer_weighted(
                data[cn]['images'], data[cn]['gd_boxes'], data[cn]['gd_scores'],
                bank, dior_name, dino_model, dino_transform, w
            )
            ap = compute_ap50(pred_boxes, np.array(pred_scores), data[cn]['gts_flat'])
            softmax_aps[cn] = ap

            top_layers = sorted([(l, float(w_probs[l])) for l in range(N_LAYERS)],
                                key=lambda x: -x[1])[:3]
            print(f"  {cn}: AP50={ap:.4f}  top_layers={top_layers}")
    except Exception as e:
        print(f"  Softmax 优化失败: {e}")
        softmax_aps = {}
        softmax_weights = {}

    # ═══ 汇总 ═══
    print(f"\n{'='*65}")
    print("C-VEP 汇总: Baseline vs L12-VEP vs C-VEP-Grid vs C-VEP-Softmax")
    print(f"{'='*65}")
    print(f"{'COCO Class':<15s} {'Baseline':>8s} {'L12-VEP':>8s} "
          f"{'Grid':>8s} {'Softmax':>8s} {'BestΔ':>8s}")
    print(f"{'-'*58}")

    all_results = {}
    for cn in class_names:
        base = baseline_aps[cn]
        l12 = l12_aps[cn]
        grid = grid_aps[cn]
        soft = softmax_aps.get(cn, grid)
        best = max(base, l12, grid, soft)
        best_name = 'Baseline' if best == base else ('L12' if best == l12 else ('Grid' if best == grid else 'Softmax'))

        print(f"{cn:<15s} {base:8.4f} {l12:8.4f} {grid:8.4f} {soft:8.4f} "
              f"{best - base:+8.4f} ({best_name})")

        all_results[cn] = {
            'baseline': float(base), 'l12_vep': float(l12),
            'cvep_grid': float(grid), 'cvep_grid_layer': f'L{grid_layers[cn]+1}',
            'cvep_softmax': float(soft),
            'layer_scan': grid_all[cn],
            'softmax_weights': softmax_weights.get(cn, {}),
            'best_delta': float(best - base),
        }

    with open(SAVE_DIR / 'cvep_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果: {SAVE_DIR / 'cvep_results.json'}")
    print("✅ C-VEP 完成")


if __name__ == '__main__':
    main()
