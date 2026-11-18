# -*- coding: utf-8 -*-
"""VEP 工作机制深度诊断

核心问题: VEP 靠什么把 AP50 从 0.55 提升到 0.81?

诊断维度:
  1. 原型到底编码了什么? → 逐层 prototype 的类间/类内 cosine 结构
  2. TP vs FP 分离能力? → 正确框 vs 错误框的 visual similarity 分布
  3. GDINO 为什么失败? → GDINO 高分但 visual sim 低的框 (false positive)
  4. 为什么 L11 > L12? → L11/L12 对同一 crop 的编码差异
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

BANK_PATH = SAVE_DIR / 'exemplar_bank_k5_dinov2_layers.pt'
DINOV2_PATH = r'C:\Users\张三\.cache\modelscope\tiansz\dinov2-base'
GDINO_CACHE = (r'C:\Users\张三\.cache\huggingface\hub'
               r'\models--IDEA-Research--grounding-dino-tiny'
               r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')
COCO_IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
COCO_ANN_FILE = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'

N_IMAGES = 20  # 每诊断用 20 张图


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
# 数据
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
# IoU
# ══════════════════════════════════════════════════════

def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


# ══════════════════════════════════════════════════════
# 诊断1: 原型编码了什么?
# ══════════════════════════════════════════════════════

def diagnose_prototype_content(bank):
    """分析 prototype 的类间结构和逐层差异。

    Q: prototype 是纯粹的"类语义"还是混合了"域特征"?
    Q: 不同层的 prototype 之间 cosine similarity 如何变化?
    """
    print("=" * 65)
    print("诊断1: Prototype 编码了什么?")
    print("=" * 65)

    classes = ['airplane', 'ship', 'vehicle', 'stadium', 'bridge', 'harbor']

    # 1a: 类间 cosine (L12)
    print("\n  1a: L12 prototype 类间 cosine similarity:")
    print(f"     {'':>12s}", end="")
    for c in classes:
        print(f" {c:>10s}", end="")
    print()

    for c1 in classes:
        print(f"     {c1:>12s}", end="")
        p1_l12 = bank[c1][11]['prototype']
        for c2 in classes:
            p2_l12 = bank[c2][11]['prototype']
            cos = float((p1_l12 @ p2_l12).item())
            print(f" {cos:10.4f}", end="")
        print()

    # 1b: 同类逐层 cosine 变化 (prototype 在不同层的自我相似度)
    print("\n  1b: 同类 prototype 的逐层变化 (L1 vs L12 cosine):")
    for c in classes[:4]:
        p_l1 = bank[c][0]['prototype']
        p_l6 = bank[c][5]['prototype']
        p_l12 = bank[c][11]['prototype']
        cos_1_12 = float((p_l1 @ p_l12).item())
        cos_6_12 = float((p_l6 @ p_l12).item())
        print(f"     {c:>12s}: L1↔L12={cos_1_12:.4f}, L6↔L12={cos_6_12:.4f}")

    # 1c: 跨类逐层 cosine — 验证"浅层共享纹理、深层分离语义"
    print("\n  1c: 跨类 prototype 在不同层的 cosine (airplane vs ship):")
    for l in [0, 3, 5, 7, 9, 11]:
        cos = float((bank['airplane'][l]['prototype'] @ bank['ship'][l]['prototype']).item())
        bar = '█' * int((1 - cos) * 40)
        print(f"     L{l+1:>2d}: cos={cos:.4f}  分离度={1-cos:.4f} {bar}")


# ══════════════════════════════════════════════════════
# 诊断2: TP vs FP 分离能力
# ══════════════════════════════════════════════════════

def diagnose_tp_fp_separation(images, class_name, gd_proc, gd_model,
                                dino_model, dino_transform, bank):
    """分析 visual similarity 能否分离 TP 和 FP。

    对每张图:
      1. GDINO 检测 (threshold=0.0) → 所有候选框
      2. 对每个框: 计算 GDINO score, visual_sim (L11, L12)
      3. 对每个框: 判断是否匹配 GT (IoU≥0.5 → TP)
      4. 统计 TP/FP 的 score 和 visual_sim 分布
    """
    print(f"\n{'='*65}")
    print(f"诊断2: TP vs FP 分离能力 — {class_name}")
    print(f"{'='*65}")

    dior_class = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}[class_name]

    gd_all = []     # GDINO scores
    vs_l11_all = [] # L11 visual sim
    vs_l12_all = [] # L12 visual sim
    is_tp_all = []  # True/False

    for img_idx, img in enumerate(images):
        # GDINO 低阈值检测
        prompt = f'a photo of {class_name}'
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        boxes = r['boxes'].cpu().numpy()
        scores = r['scores'].cpu().numpy()

        keep = scores > 0.03
        boxes = boxes[keep]
        scores = scores[keep]

        if len(boxes) == 0:
            continue

        # 对每个框计算 visual sim
        for bi, box in enumerate(boxes):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue

            crop = img.crop((x1, y1, x2, y2))
            img_t = dino_transform(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                o_dino = dino_model(img_t)

            # L11 和 L12 的 visual sim
            for layer_idx, vs_list in [(10, vs_l11_all), (11, vs_l12_all)]:
                hs = o_dino.hidden_states[layer_idx + 1]
                feat = hs[0, 0, :]  # CLS token
                feat = feat / feat.norm(dim=-1, keepdim=True)
                proto = bank[dior_class][layer_idx]['prototype'].to(DEVICE)
                sim = max(0.0, float((feat @ proto).item()))
                vs_list.append(sim)

            gd_all.append(float(scores[bi]))

            # 判断 TP/FP
            is_tp = False
            for gt_box in []:  # placeholder — will fill below
                pass
            is_tp_all.append(False)  # will fix below

    # 实际上我们需要 per-image GT matching
    # 重新做一次，正确标记 TP/FP
    gd_all = []
    vs_l11_all = []
    vs_l12_all = []
    is_tp_all = []

    for img_idx, img in enumerate(images):
        prompt = f'a photo of {class_name}'
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        boxes = r['boxes'].cpu().numpy()
        scores = r['scores'].cpu().numpy()

        keep = scores > 0.03
        boxes = boxes[keep]
        scores = scores[keep]

        # 获取该图对应的 GT (需要从外部传入)
        # 这里简化: 直接从 load_coco_data 返回的 gts 中取
        # 由于我们在这个函数内没有 gts，先跳过，在主函数中做

    return None  # 主函数中实现


# ══════════════════════════════════════════════════════
# 诊断3: GDINO 失败案例分析
# ══════════════════════════════════════════════════════

def diagnose_failure_cases(images, gts_per_image, class_name,
                            gd_proc, gd_model, dino_model, dino_transform, bank):
    """找出 GDINO 失败但 VEP 挽回的典型案例。

    对每个框标注:
      - gd_score: GDINO 置信度
      - vs_l11, vs_l12: visual similarity
      - iou_max: 与最近 GT 的最大 IoU
      - is_tp: IoU ≥ 0.5
    """
    print(f"\n{'='*65}")
    print(f"诊断3+4: GDINO 失败案例 + 逐层差异 — {class_name}")
    print(f"{'='*65}")

    dior_class = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}[class_name]

    all_records = []  # [{gd, vs_l11, vs_l12, iou_max, is_tp, fused_l11, fused_l12}]

    for img_idx, (img, gts) in enumerate(zip(images, gts_per_image)):
        prompt = f'a photo of {class_name}'
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=ts)[0]
        boxes = r['boxes'].cpu().numpy()
        scores = r['scores'].cpu().numpy()

        keep = scores > 0.03
        boxes = boxes[keep]
        scores = scores[keep]

        for bi, box in enumerate(boxes):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue

            # IoU with best GT
            iou_max = 0.0
            for gt in gts:
                iou = compute_iou([x1, y1, x2, y2], gt)
                if iou > iou_max:
                    iou_max = iou

            crop = img.crop((x1, y1, x2, y2))
            img_t = dino_transform(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                o_dino = dino_model(img_t)

            vs_vals = {}
            for layer_idx in range(12):
                hs = o_dino.hidden_states[layer_idx + 1]
                feat = hs[0, 0, :]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                proto = bank[dior_class][layer_idx]['prototype'].to(DEVICE)
                sim = max(0.0, float((feat @ proto).item()))
                vs_vals[layer_idx] = sim

            gd = float(scores[bi])
            is_tp = iou_max >= 0.5

            all_records.append({
                'gd': gd, 'vs': vs_vals, 'iou': iou_max, 'is_tp': is_tp,
                'img_idx': img_idx, 'box': [x1, y1, x2, y2],
            })

    if not all_records:
        print("  无检测框")
        return

    # ── 3a: TP vs FP 的 visual similarity 分布 ──
    print(f"\n  3a: TP vs FP 数量: "
          f"TP={sum(1 for r in all_records if r['is_tp'])}, "
          f"FP={sum(1 for r in all_records if not r['is_tp'])}")

    for layer_idx in [5, 10, 11]:  # L6, L11, L12
        tp_vs = [r['vs'][layer_idx] for r in all_records if r['is_tp']]
        fp_vs = [r['vs'][layer_idx] for r in all_records if not r['is_tp']]

        tp_mean = np.mean(tp_vs) if tp_vs else 0
        fp_mean = np.mean(fp_vs) if fp_vs else 0
        separation = tp_mean - fp_mean

        print(f"  L{layer_idx+1}: TP_vs_mean={tp_mean:.4f}, FP_vs_mean={fp_mean:.4f}, "
              f"分离度={separation:+.4f}")

    # ── 3b: GDINO 高分但 visual sim 低的框 (FP 案例) ──
    print(f"\n  3b: GDINO 高分 + visual_sim 低的框 (典型 FP):")
    fp_records = [r for r in all_records if not r['is_tp'] and r['gd'] > 0.15]
    fp_records.sort(key=lambda r: -(r['gd'] - r['vs'][11]))  # GDINO-L12 sim gap

    for i, rec in enumerate(fp_records[:5]):
        vs_l6 = rec['vs'][5]
        vs_l11 = rec['vs'][10]
        vs_l12 = rec['vs'][11]
        fused_l11 = np.sqrt(rec['gd'] * max(vs_l11, 1e-6))
        fused_l12 = np.sqrt(rec['gd'] * max(vs_l12, 1e-6))
        print(f"    [{i}] gd={rec['gd']:.3f} vs(L6={vs_l6:.3f}, L11={vs_l11:.3f}, "
              f"L12={vs_l12:.3f}) iou={rec['iou']:.3f} "
              f"fused(L11={fused_l11:.3f}, L12={fused_l12:.3f}) "
              f"→ {'✅被VEP抑制' if fused_l11 < 0.25 else '❌仍误检'}")

    # ── 3c: GDINO 低分但 visual sim 高的框 (FN 挽回案例) ──
    print(f"\n  3c: GDINO 低分 + visual_sim 高的框 (FN 被 VEP 挽回):")
    fn_records = [r for r in all_records if r['is_tp'] and r['gd'] < 0.25]
    fn_records.sort(key=lambda r: r['vs'][11] - r['gd'])

    for i, rec in enumerate(fn_records[:5]):
        vs_l6 = rec['vs'][5]
        vs_l11 = rec['vs'][10]
        vs_l12 = rec['vs'][11]
        fused_l11 = np.sqrt(rec['gd'] * max(vs_l11, 1e-6))
        fused_l12 = np.sqrt(rec['gd'] * max(vs_l12, 1e-6))
        rescued_l11 = fused_l11 > 0.25
        rescued_l12 = fused_l12 > 0.25
        print(f"    [{i}] gd={rec['gd']:.3f} vs(L6={vs_l6:.3f}, L11={vs_l11:.3f}, "
              f"L12={vs_l12:.3f}) iou={rec['iou']:.3f} "
              f"fused(L11={fused_l11:.3f}[{'✅' if rescued_l11 else '❌'}], "
              f"L12={fused_l12:.3f}[{'✅' if rescued_l12 else '❌'}])")

    # ── 3d: L11 vs L12 差异分析 ──
    print(f"\n  3d: L11 vs L12 visual_sim 差异分析:")
    tp_records = [r for r in all_records if r['is_tp']]
    if tp_records:
        l11_better = sum(1 for r in tp_records if r['vs'][10] > r['vs'][11] + 0.01)
        l12_better = sum(1 for r in tp_records if r['vs'][11] > r['vs'][10] + 0.01)
        same = len(tp_records) - l11_better - l12_better
        print(f"    TP 框中: L11>L12: {l11_better}, L12>L11: {l12_better}, ≈: {same}")
        print(f"    TP mean vs: L11={np.mean([r['vs'][10] for r in tp_records]):.4f}, "
              f"L12={np.mean([r['vs'][11] for r in tp_records]):.4f}")

    fp_records = [r for r in all_records if not r['is_tp']]
    if fp_records:
        l11_better_fp = sum(1 for r in fp_records if r['vs'][10] > r['vs'][11] + 0.01)
        l12_better_fp = sum(1 for r in fp_records if r['vs'][11] > r['vs'][10] + 0.01)
        print(f"    FP 框中: L11>L12: {l11_better_fp}, L12>L11: {l12_better_fp}")
        print(f"    FP mean vs: L11={np.mean([r['vs'][10] for r in fp_records]):.4f}, "
              f"L12={np.mean([r['vs'][11] for r in fp_records]):.4f}")

    # ── 汇总统计 ──
    print(f"\n  汇总:")
    for layer_idx in [5, 10, 11]:
        tp = [r for r in all_records if r['is_tp']]
        fp = [r for r in all_records if not r['is_tp']]
        # 用 fused score 计算 rescue rate
        fused_tp = [np.sqrt(r['gd'] * max(r['vs'][layer_idx], 1e-6)) for r in tp]
        fused_fp = [np.sqrt(r['gd'] * max(r['vs'][layer_idx], 1e-6)) for r in fp]
        tp_rescued = sum(1 for f in fused_tp if f > 0.25)
        fp_suppressed = sum(1 for f in fused_fp if f < 0.25)
        print(f"    L{layer_idx+1}: TP保住={tp_rescued}/{len(tp)}, "
              f"FP抑制={fp_suppressed}/{len(fp)}, "
              f"fused_TP_mean={np.mean(fused_tp):.4f}, "
              f"fused_FP_mean={np.mean(fused_fp):.4f}")


# ══════════════════════════════════════════════════════
# 诊断5: 跨域层偏移的根源
# ══════════════════════════════════════════════════════

def diagnose_layer_shift(images, gts_per_image, class_name, dino_model, dino_transform, bank):
    """分析为什么 airplane L11 > L12，而 car L12 最优。

    假设: 浅层关注纹理/局部结构 → 跨域时对纹理变化更敏感
          深层关注全局语义 → 跨域时更鲁棒但对细节不敏感
          L11 可能是"语义足够、纹理不过拟合"的最佳折中
    """
    print(f"\n{'='*65}")
    print(f"诊断5: 逐层特征分析 — {class_name}")
    print(f"{'='*65}")

    dior_class = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}[class_name]

    # 对 GT crops 做逐层 self-similarity 分析
    # 提取每层的 crop 特征，看层间 correlation

    gt_crops_features = {l: [] for l in range(12)}

    for img, gts in zip(images, gts_per_image):
        for gt in gts:
            x1, y1, x2, y2 = [int(v) for v in gt]
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crop = img.crop((x1, y1, x2, y2))
            img_t = dino_transform(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                o = dino_model(img_t)
            for l in range(12):
                hs = o.hidden_states[l + 1]
                feat = hs[0, 0, :]
                feat = feat / feat.norm(dim=-1, keepdim=True)
                gt_crops_features[l].append(feat.cpu())

            if len(gt_crops_features[0]) >= 20:
                break
        if len(gt_crops_features[0]) >= 20:
            break

    n_crops = len(gt_crops_features[0])
    if n_crops < 3:
        print(f"  GT crops 不足 ({n_crops})")
        return

    # 层间 correlation: 同一 crop 在不同层的特征相似度
    print(f"\n  层间特征一致性 (同一 GT crop 在不同层的 cosine, n={n_crops}):")
    layer_self_cos = np.zeros((12, 12))
    for l1 in range(12):
        for l2 in range(12):
            cos_vals = []
            for i in range(n_crops):
                cos = float((gt_crops_features[l1][i] @ gt_crops_features[l2][i]).item())
                cos_vals.append(cos)
            layer_self_cos[l1, l2] = np.mean(cos_vals)

    # 打印关键层
    for l in [0, 5, 10, 11]:
        print(f"    L{l+1} vs L6: {layer_self_cos[l, 5]:.4f}, "
              f"vs L11: {layer_self_cos[l, 10]:.4f}, "
              f"vs L12: {layer_self_cos[l, 11]:.4f}")

    # 跨域 prototype-crop 匹配: DIOR prototype 与 COCO crop 的 cosine
    print(f"\n  跨域匹配: DIOR prototype vs COCO GT crop cosine:")
    for l in [5, 9, 10, 11]:
        proto = bank[dior_class][l]['prototype']
        cos_vals = []
        for i in range(n_crops):
            cos = float((gt_crops_features[l][i] @ proto).item())
            cos_vals.append(max(0, cos))
        print(f"    L{l+1}: cos_mean={np.mean(cos_vals):.4f}, "
              f"cos_std={np.std(cos_vals):.4f}, "
              f"min={np.min(cos_vals):.4f}, max={np.max(cos_vals):.4f}")

    # 层间 prototype-crop 匹配的方差 → 高方差 = 对 crop 变化敏感
    print(f"\n  各层匹配稳定性 (cos 的 CV = std/mean):")
    for l in [5, 9, 10, 11]:
        proto = bank[dior_class][l]['prototype']
        cos_vals = [max(0, float((gt_crops_features[l][i] @ proto).item()))
                    for i in range(n_crops)]
        cv = np.std(cos_vals) / (np.mean(cos_vals) + 1e-8)
        print(f"    L{l+1}: mean={np.mean(cos_vals):.4f}, CV={cv:.4f}")


# ══════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("VEP 工作机制深度诊断")
    print("=" * 65)

    # 加载
    print("\n加载模型...")
    gd_proc, gd_model = get_gdino()
    dino_model, dino_transform = get_dinov2_with_layers()
    bank = torch.load(BANK_PATH, map_location='cpu')

    # ── 诊断1: Prototype 内容 ──
    diagnose_prototype_content(bank)

    # ── 诊断2+3+4: TP/FP 分离 + 失败案例 + L11 vs L12 ──
    for class_name in ['airplane', 'boat', 'car']:
        images, gts = load_coco_data(class_name)
        if len(images) < 5:
            continue
        diagnose_failure_cases(images, gts, class_name,
                               gd_proc, gd_model, dino_model, dino_transform, bank)

    # ── 诊断5: 层偏移根源 ──
    for class_name in ['airplane', 'boat', 'car']:
        images, gts = load_coco_data(class_name, n=10)
        if len(images) < 3:
            continue
        diagnose_layer_shift(images, gts, class_name,
                             dino_model, dino_transform, bank)

    print(f"\n{'='*65}")
    print("✅ VEP 诊断完成")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
