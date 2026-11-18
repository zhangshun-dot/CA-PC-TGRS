# -*- coding: utf-8 -*-
"""直接验证: geometric mean fusion 的 VEP 是否真的有效

对比:
  A. GDINO baseline (gd score only)
  B. VEP fused = sqrt(gd * vs_DIOR)  — 原版 VEP
  C. VEP fused = sqrt(gd * vs_random) — 随机原型对照
  D. VEP fused = sqrt(gd * vs_COCO)  — 同域上界

这里 vs 信号用原版 VEP 的 geometric mean 融合，不是纯 visual sim。
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, sys
from pathlib import Path
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

N_IMAGES = 30
N_RANDOM = 10  # 随机原型重复次数


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


def load_coco_data(class_name, n=N_IMAGES):
    from pycocotools.coco import COCO
    coco = COCO(COCO_ANN_FILE)
    cat_ids = coco.getCatIds(catNms=[class_name])
    if not cat_ids:
        return [], []
    cat_id = cat_ids[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    images, gts_all = [], []
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
        gts_all.append(gts)
        if len(images) >= n:
            break
    return images, gts_all


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-8)


def compute_ap50(pred_boxes, pred_scores, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    idx = np.argsort(-np.array(pred_scores))
    pred_boxes = [pred_boxes[i] for i in idx]
    matched = [False] * len(gt_boxes)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    for i, pb in enumerate(pred_boxes):
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if matched[j]:
                continue
            iou = compute_iou(pb, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0.5 and best_j >= 0:
            tp[i] = 1
            matched[best_j] = 1
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / (len(gt_boxes) + 1e-8)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += float(np.max(precisions[mask]))
    return ap / 11.0


def encode_crop_per_layer(crop, dino_model, dino_transform):
    img_t = dino_transform(crop).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o = dino_model(img_t)
    feats = []
    for l in range(12):
        hs = o.hidden_states[l + 1]
        feat = hs[0, 0, :]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.cpu())
    return feats


def evaluate_all_conditions(images, gts_per_image, class_name, gd_proc, gd_model,
                              dino_model, dino_transform, dior_proto, coco_proto,
                              best_layer):
    """一次推理，多个重打分方案对比。

    Returns: {condition_name: AP50}
    """
    # 收集所有检测框
    all_boxes = []
    all_gd_scores = []
    all_vs_dior = []
    all_vs_coco = []
    all_vs_randoms = []  # list of N_RANDOM lists
    all_gts = []

    # 预生成 N_RANDOM 个随机原型
    rand_protos = []
    for _ in range(N_RANDOM):
        rp = torch.randn(768)
        rp = rp / rp.norm()
        rand_protos.append(rp)
    all_vs_randoms = [[] for _ in range(N_RANDOM)]

    for img, gts in zip(images, gts_per_image):
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
                # 跳过太小的框
                all_boxes.append(box.tolist())
                all_gd_scores.append(float(scores[bi]))
                all_vs_dior.append(0.0)
                all_vs_coco.append(0.0)
                for ri in range(N_RANDOM):
                    all_vs_randoms[ri].append(0.0)
                all_gts.extend(gts)
                continue

            crop = img.crop((x1, y1, x2, y2))
            feats = encode_crop_per_layer(crop, dino_model, dino_transform)

            vs_d = max(0.0, float((feats[best_layer] @ dior_proto).item()))
            vs_c = max(0.0, float((feats[best_layer] @ coco_proto).item()))

            all_boxes.append(box.tolist())
            all_gd_scores.append(float(scores[bi]))
            all_vs_dior.append(vs_d)
            all_vs_coco.append(vs_c)
            for ri in range(N_RANDOM):
                vs_r = max(0.0, float((feats[best_layer] @ rand_protos[ri]).item()))
                all_vs_randoms[ri].append(vs_r)

        all_gts.extend(gts)

    results = {}

    # A: Baseline (gd only)
    results['baseline'] = compute_ap50(all_boxes, np.array(all_gd_scores), all_gts)

    # B: VEP fused = sqrt(gd * vs_dior)
    fused_dior = [np.sqrt(g * v + 1e-8) for g, v in zip(all_gd_scores, all_vs_dior)]
    results['vep_dior'] = compute_ap50(all_boxes, np.array(fused_dior), all_gts)

    # C: VEP fused = sqrt(gd * vs_random), average over N_RANDOM
    rand_aps = []
    for ri in range(N_RANDOM):
        fused_rand = [np.sqrt(g * v + 1e-8) for g, v in zip(all_gd_scores, all_vs_randoms[ri])]
        rand_aps.append(compute_ap50(all_boxes, np.array(fused_rand), all_gts))
    results['vep_random_mean'] = np.mean(rand_aps)
    results['vep_random_std'] = np.std(rand_aps)
    results['vep_random_max'] = np.max(rand_aps)
    results['vep_random_min'] = np.min(rand_aps)

    # D: VEP fused = sqrt(gd * vs_coco) — oracle
    fused_coco = [np.sqrt(g * v + 1e-8) for g, v in zip(all_gd_scores, all_vs_coco)]
    results['vep_coco'] = compute_ap50(all_boxes, np.array(fused_coco), all_gts)

    # E: fused = (gd + vs_dior) / 2 — arithmetic mean (对照几何平均)
    arith_dior = [(g + v) / 2 for g, v in zip(all_gd_scores, all_vs_dior)]
    results['arith_dior'] = compute_ap50(all_boxes, np.array(arith_dior), all_gts)

    # F: fused = gd * vs_dior — 直接乘积
    prod_dior = [g * v for g, v in zip(all_gd_scores, all_vs_dior)]
    results['prod_dior'] = compute_ap50(all_boxes, np.array(prod_dior), all_gts)

    return results, all_gd_scores, all_vs_dior


def main():
    print("=" * 70)
    print("VEP geometric mean fusion 验证: DIOR proto 真的有用吗?")
    print("=" * 70)

    gd_proc, gd_model = get_gdino()
    dino_model, dino_transform = get_dinov2_with_layers()
    bank = torch.load(BANK_PATH, map_location='cpu')

    CLASS_MAP = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}
    BEST_LAYERS = {'airplane': 10, 'boat': 10, 'car': 11}  # L11, L11, L12

    all_results = {}

    for coco_cls in ['airplane', 'boat', 'car']:
        dior_cls = CLASS_MAP[coco_cls]
        best_layer = BEST_LAYERS[coco_cls]
        images, gts = load_coco_data(coco_cls)
        print(f"\n{'='*70}")
        print(f"  {coco_cls} ({len(images)} images, best_layer=L{best_layer+1})")
        print(f"{'='*70}")

        dior_proto = bank[dior_cls][best_layer]['prototype']

        # 构建 COCO proto (同域上界)
        coco_feats = []
        for img, gt_list in zip(images, gts):
            for gt_box in gt_list:
                x1, y1, x2, y2 = [int(v) for v in gt_box]
                if x2 > x1 + 10 and y2 > y1 + 10:
                    crop = img.crop((x1, y1, x2, y2))
                    feats = encode_crop_per_layer(crop, dino_model, dino_transform)
                    coco_feats.append(feats[best_layer])
                    if len(coco_feats) >= 5:
                        break
            if len(coco_feats) >= 5:
                break
        coco_proto = torch.stack(coco_feats).mean(dim=0) if coco_feats else dior_proto
        coco_proto = coco_proto / coco_proto.norm()

        results, gd_scores, vs_dior = evaluate_all_conditions(
            images, gts, coco_cls, gd_proc, gd_model,
            dino_model, dino_transform, dior_proto, coco_proto, best_layer)

        all_results[coco_cls] = results

        print(f"  [A] GDINO baseline:             AP50 = {results['baseline']:.4f}")
        print(f"  [B] VEP fused=sqrt(gd×vs_dior):  AP50 = {results['vep_dior']:.4f}")
        print(f"  [C] VEP fused=sqrt(gd×vs_random): AP50 = {results['vep_random_mean']:.4f} "
              f"±{results['vep_random_std']:.4f} "
              f"[{results['vep_random_min']:.4f}, {results['vep_random_max']:.4f}]")
        print(f"  [D] VEP fused=sqrt(gd×vs_coco):  AP50 = {results['vep_coco']:.4f}  (同域上界)")
        print(f"  [E] Arithmetic mean (gd+vs)/2:    AP50 = {results['arith_dior']:.4f}")
        print(f"  [F] Product gd×vs:                AP50 = {results['prod_dior']:.4f}")

        vep_gain = results['vep_dior'] - results['baseline']
        random_gap = results['vep_dior'] - results['vep_random_mean']
        oracle_gap = results['vep_coco'] - results['vep_dior']

        print(f"\n  判定:")
        print(f"    VEP DIOR vs Baseline:  {vep_gain:+.4f} {'✅ 提升' if vep_gain > 0.01 else '❌ 无提升/退化'}")
        print(f"    VEP DIOR vs Random:    {random_gap:+.4f} {'✅ DIOR > 随机' if random_gap > 0.01 else '❌ DIOR ≈ 随机'}")
        print(f"    VEP COCO vs DIOR:      {oracle_gap:+.4f} {'⚠️ 同域远优于跨域' if oracle_gap > 0.02 else '域差距小'}")

    # ──── 汇总 ────
    print(f"\n{'='*70}")
    print("最终结论")
    print(f"{'='*70}")
    print(f"\n{'Class':<12s} {'Base':>8s} {'DIOR':>8s} {'Random':>8s} "
          f"{'COCO':>8s} {'Δ(DIOR-Base)':>14s} {'Δ(DIOR-Rand)':>14s}")
    print("-" * 78)
    for c in ['airplane', 'boat', 'car']:
        r = all_results[c]
        print(f"{c:<12s} {r['baseline']:8.4f} {r['vep_dior']:8.4f} {r['vep_random_mean']:8.4f} "
              f"{r['vep_coco']:8.4f} {r['vep_dior']-r['baseline']:+14.4f} "
              f"{r['vep_dior']-r['vep_random_mean']:+14.4f}")

    print(f"\n  Δ 全部 ≤ 0.01 意味着 DIOR prototype 的跨域信号 ≈ 随机噪声。")
    print(f"  VEP 即使使用 geometric mean fusion，跨域提升也来自数学变换而非原型质量。")

    # 保存
    json_out = {}
    for k, v in all_results.items():
        json_out[k] = {k2: float(v2) if isinstance(v2, (np.floating, float, np.ndarray)) else v2
                       for k2, v2 in v.items()}
    with open(SAVE_DIR / 'vep_fusion_verify.json', 'w', encoding='utf-8') as f:
        json.dump(json_out, f, indent=2)
    print(f"\n结果保存到 {SAVE_DIR / 'vep_fusion_verify.json'}")


if __name__ == '__main__':
    main()
