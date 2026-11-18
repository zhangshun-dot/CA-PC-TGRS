# -*- coding: utf-8 -*-
"""VEP 消融实验: 验证跨域原型的真实贡献

对照设计:
  A. GDINO baseline (no re-rank)
  B. VEP with DIOR prototype (actual VEP)
  C. VEP with random prototype (768-d 随机向量 → 相当于随机重排)
  D. VEP with anti-prototype (反向原型 → 负向信号)
  E. Shuffle baseline (直接 shuffle GDINO scores → 纯随机重排)
  F. VEP with COCO prototype (in-domain oracle → 理论上界)
  G. VEP with L6 prototype (浅层, 已知无区分力 → 接近随机)

如果 VEP 真正在做跨域语义匹配:
  - B 应该 >> C (随机原型不应有效)
  - B 应该 >> E (纯随机重排不应有效)
  - D 应该 << A (反向信号应更差)
  - F 应该 > B (域内匹配应优于跨域匹配)
  - G 应该 ≈ E (浅层无区分力)

如果 VEP 只是"分数重排的数学效应":
  - C 和 E 的 AP50 也可能改善 (任何重排")
  - B ≈ C ≈ E
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

N_IMAGES = 20
N_SEEDS = 5  # 随机实验重复次数


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
    """11-point interpolated AP at IoU≥0.5"""
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


def gdino_detect_all(img, prompt, gd_proc, gd_model):
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
    return boxes[keep], scores[keep]


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
    return feats  # list of 12 tensors


def evaluate_rerank(images, gts_per_image, class_name, gd_proc, gd_model,
                     dino_model, dino_transform, proto, layer_idx=10, seed=None):
    """用给定原型和层重打分，返回 AP50。

    proto: 768-d tensor (normalized)
    layer_idx: 0-11
    """
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []

    for img, gts in zip(images, gts_per_image):
        prompt = f'a photo of {class_name}'
        boxes, scores = gdino_detect_all(img, prompt, gd_proc, gd_model)

        if len(boxes) == 0:
            all_gt_boxes.extend(gts)
            continue

        # 计算每个框的 fused score
        fused_scores = []
        for box in boxes:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                fused_scores.append(0.0)
                continue
            crop = img.crop((x1, y1, x2, y2))
            feats = encode_crop_per_layer(crop, dino_model, dino_transform)
            vs = max(0.0, float((feats[layer_idx] @ proto).item()))
            fused_scores.append(vs)  # pure visual sim (no gd fusion, to isolate proto effect)

        all_pred_boxes.extend(boxes.tolist())
        all_pred_scores.extend(fused_scores)
        all_gt_boxes.extend(gts)

    return compute_ap50(all_pred_boxes, np.array(all_pred_scores), all_gt_boxes)


def evaluate_baseline(images, gts_per_image, class_name, gd_proc, gd_model):
    """GDINO baseline AP50"""
    all_pred_boxes, all_pred_scores, all_gt_boxes = [], [], []

    for img, gts in zip(images, gts_per_image):
        prompt = f'a photo of {class_name}'
        boxes, scores = gdino_detect_all(img, prompt, gd_proc, gd_model)

        all_pred_boxes.extend(boxes.tolist())
        all_pred_scores.extend(scores.tolist())
        all_gt_boxes.extend(gts)

    return compute_ap50(all_pred_boxes, np.array(all_pred_scores), all_gt_boxes)


def main():
    print("=" * 70)
    print("VEP 消融实验: DIOR 原型是真的在跨域匹配, 还是随机重排?")
    print("=" * 70)

    gd_proc, gd_model = get_gdino()
    dino_model, dino_transform = get_dinov2_with_layers()
    bank = torch.load(BANK_PATH, map_location='cpu')

    CLASS_MAP = {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'}
    results = {}

    for coco_cls in ['airplane', 'boat', 'car']:
        dior_cls = CLASS_MAP[coco_cls]
        images, gts = load_coco_data(coco_cls)
        print(f"\n{'='*70}")
        print(f"  {coco_cls} ({len(images)} images, {sum(len(g) for g in gts)} GT boxes)")
        print(f"{'='*70}")

        # A: Baseline
        base_ap = evaluate_baseline(images, gts, coco_cls, gd_proc, gd_model)
        print(f"  [A] GDINO baseline:       AP50 = {base_ap:.4f}")

        # F: COCO prototype (in-domain oracle) — 用 COCO GT crops 做原型
        # Build COCO prototype from GT crops
        coco_gts = []
        for img, gt_list in zip(images, gts):
            for gt_box in gt_list:
                x1, y1, x2, y2 = [int(v) for v in gt_box]
                if x2 > x1 + 10 and y2 > y1 + 10:
                    crop = img.crop((x1, y1, x2, y2))
                    feats = encode_crop_per_layer(crop, dino_model, dino_transform)
                    coco_gts.append(feats)
                    if len(coco_gts) >= 5:
                        break
            if len(coco_gts) >= 5:
                break

        if len(coco_gts) >= 3:
            # L11 prototype
            coco_proto_l11 = torch.stack([f[10] for f in coco_gts]).mean(dim=0)
            coco_proto_l11 = coco_proto_l11 / coco_proto_l11.norm()
            coco_ap_l11 = evaluate_rerank(images, gts, coco_cls, gd_proc, gd_model,
                                           dino_model, dino_transform, coco_proto_l11, 10)
            print(f"  [F] COCO in-domain proto (L11): AP50 = {coco_ap_l11:.4f}  "
                  f"(理论上界, COCO→COCO)")

            # L12 prototype
            coco_proto_l12 = torch.stack([f[11] for f in coco_gts]).mean(dim=0)
            coco_proto_l12 = coco_proto_l12 / coco_proto_l12.norm()
            coco_ap_l12 = evaluate_rerank(images, gts, coco_cls, gd_proc, gd_model,
                                           dino_model, dino_transform, coco_proto_l12, 11)
            print(f"  [F] COCO in-domain proto (L12): AP50 = {coco_ap_l12:.4f}")
        else:
            coco_proto_l11 = None
            coco_proto_l12 = None
            coco_ap_l11 = coco_ap_l12 = 0

        # B: DIOR prototype VEP
        dior_proto_l11 = bank[dior_cls][10]['prototype']  # L11
        dior_proto_l12 = bank[dior_cls][11]['prototype']  # L12

        dior_ap_l11 = evaluate_rerank(images, gts, coco_cls, gd_proc, gd_model,
                                       dino_model, dino_transform, dior_proto_l11, 10)
        dior_ap_l12 = evaluate_rerank(images, gts, coco_cls, gd_proc, gd_model,
                                       dino_model, dino_transform, dior_proto_l12, 11)
        print(f"  [B] DIOR cross-domain proto (L11): AP50 = {dior_ap_l11:.4f}")
        print(f"  [B] DIOR cross-domain proto (L12): AP50 = {dior_ap_l12:.4f}")

        # C: Random proto (多次平均)
        rand_aps_l11 = []
        rand_aps_l12 = []
        for seed in range(N_SEEDS):
            rand_proto = torch.randn(768)
            rand_proto = rand_proto / rand_proto.norm()
            rand_aps_l11.append(
                evaluate_rerank(images, gts, coco_cls, gd_proc, gd_model,
                                dino_model, dino_transform, rand_proto, 10))
            rand_proto2 = torch.randn(768)
            rand_proto2 = rand_proto2 / rand_proto2.norm()
            rand_aps_l12.append(
                evaluate_rerank(images, gts, coco_cls, gd_proc, gd_model,
                                dino_model, dino_transform, rand_proto2, 11))
        rand_mean_l11 = np.mean(rand_aps_l11)
        rand_std_l11 = np.std(rand_aps_l11)
        rand_mean_l12 = np.mean(rand_aps_l12)
        rand_std_l12 = np.std(rand_aps_l12)
        print(f"  [C] Random proto (L11, n={N_SEEDS}):   AP50 = {rand_mean_l11:.4f} ± {rand_std_l11:.4f}")
        print(f"  [C] Random proto (L12, n={N_SEEDS}):   AP50 = {rand_mean_l12:.4f} ± {rand_std_l12:.4f}")

        # D: Anti-proto (DIOR prototype × -1)
        anti_proto_l11 = -dior_proto_l11
        anti_ap_l11 = evaluate_rerank(images, gts, coco_cls, gd_proc, gd_model,
                                       dino_model, dino_transform, anti_proto_l11, 10)
        anti_proto_l12 = -dior_proto_l12
        anti_ap_l12 = evaluate_rerank(images, gts, coco_cls, gd_proc, gd_model,
                                       dino_model, dino_transform, anti_proto_l12, 11)
        print(f"  [D] Anti-proto (DIOR×-1, L11):  AP50 = {anti_ap_l11:.4f}")
        print(f"  [D] Anti-proto (DIOR×-1, L12):  AP50 = {anti_ap_l12:.4f}")

        # E: Score shuffle (纯随机重排 GDINO scores, 多次平均)
        shuffle_aps = []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed)
            all_boxes, all_scores, all_gts = [], [], []
            for img, img_gts in zip(images, gts):
                prompt = f'a photo of {coco_cls}'
                boxes, scores = gdino_detect_all(img, prompt, gd_proc, gd_model)
                shuffled = rng.permutation(scores) if len(scores) > 0 else scores
                all_boxes.extend(boxes.tolist())
                all_scores.extend(shuffled.tolist())
                all_gts.extend(img_gts)
            shuffle_aps.append(compute_ap50(all_boxes, np.array(all_scores), all_gts))
        shuffle_mean = np.mean(shuffle_aps)
        shuffle_std = np.std(shuffle_aps)
        print(f"  [E] Shuffle GDINO scores (n={N_SEEDS}): AP50 = {shuffle_mean:.4f} ± {shuffle_std:.4f}")

        # G: L6 proto (浅层，已知无区分力)
        dior_proto_l6 = bank[dior_cls][5]['prototype']
        l6_ap = evaluate_rerank(images, gts, coco_cls, gd_proc, gd_model,
                                dino_model, dino_transform, dior_proto_l6, 5)
        print(f"  [G] DIOR L6 proto (浅层无区分):   AP50 = {l6_ap:.4f}")

        # ──── 汇总 ────
        results[coco_cls] = {
            'baseline': base_ap,
            'dior_L11': dior_ap_l11, 'dior_L12': dior_ap_l12,
            'coco_L11': coco_ap_l11, 'coco_L12': coco_ap_l12,
            'random_L11': rand_mean_l11, 'random_L12': rand_mean_l12,
            'anti_L11': anti_ap_l11, 'anti_L12': anti_ap_l12,
            'shuffle': shuffle_mean,
            'L6': l6_ap,
        }

        # 关键判断
        print(f"\n  ──── 跨域信号检验 ────")
        # 如果 DIOR proto 真的在跨域匹配:
        #   dior_ap 应该接近 coco_ap (in-domain oracle)
        #   dior_ap 应该 > random_ap (随机原型不应有效)
        #   dior_ap 应该 > anti_ap (方向很重要)
        #   dior_ap 应该 > L6 (浅层无信号)
        cross_domain_gain = dior_ap_l11 - base_ap
        in_domain_gap = coco_ap_l11 - dior_ap_l11
        random_gap = dior_ap_l11 - rand_mean_l11
        anti_gap = dior_ap_l11 - anti_ap_l11
        l6_gap = dior_ap_l11 - l6_ap

        print(f"  跨域增益 (DIOR L11 - Baseline):     {cross_domain_gain:+.4f}")
        print(f"  域差距 (COCO proto - DIOR proto):    {in_domain_gap:+.4f}")
        print(f"  方向信号 (DIOR - Anti):              {anti_gap:+.4f}")
        print(f"  语义信号 (DIOR L11 - Random):        {random_gap:+.4f}")
        print(f"  层级信号 (DIOR L11 - L6):            {l6_gap:+.4f}")

        verdicts = []
        if anti_gap > 0.02:
            verdicts.append("✅方向敏感(proto方向→分数排序有意义)")
        else:
            verdicts.append("❌方向不敏感(proto方向无信息)")

        if random_gap > 0.02:
            verdicts.append("✅语义信号(类原型→超越随机)")
        else:
            verdicts.append("❌无语义信号(≈随机向量重排)")

        if l6_gap > 0.02:
            verdicts.append("✅层级信号(深层→优于浅层)")
        else:
            verdicts.append("❌无层级信号(浅层≈深层)")

        if in_domain_gap > 0.05:
            verdicts.append("⚠️域差距大(COCO→COCO远好于DIOR→COCO)")
        else:
            verdicts.append("✅域差距小(DIOR原型接近COCO原型效果)")

        print(f"  判定: {' | '.join(verdicts)}")

    # ──── 最终结论 ────
    print(f"\n{'='*70}")
    print("消融实验结论")
    print(f"{'='*70}")

    print(f"\n{'Class':<12s} {'Base':>8s} {'DIOR ':>8s} {'COCO ':>8s} {'Random':>8s} "
          f"{'Anti':>8s} {'Shuffle':>8s} {'L6':>8s} {'Δ(COCO-DIOR)':>14s}")
    print("-" * 90)
    for c in ['airplane', 'boat', 'car']:
        r = results[c]
        print(f"{c:<12s} {r['baseline']:8.4f} {r['dior_L11']:8.4f} {r['coco_L11']:8.4f} "
              f"{r['random_L11']:8.4f} {r['anti_L11']:8.4f} {r['shuffle']:8.4f} "
              f"{r['L6']:8.4f} {r['coco_L11']-r['dior_L11']:+14.4f}")

    print(f"\n逐类分析:")
    for c in ['airplane', 'boat', 'car']:
        r = results[c]
        cross_gain = r['dior_L11'] - r['baseline']
        domain_gap = r['coco_L11'] - r['dior_L11']
        semantic = r['dior_L11'] - r['random_L11']
        print(f"  {c}: 跨域增益={cross_gain:+.3f}, 域差距={domain_gap:+.3f}, "
              f"语义超越随机={semantic:+.3f}")

    print(f"\n保存详细结果到 {SAVE_DIR / 'vep_ablation.json'}")
    # 让 JSON 可序列化
    json_results = {}
    for k, v in results.items():
        json_results[k] = {k2: float(v2) if isinstance(v2, (np.floating, float)) else v2
                           for k2, v2 in v.items()}
    with open(SAVE_DIR / 'vep_ablation.json', 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)


if __name__ == '__main__':
    main()
