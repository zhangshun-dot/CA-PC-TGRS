# -*- coding: utf-8 -*-
"""CDPS 端到端 GDINO 检测验证

验证 CDPS prompt 在真实检测器上的性能:
  Baseline:  "prefix cls"
  CD-TDM:    "prefix cls, + OFAT best modifiers"
  CDPS:      "prefix cls, + selector predicted modifiers"

测试: COCO val2017 (airplane/boat/car) + DIOR test (airplane/ship/vehicle)
指标: AP50
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, glob, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 模型加载 ──

def get_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gd_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
    proc = AutoProcessor.from_pretrained(gd_cache)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        gd_cache, dtype=torch.float32
    ).eval().to(DEVICE)
    return proc, model

def get_clip():
    import clip
    clip_path = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
    model, preprocess = clip.load(clip_path, device=DEVICE)
    model.eval()
    return model, preprocess

def load_cdps_selector():
    """加载训练好的 CDPS v2 selector。"""
    from cross_domain_prompt_selector_v2 import FiLMPromptSelector
    ckpt = torch.load(SAVE_DIR / 'cdps_v2_weights.pt', map_location='cpu')
    selector = FiLMPromptSelector(d_model=512, n_axes=6, n_domains=2,
                                   learnable_domain_emb=True, dropout=0.1)
    selector.load_state_dict(ckpt['selector_state'])
    selector.domain_to_idx = ckpt['domain_to_idx']
    selector.eval().to(DEVICE)
    return selector, ckpt


# ── AP50 计算 ──

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
    pred_boxes = [pred_boxes[i] for i in idx]
    gt_matched = [False] * len(gt_boxes)
    tp, fp = np.zeros(len(pred_boxes)), np.zeros(len(pred_boxes))
    for i, pb in enumerate(pred_boxes):
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


# ── GDINO 检测 ──

def gdino_detect_all(images, prompt, gd_proc, gd_model):
    """对每张图分别检测 → 返回 (boxes, scores) 列表。"""
    all_boxes, all_scores = [], []
    for img in images:
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(
            o, threshold=0.0, target_sizes=ts
        )[0]
        all_boxes.append(r['boxes'].cpu().numpy())
        all_scores.append(r['scores'].cpu().numpy())
    return all_boxes, all_scores


# ── COCO 数据 ──

def load_coco_eval_data(class_name, n=30):
    from pycocotools.coco import COCO
    ann_file = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
    img_dir = r'C:\Users\张三\modeling\coco_data\val2017'

    coco = COCO(ann_file)
    cat_ids = coco.getCatIds(catNms=[class_name])
    if not cat_ids:
        return [], [], []
    cat_id = cat_ids[0]

    img_ids = coco.getImgIds(catIds=[cat_id])
    images, gt_boxes_all = [], []
    for iid in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[iid], catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue
        img_info = coco.loadImgs([iid])[0]
        path = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            continue
        gts = []
        for a in anns:
            x, y, w, h = a['bbox']
            gts.append([x, y, x + w, y + h])
        images.append(img)
        gt_boxes_all.append(gts)
        if len(images) >= n:
            break
    return images, gt_boxes_all


# ── DIOR 数据 ──

def load_dior_eval_data(class_name, n=30):
    import pandas as pd
    from collections import defaultdict

    test_dir = r'C:\Users\张三\modeling\dior_data\full20_images'
    parquet_path = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'

    # 先找 test image ids
    test_ids = set()
    for f in glob.glob(os.path.join(test_dir, '*.jpg')):
        parts = Path(f).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                test_ids.add(int(part))
                break

    # 加载 GT
    df = pd.read_parquet(parquet_path)
    DIOR_ID2NAME = {
        0: 'airplane', 1: 'airport', 2: 'baseball field', 3: 'basketball court',
        4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'expressway service area',
        8: 'expressway toll station', 9: 'golf course', 10: 'ground track field',
        11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
        15: 'storage tank', 16: 'tennis court', 17: 'train station',
        18: 'vehicle', 19: 'windmill',
    }
    NAME2ID = {v: k for k, v in DIOR_ID2NAME.items()}
    cls_id = NAME2ID[class_name]

    gt_index = defaultdict(list)  # img_id → [boxes]
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        if img_id not in test_ids:
            continue
        for i, cat in enumerate(row['objects']['category']):
            if int(cat) == cls_id:
                x, y, w, h = [int(v) for v in row['objects']['bbox'][i]]
                gt_index[img_id].append([x, y, x + w, y + h])

    # 找图像
    pattern = os.path.join(test_dir, f"{class_name.replace(' ','_')}_*.jpg")
    paths = sorted(glob.glob(pattern))
    if not paths:
        pattern = os.path.join(test_dir, f"{class_name}_*.jpg")
        paths = sorted(glob.glob(pattern))

    images, gt_boxes_all = [], []
    for p in paths:
        parts = Path(p).stem.split('_')
        img_id = None
        for part in reversed(parts):
            if part.isdigit():
                img_id = int(part)
                break
        if img_id is None:
            continue
        gts = gt_index.get(img_id, [])
        if not gts:
            continue
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            continue
        images.append(img)
        gt_boxes_all.append(gts)
        if len(images) >= n:
            break
    return images, gt_boxes_all


# ── 主验证 ──

def main():
    print("=" * 65)
    print("CDPS 端到端 GDINO 检测验证")
    print("=" * 65)

    from token_projection import AXIS_MODIFIERS, AXIS_ORDER
    from ofat_knowledge_base import OFATKnowledgeBase
    kb = OFATKnowledgeBase()

    # 加载模型
    print("\n[1] 加载模型...")
    gd_proc, gd_model = get_gdino()
    clip_model, preprocess = get_clip()
    selector, ckpt = load_cdps_selector()
    print(f"  CDPS 域: {selector.domain_to_idx}")

    # 测试配置
    test_configs = [
        {'name': 'COCO-airplane', 'cls': 'airplane', 'domain': 'natural_scene',
         'domain_id': 1, 'loader': load_coco_eval_data, 'prefix': 'a photo of'},
        {'name': 'COCO-boat', 'cls': 'boat', 'domain': 'natural_scene',
         'domain_id': 1, 'loader': load_coco_eval_data, 'prefix': 'a photo of'},
        {'name': 'COCO-car', 'cls': 'car', 'domain': 'natural_scene',
         'domain_id': 1, 'loader': load_coco_eval_data, 'prefix': 'a photo of'},
        {'name': 'DIOR-airplane', 'cls': 'airplane', 'domain': 'remote_sensing',
         'domain_id': 0, 'loader': load_dior_eval_data, 'prefix': 'an overhead image of'},
        {'name': 'DIOR-ship', 'cls': 'ship', 'domain': 'remote_sensing',
         'domain_id': 0, 'loader': load_dior_eval_data, 'prefix': 'an overhead image of'},
        {'name': 'DIOR-vehicle', 'cls': 'vehicle', 'domain': 'remote_sensing',
         'domain_id': 0, 'loader': load_dior_eval_data, 'prefix': 'an overhead image of'},
    ]

    all_results = {}

    for cfg in test_configs:
        print(f"\n{'─'*50}")
        print(f"🔬 {cfg['name']}")
        print(f"{'─'*50}")

        # 加载数据
        images, gts = cfg['loader'](cfg['cls'], n=30)
        print(f"  图像: {len(images)}, GT boxes: {sum(len(g) for g in gts)}")

        if len(images) == 0:
            print(f"  跳过 (无数据)")
            continue

        # ── A. Baseline: 纯前缀 ──
        baseline_prompt = f"{cfg['prefix']} {cfg['cls']}"
        boxes_a, scores_a = gdino_detect_all(images, baseline_prompt, gd_proc, gd_model)

        aps_a = []
        for i in range(len(images)):
            b = boxes_a[i]
            s = scores_a[i]
            keep = s > 0.03
            ap = compute_ap50(b[keep].tolist(), s[keep].tolist(), gts[i])
            aps_a.append(ap)

        # ── B. CD-TDM: OFAT 最优修饰 ──
        recs = kb.get_recommended_axes(cfg['domain'], cfg['cls'])
        tdm_prompt = f"{cfg['prefix']} {cfg['cls']}"
        tdm_mods = []
        if recs:
            tdm_mods = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
            tdm_prompt += ', ' + ', '.join(tdm_mods)

        boxes_b, scores_b = gdino_detect_all(images, tdm_prompt, gd_proc, gd_model)
        aps_b = []
        for i in range(len(images)):
            b = boxes_b[i]
            s = scores_b[i]
            keep = s > 0.03
            ap = compute_ap50(b[keep].tolist(), s[keep].tolist(), gts[i])
            aps_b.append(ap)

        # ── C. CDPS: Selector 预测修饰 ──
        cdps_prompt = selector.build_prompt(
            cfg['cls'], cfg['domain_id'], clip_model,
            top_k=2, threshold=0.03
        )
        boxes_c, scores_c = gdino_detect_all(images, cdps_prompt, gd_proc, gd_model)
        aps_c = []
        for i in range(len(images)):
            b = boxes_c[i]
            s = scores_c[i]
            keep = s > 0.03
            ap = compute_ap50(b[keep].tolist(), s[keep].tolist(), gts[i])
            aps_c.append(ap)

        # 汇总
        mean_a, mean_b, mean_c = np.mean(aps_a), np.mean(aps_b), np.mean(aps_c)
        best = max(mean_a, mean_b, mean_c)
        delta_tdm = mean_b - mean_a
        delta_cdps = mean_c - mean_a

        print(f"  Baseline:  '{baseline_prompt}'")
        print(f"  CD-TDM:    '{tdm_prompt}'")
        print(f"  CDPS:      '{cdps_prompt}'")
        print(f"  {'─'*40}")
        print(f"  Baseline AP50:  {mean_a:.4f}")
        print(f"  CD-TDM AP50:    {mean_b:.4f}  (Δ={delta_tdm:+.4f})")
        print(f"  CDPS AP50:      {mean_c:.4f}  (Δ={delta_cdps:+.4f})")

        winner = 'Baseline' if best == mean_a else ('CD-TDM' if best == mean_b else 'CDPS')
        print(f"  🏆 最优: {winner}")

        all_results[cfg['name']] = {
            'class': cfg['cls'], 'domain': cfg['domain'],
            'n_images': len(images),
            'baseline_prompt': baseline_prompt,
            'tdm_prompt': tdm_prompt,
            'cdps_prompt': cdps_prompt,
            'baseline_ap50': float(mean_a),
            'tdm_ap50': float(mean_b),
            'cdps_ap50': float(mean_c),
            'tdm_delta': float(delta_tdm),
            'cdps_delta': float(delta_cdps),
            'per_image_aps': {
                'baseline': [float(x) for x in aps_a],
                'tdm': [float(x) for x in aps_b],
                'cdps': [float(x) for x in aps_c],
            },
        }

    # ── 汇总 ──
    print(f"\n{'='*65}")
    print("汇总: CDPS vs CD-TDM vs Baseline")
    print(f"{'='*65}")
    print(f"{'Test':<20s} {'Baseline':>9s} {'CD-TDM':>9s} {'CDPS':>9s} {'TDMΔ':>7s} {'CDPSΔ':>7s} {'Winner':>10s}")
    print(f"{'─'*75}")

    for name, r in all_results.items():
        tdm_d = r['tdm_ap50'] - r['baseline_ap50']
        cdps_d = r['cdps_ap50'] - r['baseline_ap50']
        best_val = max(r['baseline_ap50'], r['tdm_ap50'], r['cdps_ap50'])
        if best_val == r['baseline_ap50']:
            w = 'Baseline'
        elif best_val == r['tdm_ap50']:
            w = 'CD-TDM'
        else:
            w = 'CDPS'
        print(f"{name:<20s} {r['baseline_ap50']:9.4f} {r['tdm_ap50']:9.4f} "
              f"{r['cdps_ap50']:9.4f} {tdm_d:+7.4f} {cdps_d:+7.4f} {w:>10s}")

    # 平均
    avg_base = np.mean([r['baseline_ap50'] for r in all_results.values()])
    avg_tdm = np.mean([r['tdm_ap50'] for r in all_results.values()])
    avg_cdps = np.mean([r['cdps_ap50'] for r in all_results.values()])
    print(f"{'─'*75}")
    print(f"{'AVERAGE':<20s} {avg_base:9.4f} {avg_tdm:9.4f} {avg_cdps:9.4f} "
          f"{avg_tdm-avg_base:+7.4f} {avg_cdps-avg_base:+7.4f}")

    # 一致性检查
    n_match_tdm = sum(1 for r in all_results.values()
                      if abs(r['tdm_ap50'] - r['cdps_ap50']) < 0.005)
    print(f"\n  CDPS vs CD-TDM 一致性 (|ΔAP|<0.005): {n_match_tdm}/{len(all_results)}")

    with open(SAVE_DIR / 'cdps_e2e_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果: {SAVE_DIR / 'cdps_e2e_results.json'}")
    print("✅ 端到端验证完成")


if __name__ == '__main__':
    main()
