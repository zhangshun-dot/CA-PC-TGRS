# -*- coding: utf-8 -*-
"""DPS: Domain Prefix Selector — 量化前缀选择对 GDINO 检测 AP50 的影响

之前的实验证明:
  - 修饰词 ("large and prominent" 等) 对 GDINO AP50 无增益
  - 但域前缀 ("an overhead image of" vs "a photo of") 是有效的跨域信号

本实验量化: 选对前缀 vs 选错前缀 对 AP50 的影响有多大。

对比:
  A. 固定自然场景前缀:  "a photo of {cls}"
  B. 固定遥感前缀:      "an overhead image of {cls}"
  C. CLIP 域分类自动选择 (DPS)
  D. Oracle (永远选正确前缀)

测试: DIOR test (airplane/ship/vehicle) + COCO val (airplane/boat/car)
"""

import os
os.environ['HF_HUB_OFFLINE'] = '1'

import json, glob, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
import pandas as pd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 路径 ──
CLIP_PATH = r'C:\Users\张三\Desktop\weights\clip\ViT-B-32.pt'
GDINO_CACHE = (r'C:\Users\张三\.cache\huggingface\hub'
               r'\models--IDEA-Research--grounding-dino-tiny'
               r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')
COCO_IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
COCO_ANN_FILE = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
DIOR_IMG_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
DIOR_PARQUET = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'


# ══════════════════════════════════════════════════════
# 域分类器 (复用 domain_classifier 逻辑)
# ══════════════════════════════════════════════════════

class SimpleDomainClassifier:
    """简化版 CLIP 域分类器: 遥感 vs 自然场景 (2-way)。"""

    DOMAIN_PROMPTS = {
        'remote_sensing': [
            'a remote sensing image',
            'a satellite image of terrain',
            'an aerial view photograph',
            'an overhead image from above',
        ],
        'natural_scene': [
            'a natural scene photograph',
            'a ground-level photo',
            'an everyday scene image',
            'a picture taken from eye level',
        ],
    }

    DOMAIN_PREFIXES = {
        'remote_sensing': 'an overhead image of',
        'natural_scene': 'a photo of',
    }

    def __init__(self, clip_model, device, preprocess):
        import clip as clip_module
        self.device = device
        self.clip_model = clip_model
        self._tokenize = clip_module.tokenize
        self._preprocess = preprocess

        # 预计算所有域 prompt 的文本嵌入
        self.text_features = {}
        for domain, prompts in self.DOMAIN_PROMPTS.items():
            feats = []
            for p in prompts:
                tok = self._tokenize([p]).to(device)
                with torch.no_grad():
                    f = clip_model.encode_text(tok).float()
                    f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f)
            self.text_features[domain] = torch.stack(feats)  # (N_prompts, 512)

    def classify(self, image, return_scores=False):
        """分类单张图像 → 域标签。"""
        img_t = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_f = self.clip_model.encode_image(img_t).float()
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        scores = {}
        for domain, text_feats in self.text_features.items():
            # text_feats: (N_prompts, 1, 512) → (N_prompts, 512)
            tf = text_feats.squeeze(1) if text_feats.dim() == 3 else text_feats
            sims = (img_f @ tf.T).squeeze(0)  # (1, 512) @ (512, N) → (1, N) → (N,)
            scores[domain] = float(sims.max())  # max over prompts

        best = max(scores, key=scores.get)
        if return_scores:
            return best, scores
        return best

    def get_prefix(self, image):
        """分类后返回对应的域前缀。"""
        domain = self.classify(image)
        return self.DOMAIN_PREFIXES[domain], domain


# ══════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════

def load_coco_eval_data(class_name, n=30):
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


def load_dior_eval_data(class_name, n=20):
    test_ids = set()
    for f in glob.glob(os.path.join(DIOR_IMG_DIR, '*.jpg')):
        parts = Path(f).stem.split('_')
        for part in reversed(parts):
            if part.isdigit():
                test_ids.add(int(part))
                break

    df = pd.read_parquet(DIOR_PARQUET)
    DIOR_ID2NAME = {
        0: 'airplane', 1: 'airport', 2: 'baseball field', 3: 'basketball court',
        4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'expressway service area',
        8: 'expressway toll station', 9: 'golf course', 10: 'ground track field',
        11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
        15: 'storage tank', 16: 'tennis court', 17: 'train station',
        18: 'vehicle', 19: 'windmill',
    }
    NAME2ID = {v: k for k, v in DIOR_ID2NAME.items()}
    cls_id = NAME2ID.get(class_name)
    if cls_id is None:
        return [], []

    gt_index = defaultdict(list)
    for _, row in df.iterrows():
        img_id = int(row['image_id'])
        if img_id not in test_ids:
            continue
        for i, cat in enumerate(row['objects']['category']):
            if int(cat) == cls_id:
                x, y, w, h = [int(v) for v in row['objects']['bbox'][i]]
                gt_index[img_id].append([x, y, x + w, y + h])

    # 找图像文件
    patterns = [
        os.path.join(DIOR_IMG_DIR, f"{class_name}_*.jpg"),
        os.path.join(DIOR_IMG_DIR, f"{class_name.replace(' ', '_')}_*.jpg"),
    ]
    paths = []
    for pat in patterns:
        paths = sorted(glob.glob(pat))
        if paths:
            break

    images, gt_boxes_all = [], []
    for p in paths:
        parts = Path(p).stem.split('_')
        img_id = None
        for part in reversed(parts):
            if part.isdigit():
                img_id = int(part)
                break
        if img_id is None or img_id not in test_ids:
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


# ══════════════════════════════════════════════════════
# GDINO + AP50
# ══════════════════════════════════════════════════════

def get_gdino():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    proc = AutoProcessor.from_pretrained(GDINO_CACHE)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_CACHE, dtype=torch.float32
    ).eval().to(DEVICE)
    return proc, model


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


def gdino_detect_ap50(images, prompt, gd_proc, gd_model, gts_flat, threshold=0.25):
    """对一批图像用同一 prompt 跑 GDINO，返回 AP50。"""
    all_boxes, all_scores = [], []
    for img in images:
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=threshold, target_sizes=ts)[0]
        all_boxes.extend(r['boxes'].cpu().numpy().tolist())
        all_scores.extend(r['scores'].cpu().numpy().tolist())
    return compute_ap50(all_boxes, np.array(all_scores), gts_flat)


# ══════════════════════════════════════════════════════
# DPS per-image 前缀选择
# ══════════════════════════════════════════════════════

def evaluate_prefix_strategies(images, class_name, true_domain, gd_proc, gd_model,
                                dps_classifier, gts_per_image):
    """
    四种前缀策略对比:
      A. 固定自然前缀:  "a photo of {cls}"
      B. 固定遥感前缀:  "an overhead image of {cls}"
      C. DPS 自动选择 (per-image)
      D. Oracle (永久正确前缀)
    """
    gts_flat = []
    for g in gts_per_image:
        gts_flat.extend(g)

    prefix_ns = 'a photo of'
    prefix_rs = 'an overhead image of'
    true_prefix = prefix_rs if true_domain == 'remote_sensing' else prefix_ns

    # A & B & D: 统一 prompt，batch 检测
    ap_ns = gdino_detect_ap50(images, f'{prefix_ns} {class_name}', gd_proc, gd_model, gts_flat)
    ap_rs = gdino_detect_ap50(images, f'{prefix_rs} {class_name}', gd_proc, gd_model, gts_flat)
    ap_oracle = gdino_detect_ap50(images, f'{true_prefix} {class_name}', gd_proc, gd_model, gts_flat)

    # C: DPS per-image 选择 → 需要逐张检测
    all_boxes, all_scores = [], []
    dps_choices = {'remote_sensing': 0, 'natural_scene': 0}

    for i, img in enumerate(images):
        prefix, domain_pred = dps_classifier.get_prefix(img)
        prompt = f'{prefix} {class_name}'
        dps_choices[domain_pred] += 1

        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            o = gd_model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(DEVICE)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=ts)[0]
        all_boxes.extend(r['boxes'].cpu().numpy().tolist())
        all_scores.extend(r['scores'].cpu().numpy().tolist())

    ap_dps = compute_ap50(all_boxes, np.array(all_scores), gts_flat)
    dps_accuracy = dps_choices[true_domain] / len(images)

    return {
        'natural_prefix': ap_ns,
        'remote_prefix': ap_rs,
        'dps': ap_dps,
        'oracle': ap_oracle,
        'dps_accuracy': dps_accuracy,
        'n_images': len(images),
    }


# ══════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("DPS: Domain Prefix Selector — 量化前缀选择的影响")
    print("=" * 65)

    # ── 加载模型 ──
    print("\n[1] 加载 CLIP + GDINO...")
    import clip as clip_module
    clip_model, preprocess = clip_module.load(CLIP_PATH, device=DEVICE)
    clip_model.eval()
    gd_proc, gd_model = get_gdino()
    dps = SimpleDomainClassifier(clip_model, DEVICE, preprocess)
    print("  DPS 就绪")

    # ── 测试配置 ──
    test_configs = [
        # COCO (自然场景 — 正确前缀是 "a photo of")
        {'name': 'COCO-airplane', 'cls': 'airplane', 'true_domain': 'natural_scene',
         'loader': lambda: load_coco_eval_data('airplane', 30)},
        {'name': 'COCO-boat', 'cls': 'boat', 'true_domain': 'natural_scene',
         'loader': lambda: load_coco_eval_data('boat', 30)},
        {'name': 'COCO-car', 'cls': 'car', 'true_domain': 'natural_scene',
         'loader': lambda: load_coco_eval_data('car', 30)},
        # DIOR (遥感 — 正确前缀是 "an overhead image of")
        {'name': 'DIOR-airplane', 'cls': 'airplane', 'true_domain': 'remote_sensing',
         'loader': lambda: load_dior_eval_data('airplane', 20)},
        {'name': 'DIOR-ship', 'cls': 'ship', 'true_domain': 'remote_sensing',
         'loader': lambda: load_dior_eval_data('ship', 20)},
        {'name': 'DIOR-vehicle', 'cls': 'vehicle', 'true_domain': 'remote_sensing',
         'loader': lambda: load_dior_eval_data('vehicle', 20)},
    ]

    all_results = {}

    for cfg in test_configs:
        print(f"\n{'─'*50}")
        print(f"🔬 {cfg['name']} (真域={cfg['true_domain']})")
        print(f"{'─'*50}")

        images, gts = cfg['loader']()
        if len(images) < 5:
            print(f"  跳过 (太少图像)")
            continue

        gts_flat = []
        for g in gts:
            gts_flat.extend(g)
        print(f"  {len(images)} images, {len(gts_flat)} GT boxes")

        results = evaluate_prefix_strategies(
            images, cfg['cls'], cfg['true_domain'],
            gd_proc, gd_model, dps, gts
        )

        print(f"  域分类准确率: {results['dps_accuracy']:.1%}")
        print(f"  {'─'*45}")
        print(f"  自然前缀 (a photo of):       {results['natural_prefix']:.4f}")
        print(f"  遥感前缀 (an overhead of):   {results['remote_prefix']:.4f}")
        print(f"  DPS 自动选择:               {results['dps']:.4f}")
        print(f"  Oracle (正确前缀):          {results['oracle']:.4f}")

        # 分析前缀错配对 AP50 的损害
        if cfg['true_domain'] == 'natural_scene':
            wrong_penalty = results['natural_prefix'] - results['remote_prefix']
        else:
            wrong_penalty = results['remote_prefix'] - results['natural_prefix']
        print(f"  选错前缀的 AP50 损失: {wrong_penalty:+.4f}")

        all_results[cfg['name']] = {
            'class': cfg['cls'],
            'true_domain': cfg['true_domain'],
            **{k: float(v) if isinstance(v, (np.floating, float)) else v
               for k, v in results.items()},
        }

    # ── 汇总 ──
    print(f"\n{'='*65}")
    print("DPS 汇总")
    print(f"{'='*65}")
    print(f"{'Test':<18s} {'NatPref':>8s} {'RemPref':>8s} {'DPS':>8s} {'Oracle':>8s} {'Acc':>6s} {'Penalty':>8s}")
    print(f"{'─'*68}")

    for name, r in all_results.items():
        wrong_delta = abs(r['natural_prefix'] - r['remote_prefix'])
        print(f"{name:<18s} {r['natural_prefix']:8.4f} {r['remote_prefix']:8.4f} "
              f"{r['dps']:8.4f} {r['oracle']:8.4f} {r['dps_accuracy']:5.1%} {wrong_delta:8.4f}")

    # 关键问题: DPS 是否接近 Oracle?
    dps_deltas = [r['oracle'] - r['dps'] for r in all_results.values()]
    wrong_deltas = [abs(r['natural_prefix'] - r['remote_prefix']) for r in all_results.values()]
    print(f"\n  平均 DPS→Oracle 差距: {np.mean(dps_deltas):.4f}")
    print(f"  平均 选错前缀 损失:   {np.mean(wrong_deltas):.4f}")

    with open(SAVE_DIR / 'dps_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果: {SAVE_DIR / 'dps_results.json'}")
    print("✅ DPS 完成")


if __name__ == '__main__':
    main()
