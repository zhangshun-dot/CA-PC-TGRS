# -*- coding: utf-8 -*-
"""COCO OFAT Measurement for Cross-Domain TDM.

Runs full OFAT scan on COCO 20 common classes:
    1. Find best domain prefix (5 candidates)
    2. Measure per-axis deltas (6 modifier axes)
    3. Register as 'natural_scene' domain

Uses GDINO-tiny for efficiency.
Estimated time: ~10 min (10 imgs/class) to ~20 min (20 imgs/class) on RTX 5060.
"""
import json, os, glob, numpy as np, torch, time
from PIL import Image
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
device = 'cuda'

# ── COCO 20 common classes ──
COCO_CLASSES = [
    'person', 'car', 'dog', 'chair', 'bottle', 'cup',
    'boat', 'airplane', 'bird', 'potted plant',
    'dining table', 'couch', 'motorcycle', 'cat', 'tv',
    'laptop', 'sink', 'refrigerator', 'book', 'clock',
]

# Domain prefix candidates for natural_scene
PREFIXES = [
    'a photo of',
    'an image of',
    'a picture of',
    'a natural image of',
]

# Modifier axes (same as DIOR OFAT)
AXES = {
    'granularity': 'with fine details',
    'spatial': 'in its natural setting',
    'geometric': 'with distinct structural features',
    'scale': 'large and prominent',
    'spectral': 'with natural coloring',
    'texture': 'with clearly visible surface features',
}

IMAGES_PER_CLASS = 20  # Adjust for speed vs accuracy
CONF_THRESHOLD = 0.25

# ── Load model ──
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Loading GDINO-tiny...")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

cache = (r'C:\Users\张三\.cache\huggingface\hub'
         r'\models--IDEA-Research--grounding-dino-tiny'
         r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')
proc = AutoProcessor.from_pretrained(cache)
model = (AutoModelForZeroShotObjectDetection
         .from_pretrained(cache, dtype=torch.float32)
         .eval().to(device))
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── Load COCO data ──
from pycocotools.coco import COCO
coco = COCO(r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
coco_dir = r'C:\Users\张三\modeling\coco_data\val2017'

# Get image IDs per class
cat_ids_all = coco.getCatIds(catNms=COCO_CLASSES)
cat_map = {}
for cid in cat_ids_all:
    cat_info = coco.loadCats([cid])[0]
    cat_map[cat_info['name']] = cid

# Collect images per class with ground truth boxes
class_images = defaultdict(list)
for cls_name in COCO_CLASSES:
    cid = cat_map.get(cls_name)
    if cid is None:
        print(f"  WARNING: {cls_name} not found in COCO")
        continue

    img_ids = coco.getImgIds(catIds=[cid])
    for img_id in img_ids[:IMAGES_PER_CLASS * 3]:  # oversample for filtering
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[cid])
        anns = coco.loadAnns(ann_ids)
        if anns:
            # Convert to [x1, y1, x2, y2]
            gt_boxes = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x + w, y + h])
            class_images[cls_name].append({
                'file': img_info['file_name'],
                'gt_boxes': gt_boxes,
            })
        if len(class_images[cls_name]) >= IMAGES_PER_CLASS:
            break
    print(f"  {cls_name}: {len(class_images[cls_name])} images")

# ── AP50 computation ──
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


def detect_all(image, prompt):
    """Return all boxes and scores above threshold."""
    inputs = proc(images=image, text=prompt, return_tensors='pt')
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    with torch.no_grad():
        o = model(**inputs)
    ts = torch.tensor([image.size[::-1]]).to(device)
    r = proc.post_process_grounded_object_detection(
        o, threshold=0.0, target_sizes=ts
    )[0]
    boxes = r['boxes'].cpu().numpy()
    scores = r['scores'].cpu().numpy()
    keep = scores > CONF_THRESHOLD
    return boxes[keep].tolist(), scores[keep].tolist()


def evaluate_prompt(images_info, prompt):
    """Evaluate AP50 on all images, aggregated across all GT boxes."""
    all_pred_boxes, all_pred_scores = [], []
    all_gt_boxes = []
    for info in images_info:
        img_path = os.path.join(coco_dir, info['file'])
        img = Image.open(img_path).convert('RGB')
        boxes, scores = detect_all(img, prompt)
        all_pred_boxes.extend(boxes)
        all_pred_scores.extend(scores)
        all_gt_boxes.extend(info['gt_boxes'])
    return compute_ap50(all_pred_boxes, np.array(all_pred_scores), all_gt_boxes)


# ═══════════════════════════════════════════════
# OFAT Measurement
# ═══════════════════════════════════════════════
print(f"\n{'='*60}")
print("COCO OFAT: 20-class full measurement")
print(f"{'='*60}")

ofat_results = {}
total_ops = len(COCO_CLASSES) * (len(PREFIXES) + len(AXES))
op_count = 0

for cls_name in COCO_CLASSES:
    imgs = class_images.get(cls_name, [])
    if len(imgs) < 3:
        print(f"  {cls_name}: too few images, skipping")
        continue

    t0 = time.time()

    # Step 1: Find best prefix
    best_prefix = None
    best_ap = -1.0
    prefix_scores = {}

    for prefix in PREFIXES:
        prompt = f'{prefix} {cls_name}'
        ap = evaluate_prompt(imgs, prompt)
        prefix_scores[prefix] = ap
        op_count += 1
        if ap > best_ap:
            best_ap = ap
            best_prefix = prefix

    base_ap = best_ap

    # Step 2: Measure per-axis deltas
    deltas = {}
    for axis_name, modifier_text in AXES.items():
        prompt = f'{best_prefix} {cls_name}, {modifier_text}'
        ap_mod = evaluate_prompt(imgs, prompt)
        deltas[axis_name] = ap_mod - base_ap
        op_count += 1

    elapsed = time.time() - t0
    n_pos = sum(1 for v in deltas.values() if v > 0)
    ofat_results[cls_name] = {
        'base': base_ap,
        'best_prefix': best_prefix,
        'prefix_scores': prefix_scores,
        'deltas': deltas,
    }

    print(f"  [{op_count}/{total_ops}] {cls_name:20s} prefix={best_prefix} "
          f"base={base_ap:.3f} pos_axes={n_pos}/6 ({elapsed:.1f}s)")

# ── Summary ──
mean_deltas = {}
for ax in AXES:
    vals = [c['deltas'][ax] for c in ofat_results.values()]
    mean_deltas[ax] = float(np.mean(vals))

# Find overall best prefix
prefix_wins = defaultdict(int)
for cls_data in ofat_results.values():
    prefix_wins[cls_data['best_prefix']] += 1
overall_best = max(prefix_wins, key=prefix_wins.get)

print(f"\n{'='*60}")
print("COCO OFAT SUMMARY")
print(f"{'='*60}")
print(f"Overall best prefix: '{overall_best}' ({prefix_wins[overall_best]}/{len(ofat_results)} classes)")
print(f"Mean axis deltas:")
for ax in AXES:
    v = mean_deltas[ax]
    sign = '+' if v > 0 else ' '
    bar = '#' * max(0, int(abs(v) * 50)) if v > 0 else '-' * max(0, int(abs(v) * 50))
    print(f"  {ax:15s} {sign}{v:.4f} {bar}")

# ── Save ──
output = {
    'domain': 'natural_scene',
    'prefix': overall_best,
    'dataset': 'COCO 2017 val',
    'model': 'GDINO-tiny',
    'n_images_per_class': IMAGES_PER_CLASS,
    'threshold': CONF_THRESHOLD,
    'classes': ofat_results,
    'mean_deltas': mean_deltas,
    'prefix_distribution': dict(prefix_wins),
}

out_path = r'C:\Users\张三\Desktop\实验数据\coco_ofat_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to: {out_path}")

# Register in domain registry
from domain_registry import DomainRegistry
dr = DomainRegistry()
dr.register('natural_scene', {
    'prefix': overall_best,
    'threshold': 0.03,
    'classes': ofat_results,
    'mean_deltas': mean_deltas,
}, metadata={
    'description': 'COCO 20-class natural scene',
    'dataset': 'COCO 2017 val',
    'encoder': 'BERT (GDINO-tiny)',
    'n_classes': len(ofat_results),
})
print(f"Registered 'natural_scene' domain")
print(f"Registered domains: {dr.registered_domains}")

print("\nDone.")
