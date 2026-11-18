# -*- coding: utf-8 -*-
"""COCO full metrics: DR/P/R/F1 for GDINO-tiny (satellite/overhead/TDM)"""
import json, os, numpy as np, torch, time
from PIL import Image
from collections import defaultdict
os.environ['HF_HUB_OFFLINE'] = '1'
device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pycocotools.coco import COCO

# ── Model ──
print("Loading GDINO-tiny...")
tiny_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
tiny_proc = AutoProcessor.from_pretrained(tiny_cache)
tiny_model = AutoModelForZeroShotObjectDetection.from_pretrained(tiny_cache, dtype=torch.float32).eval().to(device)
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── COCO data ──
COCO_CLASSES = ['person','car','dog','chair','bottle','cup','boat','airplane','bird',
                'potted plant','dining table','couch','motorcycle','cat','tv',
                'laptop','sink','refrigerator','book','clock']

ann_file = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
img_dir = r'C:\Users\张三\modeling\coco_data\val2017'

coco = COCO(ann_file)

# Build: per class, 20 images each with GT boxes
class_data = {}
for cls in COCO_CLASSES:
    cat_ids = coco.getCatIds(catNms=[cls])
    if not cat_ids:
        print(f"  WARNING: {cls} not in COCO categories")
        continue
    cat_id = cat_ids[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    imgs_meta = coco.loadImgs(img_ids)

    entries = []
    for meta in imgs_meta:
        if len(entries) >= 20:
            break
        path = os.path.join(img_dir, meta['file_name'])
        if not os.path.exists(path):
            continue
        # Get GT boxes for this class only
        ann_ids = coco.getAnnIds(imgIds=[meta['id']], catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']  # COCO format: x,y,w,h
            gt_boxes.append([x, y, x+w, y+h])
        if gt_boxes:
            entries.append((path, gt_boxes))

    if entries:
        class_data[cls] = entries
        print(f"  {cls}: {len(entries)} images, {sum(len(e[1]) for e in entries)} GT boxes")
    else:
        print(f"  {cls}: no valid images")

print(f"\nTotal classes with data: {len(class_data)}")

# ── Detection + Metrics ──
def detect_with_boxes(image_paths, prompt, proc, model, threshold=0.25):
    results = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            o = model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=threshold, target_sizes=ts)[0]
        results.append({
            'boxes': r['boxes'].cpu().tolist(),
            'scores': r['scores'].cpu().tolist(),
        })
    return results

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0

def compute_metrics(all_preds, all_gts, iou_threshold=0.25):
    total_tp = 0; total_fp = 0; total_fn = 0; any_detected = 0
    for pred, gt in zip(all_preds, all_gts):
        pred_boxes = pred['boxes']
        gt_boxes = gt
        if len(pred_boxes) > 0:
            any_detected += 1
        matched_gt = set()
        for pbox in pred_boxes:
            best_iou = 0; best_gi = -1
            for gi, gbox in enumerate(gt_boxes):
                if gi in matched_gt: continue
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou; best_gi = gi
            if best_iou >= iou_threshold:
                matched_gt.add(best_gi); total_tp += 1
            else:
                total_fp += 1
        total_fn += len(gt_boxes) - len(matched_gt)
    n = len(all_gts)
    total_gt_boxes = sum(len(g) for g in all_gts)
    return {
        'detection_rate': round(any_detected/n, 4) if n else 0,
        'precision': round(total_tp/(total_tp+total_fp), 4) if (total_tp+total_fp) else 0,
        'recall': round(total_tp/total_gt_boxes, 4) if total_gt_boxes else 0,
        'f1': round(2*total_tp/(2*total_tp+total_fp+total_fn), 4) if (2*total_tp+total_fp+total_fn) else 0,
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
    }

def run_exp(class_data_dict, prompt_fn, proc, model, threshold=0.25):
    per_class = {}
    all_preds_flat = []; all_gts_flat = []
    for cls_name, img_list in class_data_dict.items():
        paths = [item[0] for item in img_list]
        gt_list = [item[1] for item in img_list]
        prompt = prompt_fn(cls_name)
        preds = detect_with_boxes(paths, prompt, proc, model, threshold)
        m = compute_metrics(preds, gt_list)
        m['class'] = cls_name
        per_class[cls_name] = m
        all_preds_flat.extend(preds)
        all_gts_flat.extend(gt_list)
    overall = compute_metrics(all_preds_flat, all_gts_flat)
    mean_dr = np.mean([m['detection_rate'] for m in per_class.values()]) if per_class else 0
    return {'per_class': per_class, 'overall': overall, 'mean_dr': round(float(mean_dr), 4)}

# ── PROMOT DEFINITIONS ──
# Fixed modifiers for COCO (consistent with "固定修饰词方案")
FIXED_MODS = 'with distinct structural features, large and prominent'

def prompt_satellite(cls):
    return f'a satellite image of {cls}'

def prompt_overhead(cls):
    return f'an overhead image of {cls}'

def prompt_tdm(cls):
    return f'an overhead image of {cls}, {FIXED_MODS}'

# ═══════════════════════════════════════════
all_results = {}

for name, prompt_fn in [('satellite', prompt_satellite),
                         ('overhead', prompt_overhead),
                         ('tdm', prompt_tdm)]:
    print(f"\n{'='*60}")
    print(f"COCO GDINO-tiny: {name}")
    print(f"{'='*60}")
    r = run_exp(class_data, prompt_fn, tiny_proc, tiny_model)
    all_results[f'coco_{name}'] = r
    o = r['overall']
    print(f"  DR={r['mean_dr']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")

# ── Save ──
out = r'C:\Users\张三\Desktop\coco_full_metrics_results.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Saved: {out}")
print(f"\nSUMMARY (Table 6 COCO row):")
for name, r in all_results.items():
    o = r['overall']
    print(f"  {name}: DR={r['mean_dr']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")

# Compute oh/sat F1 ratio
sat_f1 = all_results['coco_satellite']['overall']['f1']
oh_f1 = all_results['coco_overhead']['overall']['f1']
tdm_f1 = all_results['coco_tdm']['overall']['f1']
print(f"\n  oh/sat F1 ratio: {oh_f1/max(sat_f1,0.0001):.2f}x")
print(f"  Table 6 row: sat={sat_f1:.4f} oh={oh_f1:.4f} TDM={tdm_f1:.4f} oh/sat={oh_f1/max(sat_f1,0.0001):.2f}x")

print(f"\nDone. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
