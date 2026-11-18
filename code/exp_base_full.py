# -*- coding: utf-8 -*-
"""Exp B: GDINO-base full metrics — DIOR 20class + COCO 20class, satellite/overhead/TDM"""
import json, torch, numpy as np, os, io, sys
from PIL import Image
from collections import defaultdict

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Load GDINO-base ──
print("Loading GDINO-base...")
base_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots\12bdfa3120f3e7ec7b434d90674b3396eccf88eb'
try:
    base_proc = AutoProcessor.from_pretrained(base_cache)
    base_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        base_cache, dtype=torch.float32).eval().to(device)
    print(f"  Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
except Exception as e:
    print(f"  GDINO-base not cached, trying download or fallback...")
    print(f"  Error: {e}")
    sys.exit(1)

# ── Helpers ──
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0

def compute_metrics(all_preds, all_gts):
    tp=fp=fn=0; n=len(all_gts)
    any_detected = 0
    for pred, gt in zip(all_preds, all_gts):
        if len(pred) > 0: any_detected += 1
        matched = set()
        for pb in pred:
            best=0; bg=-1
            for gi, gb in enumerate(gt):
                if gi in matched: continue
                v = compute_iou(pb, gb)
                if v > best: best=v; bg=gi
            if best >= 0.25: matched.add(bg); tp += 1
            else: fp += 1
        fn += len(gt) - len(matched)
    total_gt = sum(len(g) for g in all_gts)
    return {
        'dr': any_detected/n if n else 0,
        'p': tp/(tp+fp) if (tp+fp) else 0,
        'r': tp/total_gt if total_gt else 0,
        'f1': 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0,
        'tp': tp, 'fp': fp, 'fn': fn,
    }

def detect_batch(image_paths, prompt, proc, model):
    results = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad():
            o = model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
        results.append({'boxes': r['boxes'].cpu().tolist()})
    return results

# ── Prompt builders ──
def prompt_satellite(cls): return f'a satellite image of {cls}'
def prompt_overhead(cls): return f'an overhead image of {cls}'
def prompt_tdm(cls): return f'an overhead image of {cls}, with distinct structural features, large and prominent'

# ═══════════════════════════════════════════════════
# PART 1: DIOR 20-class full metrics
# ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 1: GDINO-base DIOR 20-class")
print("="*60)

import pyarrow.parquet as pq
DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

pq_file = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table = pq.read_table(pq_file); rows = table.to_pylist()
class_data = defaultdict(list)
for row in rows:
    img_bytes, objs = row['image']['bytes'], row['objects']
    if not objs or 'category' not in objs: continue
    gt_by_cat = defaultdict(list)
    for bbox, cat in zip(objs['bbox'], objs['category']):
        x,y,w,h = bbox; gt_by_cat[DIOR[cat]].append([x,y,x+w,y+h])
    for cat, boxes in gt_by_cat.items():
        if len(class_data[cat]) < 20:
            class_data[cat].append((img_bytes, boxes))

dior_results = {}
for name, prompt_fn in [('satellite', prompt_satellite),
                         ('overhead', prompt_overhead),
                         ('tdm', prompt_tdm)]:
    print(f"\nDIOR base: {name}")
    per_class = {}
    all_p, all_g = [], []
    for cls_name in DIOR:
        imgs = class_data.get(cls_name, [])
        if not imgs: continue
        preds_list, gts_list = [], []
        for img_bytes, gt_boxes in imgs:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            prompt = prompt_fn(cls_name)
            inputs = base_proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad():
                o = base_model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = base_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
            preds_list.append(r['boxes'].cpu().tolist())
            gts_list.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
        m = compute_metrics(preds_list, gts_list)
        m['class'] = cls_name
        per_class[cls_name] = m
        all_p.extend(preds_list); all_g.extend(gts_list)
    overall = compute_metrics(all_p, all_g)
    mean_dr = np.mean([m['dr'] for m in per_class.values()])
    dior_results[name] = {'per_class': per_class, 'overall': overall, 'mean_dr': float(mean_dr)}
    print(f"  OVERALL: DR={mean_dr:.4f} P={overall['p']:.4f} R={overall['r']:.4f} F1={overall['f1']:.4f}")

# ═══════════════════════════════════════════════════
# PART 2: COCO 20-class full metrics
# ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 2: GDINO-base COCO 20-class")
print("="*60)

from pycocotools.coco import COCO
COCO_CLASSES = ['person','car','dog','chair','bottle','cup','boat','airplane','bird',
                'potted plant','dining table','couch','motorcycle','cat','tv',
                'laptop','sink','refrigerator','book','clock']

ann_file = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
img_dir = r'C:\Users\张三\modeling\coco_data\val2017'
coco = COCO(ann_file)

coco_class_data = {}
for cls in COCO_CLASSES:
    cat_ids = coco.getCatIds(catNms=[cls])
    if not cat_ids: continue
    cat_id = cat_ids[0]
    img_ids = coco.getImgIds(catIds=[cat_id])
    imgs_meta = coco.loadImgs(img_ids)
    entries = []
    for meta in imgs_meta:
        if len(entries) >= 20: break
        path = os.path.join(img_dir, meta['file_name'])
        if not os.path.exists(path): continue
        ann_ids = coco.getAnnIds(imgIds=[meta['id']], catIds=[cat_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x+w, y+h])
        if gt_boxes:
            entries.append((path, gt_boxes))
    if entries:
        coco_class_data[cls] = entries
        print(f"  {cls}: {len(entries)} images, {sum(len(e[1]) for e in entries)} GT boxes")

coco_results = {}
for name, prompt_fn in [('satellite', prompt_satellite),
                         ('overhead', prompt_overhead),
                         ('tdm', prompt_tdm)]:
    print(f"\nCOCO base: {name}")
    per_class = {}
    all_p, all_g = [], []
    for cls_name in COCO_CLASSES:
        imgs = coco_class_data.get(cls_name, [])
        if not imgs: continue
        paths = [item[0] for item in imgs]
        gt_list = [item[1] for item in imgs]
        prompt = prompt_fn(cls_name)
        preds_dicts = detect_batch(paths, prompt, base_proc, base_model)
        preds = [r['boxes'] for r in preds_dicts]
        m = compute_metrics(preds, gt_list)
        m['class'] = cls_name
        per_class[cls_name] = m
        all_p.extend(preds); all_g.extend(gt_list)
    overall = compute_metrics(all_p, all_g)
    mean_dr = np.mean([m['dr'] for m in per_class.values()])
    coco_results[name] = {'per_class': per_class, 'overall': overall, 'mean_dr': float(mean_dr)}
    print(f"  OVERALL: DR={mean_dr:.4f} P={overall['p']:.4f} R={overall['r']:.4f} F1={overall['f1']:.4f}")

# ═══════════════════════════════════════════════════
# Save & Summary
# ═══════════════════════════════════════════════════
all_results = {'dior_base': dior_results, 'coco_base': coco_results}
out = r'C:\Users\张三\Desktop\base_full_results.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print("GDINO-BASE SUMMARY")
print(f"{'='*60}")
for dataset in ['dior_base', 'coco_base']:
    print(f"\n{dataset}:")
    print(f"  {'Method':<15} {'DR':>8} {'P':>8} {'R':>8} {'F1':>8}")
    for name in ['satellite', 'overhead', 'tdm']:
        if name in all_results[dataset]:
            o = all_results[dataset][name]['overall']
            d = all_results[dataset][name]['mean_dr']
            print(f"  {name:<15} {d:>8.4f} {o['p']:>8.4f} {o['r']:>8.4f} {o['f1']:>8.4f}")

print(f"\nSaved: {out}")
print(f"VRAM final: {torch.cuda.memory_allocated()/1e9:.1f}GB")
