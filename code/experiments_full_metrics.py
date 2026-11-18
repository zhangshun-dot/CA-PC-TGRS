# -*- coding: utf-8 -*-
"""Full experiments from parquet: Precision/Recall/F1/mAP."""
import json, os, numpy as np, torch, time
from PIL import Image
import pyarrow.parquet as pq
from collections import defaultdict
import io

os.environ['HF_HUB_OFFLINE'] = '1'
device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load models
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
print("Loading GDINO-tiny...")
tiny_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
tiny_proc = AutoProcessor.from_pretrained(tiny_cache)
tiny_model = AutoModelForZeroShotObjectDetection.from_pretrained(tiny_cache, dtype=torch.float32).eval().to(device)

print("Loading GDINO-base...")
base_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots'
base_path = os.path.join(base_cache, os.listdir(base_cache)[0])
base_proc = AutoProcessor.from_pretrained(base_path)
base_model = AutoModelForZeroShotObjectDetection.from_pretrained(base_path, dtype=torch.float32).eval().to(device)

# Load DIOR from parquet
DIOR_CLASSES = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

print("Loading DIOR from parquet...")
pq_file = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table = pq.read_table(pq_file)
rows = table.to_pylist()

# Build per-class: for each class, collect images that contain at least one GT box of that class
class_data = defaultdict(list)  # class -> [(image_bytes, gt_boxes_for_this_class)]
for row in rows:
    img_bytes = row['image']['bytes']
    objs = row['objects']
    if not objs or 'bbox' not in objs:
        continue
    # Group GT boxes by category
    gt_by_cat = defaultdict(list)
    for bbox, cat in zip(objs['bbox'], objs['category']):
        x, y, w, h = bbox
        gt_by_cat[cat].append([x, y, x+w, y+h])
    for cat_id, boxes in gt_by_cat.items():
        cls_name = DIOR_CLASSES[cat_id]
        if len(class_data[cls_name]) < 20:
            class_data[cls_name].append((img_bytes, boxes))

img_counts = {cls: len(data) for cls, data in class_data.items()}
print(f"Classes with GT: {sum(1 for v in img_counts.values() if v>0)}")
for cls, n in img_counts.items():
    if n > 0:
        print(f"  {cls}: {n} images")

# Detection + Metrics
def detect_with_boxes(images_bytes, prompt, proc, model, threshold=0.25):
    results = []
    for img_bytes in images_bytes:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad():
            o = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=threshold, target_sizes=target_sizes)[0]
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

# OFAT / TDM
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat_dior = json.load(f)['ofat_overhead']
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm(cls, ofat_data):
    if cls not in ofat_data:
        return f'an overhead image of {cls}'
    d = ofat_data[cls]['deltas']
    prefix = 'an overhead image of'
    pos = sorted([(ax, v) for ax, v in d.items() if ax != 'grammar' and v > 0.03], key=lambda x: -x[1])
    mods = [AXM[ax] for ax, _ in pos[:2]] if pos else []
    p = f'{prefix} {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

def run_exp(class_data_dict, prompt_fn, proc, model, threshold=0.25):
    per_class = {}
    all_preds_flat = []; all_gts_flat = []
    valid = {k: v for k, v in class_data_dict.items() if v}
    for cls_name, img_list in valid.items():
        img_bytes_list = [item[0] for item in img_list]
        gt_list = [item[1] for item in img_list]
        preds = detect_with_boxes(img_bytes_list, prompt_fn(cls_name), proc, model, threshold)
        m = compute_metrics(preds, gt_list)
        m['class'] = cls_name
        per_class[cls_name] = m
        all_preds_flat.extend(preds)
        all_gts_flat.extend(gt_list)
    overall = compute_metrics(all_preds_flat, all_gts_flat)
    mean_dr = np.mean([m['detection_rate'] for m in per_class.values()]) if per_class else 0
    return {'per_class': per_class, 'overall': overall, 'mean_dr': round(float(mean_dr), 4)}

all_results = {}

# ── EXP 1: DIOR tiny ──
print("\n" + "="*60)
print("EXP 1: DIOR GDINO-tiny")
for name, fn in [('satellite', lambda c: f'a satellite image of {c}'),
                  ('overhead', lambda c: f'an overhead image of {c}'),
                  ('tdm', lambda c: build_tdm(c, ofat_dior))]:
    print(f"  {name}...")
    r = run_exp(class_data, fn, tiny_proc, tiny_model)
    all_results[f'tiny_{name}'] = r
    o = r['overall']
    print(f"    DR={r['mean_dr']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")

# ── EXP 2: DIOR base ──
print("\n" + "="*60)
print("EXP 2: DIOR GDINO-base")
for name, fn in [('satellite', lambda c: f'a satellite image of {c}'),
                  ('overhead', lambda c: f'an overhead image of {c}'),
                  ('tdm', lambda c: build_tdm(c, ofat_dior))]:
    print(f"  {name}...")
    r = run_exp(class_data, fn, base_proc, base_model)
    all_results[f'base_{name}'] = r
    o = r['overall']
    print(f"    DR={r['mean_dr']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")

# ── EXP 3: Ablation - modifier count ──
print("\n" + "="*60)
print("EXP 3: Ablation (5 classes)")
ablation_cls = ['airplane','ship','vehicle','bridge','harbor']
ablation_data = {c: class_data[c] for c in ablation_cls if c in class_data and class_data[c]}
all_mods_list = ['with distinct structural features','large and prominent',
                 'with natural coloring','with fine details','with clearly visible surface features']
for n in range(6):
    mods = all_mods_list[:n]
    def make_fn(n_mods):
        return lambda c: f'an overhead image of {c}' + (', ' + ', '.join(all_mods_list[:n_mods]) if n_mods > 0 else '')
    fn = make_fn(n)
    r = run_exp(ablation_data, fn, tiny_proc, tiny_model)
    all_results[f'ablation_{n}mods'] = r
    o = r['overall']
    print(f"  {n} mods: DR={r['mean_dr']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")

# ── EXP 4: Threshold robustness ──
print("\n" + "="*60)
print("EXP 4: Threshold robustness")
robust_cls = ['airplane','ship','vehicle','bridge','harbor','windmill']
robust_data = {c: class_data[c] for c in robust_cls if c in class_data and class_data[c]}
for th in [0.15, 0.20, 0.25, 0.30, 0.35]:
    r_oh = run_exp(robust_data, lambda c: f'an overhead image of {c}', tiny_proc, tiny_model, threshold=th)
    r_tdm = run_exp(robust_data, lambda c: build_tdm(c, ofat_dior), tiny_proc, tiny_model, threshold=th)
    all_results[f'threshold_{th}_oh'] = r_oh
    all_results[f'threshold_{th}_tdm'] = r_tdm
    print(f"  th={th}: oh DR={r_oh['mean_dr']:.4f} F1={r_oh['overall']['f1']:.4f} | tdm DR={r_tdm['mean_dr']:.4f} F1={r_tdm['overall']['f1']:.4f}")

# ── Save ──
out = r'C:\Users\张三\Desktop\full_metrics_results.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Saved: {out}")
print(f"\nSUMMARY:")
for k in sorted(all_results.keys()):
    o = all_results[k]['overall']
    print(f"  {k:<30} DR={all_results[k]['mean_dr']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")
