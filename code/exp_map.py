# -*- coding: utf-8 -*-
"""Add mAP/AP50/AP75 to all main tables. GDINO-tiny, DIOR 20-class, full metrics."""

import json, torch, numpy as np, io, os, sys
from PIL import Image
from collections import defaultdict

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pyarrow.parquet as pq

tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
proc = AutoProcessor.from_pretrained(tc)
model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# Load data
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

np.random.seed(42)
test_data = {}
for cls_name, imgs in class_data.items():
    idx = np.random.permutation(len(imgs))
    n_test = min(10, len(imgs)//2)
    test_data[cls_name] = [imgs[i] for i in idx[:n_test]]

ALL_CLASSES = sorted(test_data.keys())
print(f"Test set: {len(ALL_CLASSES)} classes, {sum(len(v) for v in test_data.values())} images")

# ═══════════════════════════════════════════
# Detection with confidence scores
# ═══════════════════════════════════════════
def detect_with_scores(prompt, img_list):
    """Returns list of (boxes, scores, gt_boxes) per image"""
    results = []
    for img_bytes, gt_boxes in img_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad(): o = model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=target)[0]
        results.append({
            'boxes': r['boxes'].cpu().tolist(),
            'scores': r['scores'].cpu().tolist(),
            'gt': [(b[0],b[1],b[2],b[3]) for b in gt_boxes],
        })
    return results

def iou(b1, b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

# ═══════════════════════════════════════════
# Standard DR/P/R/F1 + mAP computation
# ═══════════════════════════════════════════
def compute_all_metrics(det_results, iou_threshold=0.25):
    """Compute DR/P/R/F1 + AP at given IoU threshold"""
    tp=fp=fn=0; n=len(det_results); dr_count=0
    all_detections = []  # for mAP: (image_id, box, score, matched_gt)

    for img_idx, det in enumerate(det_results):
        boxes = det['boxes']
        scores = det['scores']
        gt_boxes = det['gt']

        if len(boxes) > 0: dr_count += 1

        # Sort by confidence
        sorted_idx = np.argsort(scores)[::-1]
        boxes_sorted = [boxes[i] for i in sorted_idx]
        scores_sorted = [scores[i] for i in sorted_idx]

        matched_gt = set()
        for box, score in zip(boxes_sorted, scores_sorted):
            best_iou = 0; best_gi = -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt: continue
                v = iou(box, gt)
                if v > best_iou: best_iou = v; best_gi = gi
            is_tp = best_iou >= iou_threshold
            if is_tp:
                matched_gt.add(best_gi); tp += 1
            else:
                fp += 1
            all_detections.append({
                'img_idx': img_idx, 'box': box, 'score': score,
                'tp': is_tp, 'iou': best_iou
            })
        fn += len(gt_boxes) - len(matched_gt)

    total_gt = sum(len(d['gt']) for d in det_results)
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/total_gt if total_gt else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0

    # Compute AP via 11-point interpolation
    dets_sorted = sorted(all_detections, key=lambda x: x['score'], reverse=True)
    n_gt = total_gt
    tp_cum = 0; fp_cum = 0
    precisions = []; recalls = []

    for d in dets_sorted:
        if d['tp']: tp_cum += 1
        else: fp_cum += 1
        precisions.append(tp_cum / (tp_cum + fp_cum))
        recalls.append(tp_cum / n_gt if n_gt else 0)

    # 11-point interpolation
    ap = 0.0
    for t_recall in np.linspace(0, 1, 11):
        prec_at_recall = max([p for p, r in zip(precisions, recalls) if r >= t_recall] + [0])
        ap += prec_at_recall / 11.0

    return {'dr': dr_count/n, 'p': round(p,4), 'r': round(r,4), 'f1': round(f1,4),
            'ap': round(ap,4), 'tp': tp, 'fp': fp, 'fn': fn}

def compute_map_per_class(det_results, iou_threshold=0.50):
    """Compute AP at specific IoU threshold per class"""
    tp=fp=fn=0
    all_dets = []
    total_gt = sum(len(d['gt']) for d in det_results)

    for img_idx, det in enumerate(det_results):
        boxes = det['boxes']; scores = det['scores']; gts = det['gt']
        sorted_idx = np.argsort(scores)[::-1]
        matched = set()
        for i in sorted_idx:
            best_iou=0; best_gi=-1
            for gi,gt in enumerate(gts):
                if gi in matched: continue
                v=iou(boxes[i],gt)
                if v>best_iou: best_iou=v; best_gi=gi
            is_tp = best_iou >= iou_threshold
            if is_tp: matched.add(best_gi); tp+=1
            else: fp+=1
            all_dets.append({'score': scores[i], 'tp': is_tp})
        fn += len(gts)-len(matched)

    dets_sorted = sorted(all_dets, key=lambda x: x['score'], reverse=True)
    tp_cum=0; fp_cum=0; precs=[]; recs=[]
    for d in dets_sorted:
        if d['tp']: tp_cum+=1
        else: fp_cum+=1
        precs.append(tp_cum/(tp_cum+fp_cum) if (tp_cum+fp_cum) else 0)
        recs.append(tp_cum/total_gt if total_gt else 0)
    ap = 0.0
    for t in np.linspace(0,1,11):
        ap += max([p for p,r in zip(precs,recs) if r>=t]+[0]) / 11.0
    return round(ap, 4)

# ═══════════════════════════════════════════
# OFAT for TDM
# ═══════════════════════════════════════════
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)
ofat_data = ofat.get('ofat_overhead', ofat)
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm(cls):
    if cls not in ofat_data: return f'an overhead image of {cls}'
    d = ofat_data[cls]['deltas']
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'an overhead image of {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

# ═══════════════════════════════════════════
# TABLE 1: DIOR 20-class with mAP
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("TABLE 1: DIOR 20-class (DR/P/R/F1 + AP50/AP75/mAP)")
print("="*60)

methods_t1 = {
    'bare': lambda c: c,
    'satellite': lambda c: f'a satellite image of {c}',
    'satellite+TDM': lambda c: build_tdm(c).replace('an overhead image of', 'a satellite image of'),
    'overhead': lambda c: f'an overhead image of {c}',
    'overhead+TDM': build_tdm,
}

table1_results = {}
for method_name, pfn in methods_t1.items():
    print(f"\n{method_name}...", flush=True)
    per_class_ap50 = {}
    all_dets_flat = []
    all_gts_flat = []

    for cls_name in ALL_CLASSES:
        if cls_name not in test_data: continue
        dets = detect_with_scores(pfn(cls_name), test_data[cls_name])

        # Per-class AP50
        ap50 = compute_map_per_class(dets, iou_threshold=0.50)
        per_class_ap50[cls_name] = ap50

        # Collect for overall metrics
        all_dets_flat.extend(dets)

    # Overall metrics at IoU=0.25
    m25 = compute_all_metrics(all_dets_flat, iou_threshold=0.25)
    # Overall AP50
    ap50_mean = float(np.mean(list(per_class_ap50.values())))
    # Overall AP75
    ap75_per_class = {}
    for cls_name in ALL_CLASSES:
        if cls_name not in test_data: continue
        dets = detect_with_scores(pfn(cls_name), test_data[cls_name])
        ap75_per_class[cls_name] = compute_map_per_class(dets, iou_threshold=0.75)
    ap75_mean = float(np.mean(list(ap75_per_class.values())))

    table1_results[method_name] = {
        'dr': m25['dr'], 'p': m25['p'], 'r': m25['r'], 'f1': m25['f1'],
        'ap': m25['ap'],  # AP25 (same IoU as F1)
        'ap50': round(ap50_mean, 4), 'ap75': round(ap75_mean, 4),
        'mAP': round((m25['ap'] + ap50_mean + ap75_mean)/3, 4),
    }
    print(f"  DR={m25['dr']:.4f} P={m25['p']:.4f} R={m25['r']:.4f} F1={m25['f1']:.4f} "
          f"AP25={m25['ap']:.4f} AP50={ap50_mean:.4f} AP75={ap75_mean:.4f} mAP={table1_results[method_name]['mAP']:.4f}",
          flush=True)

# ═══════════════════════════════════════════
# Print table
# ═══════════════════════════════════════════
print(f"\n{'='*90}")
print(f"TABLE 1: DIOR 20-class Full Metrics (GDINO-tiny, 200 test images)")
print(f"{'='*90}")
print(f"  {'Method':<20} {'DR':>8} {'P':>8} {'R':>8} {'F1':>8} {'AP25':>8} {'AP50':>8} {'AP75':>8} {'mAP':>8}")
print(f"  {'-'*90}")
for method_name in ['bare','satellite','satellite+TDM','overhead','overhead+TDM']:
    m = table1_results[method_name]
    print(f"  {method_name:<20} {m['dr']:>8.4f} {m['p']:>8.4f} {m['r']:>8.4f} {m['f1']:>8.4f} "
          f"{m['ap']:>8.4f} {m['ap50']:>8.4f} {m['ap75']:>8.4f} {m['mAP']:>8.4f}")

tdm_delta = table1_results['overhead+TDM']['f1'] - table1_results['overhead']['f1']
tdm_delta_map = table1_results['overhead+TDM']['mAP'] - table1_results['overhead']['mAP']
print(f"\n  TDM Δ vs overhead: F1={tdm_delta:+.4f}, mAP={tdm_delta_map:+.4f}")

# ═══════════════════════════════════════════
# Save
# ═══════════════════════════════════════════
with open(r'C:\Users\张三\Desktop\map_results.json', 'w') as f:
    json.dump({'table1': table1_results, 'ap50_per_class': per_class_ap50}, f, indent=2, ensure_ascii=False)
print(f"\nSaved: map_results.json")
print("Done.")
