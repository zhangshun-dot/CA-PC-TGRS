# -*- coding: utf-8 -*-
"""Fast ablation experiments: axis ablation, hyperparameter sensitivity, threshold robustness.
Reuses existing TDM/oracle/few-shot data from final_results.json."""

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

# Load test data
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
        if len(class_data[cat]) < 20: class_data[cat].append((img_bytes, boxes))

np.random.seed(42)
test_data = {}
for cls_name, imgs in class_data.items():
    idx = np.random.permutation(len(imgs))
    n_test = min(10, len(imgs)//2)
    test_data[cls_name] = [imgs[i] for i in idx[:n_test]]

ALL_CLASSES = sorted(test_data.keys())

# Helpers
def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def detect_f1(prompt, img_list, thresh=0.25):
    tp=fp=fn=0
    for img_bytes, gt_boxes in img_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad(): o = model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=thresh, target_sizes=target)[0]
        preds = r['boxes'].cpu().tolist()
        gts = [(b[0],b[1],b[2],b[3]) for b in gt_boxes]
        matched=set()
        for pb in preds:
            best=0;bg=-1
            for gi,gb in enumerate(gts):
                if gi in matched: continue
                v=iou(pb,gb)
                if v>best: best=v;bg=gi
            if best>=thresh: matched.add(bg);tp+=1
            else: fp+=1
        fn+=len(gts)-len(matched)
    total_gt = sum(len([(b[0],b[1],b[2],b[3]) for b in gb]) for _,gb in img_list)
    total_gt = sum(sum(1 for _ in [(b[0],b[1],b[2],b[3]) for b in gb]) for _,gb in img_list)
    return 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0

# Load OFAT
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)
ofat_data = ofat.get('ofat_overhead', ofat)

AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm_variant(cls, delta_thresh=0.03, max_mods=2, drop_axis=None):
    if cls not in ofat_data: return f'an overhead image of {cls}'
    d = ofat_data[cls]['deltas']
    pos = sorted([(ax,v) for ax,v in d.items()
                  if ax != 'grammar' and ax != drop_axis and v > delta_thresh],
                 key=lambda x: -x[1])
    mods = [AXM[ax] for ax,_ in pos[:max_mods]] if pos else []
    p = f'an overhead image of {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

# ═══════════════════════════════════════════════
# TABLE A: Axis Ablation (7 axes)
# ═══════════════════════════════════════════════
print("\n=== TABLE A: Axis Ablation ===", flush=True)
AXES = ['grammar','granularity','spatial','geometric','scale','spectral','texture']

# Baseline: full TDM
tdm_f1s = []
for c in ALL_CLASSES:
    if c in test_data:
        tdm_f1s.append(detect_f1(build_tdm_variant(c), test_data[c]))
tdm_baseline = float(np.mean(tdm_f1s))
print(f"TDM baseline: F1={tdm_baseline:.4f}", flush=True)

axis_results = {}
for drop_axis in AXES:
    f1s = []
    for c in ALL_CLASSES:
        if c in test_data:
            f1s.append(detect_f1(build_tdm_variant(c, drop_axis=drop_axis), test_data[c]))
    mean_f1 = float(np.mean(f1s))
    axis_results[drop_axis] = {'f1': round(mean_f1,4), 'delta': round(mean_f1 - tdm_baseline,4)}
    print(f"  -{drop_axis:<15}: F1={mean_f1:.4f}  Δ={mean_f1-tdm_baseline:+.4f}", flush=True)

# ═══════════════════════════════════════════════
# TABLE B: Delta Threshold Sensitivity
# ═══════════════════════════════════════════════
print("\n=== TABLE B: Delta Threshold ===", flush=True)
delta_results = {}
for thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
    f1s = []
    for c in ALL_CLASSES:
        if c in test_data:
            f1s.append(detect_f1(build_tdm_variant(c, delta_thresh=thresh), test_data[c]))
    mean_f1 = float(np.mean(f1s))
    delta_results[f'd>{thresh:.2f}'] = round(mean_f1,4)
    print(f"  δ>{thresh:.2f}: F1={mean_f1:.4f}", flush=True)

# ═══════════════════════════════════════════════
# TABLE C: Max Modifier Count
# ═══════════════════════════════════════════════
print("\n=== TABLE C: Max Modifiers ===", flush=True)
mod_results = {}
for max_m in [1, 2, 3, 4]:
    f1s = []
    for c in ALL_CLASSES:
        if c in test_data:
            f1s.append(detect_f1(build_tdm_variant(c, max_mods=max_m), test_data[c]))
    mean_f1 = float(np.mean(f1s))
    mod_results[f'{max_m} modifiers'] = round(mean_f1,4)
    print(f"  max {max_m}: F1={mean_f1:.4f}", flush=True)

# ═══════════════════════════════════════════════
# TABLE D: IoU Threshold Robustness
# ═══════════════════════════════════════════════
print("\n=== TABLE D: IoU Threshold ===", flush=True)
# Use 5 representative classes for speed
sub_cls = ALL_CLASSES[:5]
thresh_results = {}
for iou_t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    tdm_sub = []; oh_sub = []
    for c in sub_cls:
        if c in test_data:
            tdm_sub.append(detect_f1(build_tdm_variant(c), test_data[c], thresh=iou_t))
            oh_sub.append(detect_f1(f'an overhead image of {c}', test_data[c], thresh=iou_t))
    thresh_results[f'IoU={iou_t:.2f}'] = {
        'tdm': round(float(np.mean(tdm_sub)),4),
        'oh': round(float(np.mean(oh_sub)),4),
        'delta': round(float(np.mean(tdm_sub)-np.mean(oh_sub)),4)
    }
    print(f"  IoU={iou_t:.2f}: TDM={np.mean(tdm_sub):.4f}  oh={np.mean(oh_sub):.4f}  Δ={np.mean(tdm_sub)-np.mean(oh_sub):+.4f}", flush=True)

# ═══════════════════════════════════════════════
# TABLE E: Comprehensive Comparison (10 methods)
# ═══════════════════════════════════════════════
print("\n=== TABLE E: Comprehensive Comparison ===", flush=True)
# Reuse existing data + add new fixed-prompt runs
fixed_prompts = {
    'Bare (class only)': lambda c: c,
    'Satellite prefix': lambda c: f'a satellite image of {c}',
    'Aerial prefix': lambda c: f'an aerial view of {c}',
    'Overhead prefix': lambda c: f'an overhead image of {c}',
    'Remote sensing prefix': lambda c: f'a remote sensing image of {c}',
    'Overhead + fixed mods': lambda c: f'an overhead image of {c}, with distinct structural features, large and prominent',
}

# Load existing data
with open(r'C:\Users\张三\Desktop\final_results.json') as f:
    final_data = json.load(f)

comparison = {}
# Fixed prompts
for name, pfn in fixed_prompts.items():
    f1s = []
    for c in ALL_CLASSES:
        if c in test_data:
            f1s.append(detect_f1(pfn(c), test_data[c]))
    comparison[name] = round(float(np.mean(f1s)), 4)
    print(f"  {name}: F1={comparison[name]:.4f}", flush=True)

# TDM (ours)
f1s_tdm = []
for c in ALL_CLASSES:
    if c in test_data:
        f1s_tdm.append(detect_f1(build_tdm_variant(c), test_data[c]))
comparison['TDM (OFAT, zero-shot)'] = round(float(np.mean(f1s_tdm)), 4)
print(f"  TDM (OFAT, zero-shot): F1={comparison['TDM (OFAT, zero-shot)']:.4f}", flush=True)

# From existing results
comparison['Oracle (upper bound)'] = round(final_data['oracle']['mean'], 4)
comparison['Random (5 trials)'] = f"{final_data['random']['mean']:.4f}±{final_data['random']['std']:.4f}"
for s in ['1shot','3shot','5shot','10shot']:
    d = final_data['fewshot'][s]
    label = s.replace('shot', '-shot')
    comparison[f'{label} selection'] = f"{d['mean_f1']:.4f}±{d['std_f1']:.4f}"

print(f"\n  Oracle: {comparison['Oracle (upper bound)']}")
print(f"  Random: {comparison['Random (5 trials)']}")
for s in ['1shot','3shot','5shot','10shot']:
    label = s.replace('shot', '-shot')
    print(f"  {label}: {comparison[f'{label} selection']}")

# ═══════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════
all_out = {
    'axis_ablation': axis_results,
    'delta_threshold': delta_results,
    'max_modifiers': mod_results,
    'iou_threshold_robustness': thresh_results,
    'comprehensive_comparison': comparison,
    'tdm_baseline_f1': tdm_baseline,
}
with open(r'C:\Users\张三\Desktop\ablation_results.json', 'w') as f:
    json.dump(all_out, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"DONE. Saved: ablation_results.json")
print(f"{'='*60}")
for section, data in all_out.items():
    if isinstance(data, dict) and len(data) > 0:
        print(f"\n  {section}:")
        for k,v in list(data.items())[:5]:
            print(f"    {k}: {v}")
