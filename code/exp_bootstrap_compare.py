# -*- coding: utf-8 -*-
"""Exp C: Bootstrap CI + Baseline Methods Comparison Table

Reuses existing data + adds bootstrap confidence intervals.
Generates the comprehensive "Table 7: Baseline Comparison" for the paper.
"""
import json, torch, numpy as np, io, os, sys, time
from PIL import Image
from collections import defaultdict

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pyarrow.parquet as pq

# ── Load existing results ──
print("Loading existing data...")

# COCO results
with open(r'C:\Users\张三\Desktop\coco_full_metrics_results.json') as f:
    coco_data = json.load(f)

# Final experiments (few-shot, TDM, oracle, random)
with open(r'C:\Users\张三\Desktop\final_results.json') as f:
    final_data = json.load(f)

# Cross-subset validation
with open(r'C:\Users\张三\Desktop\cross_subset_results.json') as f:
    subset_data = json.load(f)

# OFAT data (for TDM construction)
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat_data = json.load(f)

print("All existing data loaded.")

# ── DIOR classes ──
DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# ── Load GDINO-tiny for bootstrap ──
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
proc = AutoProcessor.from_pretrained(tc)
model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

# ── Helpers ──
def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def compute_f1(preds, gts, thresh=0.25):
    tp=fp=fn=0
    matched=set()
    for pb in preds:
        best=0;bg=-1
        for gi,gb in enumerate(gts):
            if gi in matched: continue
            v=iou(pb,gb)
            if v>best: best=v;bg=gi
        if best>=thresh: matched.add(bg);tp+=1
        else: fp+=1
    fn=len(gts)-len(matched)
    p=tp/(tp+fp) if(tp+fp)else 0;r=tp/(tp+fn) if(tp+fn)else 0
    return 2*p*r/(p+r) if(p+r)else 0

# ═══════════════════════════════════════════════════
# PART 1: Bootstrap CI for Table 1 (GDINO-tiny DIOR)
# ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 1: Bootstrap CI for Table 1")
print("="*60)

# Load DIOR test data
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

# For each method, collect per-image F1 values for bootstrap
np.random.seed(42)
n_bootstrap = 10000

def run_method_on_dior(prompt_fn, data_dict):
    """Run a prompt method on all DIOR test images, return per-image F1s"""
    f1s = []
    for cls_name, imgs in data_dict.items():
        for img_bytes, gt_boxes in imgs:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            prompt = prompt_fn(cls_name)
            inputs = proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad():
                o = model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
            preds = r['boxes'].cpu().tolist()
            gts = [(b[0],b[1],b[2],b[3]) for b in gt_boxes]
            f1s.append(compute_f1(preds, gts))
    return np.array(f1s)

def bootstrap_ci(f1_array, n_bootstrap=10000, alpha=0.05):
    """Bootstrap 95% CI for mean F1"""
    n = len(f1_array)
    means = []
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        means.append(f1_array[idx].mean())
    means = np.array(means)
    ci_low = np.percentile(means, 100*alpha/2)
    ci_high = np.percentile(means, 100*(1-alpha/2))
    return means.mean(), ci_low, ci_high

# Run bootstrap for the 5 methods in Table 1
print("Running GDINO-tiny on DIOR 20-class for bootstrap...")
methods_t1 = {
    'bare': lambda c: c,
    'satellite': lambda c: f'a satellite image of {c}',
    'satellite+TDM': lambda c: build_tdm_prompt(c, 'satellite'),
    'overhead': lambda c: f'an overhead image of {c}',
    'overhead+TDM': lambda c: build_tdm_prompt(c, 'overhead'),
}

# TDM builder
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm_prompt(cls, prefix):
    ofat_data_inner = ofat_data.get('ofat_overhead', ofat_data)
    if cls not in ofat_data_inner: return f'an overhead image of {cls}'
    d = ofat_data_inner[cls]['deltas']
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    pfx = 'a satellite image of' if prefix=='satellite' else 'an overhead image of'
    p = f'{pfx} {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

bootstrap_results_t1 = {}
for method_name, prompt_fn in methods_t1.items():
    print(f"  {method_name}...")
    f1s = run_method_on_dior(prompt_fn, class_data)
    mean, ci_low, ci_high = bootstrap_ci(f1s, n_bootstrap)
    bootstrap_results_t1[method_name] = {
        'mean_f1': float(mean), 'ci_low': float(ci_low), 'ci_high': float(ci_high),
        'ci_range': float(ci_high - ci_low)
    }
    print(f"    F1={mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

# ═══════════════════════════════════════════════════
# PART 2: Comprehensive Baseline Comparison Table
# ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 2: Baseline Methods Comparison")
print("="*60)

# Assemble from existing data
comparison = {
    'TDM (zero-shot, measurement-driven)': {
        'f1': final_data['tdm']['mean'],
        'type': 'Zero-shot',
        'description': 'OFAT measurement + TDM generation, no training'
    },
    'Random (5 trials)': {
        'f1': final_data['random']['mean'],
        'std': final_data['random']['std'],
        'type': 'Zero-shot',
        'description': 'Random prompt from 15-candidate pool'
    },
    'Oracle (pool upper bound)': {
        'f1': final_data['oracle']['mean'],
        'type': 'Upper bound',
        'description': 'Per-class best from 15-candidate pool'
    },
}

for s in [1, 3, 5, 10]:
    d = final_data['fewshot'][f'{s}shot']
    comparison[f'{s}-shot Detection Prompt Selection'] = {
        'f1': d['mean_f1'], 'std': d['std_f1'],
        'type': 'Few-shot',
        'description': f'{s} labeled images/class, select from 15 candidates'
    }

print(f"\n{'Method':<40} {'F1':>8} {'Type':>12}  Description")
print("-"*90)
for method, info in comparison.items():
    std_str = f'±{info.get("std",0):.4f}' if 'std' in info else ''
    print(f"{method:<40} {info['f1']:>8.4f} {info['type']:>12}  {info['description']}")

# ═══════════════════════════════════════════════════
# PART 3: Per-dataset summary for Table 6 extension
# ═══════════════════════════════════════════════════
print("\n" + "="*60)
print("PART 3: Multi-dataset Summary")
print("="*60)

# COCO tiny (from existing)
coco_sat = coco_data['coco_satellite']['overall']
coco_oh = coco_data['coco_overhead']['overall']
coco_tdm = coco_data['coco_tdm']['overall']

print(f"\nCOCO×tiny:  sat F1={coco_sat['f1']:.4f}  oh F1={coco_oh['f1']:.4f}  TDM F1={coco_tdm['f1']:.4f}  oh/sat={coco_oh['f1']/max(coco_sat['f1'],0.0001):.2f}×")
print(f"  DR: sat={coco_data['coco_satellite']['mean_dr']:.4f}  oh={coco_data['coco_overhead']['mean_dr']:.4f}  tdm={coco_data['coco_tdm']['mean_dr']:.4f}")

# ═══════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════
all_out = {
    'bootstrap_table1': bootstrap_results_t1,
    'baseline_comparison': comparison,
    'cross_subset_correlation': subset_data.get('correlation'),
}
with open(r'C:\Users\张三\Desktop\bootstrap_and_comparison.json', 'w') as f:
    json.dump(all_out, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print("BOOTSTRAP CI SUMMARY (Table 1)")
print(f"{'='*60}")
print(f"  {'Method':<20} {'Mean F1':>10} {'95% CI':>24}")
print(f"  {'-'*54}")
for method, info in bootstrap_results_t1.items():
    print(f"  {method:<20} {info['mean_f1']:>10.4f} [{info['ci_low']:.4f}, {info['ci_high']:.4f}]")

print(f"\nSaved: bootstrap_and_comparison.json")
print("Done.")
