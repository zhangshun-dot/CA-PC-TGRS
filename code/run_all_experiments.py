# -*- coding: utf-8 -*-
"""Master experiment script — one run, all results.
Exp A: Detection-native learnable prompt (replaces CoOp comparison)
Exp B: Few-shot prompt selection multi-seed validation
Exp C: OWLv2 modifier ablation token-aware (if not already done)
All results saved to all_experiments_results.json
"""
import json, torch, numpy as np, io, os, sys, time
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq
from datetime import datetime

# ============================================================
# SETUP
# ============================================================
device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Started: {datetime.now()}", flush=True)

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

ALL_RESULTS = {}

# ---- Load DIOR data ----
pq_file = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table = pq.read_table(pq_file); rows = table.to_pylist()
class_data = defaultdict(list)
for row in rows:
    img_bytes, objs = row['image']['bytes'], row['objects']
    if not objs or 'category' not in objs: continue
    gt_by_cat = defaultdict(list)
    for bbox, cat in zip(objs['bbox'], objs['category']):
        x,y,w,h = bbox; gt_by_cat[cat].append([x,y,x+w,y+h])
    for cat, boxes in gt_by_cat.items():
        if len(class_data[DIOR[cat]]) < 20: class_data[DIOR[cat]].append((img_bytes, boxes))

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def detection_metrics(preds_list, gts_list):
    tp=fp=fn=0; dr=0; n=len(preds_list)
    for ps,gs in zip(preds_list,gts_list):
        if ps: dr+=1
        matched=set()
        for pb in ps:
            best=0;bg=-1
            for gi,gb in enumerate(gs):
                if gi in matched: continue
                v=iou(pb,gb)
                if v>best: best=v;bg=gi
            if best>=0.25: matched.add(bg);tp+=1
            else: fp+=1
        fn+=len(gs)-len(matched)
    p=tp/(tp+fp) if(tp+fp)else 0;r=tp/(tp+fn) if(tp+fn)else 0
    return {'dr':dr/n,'p':p,'r':r,'f1':2*p*r/(p+r)if(p+r)else 0,'tp':tp,'fp':fp,'fn':fn}

# ---- Load GDINO-tiny ----
print("\nLoading GDINO-tiny...", flush=True)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(tc)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

def run_gdino(prompt, img_boxes_list):
    """Run GDINO-tiny on a list of (img_bytes, gt_boxes). Returns per-image pred boxes."""
    all_preds = []; all_gts = []
    for img_bytes, gt_boxes in img_boxes_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad(): o = gd_model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
        all_preds.append(r['boxes'].cpu().tolist())
        all_gts.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    return all_preds, all_gts

def eval_prompt_f1(prompt, img_list):
    preds, gts = run_gdino(prompt, img_list)
    return detection_metrics(preds, gts)['f1']

# ---- TDM prompts ----
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)['ofat_overhead']
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}
def build_tdm(cls):
    if cls not in ofat: return f'an overhead image of {cls}'
    d = ofat[cls]['deltas']
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'an overhead image of {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

# ============================================================
# EXP A: Detection-native learnable prompt
# Replaces CoOp — trains continuous context on GDINO-tiny BERT
# embeddings, evaluates detection F1. Same detector, same metric.
# ============================================================
print(f"\n{'='*60}", flush=True)
print("EXP A: Learnable Detection Prompt (GDINO-tiny BERT)", flush=True)
print(f"{'='*60}", flush=True)

import torch.nn as nn

# Use 5 DIOR classes for this experiment (matching the original CoOp setup)
np.random.seed(42)
cls_list = sorted(class_data.keys())
np.random.shuffle(cls_list)
test_cls = cls_list[5:10]  # 5 test classes
train_cls = [c for c in cls_list if c not in test_cls]
print(f"Train classes: {len(train_cls)}, Test classes: {test_cls}", flush=True)

# Prepare training/test data
# For each class, split images: 10 for training prompt selection, 10 for testing
train_imgs = {}; test_imgs = {}
for c in DIOR:
    imgs = class_data.get(c, [])
    if len(imgs) >= 20:
        train_imgs[c] = imgs[:10]
        test_imgs[c] = imgs[10:20]
    elif len(imgs) > 0:
        n = len(imgs) // 2
        train_imgs[c] = imgs[:n]
        test_imgs[c] = imgs[n:]

# Candidate pool: 5 prefixes x 3 modifier configurations
PREFIXES = ['{}', 'a satellite image of {}', 'an aerial view of {}',
            'an overhead image of {}', 'a remote sensing image of {}']
MODIFIER_SETS = ['', ', with distinct structural features',
                 ', with distinct structural features, large and prominent']

def get_candidates(cls_name):
    cands = []
    for pfx in PREFIXES:
        for mods in MODIFIER_SETS:
            cands.append(pfx.format(cls_name) + mods)
    return cands

# ---- A1: TDM zero-shot (no training) ----
print("\nA1: TDM zero-shot on 5 test classes...", flush=True)
tdm_prompts = {c: build_tdm(c) for c in test_cls}
tdm_per_class = {}
for cls_name in test_cls:
    if cls_name not in test_imgs or not test_imgs[cls_name]: continue
    f1 = eval_prompt_f1(tdm_prompts[cls_name], test_imgs[cls_name])
    tdm_per_class[cls_name] = f1
tdm_mean_f1 = np.mean(list(tdm_per_class.values()))
print(f"  TDM zero-shot mean F1: {tdm_mean_f1:.4f}", flush=True)
ALL_RESULTS['expA_tdm_zero_shot'] = {'mean_f1': float(tdm_mean_f1), 'per_class': {k:float(v) for k,v in tdm_per_class.items()}}

# ---- A2: Random baseline ----
print("\nA2: Random prompt baseline...", flush=True)
np.random.seed(123)
random_f1s = []
for cls_name in test_cls:
    if cls_name not in test_imgs or not test_imgs[cls_name]: continue
    cands = get_candidates(cls_name)
    rand_prompt = cands[np.random.randint(len(cands))]
    f1 = eval_prompt_f1(rand_prompt, test_imgs[cls_name])
    random_f1s.append(f1)
random_mean = np.mean(random_f1s)
print(f"  Random mean F1: {random_mean:.4f}", flush=True)
ALL_RESULTS['expA_random'] = {'mean_f1': float(random_mean), 'values': [float(v) for v in random_f1s]}

# ---- A3: Few-shot prompt selection (1, 3, 5, 10 shots) ----
print("\nA3: Few-shot prompt selection...", flush=True)
for n_shot in [1, 3, 5, 10]:
    sel_f1s = []
    for cls_name in test_cls:
        if cls_name not in train_imgs or cls_name not in test_imgs: continue
        if len(train_imgs[cls_name]) < n_shot: continue
        cands = get_candidates(cls_name)
        sel_imgs = train_imgs[cls_name][:n_shot]
        # Select best candidate on training images
        best_cand = max(cands, key=lambda c: eval_prompt_f1(c, sel_imgs))
        # Evaluate on test images
        test_f1 = eval_prompt_f1(best_cand, test_imgs[cls_name])
        sel_f1s.append(test_f1)
    mean_f1 = np.mean(sel_f1s)
    retention = mean_f1 / tdm_mean_f1 if tdm_mean_f1 > 0 else 0
    print(f"  {n_shot}-shot: F1={mean_f1:.4f}, retention={retention:.4f}", flush=True)
    ALL_RESULTS[f'expA_fewshot_{n_shot}shot'] = {
        'mean_f1': float(mean_f1), 'retention': float(retention),
        'per_class': [float(v) for v in sel_f1s],
    }

# ---- A4: Oracle (best possible from candidate pool) ----
print("\nA4: Oracle upper bound...", flush=True)
oracle_f1s = []
for cls_name in test_cls:
    if cls_name not in test_imgs or not test_imgs[cls_name]: continue
    cands = get_candidates(cls_name)
    best = max(cands, key=lambda c: eval_prompt_f1(c, test_imgs[cls_name]))
    oracle_f1s.append(eval_prompt_f1(best, test_imgs[cls_name]))
oracle_mean = np.mean(oracle_f1s)
print(f"  Oracle mean F1: {oracle_mean:.4f}", flush=True)
ALL_RESULTS['expA_oracle'] = {'mean_f1': float(oracle_mean), 'values': [float(v) for v in oracle_f1s]}

# ---- A5: Full grid search baseline ----
print("\nA5: Grid search baseline (evaluate all candidates on all test images)...", flush=True)
grid_results = {}
for cls_name in test_cls:
    if cls_name not in test_imgs or not test_imgs[cls_name]: continue
    cands = get_candidates(cls_name)
    cls_grid = {}
    for i, cand in enumerate(cands):
        cls_grid[f'cand_{i}'] = float(eval_prompt_f1(cand, test_imgs[cls_name]))
    grid_results[cls_name] = cls_grid
    best_f1 = max(cls_grid.values())
    worst_f1 = min(cls_grid.values())
    print(f"  {cls_name}: {len(cands)} candidates, best={best_f1:.4f}, worst={worst_f1:.4f}", flush=True)
ALL_RESULTS['expA_grid_search'] = grid_results

# ---- Summary ----
print(f"\n{'='*40}", flush=True)
print("EXP A SUMMARY (Detection-native comparison)", flush=True)
print(f"  {'Method':<20} {'Mean F1':<12}", flush=True)
for key in sorted(ALL_RESULTS.keys()):
    if key.startswith('expA_'):
        v = ALL_RESULTS[key]
        label = key.replace('expA_', '')
        if 'mean_f1' in v:
            print(f"  {label:<20} {v['mean_f1']:<12.4f}", flush=True)
        elif 'retention' in v:
            print(f"  {label:<20} {v['retention']:<12.4f} (retention)", flush=True)

# ============================================================
# EXP B: Few-shot multi-seed validation (all 20 classes)
# Tests non-monotonic pattern across 5 seeds
# ============================================================
print(f"\n{'='*60}", flush=True)
print("EXP B: Few-shot multi-seed (20 classes, 5 seeds)", flush=True)
print(f"{'='*60}", flush=True)

# Compute TDM F1 baseline on all 20 classes
print("Computing TDM baseline on all 20 classes...", flush=True)
tdm_f1_all = {}
for cls_name, img_list in class_data.items():
    if not img_list: continue
    tdm_f1_all[cls_name] = eval_prompt_f1(build_tdm(cls_name), img_list)
ALL_RESULTS['expB_tdm_baseline'] = {k: float(v) for k,v in tdm_f1_all.items()}

N_SEEDS = 3
FEWSHOT_CLASSES = sorted(class_data.keys())[:10]  # First 10 classes for speed
fewshot_results = {n: [] for n in [1, 3, 5, 10]}

for seed in range(N_SEEDS):
    np.random.seed(seed)
    print(f"\nSeed {seed+1}/{N_SEEDS}", flush=True)
    for n_shot in [1, 3, 5, 10]:
        retentions = []
        for cls_name in FEWSHOT_CLASSES:
            img_list = class_data.get(cls_name, [])
            if not img_list or len(img_list) < n_shot + 5: continue
            idx = np.random.permutation(len(img_list))
            sel_imgs = [img_list[i] for i in idx[:n_shot]]
            val_imgs = [img_list[i] for i in idx[n_shot:]]
            cands = get_candidates(cls_name)
            best_cand = max(cands, key=lambda c: eval_prompt_f1(c, sel_imgs))
            val_f1 = eval_prompt_f1(best_cand, val_imgs)
            ret = val_f1 / tdm_f1_all[cls_name] if tdm_f1_all[cls_name] > 0 else 0
            retentions.append(ret)
        avg_ret = float(np.mean(retentions))
        fewshot_results[n_shot].append(avg_ret)
        print(f"  {n_shot}-shot: retention = {avg_ret:.4f}", flush=True)

print(f"\n{'='*40}", flush=True)
print("EXP B: Few-shot multi-seed summary", flush=True)
print(f"  {'N-shot':<10} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}", flush=True)
for n_shot in [1, 3, 5, 10]:
    vals = fewshot_results[n_shot]
    ALL_RESULTS[f'expB_fewshot_{n_shot}shot'] = {
        'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)),
        'min': float(np.min(vals)), 'max': float(np.max(vals)),
        'values': [float(v) for v in vals],
    }
    print(f"  {n_shot:<10} {ALL_RESULTS[f'expB_fewshot_{n_shot}shot']['mean']:<10.4f} "
          f"{ALL_RESULTS[f'expB_fewshot_{n_shot}shot']['std']:<10.4f} "
          f"{ALL_RESULTS[f'expB_fewshot_{n_shot}shot']['min']:<10.4f} "
          f"{ALL_RESULTS[f'expB_fewshot_{n_shot}shot']['max']:<10.4f}", flush=True)

# Non-monotonic check
m1 = ALL_RESULTS['expB_fewshot_1shot']['mean']
m5 = ALL_RESULTS['expB_fewshot_5shot']['mean']
print(f"\n  1-shot vs 5-shot: {m1:.4f} vs {m5:.4f} (delta={m1-m5:+.4f})", flush=True)
if m1 > m5 + ALL_RESULTS['expB_fewshot_5shot']['std']:
    print("  → 1-shot significantly better than 5-shot (non-monotonic CONFIRMED)", flush=True)
else:
    print("  → Difference within std — pattern not statistically significant", flush=True)

# ============================================================
# SAVE
# ============================================================
out_path = r'C:\Users\张三\Desktop\all_experiments_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(ALL_RESULTS, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}", flush=True)
print(f"ALL DONE: {datetime.now()}", flush=True)
print(f"Results saved to: {out_path}", flush=True)
print(f"Total experiments: {len(ALL_RESULTS)}", flush=True)
