# -*- coding: utf-8 -*-
"""Exp 3: Few-shot prompt selection — multi-seed validation on GDINO-tiny."""
import json, torch, numpy as np, io, os, sys, time
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# Candidate pool: 5 prefixes x 3 modifier levels = 15 candidates
PREFIXES = ['{}', 'a satellite image of {}', 'an aerial view of {}',
            'an overhead image of {}', 'a remote sensing image of {}']
MODIFIER_SETS = ['', ', with distinct structural features',
                 ', with distinct structural features, large and prominent']

def build_candidates(cls_name):
    cands = []
    for pfx in PREFIXES:
        for mods in MODIFIER_SETS:
            cands.append(pfx.format(cls_name) + mods)
    return cands

# Load data
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

def eval_prompt_f1(proc, model, prompt, img_list):
    """Compute F1 for a prompt on a list of (img_bytes, gt_boxes)."""
    tp=fp=fn=0
    for img_bytes, gt_boxes in img_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad(): o = model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
        pred_boxes = r['boxes'].cpu().tolist()
        gt = [(b[0],b[1],b[2],b[3]) for b in gt_boxes]
        matched=set()
        for pb in pred_boxes:
            best=0;bg=-1
            for gi,gb in enumerate(gt):
                if gi in matched: continue
                v=iou(pb,gb)
                if v>best: best=v;bg=gi
            if best>=0.25: matched.add(bg);tp+=1
            else: fp+=1
        fn+=len(gt)-len(matched)
    p=tp/(tp+fp) if(tp+fp)else 0;r=tp/(tp+fn) if(tp+fn)else 0
    return 2*p*r/(p+r) if(p+r)else 0

print("Loading GDINO-tiny...", flush=True)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
tp = AutoProcessor.from_pretrained(tc)
tm = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

# TDM baseline
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

# Compute TDM full-dataset F1 per class
print("TDM baseline...", flush=True)
tdm_f1 = {}
for cls_name, img_list in class_data.items():
    if not img_list: continue
    tdm_f1[cls_name] = eval_prompt_f1(tp, tm, build_tdm(cls_name), img_list)

N_SHOTS = [1, 3, 5, 10]
N_SEEDS = 5
results = {n: [] for n in N_SHOTS}

for seed in range(N_SEEDS):
    np.random.seed(seed)
    print(f"\nSeed {seed+1}/{N_SEEDS}", flush=True)
    for n_shot in N_SHOTS:
        retentions = []
        for cls_name, img_list in class_data.items():
            if not img_list or len(img_list) < n_shot + 5: continue
            idx = np.random.permutation(len(img_list))
            sel_imgs = [img_list[i] for i in idx[:n_shot]]
            val_imgs = [img_list[i] for i in idx[n_shot:]]

            cands = build_candidates(cls_name)
            best_cand = max(cands, key=lambda c: eval_prompt_f1(tp, tm, c, sel_imgs))
            val_f1 = eval_prompt_f1(tp, tm, best_cand, val_imgs)
            ret = val_f1 / tdm_f1[cls_name] if tdm_f1[cls_name] > 0 else 0
            retentions.append(ret)

        avg_ret = np.mean(retentions)
        results[n_shot].append(avg_ret)
        print(f"  {n_shot}-shot: retention = {avg_ret:.4f}", flush=True)

print(f"\n{'='*50}", flush=True)
print("FEW-SHOT RESULTS (5 seeds x 4 shot-levels)", flush=True)
print(f"  {'N-shot':<10} {'Mean':<10} {'Std':<10}", flush=True)
summary = {}
for n_shot in N_SHOTS:
    vals = results[n_shot]
    m, s = np.mean(vals), np.std(vals, ddof=1)
    summary[n_shot] = {'mean': float(m), 'std': float(s), 'values': [float(v) for v in vals]}
    print(f"  {n_shot:<10} {m:<10.4f} {s:<10.4f}", flush=True)

print(f"\nPaper reported: 1-shot=1.000, 3-shot=0.971, 5-shot=0.950, 10-shot=0.951", flush=True)
for n in N_SHOTS:
    print(f"  Our {n}-shot: {summary[n]['mean']:.3f} ± {summary[n]['std']:.3f}", flush=True)

out = {'tdm_f1': {k: float(v) for k,v in tdm_f1.items()}, 'fewshot': summary}
with open(r'C:\Users\张三\Desktop\fewshot_multiseed_results.json','w') as f:
    json.dump(out, f, indent=2)
print("Done.", flush=True)
