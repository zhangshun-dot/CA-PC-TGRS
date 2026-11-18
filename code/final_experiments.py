# -*- coding: utf-8 -*-
"""Final experiments: TDM vs Few-shot Detection Prompt Selection vs Oracle vs Random.
All on GDINO-tiny, all measured by detection F1. Apples to apples."""
import json, torch, numpy as np, io, time
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq
from datetime import datetime

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Start: {datetime.now()}", flush=True)

OUT = r'C:\Users\张三\Desktop\final_results.json'
DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# ---- Load data (10 train / 10 test per class) ----
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

np.random.seed(42)
train_data = {}; test_data = {}
for cls_name, imgs in class_data.items():
    idx = np.random.permutation(len(imgs))
    n = min(10, len(imgs)//2)
    train_data[cls_name] = [imgs[i] for i in idx[:n]]
    test_data[cls_name] = [imgs[i] for i in idx[n:2*n]]

ALL_CLASSES = sorted(class_data.keys())
print(f"Classes: {len(ALL_CLASSES)}, Train/class: {sum(len(train_data.get(c,[])) for c in ALL_CLASSES)//len(ALL_CLASSES)}, Test/class: {sum(len(test_data.get(c,[])) for c in ALL_CLASSES)//len(ALL_CLASSES)}", flush=True)

# ---- Helpers ----
def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def metrics(preds_list, gts_list):
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
    return {'dr':dr/n,'p':p,'r':r,'f1':2*p*r/(p+r)if(p+r)else 0}

# ---- Load GDINO-tiny ----
print("Loading GDINO-tiny...", flush=True)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(tc)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

def run_gdino(prompt, img_list):
    all_p=[];all_g=[]
    for img_bytes, gt_boxes in img_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad(): o = gd_model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
        all_p.append(r['boxes'].cpu().tolist())
        all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    return metrics(all_p, all_g)

# ---- TDM ----
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

# ---- Candidate pool ----
PREFIXES = ['{}', 'a satellite image of {}', 'an aerial view of {}',
            'an overhead image of {}', 'a remote sensing image of {}']
MODIFIER_SETS = ['', ', with distinct structural features',
                 ', with distinct structural features, large and prominent']
def get_candidates(cls_name):
    return [pfx.format(cls_name) + mods for pfx in PREFIXES for mods in MODIFIER_SETS]

ALL = {'config': {'classes': ALL_CLASSES, 'n_classes': len(ALL_CLASSES), 'started': str(datetime.now())}}

def save():
    with open(OUT, 'w') as f: json.dump(ALL, f, indent=2, ensure_ascii=False)

# ============================================================
# EXP 1: TDM zero-shot (measurement-driven, no training)
# ============================================================
print(f"\n{'='*50}\nEXP 1: TDM zero-shot\n{'='*50}", flush=True)
tdm_f1s = {}
for cls_name in ALL_CLASSES:
    tdm_f1s[cls_name] = run_gdino(build_tdm(cls_name), test_data[cls_name])['f1']
tdm_mean = float(np.mean(list(tdm_f1s.values())))
print(f"  Mean F1: {tdm_mean:.4f}", flush=True)
ALL['tdm'] = {'per_class': {k:float(v) for k,v in tdm_f1s.items()}, 'mean': tdm_mean}
save()

# ============================================================
# EXP 2: Grid search (all 15 candidates on all test images)
# ============================================================
print(f"\n{'='*50}\nEXP 2: Grid search\n{'='*50}", flush=True)
grid = {}
for i, cls_name in enumerate(ALL_CLASSES):
    cands = get_candidates(cls_name)
    cls_results = {}
    for j, cand in enumerate(cands):
        cls_results[f'cand_{j}'] = float(run_gdino(cand, test_data[cls_name])['f1'])
    grid[cls_name] = cls_results
    if i % 4 == 0:
        print(f"  {cls_name} ({i+1}/{len(ALL_CLASSES)}): best={max(cls_results.values()):.4f}", flush=True)
ALL['grid_search'] = grid
save()

# ============================================================
# EXP 3: Oracle (upper bound from candidate pool)
# ============================================================
print(f"\n{'='*50}\nEXP 3: Oracle\n{'='*50}", flush=True)
oracle_vals = [max(grid[c].values()) for c in ALL_CLASSES]
oracle_mean = float(np.mean(oracle_vals))
ALL['oracle'] = {'mean': oracle_mean, 'per_class': [float(v) for v in oracle_vals]}
print(f"  Oracle mean F1: {oracle_mean:.4f}", flush=True)
save()

# ============================================================
# EXP 4: Random baseline
# ============================================================
print(f"\n{'='*50}\nEXP 4: Random\n{'='*50}", flush=True)
np.random.seed(123)
random_runs = []
for seed in range(5):
    f1s = []
    for cls_name in ALL_CLASSES:
        cands = get_candidates(cls_name)
        rc = cands[np.random.randint(len(cands))]
        f1s.append(run_gdino(rc, test_data[cls_name])['f1'])
    random_runs.append(float(np.mean(f1s)))
ALL['random'] = {'mean': float(np.mean(random_runs)), 'std': float(np.std(random_runs)), 'values': random_runs}
print(f"  Random mean F1: {ALL['random']['mean']:.4f} +/- {ALL['random']['std']:.4f}", flush=True)
save()

# ============================================================
# EXP 5: Few-shot Detection Prompt Selection
# Uses N labeled images per class to select best prompt from pool.
# This is a detection-native alternative to CoOp/PromptDet —
# same detector, same metric, same task.
# ============================================================
print(f"\n{'='*50}\nEXP 5: Few-shot Detection Prompt Selection\n{'='*50}", flush=True)

SHOTS = [1, 3, 5, 10]
N_SEEDS = 3
fewshot = {s: [] for s in SHOTS}

for seed in range(N_SEEDS):
    np.random.seed(seed)
    print(f"  Seed {seed+1}/{N_SEEDS}", flush=True)
    for n_shot in SHOTS:
        f1s = []
        for cls_name in ALL_CLASSES:
            if len(train_data.get(cls_name, [])) < n_shot: continue
            idx = np.random.permutation(len(train_data[cls_name]))
            sel_imgs = [train_data[cls_name][i] for i in idx[:n_shot]]
            cands = get_candidates(cls_name)
            best_cand = max(cands, key=lambda c: run_gdino(c, sel_imgs)['f1'])
            f1s.append(run_gdino(best_cand, test_data[cls_name])['f1'])
        mf1 = float(np.mean(f1s))
        ret = mf1 / tdm_mean if tdm_mean > 0 else 0
        fewshot[n_shot].append({'mean_f1': mf1, 'retention': ret})
        print(f"    {n_shot}-shot: F1={mf1:.4f} (ret={ret:.4f})", flush=True)

ALL['fewshot'] = {}
for s in SHOTS:
    rets = [r['retention'] for r in fewshot[s]]
    f1s = [r['mean_f1'] for r in fewshot[s]]
    ALL['fewshot'][f'{s}shot'] = {
        'mean_retention': float(np.mean(rets)), 'std_retention': float(np.std(rets, ddof=1)),
        'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s, ddof=1)),
        'runs': fewshot[s],
    }
save()

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*50}\nFINAL SUMMARY\n{'='*50}", flush=True)
print(f"  {'Method':<30} {'F1':>8} {'Retention':>10}", flush=True)
print(f"  {'-'*48}", flush=True)
print(f"  {'TDM (zero-shot, measurement)':<30} {tdm_mean:>8.4f} {'—':>10}", flush=True)
print(f"  {'Oracle (pool upper bound)':<30} {oracle_mean:>8.4f} {'—':>10}", flush=True)
print(f"  {'Random':<30} {ALL['random']['mean']:>8.4f} {'—':>10}", flush=True)
for s in SHOTS:
    d = ALL['fewshot'][f'{s}shot']
    print(f"  {f'{s}-shot selection':<30} {d['mean_f1']:>8.4f} {d['mean_retention']:>10.4f}", flush=True)

print(f"\nDone: {datetime.now()}", flush=True)
print(f"Saved: {OUT}", flush=True)
