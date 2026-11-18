# -*- coding: utf-8 -*-
"""Quick cross-subset validation: split DIOR 20→10+10, prefix ablation on both."""
import json, torch, numpy as np, io, time
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

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
        x,y,w,h = bbox; gt_by_cat[cat].append([x,y,x+w,y+h])
    for cat, boxes in gt_by_cat.items():
        if len(class_data[DIOR[cat]]) < 20: class_data[DIOR[cat]].append((img_bytes, boxes))

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

print("Loading GDINO-tiny...", flush=True)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(tc)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

def run_prefix(pfx_tmpl, cls_list):
    all_p=[];all_g=[]
    for cls_name in cls_list:
        prompt = pfx_tmpl.format(cls_name)
        for img_bytes, gt_boxes in class_data.get(cls_name, []):
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o = gd_model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = gd_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
            all_p.append(r['boxes'].cpu().tolist())
            all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    return metrics(all_p, all_g)

# Split: 3 random seeds for stability
PREFIXES = {
    'bare': '{}',
    'satellite': 'a satellite image of {}',
    'aerial': 'an aerial view of {}',
    'overhead': 'an overhead image of {}',
    'remote_sensing': 'a remote sensing image of {}',
}

results = {}
for seed in range(3):
    np.random.seed(seed)
    all_cls = sorted(class_data.keys())
    np.random.shuffle(all_cls)
    A, B = all_cls[:10], all_cls[10:]

    for label, grp in [('A', A), ('B', B)]:
        for pname, tmpl in PREFIXES.items():
            m = run_prefix(tmpl, grp)
            results[f'seed{seed}_{label}_{pname}'] = m['f1']
            print(f'  seed{seed} {label} {pname}: F1={m["f1"]:.4f}', flush=True)

# Summary: correlation between subsets
print(f"\n{'='*50}")
print("Cross-subset stability:")
for pname in PREFIXES:
    vals_A = [results[f'seed{s}_A_{pname}'] for s in range(3)]
    vals_B = [results[f'seed{s}_B_{pname}'] for s in range(3)]
    # Mean across seeds
    mA, mB = np.mean(vals_A), np.mean(vals_B)
    print(f'  {pname}: A={mA:.4f} B={mB:.4f} |Δ|={abs(mA-mB):.4f}')

# Overall A vs B correlation
all_A = []; all_B = []
for s in range(3):
    for pname in PREFIXES:
        all_A.append(results[f'seed{s}_A_{pname}'])
        all_B.append(results[f'seed{s}_B_{pname}'])
corr = np.corrcoef(all_A, all_B)[0,1]
print(f'\n  Pearson r: {corr:.4f}')
print(f'  Mean |Δ|: {np.mean([abs(a-b) for a,b in zip(all_A,all_B)]):.4f}')

with open(r'C:\Users\张三\Desktop\cross_subset_results.json', 'w') as f:
    json.dump({'results': {k: float(v) for k,v in results.items()}, 'correlation': float(corr)}, f, indent=2)
print('\nDone.')
