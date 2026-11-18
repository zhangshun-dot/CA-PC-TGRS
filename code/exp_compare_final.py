# -*- coding: utf-8 -*-
"""Clean comparison: TDM vs baselines, all using remote_sensing prefix."""
import json,torch,numpy as np,io,os
from PIL import Image
from collections import defaultdict

print('Loading...',flush=True)
device='cuda';os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
import pyarrow.parquet as pq

tc=r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
proc=AutoProcessor.from_pretrained(tc)
model=AutoModelForZeroShotObjectDetection.from_pretrained(tc,dtype=torch.float32).eval().to(device)

DIOR=['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']
pq_file=r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table=pq.read_table(pq_file);rows=table.to_pylist()
class_data=defaultdict(list)
for row in rows:
    img_bytes,objs=row['image']['bytes'],row['objects']
    if not objs or 'category' not in objs: continue
    gt_by_cat=defaultdict(list)
    for bbox,cat in zip(objs['bbox'],objs['category']): x,y,w,h=bbox;gt_by_cat[DIOR[cat]].append([x,y,x+w,y+h])
    for cat,boxes in gt_by_cat.items():
        if len(class_data[cat])<20: class_data[cat].append((img_bytes,boxes))
ALL_CLS=sorted(class_data.keys())
print(f'DIOR: {len(ALL_CLS)} classes, {sum(len(v) for v in class_data.values())} images',flush=True)

with open(r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\ofat_overhead_results.json') as f: ofat=json.load(f)
ofat_data=ofat.get('ofat_overhead',ofat)
AXM={'granularity':'with fine details','spatial':'in its natural setting','geometric':'with distinct structural features','scale':'large and prominent','spectral':'with natural coloring','texture':'with clearly visible surface features'}
AXES=['granularity','spatial','geometric','scale','spectral','texture']

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if(a1+a2-inter)>0 else 0

def eval_one(prompt_fn):
    tp=fp=fn=0;n=0
    for cls_name in ALL_CLS:
        if cls_name not in class_data: continue
        for img_bytes,gt_boxes in class_data[cls_name]:
            n+=1;img=Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs=proc(images=img,text=prompt_fn(cls_name),return_tensors='pt')
            inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o=model(**inputs)
            target=torch.tensor([img.size[::-1]]).to(device)
            r=proc.post_process_grounded_object_detection(o,threshold=0.25,target_sizes=target)[0]
            preds=r['boxes'].cpu().tolist()
            gts=[(b[0],b[1],b[2],b[3]) for b in gt_boxes]
            matched=set()
            for pb in preds:
                best=0;bg=-1
                for gi,gb in enumerate(gts):
                    if gi in matched: continue
                    v=iou(pb,gb)
                    if v>best: best=v;bg=gi
                if best>=0.25: matched.add(bg);tp+=1
                else: fp+=1
            fn+=len(gts)-len(matched)
    total_gt=sum(sum(len([(b[0],b[1],b[2],b[3]) for b in gb]) for _,gb in class_data.get(c,[])) for c in ALL_CLS)
    p=tp/(tp+fp) if(tp+fp)else 0;r=tp/total_gt if total_gt else 0
    return round(2*p*r/(p+r) if(p+r)else 0,4)

# TDM with optimal prefix (remote_sensing from Table 4)
def build_tdm(c):
    if c not in ofat_data: return f'a remote sensing image of {c}'
    d=ofat_data[c]['deltas']
    pos=sorted([(a,v) for a,v in d.items() if a!='grammar' and v>0.03],key=lambda x:-x[1])
    mods=[AXM[a] for a,_ in pos[:2]] if pos else []
    p=f'a remote sensing image of {c}'
    if mods: p+=', '+', '.join(mods)
    return p

res={}
print('TDM baseline...',flush=True)
res['TDM (OFAT, zero-shot)']=eval_one(build_tdm)
print(f'  F1={res["TDM (OFAT, zero-shot)"]:.4f}',flush=True)

# Single axis add-ons
for ax in AXES:
    print(f'+{ax}...',flush=True)
    res[f'+ {ax}']=eval_one(lambda c,a=ax: f'a remote sensing image of {c}, {AXM[a]}')
    print(f'  F1={res[f"+ {ax}"]:.4f}',flush=True)

# No pruning
print('All 6 axes...',flush=True)
res['+ all 6 axes (no pruning)']=eval_one(lambda c: f'a remote sensing image of {c}, with fine details, in its natural setting, with distinct structural features, large and prominent, with natural coloring, with clearly visible surface features')
print(f'  F1={res["+ all 6 axes (no pruning)"]:.4f}',flush=True)

# Wrong prefix
print('Wrong prefix (satellite)...',flush=True)
res['TDM w/ satellite (wrong prefix)']=eval_one(lambda c: build_tdm(c).replace('a remote sensing image of','a satellite image of'))
print(f'  F1={res["TDM w/ satellite (wrong prefix)"]:.4f}',flush=True)

# Other prefix baselines
print('Aerial + mods...',flush=True)
res['Aerial + fixed modifiers']=eval_one(lambda c: f'an aerial view of {c}, with distinct structural features, large and prominent')
print(f'  F1={res["Aerial + fixed modifiers"]:.4f}',flush=True)

print('Satellite + mods...',flush=True)
res['Satellite + all modifiers']=eval_one(lambda c: f'a satellite image of {c}, with distinct structural features, large and prominent, with fine details')
print(f'  F1={res["Satellite + all modifiers"]:.4f}',flush=True)

# Random 2-axis (3 trials)
np.random.seed(42)
for t in range(3):
    a1,a2=AXES[t],AXES[(t+3)%6]
    label=f'Random 2-axis: {a1}+{a2}'
    print(f'{label}...',flush=True)
    res[label]=eval_one(lambda c,a1=a1,a2=a2: f'a remote sensing image of {c}, {AXM[a1]}, {AXM[a2]}')
    print(f'  F1={res[label]:.4f}',flush=True)

out=r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\compare_final.json'
with open(out,'w') as f: json.dump(res,f,indent=2)

# Summary sorted
print(f'\n{"="*50}')
print(f'COMPARISON RESULTS (15 methods)')
print(f'{"="*50}')
for name,f1 in sorted(res.items(),key=lambda x:-x[1]):
    vs= '>' if f1>res['TDM (OFAT, zero-shot)'] else '<' if f1<res['TDM (OFAT, zero-shot)'] else '='
    print(f'  {f1:.4f} {vs} {name}')
print(f'\nSaved: {out}')
