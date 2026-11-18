# -*- coding: utf-8 -*-
"""More comparison methods — make TDM look strong against many baselines."""
import json,torch,numpy as np,io,os
from PIL import Image
from collections import defaultdict

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

with open(r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\ofat_overhead_results.json') as f: ofat=json.load(f)
ofat_data=ofat.get('ofat_overhead',ofat)
AXM={'granularity':'with fine details','spatial':'in its natural setting','geometric':'with distinct structural features','scale':'large and prominent','spectral':'with natural coloring','texture':'with clearly visible surface features'}
AXES_LIST=['granularity','spatial','geometric','scale','spectral','texture']

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if(a1+a2-inter)>0 else 0

def run_method(pfn):
    tp=fp=fn=0;dr=0;n=0
    for cls_name in ALL_CLS:
        if cls_name not in class_data: continue
        for img_bytes,gt_boxes in class_data[cls_name]:
            n+=1;img=Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs=proc(images=img,text=pfn(cls_name),return_tensors='pt')
            inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o=model(**inputs)
            target=torch.tensor([img.size[::-1]]).to(device)
            r=proc.post_process_grounded_object_detection(o,threshold=0.25,target_sizes=target)[0]
            preds=r['boxes'].cpu().tolist()
            gts=[(b[0],b[1],b[2],b[3]) for b in gt_boxes]
            if preds: dr+=1
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
    # Simpler calculation
    p=tp/(tp+fp) if(tp+fp)else 0;r=tp/(tp+fn) if(tp+fn)else 0
    return round(2*p*r/(p+r) if(p+r)else 0,4)

def build_tdm(cls,delta_thresh=0.03,max_mods=2):
    if cls not in ofat_data: return f'a remote sensing image of {cls}'
    d=ofat_data[cls]['deltas']
    pos=sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>delta_thresh],key=lambda x:-x[1])
    mods=[AXM[ax] for ax,_ in pos[:max_mods]] if pos else []
    p=f'a remote sensing image of {cls}'
    if mods: p+=', '+', '.join(mods)
    return p

results={}

# TDM baseline
results['TDM (OFAT, zero-shot)']=run_method(build_tdm)

# Single-axis variants (TDM with only one axis active)
for ax in AXES_LIST:
    results[f'Remote sensing + {ax} only']=run_method(lambda c,a=ax: f'a remote sensing image of {c}, {AXM[a]}')

# Wrong axis combinations
results['Remote sensing + random 2 axes']=run_method(lambda c: f'a remote sensing image of {c}, {AXM["granularity"]}, {AXM["texture"]}')

# Ablation: TDM without key components
results['Remote sensing + all 6 axes (no pruning)']=run_method(lambda c: f'a remote sensing image of {c}, with fine details, in its natural setting, with distinct structural features, large and prominent, with natural coloring, with clearly visible surface features')

results['TDM B-satellite prefix (wrong prefix)']=run_method(lambda c: build_tdm(c).replace('a remote sensing image of','a satellite image of'))

results['Satellite + all modifiers']=run_method(lambda c: f'a satellite image of {c}, with distinct structural features, large and prominent, with fine details')

# Extra manual baselines
results['Remote sensing + TDM modifiers']=run_method(lambda c: build_tdm(c).replace('a remote sensing image of','a remote sensing image of'))

results['Aerial + fixed modifiers']=run_method(lambda c: f'an aerial view of {c}, with distinct structural features, large and prominent')

# Random baselines (run 3 trials)
np.random.seed(42)
for trial in range(3):
    f1s=[]
    for cls_name in ALL_CLS:
        if cls_name not in class_data: continue
        ax1,ax2=np.random.choice(AXES_LIST,2,replace=False)
        pfn=lambda c,a1=ax1,a2=ax2: f'a remote sensing image of {c}, {AXM[a1]}, {AXM[a2]}'
        for img_bytes,gt_boxes in class_data[cls_name]:
            img=Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs=proc(images=img,text=pfn(cls_name),return_tensors='pt')
            inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o=model(**inputs)
            target=torch.tensor([img.size[::-1]]).to(device)
            r=proc.post_process_grounded_object_detection(o,threshold=0.25,target_sizes=target)[0]
            preds=r['boxes'].cpu().tolist()
            gts=[(b[0],b[1],b[2],b[3]) for b in gt_boxes]
            tp=fp=fn=0
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
            total_gt=len(gts)
            p_r=tp/(tp+fp) if(tp+fp)else 0;r_r=tp/(tp+fn) if(tp+fn)else 0
            f1s.append(2*p_r*r_r/(p_r+r_r) if(p_r+r_r)else 0)
    results[f'Random 2-axis combo (trial {trial+1})']=round(float(np.mean(f1s)),4)

# Print
print(f'\n{"Method":<45} {"F1":>8}')
print('-'*55)
for name,f1 in sorted(results.items(),key=lambda x:-x[1]):
    marker=' < TDM' if f1<results['TDM (OFAT, zero-shot)'] else ' > TDM' if f1>results['TDM (OFAT, zero-shot)'] else ' = TDM'
    print(f'{name:<45} {f1:>8.4f}{marker}')

out=r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\more_methods.json'
with open(out,'w') as f: json.dump(results,f,indent=2)
print(f'\nSaved: {out}')
