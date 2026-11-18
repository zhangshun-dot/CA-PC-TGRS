# -*- coding: utf-8 -*-
"""YOLO-World ablation: axis ablation + max modifiers on full DIOR 20."""
import json,torch,numpy as np,io,os
from PIL import Image
from collections import defaultdict

device='cuda'
from ultralytics import YOLOWorld
yw=YOLOWorld('yolov8s-worldv2.pt');yw.to(device)

DIOR=['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']
import pyarrow.parquet as pq
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
print(f'DIOR: {len(ALL_CLS)} classes, {sum(len(v) for v in class_data.values())} images')

with open(r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\ofat_overhead_results.json') as f: ofat=json.load(f)
ofat_data=ofat.get('ofat_overhead',ofat)
AXM={'granularity':'with fine details','spatial':'in its natural setting','geometric':'with distinct structural features','scale':'large and prominent','spectral':'with natural coloring','texture':'with clearly visible surface features'}
AXES=['granularity','spatial','geometric','scale','spectral','texture']

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if(a1+a2-inter)>0 else 0

def yw_metrics(prompt_template):
    tp=fp=fn=dr=ni=0
    for cls_name in ALL_CLS:
        if cls_name not in class_data: continue
        for img_bytes,gt_boxes in class_data[cls_name]:
            ni+=1;img=Image.open(io.BytesIO(img_bytes)).convert('RGB')
            prompt=prompt_template(cls_name)
            yw.set_classes([prompt])
            res=yw.predict(img,verbose=False,device=device,conf=0.25,iou=0.25)
            gts=[(b[0],b[1],b[2],b[3]) for b in gt_boxes]
            if res[0].boxes is not None:
                preds=res[0].boxes.xyxy.cpu().tolist()
            else:
                preds=[]
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
    p=tp/(tp+fp) if(tp+fp)else 0;r=tp/(tp+fn) if(tp+fn)else 0
    return {'dr':round(dr/ni,4),'p':round(p,4),'r':round(r,4),'f1':round(2*p*r/(p+r) if(p+r)else 0,4)}

def build_tdm(cls,delta_thresh=0.03,max_mods=2,drop_axis=None):
    if cls not in ofat_data: return f'an overhead image of {cls}'
    d=ofat_data[cls]['deltas']
    pos=sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and ax!=drop_axis and v>delta_thresh],key=lambda x:-x[1])
    mods=[AXM[ax] for ax,_ in pos[:max_mods]] if pos else []
    p=f'an overhead image of {cls}'
    if mods: p+=', '+', '.join(mods)
    return p

# Baseline
print('=== YOLO-World Ablation ===')
base=yw_metrics(lambda c: build_tdm(c))
print(f'Baseline: DR={base["dr"]:.4f} F1={base["f1"]:.4f}',flush=True)

# Axis ablation
print('\nAxis ablation:')
axis_res={}
for ax in AXES:
    print(f'  -{ax}...',flush=True)
    axis_res[ax]=yw_metrics(lambda c,a=ax: build_tdm(c,drop_axis=a))
    m=axis_res[ax]
    d=m['f1']-base['f1']
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} D={d:+.4f}',flush=True)

# Max modifiers
print('\nMax modifiers:')
mod_res={}
for mm in [1,2,3,4]:
    print(f'  max {mm}...',flush=True)
    mod_res[str(mm)]=yw_metrics(lambda c,m=mm: build_tdm(c,max_mods=m))
    m=mod_res[str(mm)]
    d=m['f1']-base['f1']
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} D={d:+.4f}',flush=True)

out={'baseline':base,'axis_ablation':axis_res,'max_modifiers':mod_res}
out_path=r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\yolo_ablation.json'
with open(out_path,'w') as f: json.dump(out,f,indent=2)

print(f'\n=== YOLO ABLATION SUMMARY ===')
print(f'Baseline TDM: F1={base["f1"]:.4f}')
print(f'Axis ablation:')
for ax in AXES:
    d=axis_res[ax]['f1']-base['f1']
    print(f'  -{ax:<15}: F1={axis_res[ax]["f1"]:.4f} D={d:+.4f}')
print(f'Max modifiers:')
for mm in ['1','2','3','4']:
    d=mod_res[mm]['f1']-base['f1']
    print(f'  {mm}: F1={mod_res[mm]["f1"]:.4f} D={d:+.4f}')
print(f'\nSaved: {out_path}')
