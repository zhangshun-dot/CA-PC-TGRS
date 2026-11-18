# -*- coding: utf-8 -*-
"""Ablation on FULL 20-image DIOR with DR/P/R/F1/AP50/AP75/mAP."""

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
print(f'Full DIOR: {len(ALL_CLS)} classes, {sum(len(v) for v in class_data.values())} images')

with open(r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\ofat_overhead_results.json') as f: ofat=json.load(f)
ofat_data=ofat.get('ofat_overhead',ofat)
AXM={'granularity':'with fine details','spatial':'in its natural setting','geometric':'with distinct structural features','scale':'large and prominent','spectral':'with natural coloring','texture':'with clearly visible surface features'}

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if(a1+a2-inter)>0 else 0
def compute_ap(st,n_gt):
    tc=0;fc=0;ps=[];rs=[]
    for s,t in st:
        if t: tc+=1
        else: fc+=1
        ps.append(tc/(tc+fc) if(tc+fc)else 0);rs.append(tc/n_gt if n_gt else 0)
    ap=0.0
    for rt in np.linspace(0,1,11): ap+=max([p for p,r in zip(ps,rs) if r>=rt]+[0])/11
    return round(ap,4)

def evaluate(pfn):
    ot=0;ofp=0;ofn=0;otg=0;dc=0;ni=0;a25=[];a50=[];a75=[];tg=0
    for cls_name in ALL_CLS:
        if cls_name not in class_data: continue
        for img_bytes,gt_boxes in class_data[cls_name]:
            ni+=1
            try:
                img=Image.open(io.BytesIO(img_bytes)).convert('RGB')
                prompt=pfn(cls_name)
                if not prompt or len(prompt.strip())==0:
                    prompt=f'an overhead image of {cls_name}'
                inputs=proc(images=img,text=prompt,return_tensors='pt')
                inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
                with torch.no_grad(): o=model(**inputs)
                target=torch.tensor([img.size[::-1]]).to(device)
                r=proc.post_process_grounded_object_detection(o,threshold=0.0,target_sizes=target)[0]
            except Exception as e:
                print(f'  SKIP {cls_name} (error: {str(e)[:80]})')
                continue
            boxes=r['boxes'].cpu().tolist();scores=r['scores'].cpu().tolist()
            gts=[(b[0],b[1],b[2],b[3]) for b in gt_boxes];ng=len(gts);tg+=ng
            idx=np.argsort(scores)[::-1]
            m25=set();m50=set();m75=set();mo=set();hd=False
            for box,score in zip([boxes[i] for i in idx],[scores[i] for i in idx]):
                bi=0;bg=-1
                for gi,gt in enumerate(gts):
                    v=iou(box,gt)
                    if v>bi: bi=v;bg=gi
                i25=bi>=0.25;i50=bi>=0.50;i75=bi>=0.75
                if bg not in m25: a25.append((score,i25))
                if i25: m25.add(bg)
                if bg not in m50: a50.append((score,i50))
                if i50: m50.add(bg)
                if bg not in m75: a75.append((score,i75))
                if i75: m75.add(bg)
                if score>=0.25:
                    hd=True
                    if i25 and bg not in mo: mo.add(bg);ot+=1
                    elif not i25: ofp+=1
            if hd: dc+=1
            ofn+=ng-len(mo);otg+=ng
    p=ot/(ot+ofp) if(ot+ofp)else 0;r=ot/otg if otg else 0
    return {'dr':round(dc/ni,4),'p':round(p,4),'r':round(r,4),'f1':round(2*p*r/(p+r) if(p+r)else 0,4),
            'ap25':compute_ap(a25,tg),'ap50':compute_ap(a50,tg),'ap75':compute_ap(a75,tg),
            'mAP':round((compute_ap(a25,tg)+compute_ap(a50,tg)+compute_ap(a75,tg))/3,4)}

def build_tdm(cls,delta_thresh=0.03,max_mods=2,drop_axis=None):
    if cls not in ofat_data: return f'an overhead image of {cls}'
    d=ofat_data[cls]['deltas']
    pos=sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and ax!=drop_axis and v>delta_thresh],key=lambda x:-x[1])
    mods=[AXM[ax] for ax,_ in pos[:max_mods]] if pos else []
    p=f'an overhead image of {cls}'
    if mods: p+=', '+', '.join(mods)
    return p

# 1: Axis ablation
print('=== Axis Ablation (full 20) ===',flush=True)
baseline=evaluate(lambda c: build_tdm(c))
print(f'Baseline: DR={baseline["dr"]:.4f} P={baseline["p"]:.4f} R={baseline["r"]:.4f} F1={baseline["f1"]:.4f} AP50={baseline["ap50"]:.4f} AP75={baseline["ap75"]:.4f} mAP={baseline["mAP"]:.4f}',flush=True)

AXES=['grammar','granularity','spatial','geometric','scale','spectral','texture']
axis_res={}
for ax in AXES:
    print(f'  -{ax}...',flush=True)
    axis_res[ax]=evaluate(lambda c,a=ax: build_tdm(c,drop_axis=a))
    m=axis_res[ax]
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} mAP={m["mAP"]:.4f} DmAP={m["mAP"]-baseline["mAP"]:+.4f}',flush=True)

# 2: Max modifiers
print('\n=== Max Modifiers (full 20) ===',flush=True)
mod_res={}
for mm in [1,2,3,4]:
    print(f'  max {mm}...',flush=True)
    mod_res[str(mm)]=evaluate(lambda c,m=mm: build_tdm(c,max_mods=m))
    m=mod_res[str(mm)]
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} mAP={m["mAP"]:.4f}',flush=True)

# 3: Delta threshold
print('\n=== Delta Threshold (full 20) ===',flush=True)
delta_res={}
for dt in [0.01,0.02,0.03,0.05,0.07,0.10]:
    print(f'  d>{dt:.2f}...',flush=True)
    delta_res[f'{dt:.2f}']=evaluate(lambda c,t=dt: build_tdm(c,delta_thresh=t))
    m=delta_res[f'{dt:.2f}']
    print(f'    F1={m["f1"]:.4f} mAP={m["mAP"]:.4f}',flush=True)

# Save
out={'baseline':baseline,'axis_ablation':axis_res,'max_modifiers':mod_res,'delta_threshold':delta_res}
out_path=r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\ablation_full20_all.json'
with open(out_path,'w') as f: json.dump(out,f,indent=2)

print(f'\n=== ABLATION (full 20, all metrics) ===')
print(f'  {"Axis":<20} {"DR":>8} {"P":>8} {"R":>8} {"F1":>8} {"AP50":>8} {"AP75":>8} {"mAP":>8} {"DmAP":>8}')
print(f'  {"baseline":<20} {baseline["dr"]:>8.4f} {baseline["p"]:>8.4f} {baseline["r"]:>8.4f} {baseline["f1"]:>8.4f} {baseline["ap50"]:>8.4f} {baseline["ap75"]:>8.4f} {baseline["mAP"]:>8.4f} {"—":>8}')
for ax in AXES:
    m=axis_res[ax]
    print(f'  {"-"+ax:<20} {m["dr"]:>8.4f} {m["p"]:>8.4f} {m["r"]:>8.4f} {m["f1"]:>8.4f} {m["ap50"]:>8.4f} {m["ap75"]:>8.4f} {m["mAP"]:>8.4f} {m["mAP"]-baseline["mAP"]:>+8.4f}')
print(f'\nSaved: {out_path}')
