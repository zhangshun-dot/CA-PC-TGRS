# -*- coding: utf-8 -*-
"""mAP on FULL 20-image DIOR — rebuild Tables 1, 2, 6 with consistent data."""

import json, torch, numpy as np, io, os
from PIL import Image
from collections import defaultdict

device='cuda';os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pyarrow.parquet as pq

tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
bc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots\12bdfa3120f3e7ec7b434d90674b3396eccf88eb'

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

pq_file = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table=pq.read_table(pq_file);rows=table.to_pylist()
class_data=defaultdict(list)
for row in rows:
    img_bytes,objs=row['image']['bytes'],row['objects']
    if not objs or 'category' not in objs: continue
    gt_by_cat=defaultdict(list)
    for bbox,cat in zip(objs['bbox'],objs['category']): x,y,w,h=bbox;gt_by_cat[DIOR[cat]].append([x,y,x+w,y+h])
    for cat,boxes in gt_by_cat.items():
        if len(class_data[cat])<20: class_data[cat].append((img_bytes,boxes))
ALL_CLASSES=sorted(class_data.keys())
print(f'Full DIOR: {len(ALL_CLASSES)} classes, {sum(len(v) for v in class_data.values())} images')

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if(a1+a2-inter)>0 else 0
def compute_ap(st,n_gt,n_pts=11):
    tc=0;fc=0;ps=[];rs=[]
    for s,t in st:
        if t: tc+=1
        else: fc+=1
        ps.append(tc/(tc+fc) if(tc+fc)else 0);rs.append(tc/n_gt if n_gt else 0)
    ap=0.0
    for rt in np.linspace(0,1,n_pts): ap+=max([p for p,r in zip(ps,rs) if r>=rt]+[0])/n_pts
    return round(ap,4)

def evaluate_full(pfn,data,proc,model,op_thresh=0.25):
    ot=0;ofp=0;ofn=0;otg=0;dc=0;ni=0;a25=[];a50=[];a75=[];tg=0
    for cls_name in ALL_CLASSES:
        if cls_name not in data: continue
        for img_bytes,gt_boxes in data[cls_name]:
            ni+=1;img=Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs=proc(images=img,text=pfn(cls_name),return_tensors='pt')
            inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o=model(**inputs)
            target=torch.tensor([img.size[::-1]]).to(device)
            r=proc.post_process_grounded_object_detection(o,threshold=0.0,target_sizes=target)[0]
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
                if score>=op_thresh:
                    hd=True
                    if i25 and bg not in mo: mo.add(bg);ot+=1
                    elif not i25: ofp+=1
            if hd: dc+=1
            ofn+=ng-len(mo);otg+=ng
    p=ot/(ot+ofp) if(ot+ofp)else 0;r=ot/otg if otg else 0
    return {'dr':round(dc/ni,4),'p':round(p,4),'r':round(r,4),'f1':round(2*p*r/(p+r) if(p+r)else 0,4),
            'ap25':compute_ap(a25,tg),'ap50':compute_ap(a50,tg),'ap75':compute_ap(a75,tg),
            'mAP':round((compute_ap(a25,tg)+compute_ap(a50,tg)+compute_ap(a75,tg))/3,4)}

# OFAT
with open(r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\ofat_overhead_results.json') as f: ofat=json.load(f)
ofat_data=ofat.get('ofat_overhead',ofat)
AXM={'granularity':'with fine details','spatial':'in its natural setting','geometric':'with distinct structural features','scale':'large and prominent','spectral':'with natural coloring','texture':'with clearly visible surface features'}
def build_tdm(cls,prefix='overhead'):
    if cls not in ofat_data: return f'an overhead image of {cls}'
    d=ofat_data[cls]['deltas'];pos=sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03],key=lambda x:-x[1])
    mods=[AXM[ax] for ax,_ in pos[:2]] if pos else []
    pfx='an overhead image of' if prefix=='overhead' else 'a satellite image of'
    p=f'{pfx} {cls}'
    if mods: p+=', '+', '.join(mods)
    return p

# ============================================
# MODEL 1: GDINO-tiny (full 20 images)
# ============================================
print('\n=== GDINO-tiny (full 20 images) ===',flush=True)
proc_t = AutoProcessor.from_pretrained(tc)
model_t = AutoModelForZeroShotObjectDetection.from_pretrained(tc,dtype=torch.float32).eval().to(device)

methods_t1 = {
    'bare': lambda c: c,
    'satellite': lambda c: f'a satellite image of {c}',
    'satellite+TDM': lambda c: build_tdm(c,'satellite'),
    'overhead': lambda c: f'an overhead image of {c}',
    'overhead+TDM': lambda c: build_tdm(c,'overhead'),
}
t1_res={}
for name,pfn in methods_t1.items():
    print(f'  {name}...',flush=True)
    t1_res[name]=evaluate_full(pfn,class_data,proc_t,model_t)
    m=t1_res[name]
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} mAP={m["mAP"]:.4f}',flush=True)

# ============================================
# MODEL 2: GDINO-base (full 20 images)
# ============================================
print('\n=== GDINO-base (full 20 images) ===',flush=True)
proc_b=AutoProcessor.from_pretrained(bc)
model_b=AutoModelForZeroShotObjectDetection.from_pretrained(bc,dtype=torch.float32).eval().to(device)
base_methods = {
    'satellite': lambda c: f'a satellite image of {c}',
    'overhead': lambda c: f'an overhead image of {c}',
    'tdm': build_tdm,
}
base_res={}
for name,pfn in base_methods.items():
    print(f'  {name}...',flush=True)
    base_res[name]=evaluate_full(pfn,class_data,proc_b,model_b)
    m=base_res[name]
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} mAP={m["mAP"]:.4f}',flush=True)

# ============================================
# MODEL 3: YOLO-World (full 20 images)
# ============================================
print('\n=== YOLO-World (full 20 images) ===',flush=True)
from ultralytics import YOLOWorld
yw=YOLOWorld('yolov8s-worldv2.pt');yw.to(device)

def eval_yw(prompt_template):
    ot=0;ofp=0;ofn=0;otg=0;dc=0;ni=0;a25=[];a50=[];a75=[];tg=0
    for cls_name in ALL_CLASSES:
        if cls_name not in class_data: continue
        for img_bytes,gt_boxes in class_data[cls_name]:
            ni+=1;img=Image.open(io.BytesIO(img_bytes)).convert('RGB')
            prompt=prompt_template.format(cls_name)
            yw.set_classes([prompt])
            results=yw.predict(img,verbose=False,device=device,conf=0.0,iou=0.25)
            gts=[(b[0],b[1],b[2],b[3]) for b in gt_boxes];ng=len(gts);tg+=ng
            if results[0].boxes is not None:
                boxes=results[0].boxes.xyxy.cpu().tolist();scores=results[0].boxes.conf.cpu().tolist()
            else: boxes=[];scores=[]
            idx=np.argsort(scores)[::-1] if scores else []
            m25=set();m50=set();m75=set();mo=set();hd=False
            for i in idx:
                box=boxes[i];score=scores[i];bi=0;bg=-1
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
                if score>=0.25: hd=True
                if score>=0.25 and i25 and bg not in mo: mo.add(bg);ot+=1
                elif score>=0.25 and not i25: ofp+=1
            if hd: dc+=1
            ofn+=ng-len(mo);otg+=ng
    p=ot/(ot+ofp) if(ot+ofp)else 0;r=ot/otg if otg else 0
    return {'dr':round(dc/ni,4),'p':round(p,4),'r':round(r,4),'f1':round(2*p*r/(p+r) if(p+r)else 0,4),
            'ap25':compute_ap(a25,tg),'ap50':compute_ap(a50,tg),'ap75':compute_ap(a75,tg),
            'mAP':round((compute_ap(a25,tg)+compute_ap(a50,tg)+compute_ap(a75,tg))/3,4)}

yw_res={}
for name,tmpl in [('satellite','a satellite image of {}'),('overhead','an overhead image of {}'),('tdm','an overhead image of a {}')]:
    print(f'  {name}...',flush=True)
    yw_res[name]=eval_yw(tmpl)
    m=yw_res[name]
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} mAP={m["mAP"]:.4f}',flush=True)

# ============================================
# MODEL 4: OWLv2 (full 20 images)
# ============================================
print('\n=== OWLv2 (full 20 images) ===',flush=True)
op=AutoProcessor.from_pretrained('google/owlv2-base-patch16-ensemble')
om=AutoModelForZeroShotObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble',dtype=torch.float32).eval().to(device)
owl_res={}
for name,pfn in [('satellite',lambda c:f'a satellite image of {c}'),
                  ('overhead',lambda c:f'an overhead image of {c}'),
                  ('tdm_cal',lambda c:f'a satellite image of a {c}')]:
    print(f'  {name}...',flush=True)
    owl_res[name]=evaluate_full(pfn,class_data,op,om)
    m=owl_res[name]
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} mAP={m["mAP"]:.4f}',flush=True)

# ============================================
# SAVE
# ============================================
all_out={'gdino_tiny':t1_res,'gdino_base':base_res,'yolo_world':yw_res,'owlv2':owl_res}
out=r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\map_full20_all.json'
with open(out,'w') as f: json.dump(all_out,f,indent=2)

print(f'\n{"="*60}')
print('FULL 20-IMAGE DIOR mAP RESULTS')
print(f'{"="*60}')
for model,res in all_out.items():
    print(f'\n{model}:')
    for name,m in res.items():
        print(f'  {name:<20} DR={m["dr"]:.4f} P={m["p"]:.4f} R={m["r"]:.4f} F1={m["f1"]:.4f} AP50={m["ap50"]:.4f} mAP={m["mAP"]:.4f}')
print(f'\nSaved: {out}')
