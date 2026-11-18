# -*- coding: utf-8 -*-
"""Cross-dataset: COCO-base mAP + DOTA Detection Rate."""
import json, torch, numpy as np, io, os
from PIL import Image
from collections import defaultdict

device='cuda';os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from pycocotools.coco import COCO

# Helpers
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

def eval_full(pfn,data,proc,model):
    ot=0;ofp=0;ofn=0;otg=0;dc=0;ni=0;a25=[];a50=[];a75=[];tg=0
    for path,gt_boxes in data:
        ni+=1;img=Image.open(path).convert('RGB')
        inputs=proc(images=img,text=pfn,return_tensors='pt')
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

# ============================================
# 1: COCO-base mAP
# ============================================
print('=== COCO-base mAP ===',flush=True)
bc=r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots\12bdfa3120f3e7ec7b434d90674b3396eccf88eb'
bp=AutoProcessor.from_pretrained(bc)
bm=AutoModelForZeroShotObjectDetection.from_pretrained(bc,dtype=torch.float32).eval().to(device)

COCO_CLS=['person','car','dog','chair','bottle','cup','boat','airplane','bird',
          'potted plant','dining table','couch','motorcycle','cat','tv',
          'laptop','sink','refrigerator','book','clock']
ann_file=r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
img_dir=r'C:\Users\张三\modeling\coco_data\val2017'
coco=COCO(ann_file)

all_imgs=[]
for cls in COCO_CLS:
    cat_ids=coco.getCatIds(catNms=[cls])
    if not cat_ids: continue
    cat_id=cat_ids[0];img_ids=coco.getImgIds(catIds=[cat_id])
    imgs_meta=coco.loadImgs(img_ids)
    for meta in imgs_meta:
        if sum(1 for x in all_imgs if x[0]==os.path.join(img_dir,meta['file_name']))>=20: continue
        path=os.path.join(img_dir,meta['file_name'])
        if not os.path.exists(path): continue
        ann_ids=coco.getAnnIds(imgIds=[meta['id']],catIds=[cat_id],iscrowd=False)
        anns=coco.loadAnns(ann_ids)
        gt_boxes=[[ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3]] for ann in anns]
        if gt_boxes: all_imgs.append((path,gt_boxes,cls))
        if sum(1 for x in all_imgs if x[2]==cls)>=20: break

print(f'COCO images: {len(all_imgs)}')

coco_mAP={}
for name,pfn in [('satellite',lambda c:f'a satellite image of {c}'),
                  ('overhead',lambda c:f'an overhead image of {c}'),
                  ('tdm',lambda c:f'an overhead image of {c}, with distinct structural features, large and prominent')]:
    print(f'  {name}...',flush=True)
    flat_data=[(p,gb) for p,gb,c in all_imgs]
    flat_prompts=[pfn(c) for p,gb,c in all_imgs]
    m=eval_full(flat_prompts[0],flat_data,bp,bm)  # Need per-image eval
    # Actually run per-image properly
    ot=0;ofp=0;ofn=0;otg=0;dc=0;ni=0;a25=[];a50=[];a75=[];tg=0
    for path,gt_boxes,cls_name in all_imgs:
        ni+=1;img=Image.open(path).convert('RGB')
        prompt=pfn(cls_name)
        inputs=bp(images=img,text=prompt,return_tensors='pt')
        inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad(): o=bm(**inputs)
        target=torch.tensor([img.size[::-1]]).to(device)
        r=bp.post_process_grounded_object_detection(o,threshold=0.0,target_sizes=target)[0]
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
        if ni%100==0: print(f'    {ni}/{len(all_imgs)}...',flush=True)
    p=ot/(ot+ofp) if(ot+ofp)else 0;r=ot/otg if otg else 0
    coco_mAP[name]={'dr':round(dc/ni,4),'p':round(p,4),'r':round(r,4),'f1':round(2*p*r/(p+r) if(p+r)else 0,4),
                     'ap25':compute_ap(a25,tg),'ap50':compute_ap(a50,tg),'ap75':compute_ap(a75,tg),
                     'mAP':round((compute_ap(a25,tg)+compute_ap(a50,tg)+compute_ap(a75,tg))/3,4)}
    print(f'    DR={coco_mAP[name]["dr"]:.4f} F1={coco_mAP[name]["f1"]:.4f} AP50={coco_mAP[name]["ap50"]:.4f} mAP={coco_mAP[name]["mAP"]:.4f}',flush=True)

# ============================================
# 2: DOTA Detection Rate only
# ============================================
print('\n=== DOTA DR ===',flush=True)
DOTA_MAP={'plane':'airplane','ship':'ship','storage-tank':'storage tank','baseball-diamond':'baseball field',
    'tennis-court':'tennis court','basketball-court':'basketball court','ground-track-field':'ground track field',
    'harbor':'harbor','bridge':'bridge','large-vehicle':'vehicle','small-vehicle':'vehicle'}
IMG_DIR=r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
LBL_DIR=r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'
class_imgs=defaultdict(list)
for fn in os.listdir(LBL_DIR):
    if not fn.endswith('.txt'): continue
    img_path=os.path.join(IMG_DIR,fn.replace('.txt','.png'))
    if not os.path.exists(img_path): continue
    with open(os.path.join(LBL_DIR,fn)) as f: lines=f.readlines()
    bboxes_by_cls=defaultdict(list)
    for line in lines:
        parts=line.strip().split()
        if len(parts)<9: continue
        try:
            x1,y1,x2,y2,x3,y3,x4,y4=map(float,parts[:8]);cls_name=parts[8]
            if cls_name in DOTA_MAP: bboxes_by_cls[cls_name].append([x1,y1,x2,y2,x3,y3,x4,y4])
        except: continue
    for cls_name,bboxes in bboxes_by_cls.items():
        if len(class_imgs[cls_name])<10: class_imgs[cls_name].append((img_path,bboxes))
overlap=sorted(class_imgs.keys())
print(f'DOTA overlap classes: {len(overlap)}')

tc=r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
tp=AutoProcessor.from_pretrained(tc)
tm=AutoModelForZeroShotObjectDetection.from_pretrained(tc,dtype=torch.float32).eval().to(device)

dota_dr={}
for name,pfn in [('satellite',lambda c:f'a satellite image of {c}'),
                  ('overhead',lambda c:f'an overhead image of {c}'),
                  ('tdm',lambda c:f'an overhead image of {c}, with distinct structural features, large and prominent')]:
    per_cls={};all_dr=[]
    for cls_name in overlap:
        dior_name=DOTA_MAP[cls_name];imgs=class_imgs[cls_name]
        drs=[]
        for img_path,quads in imgs:
            img=Image.open(img_path).convert('RGB')
            prompt=pfn(dior_name)
            inputs=tp(images=img,text=prompt,return_tensors='pt')
            inputs={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o=tm(**inputs)
            target=torch.tensor([img.size[::-1]]).to(device)
            r=tp.post_process_grounded_object_detection(o,threshold=0.25,target_sizes=target)[0]
            drs.append(1.0 if len(r['boxes'])>0 else 0.0)
        per_cls[dior_name]=float(np.mean(drs));all_dr.extend(drs)
    dota_dr[name]={'per_class':per_cls,'mean_dr':float(np.mean(all_dr))}
    print(f'  {name}: mean DR={np.mean(all_dr):.4f}',flush=True)

# Save
out_dir=r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据'
with open(f'{out_dir}/coco_base_mAP.json','w') as f: json.dump(coco_mAP,f,indent=2)
with open(f'{out_dir}/dota_dr.json','w') as f: json.dump(dota_dr,f,indent=2)

print(f'\n=== CROSS-DATA COMPLETE ===')
print(f'COCO-base: sat F1={coco_mAP["satellite"]["f1"]:.4f} mAP={coco_mAP["satellite"]["mAP"]:.4f}')
print(f'           oh  F1={coco_mAP["overhead"]["f1"]:.4f} mAP={coco_mAP["overhead"]["mAP"]:.4f}')
print(f'           TDM F1={coco_mAP["tdm"]["f1"]:.4f} mAP={coco_mAP["tdm"]["mAP"]:.4f}')
print(f'DOTA DR:   sat={dota_dr["satellite"]["mean_dr"]:.4f}')
print(f'           oh ={dota_dr["overhead"]["mean_dr"]:.4f}')
print(f'           TDM={dota_dr["tdm"]["mean_dr"]:.4f}')
