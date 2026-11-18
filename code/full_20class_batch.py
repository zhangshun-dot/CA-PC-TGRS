# -*- coding: utf-8 -*-
"""Full 20-class batch: prefixes + ablations on all models."""
import json, torch, numpy as np, io, os, gc
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

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
print(f"Classes: {sum(1 for v in class_data.values() if v)}, Imgs/class: {len(list(class_data.values())[0])}")

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1);a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def metrics(preds_list, gts_list):
    tp=fp=fn=0;dr=0;n=len(preds_list)
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

ALL_MODS = ['with distinct structural features','large and prominent',
            'with natural coloring','with fine details','with clearly visible surface features']
all_results = {}
N_IMG = 20  # full 20 per class

# ═══════════════════════════════════
# 1. GDINO-tiny extra prefixes
# ═══════════════════════════════════
print("\n"+"="*60)
print("GDINO-tiny extra prefixes (20 classes full)")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
tp = AutoProcessor.from_pretrained(tc)
tm = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

for pname, tmpl in [('bare','{}'),('aerial','an aerial view of {}'),('remote_sensing','a remote sensing image of {}')]:
    all_p=[];all_g=[]
    for cls_name, img_list in class_data.items():
        if not img_list: continue
        prompt = tmpl.format(cls_name)
        for img_bytes, gt_boxes in img_list[:N_IMG]:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs = tp(images=img, text=prompt, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o = tm(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = tp.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
            all_p.append(r['boxes'].cpu().tolist())
            all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    m = metrics(all_p,all_g)
    all_results[f'tiny_{pname}'] = m
    print(f"  {pname}: DR={m['dr']:.4f} F1={m['f1']:.4f}")
gc.collect();torch.cuda.empty_cache()

# ═══════════════════════════════════
# 2. YOLO-World ablations
# ═══════════════════════════════════
print("\n"+"="*60)
print("YOLO-World ablations (20 classes full)")
from ultralytics import YOLOWorld
yw = YOLOWorld('yolov8s-worldv2.pt'); yw.to(device)
for n_mods in range(6):
    mods = ALL_MODS[:n_mods]
    all_p=[];all_g=[]
    for cls_name, img_list in class_data.items():
        if not img_list: continue
        prompt = f'an overhead image of {cls_name}'
        if mods: prompt += ', ' + ', '.join(mods)
        yw.set_classes([prompt])
        for img_bytes, gt_boxes in img_list[:N_IMG]:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            r = yw.predict(img, verbose=False, device=device, conf=0.25, iou=0.25)
            pb = r[0].boxes.xyxy.cpu().tolist() if r[0].boxes is not None else []
            all_p.append(pb)
            all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    m = metrics(all_p,all_g)
    all_results[f'yw_abl_{n_mods}'] = m
    print(f"  {n_mods}mods: DR={m['dr']:.4f} F1={m['f1']:.4f}")
gc.collect();torch.cuda.empty_cache();del yw

# ═══════════════════════════════════
# 3. OWLv2 ablations
# ═══════════════════════════════════
print("\n"+"="*60)
print("OWLv2 ablations (20 classes full)")
from transformers import Owlv2Processor, Owlv2ForObjectDetection
oc = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots'
osnap = os.listdir(oc)[0];opath = os.path.join(oc, osnap)
op = Owlv2Processor.from_pretrained(opath)
om = Owlv2ForObjectDetection.from_pretrained(opath, dtype=torch.float32).eval().to(device)
for n_mods in range(6):
    mods = ALL_MODS[:n_mods]
    all_p=[];all_g=[]
    for cls_name, img_list in class_data.items():
        if not img_list: continue
        prompt = f'an overhead image of {cls_name}'
        if mods: prompt += ', ' + ', '.join(mods)
        if len(prompt)>60: prompt = f'an overhead image of {cls_name}'
        for img_bytes, gt_boxes in img_list[:N_IMG]:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs = op(text=prompt, images=img, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o = om(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = op.post_process_object_detection(o, threshold=0.1, target_sizes=target)[0]
            all_p.append(r['boxes'].cpu().tolist())
            all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    m = metrics(all_p,all_g)
    all_results[f'owl_abl_{n_mods}'] = m
    print(f"  {n_mods}mods: DR={m['dr']:.4f} F1={m['f1']:.4f}")

with open(r'C:\Users\张三\Desktop\full20_batch_results.json','w') as f:
    json.dump(all_results,f,indent=2)
print(f"\n{'='*60}")
for k in sorted(all_results.keys()):
    m=all_results[k]; print(f"  {k:<25} DR={m['dr']:.4f} F1={m['f1']:.4f}")
print("Done.")
