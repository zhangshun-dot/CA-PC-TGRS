# -*- coding: utf-8 -*-
"""Batch experiments: modifier ablation (YOLO-W, OWLv2) + extra prefixes + COCO."""
import json, torch, numpy as np, io, os
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")
torch.cuda.empty_cache()

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

ablation_cls = ['airplane','ship','vehicle','bridge','harbor']
ablation_data = {c: class_data[c][:10] for c in ablation_cls if c in class_data}  # 10 imgs/class for speed

def compute_iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def compute_mets(preds_list, gts_list):
    tp=fp=fn=0; dr=0; n=len(preds_list)
    for preds, gts in zip(preds_list, gts_list):
        if preds: dr+=1
        matched=set()
        for pb in preds:
            best_iou=0;best_gi=-1
            for gi,gb in enumerate(gts):
                if gi in matched: continue
                iou=compute_iou(pb,gb)
                if iou>best_iou: best_iou=iou;best_gi=gi
            if best_iou>=0.25: matched.add(best_gi);tp+=1
            else: fp+=1
        fn+=len(gts)-len(matched)
    p=tp/(tp+fp) if (tp+fp) else 0
    r=tp/(tp+fn) if (tp+fn) else 0
    return {'dr':dr/n,'precision':p,'recall':r,'f1':2*p*r/(p+r) if(p+r) else 0}

ALL_MODS = ['with distinct structural features','large and prominent',
            'with natural coloring','with fine details','with clearly visible surface features']

all_results = {}

# ═══════════════════════════════════════════
# 1. YOLO-World ablation
# ═══════════════════════════════════════════
print("\n"+"="*60)
print("EXP A: YOLO-World modifier ablation")
from ultralytics import YOLOWorld
yw = YOLOWorld('yolov8s-worldv2.pt'); yw.to(device)

for n_mods in range(6):
    mods = ALL_MODS[:n_mods]
    all_preds=[]; all_gts=[]
    for cls_name, img_list in ablation_data.items():
        prompt = f'an overhead image of {cls_name}'
        if mods: prompt += ', ' + ', '.join(mods)
        yw.set_classes([prompt])
        for img_bytes, gt_boxes in img_list:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            r = yw.predict(img, verbose=False, device=device, conf=0.25, iou=0.25)
            pboxes = r[0].boxes.xyxy.cpu().tolist() if r[0].boxes is not None else []
            all_preds.append(pboxes)
            all_gts.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    m = compute_mets(all_preds, all_gts)
    all_results[f'yw_ablation_{n_mods}mods'] = m
    print(f"  {n_mods} mods: DR={m['dr']:.4f} F1={m['f1']:.4f}")

torch.cuda.empty_cache()
del yw

# ═══════════════════════════════════════════
# 2. OWLv2 ablation
# ═══════════════════════════════════════════
print("\n"+"="*60)
print("EXP B: OWLv2 modifier ablation")
from transformers import Owlv2Processor, Owlv2ForObjectDetection
cache = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots'
snap = os.listdir(cache)[0]
owl_proc = Owlv2Processor.from_pretrained(os.path.join(cache, snap))
owl_model = Owlv2ForObjectDetection.from_pretrained(os.path.join(cache, snap), dtype=torch.float32).eval().to(device)

for n_mods in range(6):
    mods = ALL_MODS[:n_mods]
    all_preds=[]; all_gts=[]
    for cls_name, img_list in ablation_data.items():
        prompt = f'an overhead image of {cls_name}'
        if mods: prompt += ', ' + ', '.join(mods)
        if len(prompt) > 60: prompt = f'an overhead image of {cls_name}'  # OWLv2 token limit
        for img_bytes, gt_boxes in img_list:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs = owl_proc(text=prompt, images=img, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad():
                o = owl_model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = owl_proc.post_process_object_detection(o, threshold=0.1, target_sizes=target)[0]
            pboxes = r['boxes'].cpu().tolist()
            all_preds.append(pboxes)
            all_gts.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    m = compute_mets(all_preds, all_gts)
    all_results[f'owl_ablation_{n_mods}mods'] = m
    print(f"  {n_mods} mods: DR={m['dr']:.4f} F1={m['f1']:.4f}")

torch.cuda.empty_cache()
del owl_model, owl_proc

# ═══════════════════════════════════════════
# 3. Extra prefixes on GDINO-tiny
# ═══════════════════════════════════════════
print("\n"+"="*60)
print("EXP C: Extra prefixes GDINO-tiny")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
tiny_cache = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
tiny_proc = AutoProcessor.from_pretrained(tiny_cache)
tiny_model = AutoModelForZeroShotObjectDetection.from_pretrained(tiny_cache, dtype=torch.float32).eval().to(device)

extra_prefixes = [
    ('bare', '{}'),
    ('aerial', 'an aerial view of {}'),
    ('remote_sensing', 'a remote sensing image of {}'),
]
for pname, tmpl in extra_prefixes:
    all_preds=[]; all_gts=[]
    for cls_name, img_list in class_data.items():
        if not img_list: continue
        prompt = tmpl.format(cls_name)
        for img_bytes, gt_boxes in img_list[:5]:  # 5 images/class for speed
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs = tiny_proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o = tiny_model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = tiny_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
            all_preds.append(r['boxes'].cpu().tolist())
            all_gts.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    m = compute_mets(all_preds, all_gts)
    all_results[f'tiny_{pname}'] = m
    print(f"  {pname}: DR={m['dr']:.4f} F1={m['f1']:.4f}")

torch.cuda.empty_cache()

# ═══════════════════════════════════════════
# Save
# ═══════════════════════════════════════════
with open(r'C:\Users\张三\Desktop\batch_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*60}")
print(f"Total experiments: {len(all_results)}")
for k in sorted(all_results.keys()):
    m = all_results[k]
    print(f"  {k:<35} DR={m['dr']:.4f} F1={m['f1']:.4f}")
print("\nDone.")
