# -*- coding: utf-8 -*-
"""Remaining experiments: YOLO-World mAP + Qualitative viz + Method comparison"""

import json, torch, numpy as np, io, os, sys
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

device = 'cuda'
os.environ['HF_HUB_OFFLINE'] = '1'
import pyarrow.parquet as pq

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
    for bbox, cat in zip(objs['bbox'], objs['category']): x,y,w,h = bbox; gt_by_cat[DIOR[cat]].append([x,y,x+w,y+h])
    for cat, boxes in gt_by_cat.items():
        if len(class_data[cat]) < 20: class_data[cat].append((img_bytes, boxes))

np.random.seed(42)
test_data = {}
for cls_name, imgs in class_data.items():
    idx = np.random.permutation(len(imgs)); n_test = min(10, len(imgs)//2)
    test_data[cls_name] = [imgs[i] for i in idx[:n_test]]
ALL_CLASSES = sorted(test_data.keys())

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

# ═══════════════════════════════════════
# 1: YOLO-World mAP
# ═══════════════════════════════════════
print("="*60)
print("1: YOLO-World mAP")
print("="*60)
from ultralytics import YOLOWorld
yw_model = YOLOWorld('yolov8s-worldv2.pt')
yw_model.to(device)

def eval_yolow(prompt_template, op_thresh=0.25):
    ot=0;ofp=0;ofn=0;otg=0;dc=0;ni=0;a25=[];a50=[];a75=[];tg=0
    for cls_name in ALL_CLASSES:
        if cls_name not in test_data: continue
        for img_bytes, gt_boxes in test_data[cls_name]:
            ni+=1;img=Image.open(io.BytesIO(img_bytes)).convert('RGB')
            prompt=prompt_template.format(cls_name)
            yw_model.set_classes([prompt])
            results=yw_model.predict(img,verbose=False,device=device,conf=0.0,iou=0.25)
            gts=[(b[0],b[1],b[2],b[3]) for b in gt_boxes];ng=len(gts);tg+=ng
            if results[0].boxes is not None:
                boxes=results[0].boxes.xyxy.cpu().tolist()
                scores=results[0].boxes.conf.cpu().tolist()
            else:
                boxes=[];scores=[]
            idx=np.argsort(scores)[::-1] if scores else []
            m25=set();m50=set();m75=set();mo=set();hd=False
            for i in idx:
                box=boxes[i];score=scores[i]
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

yw_res = {}
for name, tmpl in [('satellite','a satellite image of {}'),
                    ('overhead','an overhead image of {}'),
                    ('tdm','an overhead image of a {}')]:
    print(f'  {name}...',flush=True)
    yw_res[name] = eval_yolow(tmpl)
    m=yw_res[name]
    print(f'    F1={m["f1"]:.4f} mAP={m["mAP"]:.4f}',flush=True)

# ═══════════════════════════════════════
# 2: Qualitative Visualization
# ═══════════════════════════════════════
print("\n"+"="*60)
print("2: Qualitative Detection Comparison")
print("="*60)

# Pick 4 representative classes with distinct characteristics
viz_classes = ['airplane','bridge','stadium','storage tank']
# One image per class
viz_data = {}
for cls_name in viz_classes:
    if cls_name in test_data and test_data[cls_name]:
        viz_data[cls_name] = test_data[cls_name][:2]  # 2 images per class

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
proc = AutoProcessor.from_pretrained(tc)
model = AutoModelForZeroShotObjectDetection.from_pretrained(tc,dtype=torch.float32).eval().to(device)

COLORS = {
    'gt': (0,255,0),        # green
    'bare': (255,0,0),      # red
    'satellite': (255,165,0), # orange
    'overhead': (0,0,255),  # blue
    'tdm': (255,0,255),     # magenta
}

def draw_boxes(img, boxes, color, label, width=2):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x1,y1,x2,y2 = [int(v) for v in box]
        draw.rectangle([x1,y1,x2,y2], outline=color, width=width)
        draw.text((x1,max(0,y1-10)), label, fill=color)
    return img

all_viz = {}
for cls_name, imgs in viz_data.items():
    print(f"  {cls_name}...", flush=True)
    for idx, (img_bytes, gt_boxes) in enumerate(imgs):
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        gts = [(b[0],b[1],b[2],b[3]) for b in gt_boxes]

        # Run all methods
        methods = {
            'bare': cls_name,
            'satellite': f'a satellite image of {cls_name}',
            'overhead': f'an overhead image of {cls_name}',
            'tdm': f'an overhead image of {cls_name}, with distinct structural features, large and prominent',
        }

        dets = {}
        for mname, prompt in methods.items():
            inputs = proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o = model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
            dets[mname] = r['boxes'].cpu().tolist()

        # Save per-method visualization
        for mname in ['bare','satellite','overhead','tdm']:
            vis_img = img.copy()
            vis_img = draw_boxes(vis_img, gts, COLORS['gt'], 'GT', width=3)
            vis_img = draw_boxes(vis_img, dets[mname], COLORS[mname], mname, width=2)
            key = f'{cls_name}_{idx}_{mname}'
            out_path = rf'C:\Users\张三\Desktop\viz_{key}.png'
            vis_img.save(out_path)
            all_viz[key] = out_path

        # Also save a combined comparison image
        combined = img.copy()
        combined = draw_boxes(combined, gts, COLORS['gt'], 'GT', width=3)
        combined = draw_boxes(combined, dets.get('bare',[]), COLORS['bare'], 'bare', width=1)
        combined = draw_boxes(combined, dets.get('overhead',[]), COLORS['overhead'], 'oh', width=1)
        combined = draw_boxes(combined, dets.get('tdm',[]), COLORS['tdm'], 'TDM', width=2)
        combined.save(rf'C:\Users\张三\Desktop\viz_{cls_name}_{idx}_combined.png')

num_viz = len(all_viz)
print(f"  Saved {num_viz} visualization images to Desktop")

# ═══════════════════════════════════════
# 3: Method Comparison Table (literature)
# ═══════════════════════════════════════
print("\n"+"="*60)
print("3: Literature Method Comparison")
print("="*60)

# From published papers (check their reported numbers)
# Note: different datasets/models/settings, for reference only
lit_comparison = {
    'TDM (Ours, zero-shot)': {'F1': 0.1419, 'mAP': 0.0956, 'Setting': 'GDINO-tiny, DIOR 20cls, zero-shot', 'Source': 'This paper'},
    'CoOp [11]': {'F1': '0.790*', 'mAP': '—', 'Setting': 'CLIP ViT-B/32, 5-class test, 100ep', 'Source': 'IJCV 2022'},
    'CoCoOp [12]': {'F1': '0.798*', 'mAP': '—', 'Setting': 'CLIP ViT-B/32, 5-class test, 100ep', 'Source': 'CVPR 2022'},
    'MaPLe [13]': {'F1': '0.812*', 'mAP': '—', 'Setting': 'CLIP ViT-B/32, multi-modal', 'Source': 'CVPR 2023'},
    'PromptDet [14]': {'F1': '—', 'mAP': '—', 'Setting': 'Mask R-CNN, LVIS, needs training', 'Source': 'ECCV 2022'},
    'DetPro [15]': {'F1': '—', 'mAP': '—', 'Setting': 'Faster R-CNN, LVIS, needs training', 'Source': 'CVPR 2023'},
    'GDINO-tiny baseline': {'F1': 0.1375, 'mAP': 0.0806, 'Setting': 'GDINO-tiny, DIOR 20cls, zero-shot', 'Source': 'This paper'},
    'GDINO-base baseline': {'F1': 0.1723, 'mAP': 0.0951, 'Setting': 'GDINO-base, DIOR 20cls, zero-shot', 'Source': 'This paper'},
    'OWLv2 baseline': {'F1': 0.3353, 'mAP': 0.0770, 'Setting': 'OWLv2-base, DIOR 20cls, zero-shot', 'Source': 'This paper'},
    'YOLO-World baseline': {'F1': 0.0843, 'mAP': 0.0945, 'Setting': 'YOLO-World-v2, DIOR 20cls, zero-shot', 'Source': 'This paper'},
}

print(f"  {'Method':<25} {'F1':>10} {'mAP':>10}  Setting")
print(f"  {'-'*80}")
for method, info in lit_comparison.items():
    print(f"  {method:<25} {str(info['F1']):>10} {str(info['mAP']):>10}  {info['Setting'][:50]}")

print(f"\n  * CoOp/CoCoOp/MaPLe metrics from classification not detection — not directly comparable")
print(f"  PromptDet/DetPro require training, different architecture — listed for completeness")

# ═══════════════════════════════════════
# SAVE
# ═══════════════════════════════════════
# Update cross-model mAP with YOLO-World
with open(r'C:\Users\张三\Desktop\map_cross_model.json') as f: cross_model = json.load(f)
cross_model['yolo_world'] = yw_res
with open(r'C:\Users\张三\Desktop\map_cross_model.json','w') as f: json.dump(cross_model,f,indent=2)

all_out = {
    'yolo_world': yw_res,
    'visualization': {k: str(v) for k,v in all_viz.items()},
    'literature_comparison': lit_comparison,
}
with open(r'C:\Users\张三\Desktop\remaining_results.json','w') as f: json.dump(all_out,f,indent=2,ensure_ascii=False)

print(f"\n{'='*60}")
print("ALL REMAINING EXPERIMENTS DONE")
print(f"{'='*60}")
print(f"\nCross-Model mAP (complete):")
for model_name, res in cross_model.items():
    print(f"  {model_name}:")
    for mn, m in res.items():
        print(f"    {mn:<20} F1={m.get('f1',0):.4f}  mAP={m.get('mAP',0):.4f}")
