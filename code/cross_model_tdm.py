# -*- coding: utf-8 -*-
"""Cross-model TDM: apply GDINO's OFAT to YOLO-World and OWLv2."""
import json, torch, numpy as np, io, os
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# Load OFAT data
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)['ofat_overhead']
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm(cls):
    if cls not in ofat: return f'an overhead image of {cls}'
    d = ofat[cls]['deltas']
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'an overhead image of {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

# Load data
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
        if len(class_data[DIOR[cat]]) < 20:
            class_data[DIOR[cat]].append((img_bytes, boxes))

def compute_iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

# ═══ YOLO-World TDM ═══
print("\n" + "="*60)
print("YOLO-World TDM")
from ultralytics import YOLOWorld
yw = YOLOWorld('yolov8s-worldv2.pt'); yw.to(device)

tp=fp=fn=0; dr_count=0; n_imgs=0
for cls_name, img_list in class_data.items():
    if not img_list: continue
    tdm_prompt = build_tdm(cls_name)
    yw.set_classes([tdm_prompt])
    for img_bytes, gt_boxes in img_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        results = yw.predict(img, verbose=False, device=device, conf=0.25, iou=0.25)
        pred_boxes = results[0].boxes.xyxy.cpu().tolist() if results[0].boxes is not None else []
        gt = [(b[0],b[1],b[2],b[3]) for b in gt_boxes]
        n_imgs += 1
        if pred_boxes: dr_count += 1
        matched = set()
        for pb in pred_boxes:
            best_iou=0;best_gi=-1
            for gi,gb in enumerate(gt):
                if gi in matched: continue
                iou=compute_iou(pb,gb)
                if iou>best_iou: best_iou=iou;best_gi=gi
            if best_iou>=0.25: matched.add(best_gi);tp+=1
            else: fp+=1
        fn+=len(gt)-len(matched)

yw_tdm_dr = dr_count/n_imgs if n_imgs else 0
yw_tdm_p = tp/(tp+fp) if (tp+fp) else 0
yw_tdm_r = tp/(tp+fn) if (tp+fn) else 0
yw_tdm_f1 = 2*yw_tdm_p*yw_tdm_r/(yw_tdm_p+yw_tdm_r) if (yw_tdm_p+yw_tdm_r) else 0
print(f"  DR={yw_tdm_dr:.4f} P={yw_tdm_p:.4f} R={yw_tdm_r:.4f} F1={yw_tdm_f1:.4f}")

# ═══ OWLv2 TDM ═══
print("\n" + "="*60)
print("OWLv2 TDM")
from transformers import Owlv2Processor, Owlv2ForObjectDetection
cache = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots'
snap = os.listdir(cache)[0]
owl_proc = Owlv2Processor.from_pretrained(os.path.join(cache, snap))
owl_model = Owlv2ForObjectDetection.from_pretrained(os.path.join(cache, snap), dtype=torch.float32).eval().to(device)

tp=fp=fn=0; dr_count=0; n_imgs=0
for cls_name, img_list in class_data.items():
    if not img_list: continue
    tdm_prompt = build_tdm(cls_name)
    # Truncate long prompts for OWLv2 (CLIP 77-token limit)
    if len(tdm_prompt) > 60:
        tdm_prompt = f'an overhead image of {cls_name}'
    for img_bytes, gt_boxes in img_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        try:
            inputs = owl_proc(text=tdm_prompt, images=img, return_tensors='pt')
        except:
            tdm_prompt = f'an overhead image of {cls_name}'
            inputs = owl_proc(text=tdm_prompt, images=img, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad():
            o = owl_model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = owl_proc.post_process_object_detection(o, threshold=0.1, target_sizes=target)[0]
        pred_boxes = r['boxes'].cpu().tolist()
        gt = [(b[0],b[1],b[2],b[3]) for b in gt_boxes]
        n_imgs += 1
        if pred_boxes: dr_count += 1
        matched = set()
        for pb in pred_boxes:
            best_iou=0;best_gi=-1
            for gi,gb in enumerate(gt):
                if gi in matched: continue
                iou=compute_iou(pb,gb)
                if iou>best_iou: best_iou=iou;best_gi=gi
            if best_iou>=0.25: matched.add(best_gi);tp+=1
            else: fp+=1
        fn+=len(gt)-len(matched)

owl_tdm_dr = dr_count/n_imgs if n_imgs else 0
owl_tdm_p = tp/(tp+fp) if (tp+fp) else 0
owl_tdm_r = tp/(tp+fn) if (tp+fn) else 0
owl_tdm_f1 = 2*owl_tdm_p*owl_tdm_r/(owl_tdm_p+owl_tdm_r) if (owl_tdm_p+owl_tdm_r) else 0
print(f"  DR={owl_tdm_dr:.4f} P={owl_tdm_p:.4f} R={owl_tdm_r:.4f} F1={owl_tdm_f1:.4f}")

# ═══ Summary ═══
print("\n" + "="*60)
print("CROSS-MODEL TDM SUMMARY (GDINO OFAT transferred)")
print("="*60)
print(f"{'Model':<20} {'Prompt':<12} {'DR':>8} {'P':>8} {'R':>8} {'F1':>8}")
print(f"{'YOLO-World':<20} {'satellite':<12} {0.1450:>8.4f} {0.5581:>8.4f} {0.0276:>8.4f} {0.0527:>8.4f}")
print(f"{'YOLO-World':<20} {'overhead':<12} {0.2800:>8.4f} {0.4938:>8.4f} {0.0461:>8.4f} {0.0843:>8.4f}")
print(f"{'YOLO-World':<20} {'TDM(x)':<12} {yw_tdm_dr:>8.4f} {yw_tdm_p:>8.4f} {yw_tdm_r:>8.4f} {yw_tdm_f1:>8.4f}")
print(f"{'OWLv2':<20} {'satellite':<12} {0.8900:>8.4f} {0.2381:>8.4f} {0.5665:>8.4f} {0.3353:>8.4f}")
print(f"{'OWLv2':<20} {'overhead':<12} {0.8475:>8.4f} {0.2426:>8.4f} {0.4957:>8.4f} {0.3258:>8.4f}")
print(f"{'OWLv2':<20} {'TDM(x)':<12} {owl_tdm_dr:>8.4f} {owl_tdm_p:>8.4f} {owl_tdm_r:>8.4f} {owl_tdm_f1:>8.4f}")

results = {
    'yolow_tdm': {'dr':yw_tdm_dr,'p':yw_tdm_p,'r':yw_tdm_r,'f1':yw_tdm_f1},
    'owlv2_tdm': {'dr':owl_tdm_dr,'p':owl_tdm_p,'r':owl_tdm_r,'f1':owl_tdm_f1},
}
with open(r'C:\Users\张三\Desktop\cross_tdm_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nDone.")
