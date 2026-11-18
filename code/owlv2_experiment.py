# -*- coding: utf-8 -*-
"""OWLv2 on DIOR: satellite vs overhead comparison."""
import json, torch, numpy as np, os, io
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

from transformers import Owlv2Processor, Owlv2ForObjectDetection

print("Loading OWLv2...")
cache = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots'
snap = os.listdir(cache)[0]
model_path = os.path.join(cache, snap)
proc = Owlv2Processor.from_pretrained(model_path)
owl_model = Owlv2ForObjectDetection.from_pretrained(model_path, dtype=torch.float32).eval().to(device)
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# Load data
pq_file = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table = pq.read_table(pq_file); rows = table.to_pylist()
class_data = defaultdict(list)
for row in rows:
    img_bytes, objs = row['image']['bytes'], row['objects']
    if not objs or 'category' not in objs: continue
    # Group GT boxes by category
    gt_by_cat = defaultdict(list)
    for bbox, cat in zip(objs['bbox'], objs['category']):
        x,y,w,h = bbox; gt_by_cat[cat].append([x,y,x+w,y+h])
    for cat, boxes in gt_by_cat.items():
        if len(class_data[DIOR[cat]]) < 20:
            class_data[DIOR[cat]].append((img_bytes, boxes))

print(f"Classes: {sum(1 for v in class_data.values() if v)}")

def compute_iou(b1, b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def run_owlv2(prompt_fn):
    per_class = {}
    all_tp=all_fp=all_fn=0; any_det=0; n_imgs=0
    for cls_name, img_list in class_data.items():
        if not img_list: continue
        tp=fp=fn=0; dr_count=0
        for img_bytes, gt_boxes in img_list:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            prompt = prompt_fn(cls_name)
            inputs = proc(text=prompt, images=img, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad():
                o = owl_model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = proc.post_process_object_detection(o, threshold=0.1, target_sizes=target)[0]
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
        n_cls = len(img_list)
        per_class[cls_name] = {
            'dr': round(dr_count/n_cls,4) if n_cls else 0,
            'precision': round(tp/(tp+fp),4) if (tp+fp) else 0,
            'recall': round(tp/(tp+fn),4) if (tp+fn) else 0,
            'f1': round(2*tp/(2*tp+fp+fn),4) if (2*tp+fp+fn) else 0,
        }
        all_tp+=tp;all_fp+=fp;all_fn+=fn
    mean_dr = np.mean([m['dr'] for m in per_class.values()])
    overall = {
        'precision': round(all_tp/(all_tp+all_fp),4) if (all_tp+all_fp) else 0,
        'recall': round(all_tp/(all_tp+all_fn),4) if (all_tp+all_fn) else 0,
        'f1': round(2*all_tp/(2*all_tp+all_fp+all_fn),4) if (2*all_tp+all_fp+all_fn) else 0,
    }
    return {'per_class': per_class, 'overall': overall, 'mean_dr': round(float(mean_dr),4)}

# Run
results = {}
for name, fn in [('satellite', lambda c: f'a satellite image of {c}'),
                  ('overhead', lambda c: f'an overhead image of {c}')]:
    print(f"\nOWLv2 {name}...")
    r = run_owlv2(fn)
    results[f'owlv2_{name}'] = r
    o = r['overall']
    print(f"  DR={r['mean_dr']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}")

with open(r'C:\Users\张三\Desktop\owlv2_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nDone.")
