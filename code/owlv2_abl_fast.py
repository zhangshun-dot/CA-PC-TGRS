# -*- coding: utf-8 -*-
"""OWLv2 modifier count ablation — FAST PILOT (5 classes × 10 images).
Fixes the 60-char bug: uses proper token-level truncation to 16 tokens.
"""
import json, torch, numpy as np, io, os
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq
from datetime import datetime

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Start: {datetime.now()}", flush=True)

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# 5 representative classes (same as dual-axis ablation) + 1 long-name class for truncation test
SAMPLED = ['airplane', 'airport', 'stadium', 'golf course', 'vehicle', 'expressway service area']

ALL_MODS = ['with distinct structural features','large and prominent',
            'with natural coloring','with fine details','with clearly visible surface features']

# Load test data
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
        if DIOR[cat] in SAMPLED and len(class_data[DIOR[cat]]) < 10:
            class_data[DIOR[cat]].append((img_bytes, boxes))

print(f"Classes: {sorted(class_data.keys())}", flush=True)
for cls_name in sorted(class_data.keys()):
    print(f"  {cls_name}: {len(class_data[cls_name])} images", flush=True)

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
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

# Load OWLv2
print("Loading OWLv2...", flush=True)
from transformers import Owlv2Processor, Owlv2ForObjectDetection
oc = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots'
osnap = os.listdir(oc)[0]; opath = os.path.join(oc, osnap)
op = Owlv2Processor.from_pretrained(opath)
om = Owlv2ForObjectDetection.from_pretrained(opath, dtype=torch.float32).eval().to(device)
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

max_pos = om.config.text_config.max_position_embeddings
print(f"Max position embeddings: {max_pos}", flush=True)

results = {}

print(f"\n{'='*60}")
print(f"OWLv2 MODIFIER ABLATION ({len(SAMPLED)} classes × 10 imgs, token-level trunc)")
print(f"{'='*60}")

for n_mods in range(6):
    mods = ALL_MODS[:n_mods]
    all_p = []; all_g = []
    trunc_info = []

    for cls_name in SAMPLED:
        img_list = class_data.get(cls_name, [])
        if not img_list: continue

        # Build prompt
        prompt = f'an overhead image of {cls_name}'
        if mods:
            prompt += ', ' + ', '.join(mods)

        # Tokenize and truncate to max_pos
        tokens = op.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_pos)
        prompt_used = op.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
        n_tokens = len(tokens['input_ids'][0])

        if prompt_used != prompt:
            trunc_info.append(f"{cls_name}: {n_tokens}tk '{prompt_used[:80]}...'")

        for img_bytes, gt_boxes in img_list:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs = op(text=prompt_used, images=img, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad():
                o = om(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = op.post_process_object_detection(o, threshold=0.1, target_sizes=target)[0]
            all_p.append(r['boxes'].cpu().tolist())
            all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])

    m = metrics(all_p, all_g)
    results[f'owl_abl_{n_mods}'] = m
    print(f"  {n_mods}mods: F1={m['f1']:.4f}  P={m['p']:.4f}  R={m['r']:.4f}  DR={m['dr']:.4f}")
    if trunc_info:
        for ti in trunc_info[:3]:
            print(f"    TRUNC: {ti}")

print(f"\n{'='*60}")
print("SUMMARY:")
for k in sorted(results.keys()):
    m = results[k]
    print(f"  {k}: F1={m['f1']:.4f}  DR={m['dr']:.4f}")

# Save pilot results
out = {
    'pilot_owlv2_abl_fixed': results,
    'note': '5 classes × 10 imgs, token-level truncation to 16 tokens',
    'classes_used': SAMPLED,
}
with open(r'C:\Users\张三\Desktop\owlv2_abl_pilot.json', 'w') as f:
    json.dump(out, f, indent=2)

print(f"\nDone: {datetime.now()}")
torch.cuda.empty_cache()
