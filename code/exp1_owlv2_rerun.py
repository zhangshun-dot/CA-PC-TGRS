# -*- coding: utf-8 -*-
"""Exp 1 (v3): OWLv2 modifier ablation — token-aware truncation at 16-token limit."""
import json, torch, numpy as np, io, os, sys, time
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

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
n_cls = sum(1 for v in class_data.values() if v)
n_img = sum(len(v) for v in class_data.values())
print(f"Data: {n_cls} classes, {n_img} images", flush=True)

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def metrics(preds_list, gts_list):
    tp=fp=fn=0; dr=0; n=len(preds_list)
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
    return {'dr':dr/n,'p':p,'r':r,'f1':2*p*r/(p+r)if(p+r)else 0,'tp':tp,'fp':fp,'fn':fn}

print("Loading OWLv2...", flush=True)
from transformers import Owlv2Processor, Owlv2ForObjectDetection
oc = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots'
osnap = os.listdir(oc)[0]; opath = os.path.join(oc, osnap)
op = Owlv2Processor.from_pretrained(opath)
om = Owlv2ForObjectDetection.from_pretrained(opath, dtype=torch.float32).eval().to(device)
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

ALL_MODS = ['with distinct structural features','large and prominent',
            'with natural coloring','with fine details','with clearly visible surface features']
MAX_TOKENS = 16  # OWLv2 hard limit for position embeddings

def build_prompt_owl(cls_name, n_mods):
    """Build prompt respecting OWLv2's 16-token limit."""
    prompt = f'an overhead image of {cls_name}'
    if n_mods == 0:
        return prompt
    # Try adding modifiers one by one, checking token count
    result = prompt
    for mod in ALL_MODS[:n_mods]:
        candidate = result + ', ' + mod
        try:
            inputs = op(text=candidate, return_tensors='pt')
            n_tok = inputs['input_ids'].shape[1]
        except:
            n_tok = MAX_TOKENS + 1
        if n_tok <= MAX_TOKENS:
            result = candidate
        else:
            break  # Stop adding modifiers when we hit the limit
    return result

# Show token-level truncation
print("\nPrompt token analysis (16-token limit):", flush=True)
for cls_name in ['dam', 'airplane', 'expressway service area']:
    for n in range(4):
        p = build_prompt_owl(cls_name, n)
        try:
            tok = op(text=p, return_tensors='pt')['input_ids'].shape[1]
        except:
            tok = 'ERR'
        print(f"  {cls_name} +{n}mods -> {tok} tokens: '{p[:80]}...' " if len(p)>80 else f"  {cls_name} +{n}mods -> {tok} tokens: '{p}'", flush=True)

results = {}
print(f"\n{'='*50}", flush=True)
print("OWLv2 Modifier Ablation (token-aware, max 16 tokens)", flush=True)

for n_mods in [0, 1, 2, 3, 4, 5]:
    all_p=[]; all_g=[]
    t0 = time.time()
    processed = 0
    n_truncated = 0

    for cls_name, img_list in class_data.items():
        if not img_list: continue
        prompt = build_prompt_owl(cls_name, n_mods)
        # Check how many modifiers actually fit
        expected_mod_count = min(n_mods, len(ALL_MODS))
        actual_comma_count = prompt.count(',')
        if actual_comma_count < expected_mod_count:
            n_truncated += 1

        for img_bytes, gt_boxes in img_list:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs = op(text=prompt, images=img, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o = om(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = op.post_process_object_detection(o, threshold=0.1, target_sizes=target)[0]
            all_p.append(r['boxes'].cpu().tolist())
            all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
            processed += 1
            if processed % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{n_mods}mods] {processed}/{n_img} images, {elapsed:.0f}s", flush=True)

    m = metrics(all_p, all_g)
    results[f'owl_abl_{n_mods}'] = m
    elapsed = time.time() - t0
    trunc_msg = f" ({n_truncated}/20 classes token-truncated)" if n_truncated > 0 else ""
    print(f"  => {n_mods}mods: DR={m['dr']:.4f} P={m['p']:.4f} R={m['r']:.4f} F1={m['f1']:.4f} (TP={m['tp']} FP={m['fp']} FN={m['fn']}) [{elapsed:.0f}s]{trunc_msg}", flush=True)

with open(r'C:\Users\张三\Desktop\owlv2_rerun_results.json','w') as f:
    json.dump(results, f, indent=2)

# Comparison
print(f"\n{'='*50}", flush=True)
print("BEFORE vs AFTER:", flush=True)
print(f"  {'Count':<10} {'Old (buggy)':<15} {'New (token-aware)':<20}", flush=True)
old_f1 = {0:0.325766, 1:0.257576, 2:0.325766, 3:0.325766, 4:0.325766, 5:0.325766}
for n in range(6):
    of1 = old_f1[n]
    nf1 = results[f'owl_abl_{n}']['f1']
    print(f"  {n:<10} {of1:<15.4f} {nf1:<20.4f}", flush=True)

print("\nDone.", flush=True)
