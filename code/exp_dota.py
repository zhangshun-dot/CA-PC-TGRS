# -*- coding: utf-8 -*-
"""DOTA v1.5 cross-dataset prefix ablation on GDINO-tiny."""
import json, torch, numpy as np, os, time
from PIL import Image
from collections import defaultdict

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# DOTA v1.5 → DIOR class mapping
DOTA_CLASSES = ['plane','ship','storage-tank','baseball-diamond','tennis-court',
    'basketball-court','ground-track-field','harbor','bridge',
    'large-vehicle','small-vehicle','helicopter','roundabout','soccer-ball-field',
    'swimming-pool','container-crane','helipad']

DIOR_MAP = {
    'plane': 'airplane', 'ship': 'ship', 'storage-tank': 'storage tank',
    'baseball-diamond': 'baseball field', 'tennis-court': 'tennis court',
    'basketball-court': 'basketball court', 'ground-track-field': 'ground track field',
    'harbor': 'harbor', 'bridge': 'bridge',
    'large-vehicle': 'vehicle', 'small-vehicle': 'vehicle',
}

# Parse DOTA labels
IMG_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
LBL_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'

# Collect images per class (max 10 per class)
class_images = defaultdict(list)
for fn in os.listdir(LBL_DIR):
    if not fn.endswith('.txt'): continue
    img_name = fn.replace('.txt', '.png')
    img_path = os.path.join(IMG_DIR, img_name)
    if not os.path.exists(img_path): continue

    with open(os.path.join(LBL_DIR, fn)) as f:
        lines = f.readlines()
    bboxes_by_cls = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9: continue  # skip header lines
        try:
            x1,y1,x2,y2,x3,y3,x4,y4 = map(float, parts[:8])
            cls_name = parts[8]
            if cls_name in DIOR_MAP:
                bboxes_by_cls[cls_name].append([x1,y1,x2,y2,x3,y3,x4,y4])
        except: continue

    for cls_name, bboxes in bboxes_by_cls.items():
        if len(class_images[cls_name]) < 10:
            class_images[cls_name].append((img_path, bboxes))

overlap = sorted(class_images.keys())
print(f"Overlap classes: {len(overlap)}: {overlap}", flush=True)
for c in overlap:
    print(f"  {c}→{DIOR_MAP[c]}: {len(class_images[c])} images", flush=True)

# Helpers
def compute_iou(b1, b2):
    # b1: [x1,y1,x2,y2] (axis-aligned from quad min/max)
    # b2: same
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0

def quad_to_bbox(quad):
    xs = [quad[0],quad[2],quad[4],quad[6]]
    ys = [quad[1],quad[3],quad[5],quad[7]]
    return [min(xs), min(ys), max(xs), max(ys)]

def metrics(preds_list, gts_list):
    tp=fp=fn=0; dr=0; n=len(preds_list)
    for ps,gs in zip(preds_list,gts_list):
        if ps: dr += 1
        matched = set()
        for pb in ps:
            best=0; bg=-1
            for gi,gb in enumerate(gs):
                if gi in matched: continue
                v = compute_iou(pb, gb)
                if v > best: best=v; bg=gi
            if best >= 0.25: matched.add(bg); tp += 1
            else: fp += 1
        fn += len(gs) - len(matched)
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    return {'dr': dr/n, 'p': p, 'r': r, 'f1': 2*p*r/(p+r) if (p+r) else 0}

print("\nLoading GDINO-tiny...", flush=True)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(tc)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

PREFIXES = {
    'bare': 'a {}',
    'satellite': 'a satellite image of a {}',
    'aerial': 'an aerial view of a {}',
    'overhead': 'an overhead image of a {}',
    'remote_sensing': 'a remote sensing image of a {}',
}

results = {}
for cls_name in overlap:
    dior_name = DIOR_MAP[cls_name]
    img_list = class_images[cls_name]
    print(f"\n{cls_name}→{dior_name} ({len(img_list)} imgs)", flush=True)

    for pname, tmpl in PREFIXES.items():
        prompt = tmpl.format(dior_name)
        all_p, all_g = [], []
        for img_path, quads in img_list:
            img = Image.open(img_path).convert('RGB')
            # Convert quads to axis-aligned bboxes for evaluation
            gt_bboxes = [quad_to_bbox(q) for q in quads]

            inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad(): o = gd_model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = gd_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
            all_p.append(r['boxes'].cpu().tolist())
            all_g.append(gt_bboxes)

        m = metrics(all_p, all_g)
        results[f'{cls_name}_{pname}'] = m['f1']
        print(f"  {pname}: F1={m['f1']:.4f}", flush=True)

# Average per prefix
print(f"\n{'='*50}")
dota_avg = {}
for pname in PREFIXES:
    vals = [results[f'{c}_{pname}'] for c in overlap]
    dota_avg[pname] = float(np.mean(vals))
    print(f"  {pname}: avg F1={dota_avg[pname]:.4f}", flush=True)

# DIOR reference
dior_avg = {'bare': 0.0529, 'satellite': 0.0713, 'aerial': 0.0727, 'overhead': 0.1375, 'remote_sensing': 0.1401}

# Cross-dataset correlation
keys = ['bare','satellite','aerial','overhead','remote_sensing']
dota_vals = [dota_avg[k] for k in keys]
dior_vals = [dior_avg[k] for k in keys]
corr = np.corrcoef(dota_vals, dior_vals)[0,1]

dota_rank = sorted(PREFIXES.keys(), key=lambda p: dota_avg[p], reverse=True)
dior_rank = sorted(PREFIXES.keys(), key=lambda p: dior_avg[p], reverse=True)

print(f"\nDIOR top-2: {dior_rank[:2]}")
print(f"DOTA top-2: {dota_rank[:2]}")
print(f"Cross-dataset Pearson r: {corr:.4f}")

out = {'dota': dota_avg, 'dior': dior_avg, 'correlation': float(corr),
       'dota_rank': dota_rank, 'dior_rank': dior_rank, 'classes': overlap}
with open(r'C:\Users\张三\Desktop\dota_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: dota_results.json")
