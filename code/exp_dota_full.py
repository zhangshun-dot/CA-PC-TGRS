# -*- coding: utf-8 -*-
"""Exp A: DOTA v1.5 full metrics — DR/P/R/F1 per class for satellite/overhead/TDM"""
import json, torch, numpy as np, os, sys
from PIL import Image
from collections import defaultdict

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")

# DOTA → DIOR class mapping (11 overlap classes)
DOTA_MAP = {
    'plane': 'airplane', 'ship': 'ship', 'storage-tank': 'storage tank',
    'baseball-diamond': 'baseball field', 'tennis-court': 'tennis court',
    'basketball-court': 'basketball court', 'ground-track-field': 'ground track field',
    'harbor': 'harbor', 'bridge': 'bridge',
    'large-vehicle': 'vehicle', 'small-vehicle': 'vehicle',
}

IMG_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\images'
LBL_DIR = r'C:\Users\张三\modeling\dota_data\DOTA_V1.5\train\labelTxt-v1.5\DOTA-v1.5_train'

# Collect images per class
class_images = defaultdict(list)
for fn in os.listdir(LBL_DIR):
    if not fn.endswith('.txt'): continue
    img_path = os.path.join(IMG_DIR, fn.replace('.txt', '.png'))
    if not os.path.exists(img_path): continue
    with open(os.path.join(LBL_DIR, fn)) as f:
        lines = f.readlines()
    bboxes_by_cls = defaultdict(list)
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9: continue
        try:
            x1,y1,x2,y2,x3,y3,x4,y4 = map(float, parts[:8])
            cls_name = parts[8]
            if cls_name in DOTA_MAP:
                bboxes_by_cls[cls_name].append([x1,y1,x2,y2,x3,y3,x4,y4])
        except: continue
    for cls_name, bboxes in bboxes_by_cls.items():
        if len(class_images[cls_name]) < 20:
            class_images[cls_name].append((img_path, bboxes))

overlap = sorted(class_images.keys())
print(f"Overlap classes: {len(overlap)}: {overlap}")
for c in overlap:
    print(f"  {c}→{DOTA_MAP[c]}: {len(class_images[c])} images")

# ── Helpers ──
def quad_to_bbox(quad):
    xs = [quad[0],quad[2],quad[4],quad[6]]
    ys = [quad[1],quad[3],quad[5],quad[7]]
    return [min(xs), min(ys), max(xs), max(ys)]

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) > 0 else 0

def compute_metrics(all_preds, all_gts):
    tp=fp=fn=0; n=len(all_gts)
    any_detected = 0
    for pred, gt in zip(all_preds, all_gts):
        if len(pred) > 0: any_detected += 1
        matched = set()
        for pb in pred:
            best=0; bg=-1
            for gi, gb in enumerate(gt):
                if gi in matched: continue
                v = compute_iou(pb, gb)
                if v > best: best=v; bg=gi
            if best >= 0.25: matched.add(bg); tp += 1
            else: fp += 1
        fn += len(gt) - len(matched)
    total_gt = sum(len(g) for g in all_gts)
    return {
        'dr': any_detected/n if n else 0,
        'p': tp/(tp+fp) if (tp+fp) else 0,
        'r': tp/total_gt if total_gt else 0,
        'f1': 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0,
        'tp': tp, 'fp': fp, 'fn': fn,
    }

# ── Load model ──
print("\nLoading GDINO-tiny...")
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
proc = AutoProcessor.from_pretrained(tc)
model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

# ── Prompt builders ──
FIXED_MODS = 'with distinct structural features, large and prominent'

def prompt_satellite(cls): return f'a satellite image of {cls}'
def prompt_overhead(cls): return f'an overhead image of {cls}'
def prompt_tdm(cls):
    # Use overhead as base (best DIOR prefix) + fixed modifiers
    return f'an overhead image of {cls}, {FIXED_MODS}'

# ── Run ──
all_results = {}
for name, prompt_fn in [('satellite', prompt_satellite),
                         ('overhead', prompt_overhead),
                         ('tdm', prompt_tdm)]:
    print(f"\n{'='*50}\nDOTA GDINO-tiny: {name}\n{'='*50}")
    per_class = {}
    all_p, all_g = [], []
    for cls_name in overlap:
        dior_name = DOTA_MAP[cls_name]
        imgs = class_images[cls_name]
        preds_list, gts_list = [], []
        for img_path, quads in imgs:
            img = Image.open(img_path).convert('RGB')
            gts = [quad_to_bbox(q) for q in quads]
            prompt = prompt_fn(dior_name)
            inputs = proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad():
                o = model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
            preds_list.append(r['boxes'].cpu().tolist())
            gts_list.append(gts)
        m = compute_metrics(preds_list, gts_list)
        m['class'] = dior_name
        per_class[dior_name] = m
        all_p.extend(preds_list); all_g.extend(gts_list)
        print(f"  {dior_name}: DR={m['dr']:.3f} P={m['p']:.3f} R={m['r']:.3f} F1={m['f1']:.3f}")
    overall = compute_metrics(all_p, all_g)
    mean_dr = np.mean([m['dr'] for m in per_class.values()])
    all_results[name] = {'per_class': per_class, 'overall': overall, 'mean_dr': float(mean_dr)}
    print(f"  OVERALL: DR={mean_dr:.4f} P={overall['p']:.4f} R={overall['r']:.4f} F1={overall['f1']:.4f}")

# ── Save ──
out = r'C:\Users\张三\Desktop\dota_full_results.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# Summary
print(f"\n{'='*60}")
print("DOTA TABLE SUMMARY:")
print(f"  {'Method':<15} {'DR':>8} {'P':>8} {'R':>8} {'F1':>8}")
for name in ['satellite', 'overhead', 'tdm']:
    o = all_results[name]['overall']
    print(f"  {name:<15} {all_results[name]['mean_dr']:>8.4f} {o['p']:>8.4f} {o['r']:>8.4f} {o['f1']:>8.4f}")

sat_f1 = all_results['satellite']['overall']['f1']
oh_f1 = all_results['overhead']['overall']['f1']
tdm_f1 = all_results['tdm']['overall']['f1']
print(f"\n  oh/sat F1 ratio: {oh_f1/max(sat_f1,0.0001):.2f}x")
print(f"  Table row: sat={sat_f1:.4f} oh={oh_f1:.4f} TDM={tdm_f1:.4f} oh/sat={oh_f1/max(sat_f1,0.0001):.2f}x")
print(f"\nSaved: {out}")
