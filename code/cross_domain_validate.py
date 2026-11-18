# -*- coding: utf-8 -*-
"""Cross-Domain Validation Experiment.

Three-way comparison on COCO 20-class:
    A. Baseline: best COCO prefix + class name (no modifiers)
    B. Cross-Domain TDM: COCO prefix + DIOR OFAT modifiers
    C. Gold Standard: COCO prefix + COCO OFAT modifiers

Key metric: Δ_cross = F1(B) - F1(A) vs. Δ_gold = F1(C) - F1(A)
Retention rate = Δ_cross / Δ_gold → "what % of gold gain does cross-domain capture?"
"""
import json, os, glob, numpy as np, torch, time
from PIL import Image
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
device = 'cuda'

COCO_CLASSES = ['person','car','dog','chair','bottle','cup','boat','airplane',
                'bird','potted plant','dining table','couch','motorcycle','cat',
                'tv','laptop','sink','refrigerator','book','clock']

AXIS_MODIFIERS = {
    'scale': 'large and prominent',
    'spatial': 'in its natural setting',
    'granularity': 'with fine details',
    'geometric': 'with distinct structural features',
    'spectral': 'with natural coloring',
    'texture': 'with clearly visible surface features',
}
THRESHOLD = 0.03
IMAGES_PER_CLASS = 20

# ── Load data ──
print("Loading data...")
from pycocotools.coco import COCO
coco_api = COCO(r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
coco_dir = r'C:\Users\张三\modeling\coco_data\val2017'

# Load OFAT data
with open(r'C:\Users\张三\Desktop\实验数据\coco_ofat_results.json') as f:
    coco_ofat = json.load(f)

with open(r'C:\Users\张三\Desktop\论文版本历史\初稿4_论文文件\实验数据\ofat_overhead_results.json') as f:
    dior_ofat = json.load(f)['ofat_overhead']

# Compute DIOR mean deltas (for cross-domain transfer to non-overlapping classes)
dior_mean_deltas = {}
axes_list = ['granularity','spatial','geometric','scale','spectral','texture']
for ax in axes_list:
    vals = [c['deltas'].get(ax, 0) for c in dior_ofat.values()]
    dior_mean_deltas[ax] = float(np.mean(vals))

print(f"DIOR mean deltas: { {ax: round(v,4) for ax,v in dior_mean_deltas.items()} }")

# ── Collection images with GT boxes ──
cat_map = {}
for cid in coco_api.getCatIds(catNms=COCO_CLASSES):
    info = coco_api.loadCats([cid])[0]
    cat_map[info['name']] = cid

class_data = {}
for cls_name in COCO_CLASSES:
    cid = cat_map.get(cls_name)
    if cid is None: continue
    img_ids = coco_api.getImgIds(catIds=[cid])
    images = []
    for img_id in img_ids:
        if len(images) >= IMAGES_PER_CLASS: break
        img_info = coco_api.loadImgs([img_id])[0]
        ann_ids = coco_api.getAnnIds(imgIds=[img_id], catIds=[cid])
        anns = coco_api.loadAnns(ann_ids)
        if anns:
            gt_boxes = []
            for ann in anns:
                x,y,w,h = ann['bbox']
                gt_boxes.append([x, y, x+w, y+h])
            images.append({'file': img_info['file_name'], 'gt_boxes': gt_boxes})
    class_data[cls_name] = images
    print(f"  {cls_name}: {len(images)} images with GT")

# ── Load model ──
print("Loading GDINO-tiny...")
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
cache = (r'C:\Users\张三\.cache\huggingface\hub'
         r'\models--IDEA-Research--grounding-dino-tiny'
         r'\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71')
proc = AutoProcessor.from_pretrained(cache)
model = (AutoModelForZeroShotObjectDetection
         .from_pretrained(cache, dtype=torch.float32).eval().to(device))

# ── Helpers ──
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def evaluate(prompt, images_info):
    """Run detection and compute P/R/F1/DR."""
    tp = fp = fn = dr_count = 0
    n_imgs = len(images_info)

    for info in images_info:
        img = Image.open(os.path.join(coco_dir, info['file'])).convert('RGB')
        gt_boxes = info['gt_boxes']

        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k,v in inputs.items()}
        with torch.no_grad():
            o = model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(
            o, threshold=0.25, target_sizes=ts)[0]

        pred_boxes = r['boxes'].cpu().tolist()
        if pred_boxes:
            dr_count += 1

        matched = set()
        for pb in pred_boxes:
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched: continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= 0.25:
                matched.add(best_gi); tp += 1
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)

    p = tp/(tp+fp) if (tp+fp)>0 else 0
    r_val = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*p*r_val/(p+r_val) if (p+r_val)>0 else 0
    dr = dr_count/n_imgs if n_imgs>0 else 0
    return {'DR': dr, 'P': p, 'R': r_val, 'F1': f1, 'TP': tp, 'FP': fp, 'FN': fn}


# ═══════════════════════════════════════
# Three-way experiment
# ═══════════════════════════════════════
print(f"\n{'='*60}")
print("CROSS-DOMAIN VALIDATION: 3-way comparison")
print(f"{'='*60}")

results = {}
total = len(COCO_CLASSES) * 3
count = 0

for cls_name in COCO_CLASSES:
    imgs = class_data.get(cls_name, [])
    if len(imgs) < 3: continue

    coco_prefix = coco_ofat['classes'][cls_name]['best_prefix']
    t0 = time.time()

    # A. Baseline: best prefix + class name
    prompt_a = f'{coco_prefix} {cls_name}'
    mets_a = evaluate(prompt_a, imgs)
    count += 1

    # B. Cross-Domain: COCO prefix + DIOR modifiers
    dior_deltas = dior_ofat.get(cls_name, {}).get('deltas', dior_mean_deltas)
    # For non-overlapping classes, use DIOR mean deltas
    if cls_name not in dior_ofat:
        dior_deltas = dior_mean_deltas
    else:
        dior_deltas = dior_ofat[cls_name]['deltas']

    dior_mods = [AXIS_MODIFIERS[ax] for ax in axes_list
                 if dior_deltas.get(ax, 0) > THRESHOLD]
    prompt_b = f'{coco_prefix} {cls_name}'
    if dior_mods:
        prompt_b += ', ' + ', '.join(dior_mods)
    mets_b = evaluate(prompt_b, imgs)
    count += 1

    # C. Gold: COCO prefix + COCO modifiers
    coco_deltas = coco_ofat['classes'][cls_name]['deltas']
    coco_mods = [AXIS_MODIFIERS[ax] for ax in axes_list
                 if coco_deltas.get(ax, 0) > THRESHOLD]
    prompt_c = f'{coco_prefix} {cls_name}'
    if coco_mods:
        prompt_c += ', ' + ', '.join(coco_mods)
    mets_c = evaluate(prompt_c, imgs)
    count += 1

    elapsed = time.time() - t0
    results[cls_name] = {
        'A_baseline': mets_a, 'B_cross_domain': mets_b, 'C_gold': mets_c,
        'dior_mods': dior_mods, 'coco_mods': coco_mods,
        'prompts': {'baseline': prompt_a, 'cross': prompt_b, 'gold': prompt_c},
    }

    d_cross = mets_b['F1'] - mets_a['F1']
    d_gold = mets_c['F1'] - mets_a['F1']
    ret = d_cross/d_gold if d_gold > 0 else (1.0 if d_cross == d_gold else float('nan'))
    print(f"  [{count}/{total}] {cls_name:20s} "
          f"base={mets_a['F1']:.4f} cross={mets_b['F1']:.4f} gold={mets_c['F1']:.4f} "
          f"Δc={d_cross:+.4f} Δg={d_gold:+.4f} ret={ret:.2%} "
          f"({elapsed:.1f}s)")

# ═══════════════════════════════════════
# Aggregate results
# ═══════════════════════════════════════
print(f"\n{'='*60}")
print("AGGREGATE RESULTS (macro-average across 20 classes)")
print(f"{'='*60}")

def macro_mean(key):
    vals = [r[key]['F1'] for r in results.values()]
    return float(np.mean(vals))

f1_a = macro_mean('A_baseline')
f1_b = macro_mean('B_cross_domain')
f1_c = macro_mean('C_gold')
d_cross = f1_b - f1_a
d_gold = f1_c - f1_a
retention = d_cross / d_gold if d_gold > 0 else float('nan')

print(f"Baseline:          F1 = {f1_a:.4f}")
print(f"Cross-Domain TDM:  F1 = {f1_b:.4f}  (Δ = {d_cross:+.4f})")
print(f"Gold (COCO OFAT):  F1 = {f1_c:.4f}  (Δ = {d_gold:+.4f})")
print(f"Retention rate:    {retention:.1%}")

# Also compute mean DR/P/R
for metric in ['DR', 'P', 'R']:
    ba = np.mean([r['A_baseline'][metric] for r in results.values()])
    bb = np.mean([r['B_cross_domain'][metric] for r in results.values()])
    bc = np.mean([r['C_gold'][metric] for r in results.values()])
    print(f"{metric:3s}: base={ba:.4f}  cross={bb:.4f}  gold={bc:.4f}")

# Modifier agreement
same_mods = sum(1 for r in results.values()
                if set(r['dior_mods']) == set(r['coco_mods']))
print(f"\nModifier agreement (DIOR vs COCO): {same_mods}/{len(results)} classes")

# ── Save ──
output = {
    'per_class': results,
    'summary': {
        'baseline_f1': f1_a, 'cross_domain_f1': f1_b, 'gold_f1': f1_c,
        'delta_cross': d_cross, 'delta_gold': d_gold, 'retention': retention,
    },
}
out_path = r'C:\Users\张三\Desktop\实验数据\cross_domain_validation.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {out_path}")
print("Done.")
