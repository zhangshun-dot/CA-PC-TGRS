# -*- coding: utf-8 -*-
"""Dual-axis ablation FAST: 5 classes × 8 top axis pairs × DR metric. ~10 min."""
import json, torch, numpy as np, io
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq
from itertools import combinations
from datetime import datetime
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Start: {datetime.now()}", flush=True)

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

AXM = {'granularity': 'with fine details',
       'spatial': 'in its natural setting',
       'geometric': 'with distinct structural features',
       'scale': 'large and prominent',
       'spectral': 'with natural coloring',
       'texture': 'with clearly visible surface features'}

AXIS_NAMES = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']
AXIS_PAIRS = list(combinations(AXIS_NAMES, 2))  # 15 pairs

SAMPLED = ['airplane', 'airport', 'stadium', 'golf course', 'vehicle']  # 5 classes with high DR

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
        if cat < len(DIOR) and len(class_data[DIOR[cat]]) < 10:
            class_data[DIOR[cat]].append((img_bytes, boxes))

# Simple DR metric
def detect_dr(images, prompt, proc, model):
    dets = []
    for img in images:
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.no_grad(): o = model(**inputs)
        ts = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=ts)[0]
        dets.append(float(len(r['boxes']) > 0))
    return np.mean(dets)

print("Loading GDINO-tiny...", flush=True)
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(tc)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)
print("Model loaded.", flush=True)

results = {}
print(f"\n{'='*60}")
print(f"DUAL-AXIS ABLATION: {len(SAMPLED)} classes × {len(AXIS_PAIRS)} pairs (DR metric)")
print(f"{'='*60}")

for cls_name in SAMPLED:
    imgs_data = class_data[cls_name][:5]
    imgs = [Image.open(io.BytesIO(b)).convert('RGB') for b, _ in imgs_data]

    # Baseline DR (no modifiers)
    baseline_prompt = f'an overhead image of {cls_name}'
    baseline_dr = detect_dr(imgs, baseline_prompt, gd_proc, gd_model)

    # Single-axis DR
    single_dr = {}
    for axis in AXIS_NAMES:
        prompt = f'an overhead image of {cls_name}, {AXM[axis]}'
        single_dr[axis] = detect_dr(imgs, prompt, gd_proc, gd_model)

    # Dual-axis DR
    pair_results = {}
    for ax1, ax2 in AXIS_PAIRS:
        prompt = f'an overhead image of {cls_name}, {AXM[ax1]}, {AXM[ax2]}'
        actual = detect_dr(imgs, prompt, gd_proc, gd_model)
        # Correct additive prediction: baseline + delta_ax1 + delta_ax2
        # = single_ax1 + single_ax2 - baseline
        predicted = single_dr[ax1] + single_dr[ax2] - baseline_dr
        pair_results[f'{ax1}+{ax2}'] = {
            'actual': float(actual),
            'baseline': float(baseline_dr),
            'predicted_additive': float(predicted),
            'interaction': float(actual - predicted),
        }

    results[cls_name] = {'baseline': float(baseline_dr), 'single': single_dr, 'pairs': pair_results}
    # Show top interactions
    inters = sorted([(k, v['interaction']) for k, v in pair_results.items()], key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"  {cls_name:<20} Top5: " + ' | '.join([f'{k}:{v:+.3f}' for k, v in inters]))

# Summary
all_inter = []
for cls_name in SAMPLED:
    for pair_name, pair_data in results[cls_name]['pairs'].items():
        all_inter.append(pair_data['interaction'])

mean_inter = float(np.mean(all_inter))
std_inter = float(np.std(all_inter))
mae = float(np.mean(np.abs(all_inter)))

print(f"\n{'='*60}")
print(f"SUMMARY: Dual-Axis Joint Ablation (DR metric)")
print(f"{'='*60}")
print(f"  Mean interaction: {mean_inter:+.4f} ± {std_inter:.4f}")
print(f"  MAE (|interaction|): {mae:.4f}")
n_neg = sum(1 for v in all_inter if v < 0)
n_pos = sum(1 for v in all_inter if v > 0)
print(f"  Negative/Positive: {n_neg}/{n_pos}")
print(f"  Range: [{min(all_inter):+.4f}, {max(all_inter):+.4f}]")

conclusion = (
    f"交互项MAE={mae:.4f}，" +
    ("主效应可加性近似成立" if mae < 0.10 else
     "存在可检测交互但主效应捕获主要趋势" if mae < 0.20 else
     "交互项不可忽略，需全因子设计")
)

out = {
    'per_class': results,
    'summary': {
        'n_classes': len(SAMPLED), 'n_pairs': len(AXIS_PAIRS),
        'mean_interaction': mean_inter, 'std_interaction': std_inter,
        'mae_interaction': mae, 'n_neg': n_neg, 'n_pos': n_pos,
        'conclusion': conclusion,
    }
}

with open(r'C:\Users\张三\Desktop\dual_axis_ablation_results.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"\n结论: {conclusion}")
print(f"Done: {datetime.now()}")
torch.cuda.empty_cache()
