# -*- coding: utf-8 -*-
"""Dual-axis joint ablation — 验证OFAT主效应模型的可加性假设。

对6个修饰轴（grammar以外的6轴），选取C(6,2)=15个轴对，
每对同时激活两个轴 → 测量实际F1 → 与OFAT预测（δ_i + δ_j）比较。
若实测F1 ≈ OFAT预测F1，则主效应可加性假设成立；
若系统偏离，则交互项不可忽略。
"""
import json, torch, numpy as np, io
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq
from datetime import datetime
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Start: {datetime.now()}", flush=True)

OUT = r'C:\Users\张三\Desktop\dual_axis_ablation_results.json'

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# ── OFAT modifiers ──
AXM = {'granularity': 'with fine details',
       'spatial': 'in its natural setting',
       'geometric': 'with distinct structural features',
       'scale': 'large and prominent',
       'spectral': 'with natural coloring',
       'texture': 'with clearly visible surface features'}

AXIS_NAMES = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']

# All C(6,2) = 15 pairs
from itertools import combinations
AXIS_PAIRS = list(combinations(AXIS_NAMES, 2))

# ── Sample classes (10 representative classes to keep runtime reasonable) ──
SAMPLED_CLASSES = ['airplane', 'airport', 'bridge', 'vehicle', 'stadium',
                   'storage tank', 'baseball field', 'ship', 'dam', 'golf course']

# ── Load data ──
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
        if cat < len(DIOR) and len(class_data[DIOR[cat]]) < 20:
            class_data[DIOR[cat]].append((img_bytes, boxes))

np.random.seed(42)
test_data = {}
for cls_name in class_data:
    imgs = class_data[cls_name]
    idx = np.random.permutation(len(imgs))
    n = min(10, len(imgs)//2)
    test_data[cls_name] = [imgs[i] for i in idx[n:2*n]]

# ── Helpers ──
def iou(b1, b2):
    x1,y1,x2,y2 = max(b1[0],b2[0]), max(b1[1],b2[1]), min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1,a2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def metrics(preds_list, gts_list):
    tp=fp=fn=0; dr=0; n=len(preds_list)
    for ps,gs in zip(preds_list, gts_list):
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
    return {'dr':dr/n,'precision':p,'recall':r,'f1':2*p*r/(p+r)if(p+r)else 0}

# ── Load GDINO-tiny ──
print("Loading GDINO-tiny...", flush=True)
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(tc)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

# ── Load OFAT baseline measurements ──
with open(r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\ofat_overhead_results.json') as f:
    ofat_data = json.load(f)['ofat_overhead']

# Get baseline F1 (bare = overhead prefix, no modifiers) per sampled class
baseline_f1 = {}
for cls_name in SAMPLED_CLASSES:
    if cls_name in ofat_data:
        baseline_f1[cls_name] = ofat_data[cls_name].get('baseline_f1', None)

# ── Run dual-axis ablation ──
results = {'per_class': {}, 'summary': {}}

def run_prompt(prompt, img_list):
    """Run GDINO with a given prompt and compute metrics."""
    all_p=[];all_g=[]
    for img_bytes, gt_boxes in img_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad(): o = gd_model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = gd_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
        all_p.append(r['boxes'].cpu().tolist())
        all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    return metrics(all_p, all_g)

print(f"\n{'='*60}")
print(f"DUAL-AXIS JOINT ABLATION: {len(SAMPLED_CLASSES)} classes × {len(AXIS_PAIRS)} axis pairs")
print(f"{'='*60}\n")

total_experiments = 0
for cls_name in SAMPLED_CLASSES:
    if cls_name not in test_data:
        print(f"  {cls_name}: SKIP (no test data)")
        continue

    cls_results = {}
    imgs = test_data[cls_name]

    # Single-axis baselines
    single_axis_f1 = {}
    for axis in AXIS_NAMES:
        prompt = f'an overhead image of {cls_name}, {AXM[axis]}'
        m = run_prompt(prompt, imgs)
        single_axis_f1[axis] = m['f1']
        total_experiments += 1

    # Dual-axis pairs
    pair_results = {}
    for ax1, ax2 in AXIS_PAIRS:
        prompt = f'an overhead image of {cls_name}, {AXM[ax1]}, {AXM[ax2]}'
        m = run_prompt(prompt, imgs)
        ofat_pred = single_axis_f1[ax1] + single_axis_f1[ax2]  # Additive prediction (simplified)
        pair_results[f'{ax1}+{ax2}'] = {
            'actual_f1': m['f1'],
            'single_sum': ofat_pred,
            'interaction': m['f1'] - ofat_pred,  # Deviation from additivity
            'ax1_name': ax1, 'ax2_name': ax2,
            'ax1_f1': single_axis_f1[ax1],
            'ax2_f1': single_axis_f1[ax2],
        }
        total_experiments += 1

    cls_results['single_axis'] = single_axis_f1
    cls_results['pairs'] = pair_results

    # Print top interactions
    interactions = [(k, v['interaction']) for k, v in pair_results.items()]
    interactions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_inter = interactions[:3]
    inter_str = ' | '.join([f'{k}: {v:+.4f}' for k, v in top_inter])
    print(f"  {cls_name:<25} Top interactions: {inter_str}")

    results['per_class'][cls_name] = cls_results

# ── Summary statistics ──
all_interactions = []
for cls_name in SAMPLED_CLASSES:
    if 'pairs' not in results['per_class'].get(cls_name, {}): continue
    for pair_name, pair_data in results['per_class'][cls_name]['pairs'].items():
        all_interactions.append({
            'class': cls_name,
            'pair': pair_name,
            'interaction': pair_data['interaction'],
            'actual_f1': pair_data['actual_f1'],
            'ofat_pred_sum': pair_data['single_sum'],
        })

interaction_values = [x['interaction'] for x in all_interactions]
mean_interaction = np.mean(interaction_values)
std_interaction = np.std(interaction_values)
mae = np.mean(np.abs(interaction_values))
rms = np.sqrt(np.mean(np.square(interaction_values)))

# Check: are interactions systematically biased?
n_neg = sum(1 for v in interaction_values if v < 0)
n_pos = sum(1 for v in interaction_values if v > 0)

results['summary'] = {
    'n_classes': len(SAMPLED_CLASSES),
    'n_axis_pairs': len(AXIS_PAIRS),
    'total_experiments': total_experiments,
    'mean_interaction': float(mean_interaction),
    'std_interaction': float(std_interaction),
    'mae_interaction': float(mae),
    'rms_interaction': float(rms),
    'n_negative_interactions': n_neg,
    'n_positive_interactions': n_pos,
    'interaction_range': [float(np.min(interaction_values)), float(np.max(interaction_values))],
    'conclusion': (
        f"交互项均值={mean_interaction:.4f}±{std_interaction:.4f}，" +
        (f"MAE={mae:.4f}，主效应可加性近似成立" if mae < 0.02 else
         f"MAE={mae:.4f}，存在可检测的轴间交互，但主效应捕获主要趋势" if mae < 0.05 else
         f"MAE={mae:.4f}，交互项不可忽略，主效应模型存在系统性偏差")
    ),
}

print(f"\n{'='*60}")
print(f"SUMMARY: Dual-Axis Joint Ablation")
print(f"{'='*60}")
print(f"  Mean interaction: {mean_interaction:+.4f} ± {std_interaction:.4f}")
print(f"  MAE (|interaction|): {mae:.4f}")
print(f"  RMS interaction: {rms:.4f}")
print(f"  Negative/Positive: {n_neg}/{n_pos}")
print(f"  Range: [{np.min(interaction_values):+.4f}, {np.max(interaction_values):+.4f}]")
print(f"\n  Conclusion: {results['summary']['conclusion']}")

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved: {OUT}")
print(f"Done: {datetime.now()}")
torch.cuda.empty_cache()
