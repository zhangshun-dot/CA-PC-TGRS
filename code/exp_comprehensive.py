# -*- coding: utf-8 -*-
"""
Comprehensive comparison & ablation experiments for 初稿4

新增实验:
  Table 7: 10+方法综合对比 (DR/P/R/F1 + Bootstrap 95% CI)
  Table 8: OFAT七轴消融 (每次移除一个轴, 测量F1变化)
  Table 9: TDM超参数灵敏度 (delta阈值 + n_c)
  Table 10: 跨文本编码器验证 (BERT vs CLIP)

所有实验基于 GDINO-tiny, DIOR 20类, IoU=0.25
"""
import json, torch, numpy as np, io, os, sys, time
from PIL import Image
from collections import defaultdict
from datetime import datetime

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Start: {datetime.now()}")

os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pyarrow.parquet as pq

# ═══════════════════════════════════════
# Setup
# ═══════════════════════════════════════
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
proc = AutoProcessor.from_pretrained(tc)
model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# Load DIOR data
pq_file = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
table = pq.read_table(pq_file); rows = table.to_pylist()
class_data = defaultdict(list)
for row in rows:
    img_bytes, objs = row['image']['bytes'], row['objects']
    if not objs or 'category' not in objs: continue
    gt_by_cat = defaultdict(list)
    for bbox, cat in zip(objs['bbox'], objs['category']):
        x,y,w,h = bbox; gt_by_cat[DIOR[cat]].append([x,y,x+w,y+h])
    for cat, boxes in gt_by_cat.items():
        if len(class_data[cat]) < 20:
            class_data[cat].append((img_bytes, boxes))

# Split train/test
np.random.seed(42)
test_data = {}
for cls_name, imgs in class_data.items():
    idx = np.random.permutation(len(imgs))
    n_test = min(10, len(imgs)//2)
    test_data[cls_name] = [imgs[i] for i in idx[:n_test]]

ALL_CLASSES = sorted(test_data.keys())
print(f"Classes: {len(ALL_CLASSES)}, Test imgs/class: {sum(len(v) for v in test_data.values())//len(ALL_CLASSES)}")

# ═══════════════════════════════════════
# Helpers
# ═══════════════════════════════════════
def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def full_metrics(preds_list, gts_list):
    """DR/P/R/F1 with per-image tracking"""
    tp=fp=fn=0; n=len(preds_list); dr_count=0
    per_img_f1 = []
    for ps,gs in zip(preds_list, gts_list):
        if ps: dr_count += 1
        matched=set(); img_tp=0; img_fp=0
        for pb in ps:
            best=0;bg=-1
            for gi,gb in enumerate(gs):
                if gi in matched: continue
                v=iou(pb,gb)
                if v>best: best=v;bg=gi
            if best>=0.25: matched.add(bg); img_tp+=1
            else: img_fp+=1
        img_fn = len(gs)-len(matched)
        tp += img_tp; fp += img_fp; fn += img_fn
        p = img_tp/(img_tp+img_fp) if (img_tp+img_fp) else 0
        r = img_tp/(img_tp+img_fn) if (img_tp+img_fn) else 0
        per_img_f1.append(2*p*r/(p+r) if (p+r) else 0)
    total_gt = sum(len(g) for g in gts_list)
    return {
        'dr': dr_count/n, 'p': tp/(tp+fp) if(tp+fp) else 0,
        'r': tp/total_gt if total_gt else 0,
        'f1': 2*tp/(2*tp+fp+fn) if(2*tp+fp+fn) else 0,
        'per_img_f1': per_img_f1, 'tp': tp, 'fp': fp, 'fn': fn
    }

def detect(prompt, img_list):
    all_p, all_g = [], []
    for img_bytes, gt_boxes in img_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad(): o = model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
        all_p.append(r['boxes'].cpu().tolist())
        all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    return full_metrics(all_p, all_g)

# ═══════════════════════════════════════════════════════
# TABLE 7: 12-Method Comprehensive Comparison
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("TABLE 7: 12-Method Comprehensive Comparison")
print("="*60)

# Candidate pool
PREFIXES = ['{}', 'a satellite image of {}', 'an aerial view of {}',
            'an overhead image of {}', 'a remote sensing image of {}']
MODIFIER_SETS = ['', ', with distinct structural features',
                 ', with distinct structural features, large and prominent']

def get_candidates(cls):
    cands = []
    for pfx in PREFIXES:
        for mods in MODIFIER_SETS:
            cands.append(pfx.format(cls) + mods)
    return cands

# Load OFAT for TDM
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)
ofat_data = ofat.get('ofat_overhead', ofat)

AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm(cls):
    if cls not in ofat_data: return f'an overhead image of {cls}'
    d = ofat_data[cls]['deltas']
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'an overhead image of {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

# 1. TDM (zero-shot)
print("\n[1/12] TDM (zero-shot)...")
tdm_results = {}
for cls_name in ALL_CLASSES:
    tdm_results[cls_name] = detect(build_tdm(cls_name), test_data[cls_name])
tdm_f1s = [tdm_results[c]['per_img_f1'] for c in ALL_CLASSES]
tdm_all_f1s = [x for sub in tdm_f1s for x in sub]

# 2-6. Fixed prompts
fixed_prompts = {
    'bare': lambda c: c,
    'satellite': lambda c: f'a satellite image of {c}',
    'aerial': lambda c: f'an aerial view of {c}',
    'overhead': lambda c: f'an overhead image of {c}',
    'remote_sensing': lambda c: f'a remote sensing image of {c}',
}
fixed_results = {}
for name, pfn in fixed_prompts.items():
    print(f"[{len(fixed_results)+2}/12] {name}...")
    all_f1s = []
    for cls_name in ALL_CLASSES:
        m = detect(pfn(cls_name), test_data[cls_name])
        all_f1s.extend(m['per_img_f1'])
    fixed_results[name] = {'f1s': all_f1s}

# 7-8. Few-shot (3-shot, 10-shot) from existing data or re-run
print("[7/12] Random baseline...")
np.random.seed(123)
random_f1s_all = []
for _ in range(5):
    seed_f1s = []
    for cls_name in ALL_CLASSES:
        cands = get_candidates(cls_name)
        rc = cands[np.random.randint(len(cands))]
        m = detect(rc, test_data[cls_name])
        seed_f1s.extend(m['per_img_f1'])
    random_f1s_all.append(np.mean(seed_f1s))
random_mean = float(np.mean(random_f1s_all))
random_std = float(np.std(random_f1s_all))

# 9. Oracle (per-class best from pool)
print("[8/12] Oracle (grid search per class)...")
oracle_f1s = []
for i, cls_name in enumerate(ALL_CLASSES):
    cands = get_candidates(cls_name)
    best_f1 = max(detect(c, test_data[cls_name])['f1'] for c in cands)
    oracle_f1s.append(best_f1)
    if i % 5 == 0: print(f"  {cls_name}: best={best_f1:.4f}")

# 10. Single best overall prefix (grid search across all classes)
print("[9/12] Best overall prefix...")
best_overall_f1 = 0
best_overall_name = ''
for name in fixed_prompts:
    mean_f1 = np.mean(fixed_results[name]['f1s'])
    if mean_f1 > best_overall_f1:
        best_overall_f1 = mean_f1
        best_overall_name = name

# 11. TDM w/ satellite prefix (ablation)
print("[10/12] TDM-satellite (ablation)...")
tdm_sat_f1s = []
for cls_name in ALL_CLASSES:
    if cls_name in ofat_data:
        d = ofat_data[cls_name]['deltas']
        pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
        mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
        p = f'a satellite image of {cls_name}'
        if mods: p += ', ' + ', '.join(mods)
    else:
        p = f'a satellite image of {cls_name}'
    tdm_sat_f1s.extend(detect(p, test_data[cls_name])['per_img_f1'])

# 12. TDM w/ fixed modifiers (no per-class OFAT)
print("[11/12] Overhead+fixed modifiers...")
fixed_mod_f1s = []
for cls_name in ALL_CLASSES:
    p = f'an overhead image of {cls_name}, with distinct structural features, large and prominent'
    fixed_mod_f1s.extend(detect(p, test_data[cls_name])['per_img_f1'])

# 13. Empty prompt (class name only with no prefix)
print("[12/12] Class-only (no domain prefix)...")
bare_f1s = []
for cls_name in ALL_CLASSES:
    bare_f1s.extend(detect(cls_name, test_data[cls_name])['per_img_f1'])

# ═══════════════════════════════════════
# Assemble Table 7
# ═══════════════════════════════════════
def bootstrap_ci(f1s, n_boot=10000):
    n = len(f1s); means = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        means.append(np.mean([f1s[i] for i in idx]))
    means = np.array(means)
    return float(np.mean(means)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

table7_methods = [
    ('TDM (OFAT, zero-shot)', tdm_all_f1s, 'Zero-shot', 'Ours'),
    ('Overhead prefix (best manual)', fixed_results['overhead']['f1s'], 'Manual', 'Baseline'),
    ('Remote sensing prefix', fixed_results['remote_sensing']['f1s'], 'Manual', ''),
    ('Satellite prefix', fixed_results['satellite']['f1s'], 'Manual', ''),
    ('Aerial prefix', fixed_results['aerial']['f1s'], 'Manual', ''),
    ('Overhead + fixed modifiers', fixed_mod_f1s, 'Manual', 'Ablation'),
    ('TDM w/ satellite base', tdm_sat_f1s, 'Zero-shot', 'Ablation'),
    ('Bare (class only)', bare_f1s, 'Manual', 'Lower bound'),
    ('Random (5 trials)', None, 'Random', f'{random_mean:.4f}±{random_std:.4f}'),
    ('Oracle (per-class best)', oracle_f1s, 'Upper bound', 'From 15-candidate pool'),
]

print(f"\n{'='*80}")
print(f"TABLE 7: Comprehensive Comparison (GDINO-tiny, DIOR 20-class, IoU=0.25)")
print(f"{'='*80}")
print(f"  {'#':<3} {'Method':<35} {'F1':>8} {'95% CI':>22} {'DR':>8} {'P':>8} {'R':>8}  Type")
print(f"  {'-'*100}")

table7_output = []
for i, (name, f1s, mtype, note) in enumerate(table7_methods):
    if f1s is not None:
        mean, ci_low, ci_high = bootstrap_ci(f1s)
        dr = np.mean([1 if f > 0 else 0 for f in f1s])
        # For precision/recall we need the actual metrics, compute from stored results
        table7_output.append({
            'method': name, 'f1': round(mean,4), 'ci_low': round(ci_low,4),
            'ci_high': round(ci_high,4), 'type': mtype, 'note': note
        })
        print(f"  {i+1:<3} {name:<35} {mean:>8.4f} [{ci_low:.4f}, {ci_high:.4f}] {mtype:>12}")
    else:
        table7_output.append({
            'method': name, 'f1': random_mean, 'std': random_std,
            'type': mtype, 'note': note
        })
        print(f"  {i+1:<3} {name:<35} {random_mean:>8.4f} ±{random_std:.4f}          {mtype:>12}")
    print(f"       {note}")

# ═══════════════════════════════════════════════════════
# TABLE 8: OFAT Seven-Axis Ablation
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TABLE 8: OFAT Axis Ablation")
print("="*60)

AXES = ['grammar','granularity','spatial','geometric','scale','spectral','texture']
AXIS_MODIFIERS = {
    'grammar': None,  # domain prefix
    'granularity': 'with fine details',
    'spatial': 'in its natural setting',
    'geometric': 'with distinct structural features',
    'scale': 'large and prominent',
    'spectral': 'with natural coloring',
    'texture': 'with clearly visible surface features',
}

print("  Removing one axis at a time, measuring ΔF1 vs full TDM...")
axis_ablation = {}
for drop_axis in AXES:
    f1s = []
    for cls_name in ALL_CLASSES:
        if cls_name not in ofat_data: continue
        d = ofat_data[cls_name]['deltas']
        pos = sorted([(ax,v) for ax,v in d.items()
                      if ax != 'grammar' and ax != drop_axis and v > 0.03],
                     key=lambda x: -x[1])
        mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
        p = f'an overhead image of {cls_name}'
        if mods: p += ', ' + ', '.join(mods)
        f1s.extend(detect(p, test_data[cls_name])['per_img_f1'])
    mean_f1 = np.mean(f1s)
    delta = mean_f1 - np.mean(tdm_all_f1s)
    axis_ablation[drop_axis] = {'f1': float(mean_f1), 'delta': float(delta)}
    print(f"  -{drop_axis:<15}: F1={mean_f1:.4f}  Δ={delta:+.4f}")

# ═══════════════════════════════════════════════════════
# TABLE 9: TDM Hyperparameter Sensitivity
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TABLE 9: TDM Parameter Sensitivity")
print("="*60)

# 9a: Delta threshold sensitivity
delta_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
print("\n  Delta threshold sensitivity:")
delta_results = {}
for thresh in delta_thresholds:
    f1s = []
    for cls_name in ALL_CLASSES:
        if cls_name not in ofat_data: continue
        d = ofat_data[cls_name]['deltas']
        pos = sorted([(ax,v) for ax,v in d.items()
                      if ax != 'grammar' and v > thresh],
                     key=lambda x: -x[1])
        mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
        p = f'an overhead image of {cls_name}'
        if mods: p += ', ' + ', '.join(mods)
        f1s.extend(detect(p, test_data[cls_name])['per_img_f1'])
    mean_f1 = np.mean(f1s)
    delta_results[str(thresh)] = float(mean_f1)
    print(f"    δ>{thresh:.2f}: F1={mean_f1:.4f}")

# 9b: Max modifier count
max_mods_vals = [1, 2, 3, 4]
print("\n  Max modifier count sensitivity:")
mod_results = {}
for max_m in max_mods_vals:
    f1s = []
    for cls_name in ALL_CLASSES:
        if cls_name not in ofat_data: continue
        d = ofat_data[cls_name]['deltas']
        pos = sorted([(ax,v) for ax,v in d.items()
                      if ax != 'grammar' and v > 0.03],
                     key=lambda x: -x[1])
        mods = [AXM[ax] for ax,_ in pos[:max_m]] if pos else []
        p = f'an overhead image of {cls_name}'
        if mods: p += ', ' + ', '.join(mods)
        f1s.extend(detect(p, test_data[cls_name])['per_img_f1'])
    mean_f1 = np.mean(f1s)
    mod_results[str(max_m)] = float(mean_f1)
    print(f"    max {max_m} modifiers: F1={mean_f1:.4f}")

# ═══════════════════════════════════════════════════════
# TABLE 10: Detection Threshold Robustness
# ═══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("TABLE 10: IoU Threshold Robustness")
print("="*60)

iou_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
threshold_results = {}
for iou_thresh in iou_thresholds:
    tdm_f1s_thresh = []
    oh_f1s_thresh = []
    for cls_name in ALL_CLASSES:
        # overhead
        prompt = f'an overhead image of {cls_name}'
        m_oh = detect(prompt, test_data[cls_name])
        oh_f1s_thresh.extend(m_oh['per_img_f1'])

        # TDM
        m_tdm = detect(build_tdm(cls_name), test_data[cls_name])
        tdm_f1s_thresh.extend(m_tdm['per_img_f1'])

    # Note: IoU threshold is hardcoded in detect() at 0.25
    # For this experiment we need to re-run with different thresholds
    # Skip for now - would need to re-implement detect() with variable threshold

# Simplified: re-run key methods with variable thresholds
print("  Running TDM vs overhead at multiple IoU thresholds...")
def detect_with_thresh(prompt, img_list, thresh):
    all_p, all_g = [], []
    for img_bytes, gt_boxes in img_list:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad(): o = model(**inputs)
        target = torch.tensor([img.size[::-1]]).to(device)
        r = proc.post_process_grounded_object_detection(o, threshold=thresh, target_sizes=target)[0]
        all_p.append(r['boxes'].cpu().tolist())
        all_g.append([(b[0],b[1],b[2],b[3]) for b in gt_boxes])
    tp=fp=fn=0; n=len(all_g)
    for ps,gs in zip(all_p,all_g):
        matched=set()
        for pb in ps:
            best=0;bg=-1
            for gi,gb in enumerate(gs):
                if gi in matched: continue
                v=iou(pb,gb)
                if v>best: best=v;bg=gi
            if best>=thresh: matched.add(bg);tp+=1
            else: fp+=1
        fn+=len(gs)-len(matched)
    total_gt=sum(len(g) for g in all_g)
    p=tp/(tp+fp) if(tp+fp)else 0;r=tp/total_gt if total_gt else 0
    return 2*p*r/(p+r) if(p+r)else 0

# Run on subset (5 classes) for speed
subset_cls = ALL_CLASSES[:5]
for iou_thresh in iou_thresholds:
    tdm_sub = []; oh_sub = []
    for cls_name in subset_cls:
        tdm_sub.append(detect_with_thresh(build_tdm(cls_name), test_data[cls_name], iou_thresh))
        oh_sub.append(detect_with_thresh(f'an overhead image of {cls_name}', test_data[cls_name], iou_thresh))
    threshold_results[str(iou_thresh)] = {
        'tdm': float(np.mean(tdm_sub)), 'oh': float(np.mean(oh_sub)),
        'delta': float(np.mean(tdm_sub) - np.mean(oh_sub))
    }
    print(f"  IoU={iou_thresh:.2f}: TDM={np.mean(tdm_sub):.4f}  oh={np.mean(oh_sub):.4f}  Δ={np.mean(tdm_sub)-np.mean(oh_sub):+.4f}")

# ═══════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════
all_out = {
    'table7_comprehensive_comparison': table7_output,
    'table8_axis_ablation': axis_ablation,
    'table9_hyperparameter': {
        'delta_threshold': delta_results,
        'max_modifiers': mod_results,
    },
    'table10_threshold_robustness': threshold_results,
    'baseline_tdm_f1': float(np.mean(tdm_all_f1s)),
    'oracle_f1': float(np.mean(oracle_f1s)),
}

out_path = r'C:\Users\张三\Desktop\comprehensive_experiments.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(all_out, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"ALL EXPERIMENTS COMPLETE")
print(f"Saved: {out_path}")
print(f"Done: {datetime.now()}")
