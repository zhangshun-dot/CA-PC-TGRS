# -*- coding: utf-8 -*-
"""CLS受控消融：同模型、同forward、仅改logit聚合方式。
- Max (default): 逐Token点积 → 标准GDINO行为
- Mean: 所有Token logit均值 → 模拟[CLS]池化
- CLS-only: 仅取[CLS] Token logit → 严格[CLS]池化

这是对论文核心声称（逐Token点积 > CLS池化）的最直接受控实验。
"""
import json, torch, numpy as np, io, os
from PIL import Image
from collections import defaultdict
import pyarrow.parquet as pq
from datetime import datetime
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

device = 'cuda'
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Start: {datetime.now()}", flush=True)

OUT = r'C:\Users\张三\Desktop\cls_ablation_results.json'

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# ── Load data (10 test per class) ──
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

np.random.seed(42)
test_data = {}
for cls_name, imgs in class_data.items():
    idx = np.random.permutation(len(imgs))
    n = min(10, len(imgs)//2)
    test_data[cls_name] = [imgs[i] for i in idx[n:2*n]]

ALL_CLASSES = sorted(class_data.keys())
print(f"Classes: {len(ALL_CLASSES)}, Test imgs/class: {sum(len(test_data.get(c,[])) for c in ALL_CLASSES)//len(ALL_CLASSES)}")

# ── Helpers ──
def iou(b1, b2):
    x1,y1,x2,y2 = max(b1[0],b2[0]), max(b1[1],b2[1]), min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    a1,a2 = (b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def compute_metrics(pred_boxes, pred_scores, gt_boxes, conf_thresh=0.25, iou_thresh=0.25):
    """Compute DR, P, R, F1 from raw boxes and scores."""
    # Filter by confidence
    keep = pred_scores >= conf_thresh
    preds = [b for b, k in zip(pred_boxes, keep) if k]

    tp=fp=fn=0; dr = 1 if preds else 0
    matched = set()
    for pb in preds:
        best_iou=0; best_gi=-1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched: continue
            v = iou(pb, gb)
            if v > best_iou: best_iou=v; best_gi=gi
        if best_iou >= iou_thresh: matched.add(best_gi); tp+=1
        else: fp+=1
    fn += len(gt_boxes)-len(matched)
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return {'dr': dr, 'precision': p, 'recall': r, 'f1': f1}

# ── Load GDINO-tiny ──
print("Loading GDINO-tiny...", flush=True)
tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
gd_proc = AutoProcessor.from_pretrained(tc)
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

# ── OFAT data for TDM prompts ──
with open(r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据\ofat_overhead_results.json') as f:
    ofat = json.load(f)['ofat_overhead']
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm_prompt(cls):
    """TDM optimized prompt (overhead prefix + best 2 modifiers)."""
    if cls not in ofat:
        return f'an overhead image of {cls}'
    d = ofat[cls]['deltas']
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    p = f'an overhead image of {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

# ── Run CLS ablation ──
results = {}
print(f"\n{'='*60}")
print("CLS CONTROLLED ABLATION: Max vs Mean vs CLS-only")
print(f"{'='*60}\n")

for cls_name in ALL_CLASSES:
    prompt = build_tdm_prompt(cls_name)
    imgs = test_data[cls_name]

    f1s_max = []; f1s_mean = []; f1s_cls = []

    for img_bytes, gt_boxes in imgs:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        inputs = gd_proc(images=img, text=prompt, return_tensors='pt')
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            o = gd_model(**inputs, output_hidden_states=True)

        # Raw outputs
        logits = o.logits[0].cpu()        # [900, 256]
        scores_raw = torch.sigmoid(logits) # [900, 256]

        # Target size for box scaling
        target_size = torch.tensor([img.size[::-1]])
        # Standard post-processing for boxes
        with torch.no_grad():
            r_std = gd_proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target_size)[0]
        std_boxes = r_std['boxes'].tolist()

        # ── Method 1: Max (standard token-wise) ──
        scores_max = scores_raw.max(dim=1).values.numpy()
        m = compute_metrics(std_boxes, scores_max, gt_boxes)
        f1s_max.append(m['f1'])

        # ── Method 2: Mean (simulates [CLS] pooling) ──
        scores_mean = scores_raw.mean(dim=1).numpy()
        m = compute_metrics(std_boxes, scores_mean, gt_boxes)
        f1s_mean.append(m['f1'])

        # ── Method 3: CLS-only (first token = [CLS]) ──
        scores_cls = scores_raw[:, 0].numpy()
        m = compute_metrics(std_boxes, scores_cls, gt_boxes)
        f1s_cls.append(m['f1'])

    f1_max = np.mean(f1s_max)
    f1_mean = np.mean(f1s_mean)
    f1_cls = np.mean(f1s_cls)

    results[cls_name] = {
        'prompt': prompt,
        'max_f1': float(f1_max),
        'mean_f1': float(f1_mean),
        'cls_only_f1': float(f1_cls),
        'delta_max_vs_mean': float(f1_max - f1_mean),
        'delta_max_vs_cls': float(f1_max - f1_cls),
    }

    delta_mean = f1_max - f1_mean
    delta_cls = f1_max - f1_cls
    print(f"  {cls_name:<25} Max={f1_max:.4f} Mean={f1_mean:.4f} (Δ={delta_mean:+.4f}) CLS-only={f1_cls:.4f} (Δ={delta_cls:+.4f})")

# ── Summary ──
all_max = [v['max_f1'] for v in results.values()]
all_mean = [v['mean_f1'] for v in results.values()]
all_cls = [v['cls_only_f1'] for v in results.values()]

summary = {
    'max_mean': float(np.mean(all_max)),
    'mean_mean': float(np.mean(all_mean)),
    'cls_only_mean': float(np.mean(all_cls)),
    'max_per_class': {k: v['max_f1'] for k,v in results.items()},
    'mean_per_class': {k: v['mean_f1'] for k,v in results.items()},
    'cls_only_per_class': {k: v['cls_only_f1'] for k,v in results.items()},
    'delta_max_vs_mean': float(np.mean(all_max) - np.mean(all_mean)),
    'delta_max_vs_cls': float(np.mean(all_max) - np.mean(all_cls)),
    'wins_max_over_mean': sum(1 for a,b in zip(all_max, all_mean) if a > b),
    'wins_max_over_cls': sum(1 for a,b in zip(all_max, all_cls) if a > b),
    'n_classes': len(all_max),
}

print(f"\n{'='*60}")
print(f"SUMMARY: CLS Controlled Ablation (DIOR 20-class, GDINO-tiny)")
print(f"{'='*60}")
print(f"  Max (token-wise):           F1 = {summary['max_mean']:.4f}")
print(f"  Mean (CLS-like pooling):    F1 = {summary['mean_mean']:.4f}  Δ = {summary['delta_max_vs_mean']:+.4f}")
print(f"  CLS-only (strict [CLS]):   F1 = {summary['cls_only_mean']:.4f}  Δ = {summary['delta_max_vs_cls']:+.4f}")
print(f"  Max > Mean: {summary['wins_max_over_mean']}/{summary['n_classes']} classes")
print(f"  Max > CLS-only: {summary['wins_max_over_cls']}/{summary['n_classes']} classes")
if summary['mean_mean'] > 0:
    print(f"  Relative gain (Max/Mean):   {summary['max_mean']/summary['mean_mean']:.2f}×")
else:
    print(f"  Relative gain (Max/Mean):   ∞ (Mean F1=0)")
if summary['cls_only_mean'] > 0:
    print(f"  Relative gain (Max/CLS):    {summary['max_mean']/summary['cls_only_mean']:.2f}×")
else:
    print(f"  Relative gain (Max/CLS):    ∞ (CLS-only F1=0)")

with open(OUT, 'w') as f:
    json.dump({'per_class': results, 'summary': summary}, f, indent=2, ensure_ascii=False)

print(f"\nSaved: {OUT}")
print(f"Done: {datetime.now()}")
torch.cuda.empty_cache()
