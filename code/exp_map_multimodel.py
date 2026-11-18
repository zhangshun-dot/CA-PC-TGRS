# -*- coding: utf-8 -*-
"""mAP for Table 2 (cross-model) and Table 6 (cross-dataset)."""

import json, torch, numpy as np, io, os
from PIL import Image
from collections import defaultdict

device = 'cuda'
os.environ['HF_HUB_OFFLINE'] = '1'
import pyarrow.parquet as pq

DIOR = ['airplane','airport','baseball field','basketball court','bridge','chimney','dam',
    'expressway service area','expressway toll station','golf course','ground track field',
    'harbor','overpass','ship','stadium','storage tank','tennis court','train station','vehicle','windmill']

# Load test data
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
        if len(class_data[cat]) < 20: class_data[cat].append((img_bytes, boxes))

np.random.seed(42)
test_data = {}
for cls_name, imgs in class_data.items():
    idx = np.random.permutation(len(imgs)); n_test = min(10, len(imgs)//2)
    test_data[cls_name] = [imgs[i] for i in idx[:n_test]]
ALL_CLASSES = sorted(test_data.keys())

def iou(b1,b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1]);x2=min(b1[2],b2[2]);y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]);a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter)>0 else 0

def compute_ap(scores_tp, n_gt, n_pts=11):
    tp_cum=0;fp_cum=0;precs=[];recs=[]
    for score, is_tp in scores_tp:
        if is_tp: tp_cum+=1
        else: fp_cum+=1
        precs.append(tp_cum/(tp_cum+fp_cum) if(tp_cum+fp_cum)else 0)
        recs.append(tp_cum/n_gt if n_gt else 0)
    ap=0.0
    for t in np.linspace(0,1,n_pts):
        ap += max([p for p,r in zip(precs,recs) if r>=t]+[0]) / n_pts
    return round(ap,4)

def evaluate_with_model(pfn, data, proc, model, op_thresh=0.25):
    op_tp=0;op_fp=0;op_fn=0;op_tg=0;dc=0;ni=0
    ap25=[];ap50=[];ap75=[];tg=0
    for cls_name in ALL_CLASSES:
        if cls_name not in data: continue
        for img_bytes, gt_boxes in data[cls_name]:
            ni+=1
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            prompt = pfn(cls_name)
            inputs = proc(images=img, text=prompt, return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o = model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=target)[0]
            boxes=r['boxes'].cpu().tolist();scores=r['scores'].cpu().tolist()
            gts=[(b[0],b[1],b[2],b[3]) for b in gt_boxes];ng=len(gts);tg+=ng
            idx=np.argsort(scores)[::-1]
            bs=[boxes[i] for i in idx];ss=[scores[i] for i in idx]
            m25=set();m50=set();m75=set();mo=set();hd=False
            for box,score in zip(bs,ss):
                bi=0;bg=-1
                for gi,gt in enumerate(gts):
                    v=iou(box,gt)
                    if v>bi: bi=v;bg=gi
                i25=bi>=0.25;i50=bi>=0.50;i75=bi>=0.75
                if bg not in m25: ap25.append((score,i25))
                if i25: m25.add(bg)
                if bg not in m50: ap50.append((score,i50))
                if i50: m50.add(bg)
                if bg not in m75: ap75.append((score,i75))
                if i75: m75.add(bg)
                if score>=op_thresh:
                    hd=True
                    if i25 and bg not in mo: mo.add(bg);op_tp+=1
                    elif not i25: op_fp+=1
            if hd: dc+=1
            op_fn+=ng-len(mo);op_tg+=ng
    p=op_tp/(op_tp+op_fp) if(op_tp+op_fp)else 0
    r=op_tp/op_tg if op_tg else 0
    f1=2*p*r/(p+r) if(p+r)else 0
    return {'dr':round(dc/ni,4),'p':round(p,4),'r':round(r,4),'f1':round(f1,4),
            'ap25':compute_ap(ap25,tg),'ap50':compute_ap(ap50,tg),'ap75':compute_ap(ap75,tg),
            'mAP':round((compute_ap(ap25,tg)+compute_ap(ap50,tg)+compute_ap(ap75,tg))/3,4)}

# OFAT
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)
ofat_data = ofat.get('ofat_overhead', ofat)
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}
def build_tdm(cls):
    if cls not in ofat_data: return f'an overhead image of {cls}'
    d=ofat_data[cls]['deltas']
    pos=sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods=[AXM[ax] for ax,_ in pos[:2]] if pos else []
    p=f'an overhead image of {cls}'
    if mods: p+=', '+', '.join(mods)
    return p

# ═══════════════════════════════════════
# MODEL 1: GDINO-base
# ═══════════════════════════════════════
print("="*60)
print("MODEL 1: GDINO-base DIOR mAP")
print("="*60)
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
bc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-base\snapshots\12bdfa3120f3e7ec7b434d90674b3396eccf88eb'
bp = AutoProcessor.from_pretrained(bc)
bm = AutoModelForZeroShotObjectDetection.from_pretrained(bc, dtype=torch.float32).eval().to(device)

base_methods = {
    'satellite': lambda c: f'a satellite image of {c}',
    'overhead': lambda c: f'an overhead image of {c}',
    'tdm': build_tdm,
}
base_res = {}
for name, pfn in base_methods.items():
    print(f'  {name}...', flush=True)
    base_res[name] = evaluate_with_model(pfn, test_data, bp, bm)
    m=base_res[name]
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} AP50={m["ap50"]:.4f} mAP={m["mAP"]:.4f}', flush=True)

# ═══════════════════════════════════════
# MODEL 2: YOLO-World-v2
# ═══════════════════════════════════════
print("\n"+"="*60)
print("MODEL 2: YOLO-World-v2 DIOR mAP")
print("="*60)
yc = r'C:\Users\张三\.cache\huggingface\hub\models--wondervictor--YOLO-World\snapshots\1732cdaeeabef06e9c995d95385b3fc8dc249b60'
yp = AutoProcessor.from_pretrained(yc)
ym = AutoModelForZeroShotObjectDetection.from_pretrained(yc, dtype=torch.float32).eval().to(device)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

yolo_methods = {
    'satellite': lambda c: f'a satellite image of {c}',
    'overhead': lambda c: f'an overhead image of {c}',
    'tdm': build_tdm,
}
yolo_res = {}
for name, pfn in yolo_methods.items():
    print(f'  {name}...', flush=True)
    yolo_res[name] = evaluate_with_model(pfn, test_data, yp, ym)
    m=yolo_res[name]
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} AP50={m["ap50"]:.4f} mAP={m["mAP"]:.4f}', flush=True)

# ═══════════════════════════════════════
# MODEL 3: OWLv2-base
# ═══════════════════════════════════════
print("\n"+"="*60)
print("MODEL 3: OWLv2-base DIOR mAP")
print("="*60)
oc = r'C:\Users\张三\.cache\huggingface\hub\models--google--owlv2-base-patch16-ensemble\snapshots\82b59965d8f32b7fd1a480cbe29bbcb11e2878a3'
op = AutoProcessor.from_pretrained(oc)
om = AutoModelForZeroShotObjectDetection.from_pretrained(oc, dtype=torch.float32).eval().to(device)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

# OWLv2 uses CLIP, need calibrated prefix (from paper: satellite is best for OWLv2)
owl_methods = {
    'satellite': lambda c: f'a satellite image of {c}',
    'overhead': lambda c: f'an overhead image of {c}',
    'tdm_calibrated': lambda c: f'a satellite image of {c}, with distinct structural features, large and prominent',
}
owl_res = {}
for name, pfn in owl_methods.items():
    print(f'  {name}...', flush=True)
    owl_res[name] = evaluate_with_model(pfn, test_data, op, om)
    m=owl_res[name]
    print(f'    DR={m["dr"]:.4f} F1={m["f1"]:.4f} AP50={m["ap50"]:.4f} mAP={m["mAP"]:.4f}', flush=True)

# ═══════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════
# Load tiny results for comparison
with open(r'C:\Users\张三\Desktop\map_results.json') as f:
    tiny_res = json.load(f)

print(f"\n{'='*90}")
print("TABLE 2: Cross-Model mAP (DIOR 20-class, IoU operational=0.25)")
print(f"{'='*90}")
print(f"  {'Model':<20} {'Method':<15} {'DR':>8} {'F1':>8} {'AP50':>8} {'AP75':>8} {'mAP':>8}")
print(f"  {'-'*75}")

all_out = {'gdino_tiny': tiny_res, 'gdino_base': base_res, 'yolo_world': yolo_res, 'owlv2': owl_res}

for model_name, res in [('GDINO-tiny', tiny_res), ('GDINO-base', base_res),
                          ('YOLO-World', yolo_res), ('OWLv2', owl_res)]:
    for method_name in res:
        m = res[method_name]
        print(f"  {model_name:<20} {method_name:<15} {m['dr']:>8.4f} {m['f1']:>8.4f} {m['ap50']:>8.4f} {m['ap75']:>8.4f} {m['mAP']:>8.4f}")

with open(r'C:\Users\张三\Desktop\map_cross_model.json', 'w') as f:
    json.dump(all_out, f, indent=2, ensure_ascii=False)
print(f"\nSaved: map_cross_model.json")
print("Done.")
