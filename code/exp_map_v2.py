# -*- coding: utf-8 -*-
"""mAP metrics: DR/P/R/F1 at thresh=0.25, AP/AP50/AP75 from PR curve"""

import json, torch, numpy as np, io, os
from PIL import Image
from collections import defaultdict

device = 'cuda'
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pyarrow.parquet as pq

tc = r'C:\Users\张三\.cache\huggingface\hub\models--IDEA-Research--grounding-dino-tiny\snapshots\a2bb814dd30d776dcf7e30523b00659f4f141c71'
proc = AutoProcessor.from_pretrained(tc)
model = AutoModelForZeroShotObjectDetection.from_pretrained(tc, dtype=torch.float32).eval().to(device)

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

def compute_ap(scores_tp_list, n_gt, n_pts=11):
    tp_cum=0; fp_cum=0; precs=[]; recs=[]
    for score, is_tp in scores_tp_list:
        if is_tp: tp_cum+=1
        else: fp_cum+=1
        precs.append(tp_cum/(tp_cum+fp_cum) if(tp_cum+fp_cum)else 0)
        recs.append(tp_cum/n_gt if n_gt else 0)
    ap=0.0
    for t in np.linspace(0,1,n_pts):
        ap += max([p for p,r in zip(precs,recs) if r>=t]+[0]) / n_pts
    return round(ap,4)

def evaluate(pfn, data, op_thresh=0.25):
    op_tp=0; op_fp=0; op_fn=0; op_total_gt=0; dr_count=0; n_imgs=0
    ap25_list=[]; ap50_list=[]; ap75_list=[]; total_gt_all=0

    for cls_name in ALL_CLASSES:
        if cls_name not in data: continue
        imgs = data[cls_name]
        for img_bytes, gt_boxes in imgs:
            n_imgs += 1
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            inputs = proc(images=img, text=pfn(cls_name), return_tensors='pt')
            inputs = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            with torch.no_grad(): o = model(**inputs)
            target = torch.tensor([img.size[::-1]]).to(device)
            r = proc.post_process_grounded_object_detection(o, threshold=0.0, target_sizes=target)[0]
            boxes = r['boxes'].cpu().tolist()
            scores = r['scores'].cpu().tolist()
            gts = [(b[0],b[1],b[2],b[3]) for b in gt_boxes]
            n_gt = len(gts)
            total_gt_all += n_gt

            idx = np.argsort(scores)[::-1]
            boxes_s = [boxes[i] for i in idx]; scores_s = [scores[i] for i in idx]

            matched_25=set(); matched_50=set(); matched_75=set()
            matched_op=set(); has_op_det=False

            for box, score in zip(boxes_s, scores_s):
                best_iou=0;best_gi=-1
                for gi,gt in enumerate(gts):
                    v=iou(box,gt);
                    if v>best_iou: best_iou=v;best_gi=gi

                is_25 = best_iou >= 0.25
                is_50 = best_iou >= 0.50
                is_75 = best_iou >= 0.75

                if best_gi not in matched_25:
                    ap25_list.append((score, is_25))
                    if is_25: matched_25.add(best_gi)
                if best_gi not in matched_50:
                    ap50_list.append((score, is_50))
                    if is_50: matched_50.add(best_gi)
                if best_gi not in matched_75:
                    ap75_list.append((score, is_75))
                    if is_75: matched_75.add(best_gi)

                if score >= op_thresh:
                    has_op_det = True
                    if is_25 and best_gi not in matched_op:
                        matched_op.add(best_gi); op_tp += 1
                    elif not is_25:
                        op_fp += 1

            if has_op_det: dr_count += 1
            op_fn += n_gt - len(matched_op)
            op_total_gt += n_gt

    p = op_tp/(op_tp+op_fp) if(op_tp+op_fp)else 0
    r = op_tp/op_total_gt if op_total_gt else 0
    f1 = 2*p*r/(p+r) if(p+r)else 0

    ap25 = compute_ap(ap25_list, total_gt_all)
    ap50 = compute_ap(ap50_list, total_gt_all)
    ap75 = compute_ap(ap75_list, total_gt_all)
    mAP = round((ap25+ap50+ap75)/3, 4)

    return {'dr':round(dr_count/n_imgs,4), 'p':round(p,4), 'r':round(r,4),
            'f1':round(f1,4), 'ap25':ap25, 'ap50':ap50, 'ap75':ap75, 'mAP':mAP}

# OFAT
with open(r'C:\Users\张三\Desktop\ofat_overhead_results.json') as f:
    ofat = json.load(f)
ofat_data = ofat.get('ofat_overhead', ofat)
AXM = {'granularity':'with fine details','spatial':'in its natural setting',
       'geometric':'with distinct structural features','scale':'large and prominent',
       'spectral':'with natural coloring','texture':'with clearly visible surface features'}

def build_tdm(cls, prefix='overhead'):
    if cls not in ofat_data: return f'an overhead image of {cls}'
    d = ofat_data[cls]['deltas']
    pos = sorted([(ax,v) for ax,v in d.items() if ax!='grammar' and v>0.03], key=lambda x:-x[1])
    mods = [AXM[ax] for ax,_ in pos[:2]] if pos else []
    pfx = 'an overhead image of' if prefix=='overhead' else 'a satellite image of'
    p = f'{pfx} {cls}'
    if mods: p += ', ' + ', '.join(mods)
    return p

methods = {
    'bare': lambda c: c,
    'satellite': lambda c: f'a satellite image of {c}',
    'satellite+TDM': lambda c: build_tdm(c, 'satellite'),
    'overhead': lambda c: f'an overhead image of {c}',
    'overhead+TDM': lambda c: build_tdm(c, 'overhead'),
}

results = {}
for name, pfn in methods.items():
    print(f'{name}...', flush=True)
    results[name] = evaluate(pfn, test_data)
    m = results[name]
    print(f'  DR={m["dr"]:.4f} P={m["p"]:.4f} R={m["r"]:.4f} F1={m["f1"]:.4f} '
          f'AP25={m["ap25"]:.4f} AP50={m["ap50"]:.4f} AP75={m["ap75"]:.4f} mAP={m["mAP"]:.4f}', flush=True)

print()
print(f'{"Method":<20} {"DR":>8} {"P":>8} {"R":>8} {"F1":>8} {"AP25":>8} {"AP50":>8} {"AP75":>8} {"mAP":>8}')
print('-'*90)
for name in ['bare','satellite','satellite+TDM','overhead','overhead+TDM']:
    m=results[name]
    print(f'{name:<20} {m["dr"]:>8.4f} {m["p"]:>8.4f} {m["r"]:>8.4f} {m["f1"]:>8.4f} '
          f'{m["ap25"]:>8.4f} {m["ap50"]:>8.4f} {m["ap75"]:>8.4f} {m["mAP"]:>8.4f}')

tdm_delta_f1 = results['overhead+TDM']['f1'] - results['overhead']['f1']
tdm_delta_map = results['overhead+TDM']['mAP'] - results['overhead']['mAP']
print(f'\nTDM vs overhead: delta_F1={tdm_delta_f1:+.4f}, delta_mAP={tdm_delta_map:+.4f}')

with open(r'C:\Users\张三\Desktop\map_results.json','w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('Saved: map_results.json')
