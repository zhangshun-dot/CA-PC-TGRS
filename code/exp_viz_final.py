# -*- coding: utf-8 -*-
"""Publication-quality figure matching teacher's reference."""
import json, torch, numpy as np, io, os
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

device = 'cuda'; os.environ['HF_HUB_OFFLINE'] = '1'
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
        x, y, w, h = bbox; gt_by_cat[DIOR[cat]].append([x, y, x+w, y+h])
    for cat, boxes in gt_by_cat.items():
        if len(class_data[cat]) < 20: class_data[cat].append((img_bytes, boxes))

np.random.seed(42)
test_data = {}
for cls_name, imgs in class_data.items():
    idx = np.random.permutation(len(imgs)); n_test = min(10, len(imgs)//2)
    test_data[cls_name] = [imgs[i] for i in idx[:n_test]]

# Pick best-looking images (one per class, pick image with most GT boxes)
viz_picks = {}
for cls_name in ['airplane', 'bridge', 'stadium', 'storage tank', 'harbor', 'ship']:
    if cls_name in test_data and test_data[cls_name]:
        best = max(test_data[cls_name], key=lambda x: len(x[1]))
        viz_picks[cls_name] = best

# Colors (tableau10)
GT_C = (44, 160, 44)     # green
BARE_C = (214, 39, 40)   # red
SAT_C = (148, 103, 189)  # purple
OH_C = (31, 119, 180)    # blue
TDM_C = (255, 127, 14)   # orange
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def detect(img_bytes, prompt):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    inputs = proc(images=img, text=prompt, return_tensors='pt')
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad(): o = model(**inputs)
    target = torch.tensor([img.size[::-1]]).to(device)
    r = proc.post_process_grounded_object_detection(o, threshold=0.25, target_sizes=target)[0]
    return img, r['boxes'].cpu().tolist()

def draw_boxes(img, boxes, color, w=3):
    d = ImageDraw.Draw(img)
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        d.rectangle([x1, y1, x2, y2], outline=color, width=w)
    return img

# Panel size
PW, PH = 280, 280
GAP = 8
LABEL_H = 26
LEGEND_H = 40
NCOLS = 5
NROWS = len(viz_picks)

W = PW * NCOLS + GAP * (NCOLS + 1)
H = (PH + LABEL_H) * NROWS + GAP * (NROWS + 1) + LEGEND_H

canvas = Image.new('RGB', (W, H), WHITE)

for row, (cls_name, (img_bytes, gt_boxes)) in enumerate(viz_picks.items()):
    gts = [(b[0], b[1], b[2], b[3]) for b in gt_boxes]
    base_img, _ = detect(img_bytes, cls_name)
    base_img = base_img.resize((PW, PH), Image.LANCZOS)

    methods = [
        ('GT',       gts, GT_C, None),
        ('bare',     None, BARE_C, cls_name),
        ('satellite', None, SAT_C, f'a satellite image of {cls_name}'),
        ('overhead', None, OH_C, f'an overhead image of {cls_name}'),
        ('TDM',      None, TDM_C, f'an overhead image of {cls_name}, with distinct structural features, large and prominent'),
    ]

    for col, (label, pre_boxes, color, prompt) in enumerate(methods):
        panel = base_img.copy()
        if pre_boxes:
            panel = draw_boxes(panel, pre_boxes, color, w=4)
        else:
            det_img, det_boxes = detect(img_bytes, prompt)
            panel = det_img.resize((PW, PH), Image.LANCZOS)
            panel = draw_boxes(panel, det_boxes, color, w=3)

        # Label bar at top
        d = ImageDraw.Draw(panel)
        d.rectangle([(0, 0), (PW, LABEL_H)], fill=color)
        d.text((6, 4), label, fill=WHITE)

        x = GAP + col * (PW + GAP)
        y = GAP + row * (PH + LABEL_H + GAP)
        canvas.paste(panel, (x, y))

    # Class label on far left
    d = ImageDraw.Draw(canvas)
    cy = GAP + row * (PH + LABEL_H + GAP) + (PH + LABEL_H) // 2
    d.text((GAP + PW * NCOLS + GAP * NCOLS + 4, cy - 8), cls_name, fill=BLACK)

# Legend
leg_y = H - LEGEND_H + 5
d = ImageDraw.Draw(canvas)
labels = [('Ground Truth', GT_C), ('Bare', BARE_C), ('Satellite', SAT_C),
          ('Overhead', OH_C), ('TDM (Ours)', TDM_C)]
leg_x = 20
for label, color in labels:
    d.rectangle([(leg_x, leg_y), (leg_x + 18, leg_y + 18)], fill=color)
    d.text((leg_x + 24, leg_y + 1), label, fill=BLACK)
    leg_x += 140

# Title
d.text((W // 2 - 200, H - 18), 'GDINO-tiny | DIOR 20-class | IoU=0.25 | All zero-shot', fill=(128, 128, 128))

out_path = r'C:\Users\张三\Desktop\fig_detection_final.png'
canvas.save(out_path)
print(f'Saved: {out_path}')
print(f'{NROWS} classes x {NCOLS} methods = {NROWS * NCOLS} panels')
print('Done.')
