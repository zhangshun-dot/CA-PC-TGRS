# -*- coding: utf-8 -*-
"""cross_domain_inventory.py — 跨域数据对盘点（不占 GPU）

目的：把主干补强第 3 项「更多跨域类别/跨域数据对」摸清底数。
统计：DIOR full20_images 各类图/GT 覆盖；COCO val2017 各类本地图数；
      语义映射分级 → 得出可扩的跨域对清单与受限项。
"""
import os, glob, json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import torch

DIOR_IMG_DIR = r'C:\Users\张三\modeling\dior_data\full20_images'
DIOR_PARQUET = r'C:\Users\张三\modeling\dior_data\data\train-00000-of-00012.parquet'
COCO_IMG_DIR = r'C:\Users\张三\modeling\coco_data\val2017'
COCO_ANN_FILE = r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json'
BANK_PATH = r'C:\Users\张三\Desktop\实验数据\exemplar_bank_k5_dinov2_layers.pt'
OUT = r'C:\Users\张三\Desktop\实验数据\cross_domain_inventory.json'

DIOR_ID2NAME = {
    0: 'airplane', 1: 'airport', 2: 'baseball field', 3: 'basketball court',
    4: 'bridge', 5: 'chimney', 6: 'dam', 7: 'expressway service area',
    8: 'expressway toll station', 9: 'golf course', 10: 'ground track field',
    11: 'harbor', 12: 'overpass', 13: 'ship', 14: 'stadium',
    15: 'storage tank', 16: 'tennis court', 17: 'train station',
    18: 'vehicle', 19: 'windmill',
}

# ── 1. DIOR 各类图数 + GT 覆盖（在 full20_images 测试图内） ──
test_ids = set()
for f in glob.glob(os.path.join(DIOR_IMG_DIR, '*.jpg')):
    parts = Path(f).stem.split('_')
    for part in reversed(parts):
        if part.isdigit():
            test_ids.add(int(part))
            break

df = pd.read_parquet(DIOR_PARQUET)
gt_by_img = defaultdict(list)          # img_id -> [(cls_id, bbox)]
for _, row in df.iterrows():
    iid = int(row['image_id'])
    if iid not in test_ids:
        continue
    for i, cat in enumerate(row['objects']['category']):
        gt_by_img[iid].append((int(cat), row['objects']['bbox'][i]))

dior_img_count = defaultdict(int)      # 类名 -> (测试图数, GT框数)
dior_gt_count = defaultdict(int)
for f in glob.glob(os.path.join(DIOR_IMG_DIR, '*.jpg')):
    stem = Path(f).stem
    parts = stem.split('_')
    cls = stem.rsplit('_', 1)[0].replace('_', ' ')
    if cls not in DIOR_ID2NAME.values():
        # 带空格类名在文件名里是下划线，回退到映射
        cls = stem.rsplit('_', 1)[0]
        cls = {v.replace(' ', '_'): v for v in DIOR_ID2NAME.values()}.get(cls, cls)
    img_id = None
    for part in reversed(parts):
        if part.isdigit():
            img_id = int(part)
            break
    dior_img_count[cls] += 1
    if img_id is not None:
        dior_gt_count[cls] += len(gt_by_img.get(img_id, []))

# ── 2. COCO 各类本地图数（完整统计） ──
from pycocotools.coco import COCO
coco = COCO(COCO_ANN_FILE)
cats = coco.loadCats(coco.getCatIds())
coco_count = {}
for c in cats:
    iids = coco.getImgIds(catIds=[c['id']])
    n_local = 0
    for iid in iids:
        info = coco.loadImgs([iid])[0]
        if os.path.exists(os.path.join(COCO_IMG_DIR, info['file_name'])):
            n_local += 1
    coco_count[c['name']] = (len(iids), n_local)

# ── 3. 语义映射分级 ──
# 强: 同一实体，俯视(源) vs 自然场景(目标)，语义严格对齐
# 中: 语义相关但粒度/范围不同（区域级 -> 物体级）
# 弱: 实体不同/无对应
SEMANTIC_MAP = {
    'airplane': {'strong': ['airplane'], 'weak': ['airport']},
    'ship':     {'strong': ['boat'], 'weak': ['harbor']},
    'vehicle':  {'strong': ['car', 'truck', 'bus', 'motorcycle'], 'weak': ['bicycle', 'train']},
    'train station': {'medium': ['train'], 'weak': []},
    'tennis court':  {'medium': ['sports ball', 'tennis racket'], 'weak': []},
    'baseball field': {'medium': ['sports ball', 'baseball bat', 'baseball glove'], 'weak': []},
    'basketball court': {'medium': ['sports ball'], 'weak': []},
    'golf course':     {'medium': ['sports ball'], 'weak': []},
    'storage tank': {'strong': [], 'weak': []},
    'windmill':     {'strong': [], 'weak': []},
    'bridge': {'weak': []}, 'dam': {'weak': []}, 'chimney': {'weak': []},
    'overpass': {'weak': []}, 'stadium': {'weak': []},
    'expressway service area': {'weak': []}, 'expressway toll station': {'weak': []},
    'ground track field': {'weak': []}, 'harbor': {'weak': []}, 'airport': {'weak': ['airplane']},
}

bank = torch.load(BANK_PATH, map_location='cpu')

report = {'dior': {}, 'coco': {}, 'pairs': [], 'candidates': []}

for cls in DIOR_ID2NAME.values():
    report['dior'][cls] = {'images': dior_img_count.get(cls, 0),
                           'gt_boxes': dior_gt_count.get(cls, 0),
                           'has_bank_proto': cls in bank}

for name, (tot, local) in sorted(coco_count.items(), key=lambda x: -x[1][1]):
    report['coco'][name] = {'total_imgs': tot, 'local_imgs': local}

# 已有跨域对
for dior_cls, targets in [('airplane', ['airplane']), ('ship', ['boat']), ('vehicle', ['car'])]:
    report['pairs'].append({'src': f'DIOR-{dior_cls}', 'tgt': f'COCO-{targets[0]}',
                            'grade': 'strong', 'status': '已用'})

# 候选扩类：strong 优先，medium 备选；判定可部署性（COCO 本地图数 >= K=10 + 测试30）
for dior_cls, m in SEMANTIC_MAP.items():
    for grade in ('strong', 'medium', 'weak'):
        for tgt in m.get(grade, []):
            if tgt not in coco_count:
                continue
            tot, local = coco_count[tgt]
            deploy = local >= 40  # few-shot K=10 + 测试 >=30
            report['candidates'].append({
                'src': f'DIOR-{dior_cls}', 'tgt': f'COCO-{tgt}',
                'grade': grade, 'coco_imgs': local, 'deployable': deploy,
                'status': '已用' if (dior_cls in ('airplane', 'ship', 'vehicle')
                                    and tgt in ('airplane', 'boat', 'car') and grade == 'strong') else '可扩'
            })

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# ── 控制台摘要 ──
print('=' * 78)
print('DIOR full20_images 各类（图数 / GT框数 / bank原型）')
print('-' * 78)
for cls, d in report['dior'].items():
    flag = '✅' if (d['images'] >= 20 and d['gt_boxes'] > 0 and d['has_bank_proto']) else '⚠️'
    print(f"  {flag} {cls:<24s} 图 {d['images']:>3d}  GT {d['gt_boxes']:>4d}  原型 {d['has_bank_proto']}")

print()
print('=' * 78)
print('跨域对候选（按映射强度）')
print('-' * 78)
for c in sorted(report['candidates'], key=lambda x: (0 if x['grade'] == 'strong' else 1, -x['coco_imgs'])):
    dep = '可部署' if c['deployable'] else '图不足'
    print(f"  [{c['grade']:<7s}] {c['src']:<22s} → {c['tgt']:<18s} COCO本地图 {c['coco_imgs']:>4d}  {dep:>4s}  {c['status']}")

print()
print(f"COCO 本地图 >= 100 的类数: {sum(1 for v in report['coco'].values() if v['local_imgs'] >= 100)}")
print(f"结果 → {OUT}")
