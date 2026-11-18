# -*- coding: utf-8 -*-
"""dota_inventory.py — Gate 1：DIOR→DOTA 严格跨域对清点（纯 CPU，本地 DOTA V1.5）

回答三个问题：
  1. DOTA V1.5（本地 train 1411 + val 458）能贡献多少个严格 DIOR→DOTA 跨域对？
  2. 与现有 DIOR→COCO（6 对 / 3 源概念）合计后，源概念数、目标域数、总对数是多少？
  3. 若按每类 ~100 张抽样做检测+编码缓存，GPU 量级估算（含 DOTA 大图切块）？

同时输出 DOTA 类的逐类图/框分布（10px AABB 过滤后），供写「双轴诊断」与
VLM 方向扩数据决策用。

OBB→AABB：DOTA 是旋转框（8 坐标），本流水线用轴对齐框 → 取多边形 min/max。
"""
import os
import json
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

DOTA_ROOT = Path(r'C:\Users\张三\modeling\dota_data\DOTA_V1.5')
SPLITS = ['train', 'val']
MIN_SIDE = 10  # 与冻结协议一致：小于 10px 的框过滤

# DIOR 20 类 → DOTA 类映射（严格语义对齐）
DIOR2DOTA = {
    'airplane': ['plane'],
    'ship': ['ship'],
    'vehicle': ['large-vehicle', 'small-vehicle'],
    'storage tank': ['storage-tank'],
    'bridge': ['bridge'],
    'harbor': ['harbor'],
    'baseball field': ['baseball-diamond'],
    'basketball court': ['basketball-court'],
    'ground track field': ['ground-track-field'],
    'tennis court': ['tennis-court'],
}
DIOR_CLASSES = ['airplane', 'airport', 'baseball field', 'basketball court', 'bridge',
                'chimney', 'dam', 'expressway service area', 'expressway toll station',
                'golf course', 'ground track field', 'harbor', 'overpass', 'ship',
                'stadium', 'storage tank', 'tennis court', 'train station', 'vehicle', 'windmill']


def parse_label(path):
    """返回 [(x1,y1,x2,y2,cls,diff)]，OBB 转 AABB。跳过 8 坐标退化为 4 坐标（无旋转）的情况。"""
    boxes = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('imagesource') or line.startswith('gsd'):
            continue
        parts = line.split()
        if len(parts) == 10:
            x1, y1, x2, y2, x3, y3, x4, y4, cls, diff = parts
            xs = [float(v) for v in (x1, x2, x3, x4)]
            ys = [float(v) for v in (y1, y2, y3, y4)]
            boxes.append((min(xs), min(ys), max(xs), max(ys), cls, int(diff)))
        elif len(parts) == 6:  # 已退化的 HBB
            x1, y1, x2, y2, cls, diff = parts
            boxes.append((float(x1), float(y1), float(x2), float(y2), cls, int(diff)))
    return boxes


def main():
    per_split = {}
    total = Counter()
    all_images_with_boxes = 0
    img_sizes = []
    for sp in SPLITS:
        img_dir = DOTA_ROOT / sp / 'images'
        lab_dir = DOTA_ROOT / sp / 'labelTxt-v1.5'
        lab_sub = f'DOTA-v1.5_{sp}'
        img_paths = sorted(img_dir.glob('*.png'))
        class_imgs = defaultdict(int)     # cls -> #图(含框)
        class_boxes = Counter()           # cls -> #框
        class_boxes_10 = Counter()        # cls -> #框(10px 过滤后)
        for img_path in img_paths:
            lab = lab_dir / lab_sub / (img_path.stem + '.txt')
            if not lab.exists():
                continue
            boxes = parse_label(lab)
            if not boxes:
                continue
            all_images_with_boxes += 1
            img_sizes.append(Image.open(img_path).size)
            seen = set()
            for (x1, y1, x2, y2, cls, diff) in boxes:
                class_boxes[cls] += 1
                seen.add(cls)
                w = x2 - x1
                h = y2 - y1
                if min(w, h) >= MIN_SIDE:
                    class_boxes_10[cls] += 1
            for c in seen:
                class_imgs[c] += 1
        per_split[sp] = {
            'images': len(img_paths),
            'classes': sorted(set(class_boxes.keys())),
            'class_imgs': dict(class_imgs),
            'class_boxes': dict(class_boxes),
            'class_boxes_10': dict(class_boxes_10),
        }
        for c in class_boxes:
            total[c] += class_boxes[c]

    # ── DIOR→DOTA 严格对 ──
    pairs = []
    used_dior = []
    for dior, dota_cls in DIOR2DOTA.items():
        for dc in dota_cls:
            tr = per_split['train']['class_imgs'].get(dc, 0)
            va = per_split['val']['class_imgs'].get(dc, 0)
            n_tr = per_split['train']['class_boxes_10'].get(dc, 0)
            n_va = per_split['val']['class_boxes_10'].get(dc, 0)
            pairs.append({
                'dior_src': dior, 'dota_tgt': dc,
                'train_imgs': tr, 'val_imgs': va,
                'train_boxes10': n_tr, 'val_boxes10': n_va,
                'total_imgs': tr + va,
            })
            used_dior.append(dior)

    # ── 算力估算：每类 ~100 张，DOTA 大图切 1024×1024 ──
    TILE = 1024
    sampled = {p['dota_tgt']: p for p in pairs}
    est = {}
    per_img_tiles = []
    for (w, h) in img_sizes:
        per_img_tiles.append(((w + TILE - 1) // TILE) * ((h + TILE - 1) // TILE))
    avg_tiles = sum(per_img_tiles) / max(1, len(per_img_tiles))
    # 保守：GDINO+DINOv2 每 1024 tile ~0.4s（RTX 5060）
    for p in pairs:
        n = min(100, p['total_imgs'])
        tiles = int(n * avg_tiles)
        est[p['dota_tgt']] = {'sample_imgs': n, 'est_tiles': tiles,
                              'est_min': round(tiles * 0.4 / 60, 1)}

    total_pairs = len(pairs)
    n_concepts = len(set(used_dior))
    total_est_min = sum(e['est_min'] for e in est.values())

    # ── 报告 ──
    L = []
    L.append('## Gate 1 — DOTA V1.5 清点（本地，纯 CPU）\n')
    L.append(f'数据源：DOTA V1.5 train {per_split["train"]["images"]} 图 / val '
             f'{per_split["val"]["images"]} 图（含框 {all_images_with_boxes} 图）。'
             f'平均切块数（1024px）≈ {avg_tiles:.1f}/图。\n')
    L.append('### 1. DOTA 类分布（train+val）\n')
    L.append('| 类 | 图数(train) | 图数(val) | 框(train,10px) | 框(val,10px) |')
    L.append('|---|---|---|---|---|')
    for c in sorted(total.keys()):
        L.append(f'| {c} | {per_split["train"]["class_imgs"].get(c,0)} | '
                 f'{per_split["val"]["class_imgs"].get(c,0)} | '
                 f'{per_split["train"]["class_boxes_10"].get(c,0)} | '
                 f'{per_split["val"]["class_boxes_10"].get(c,0)} |')
    L.append('')
    L.append(f'### 2. DIOR→DOTA 严格对：**{total_pairs} 对 / {n_concepts} 个 DIOR 源概念**\n')
    L.append('| DIOR 源 | DOTA 目标 | 图数(tr+va) | 框10px(tr+va) | 抽样100张估算(tile/分钟) |')
    L.append('|---|---|---|---|---|')
    for p in pairs:
        e = est[p['dota_tgt']]
        L.append(f'| {p["dior_src"]} | {p["dota_tgt"]} | {p["total_imgs"]} | '
                 f'{p["train_boxes10"]+p["val_boxes10"]} | '
                 f'{e["sample_imgs"]}张/{e["est_tiles"]}tile/{e["est_min"]}min |')
    L.append('')
    L.append(f'**合计：抽样 {len(pairs)*100} 张目标图，估算 GPU 约 {round(total_est_min)} 分钟** '
             f'（detect+encode 一次性落盘缓存，之后全部分析纯 CPU）。\n')
    L.append('### 3. 与现有 DIOR→COCO 合并\n')
    L.append('| 维度 | 现有（DIOR→COCO） | 新增（DIOR→DOTA） | 合计 |')
    L.append('|---|---|---|---|')
    L.append('| 严格语义对 | 6 | ' + str(total_pairs) + ' | ' + str(6 + total_pairs) + ' |')
    L.append('| DIOR 源概念 | 3 (airplane/ship/vehicle) | ' + str(n_concepts) + ' | '
             + str(len(set(['airplane', 'ship', 'vehicle'] + used_dior))) + ' |')
    L.append('| 目标域 | 1 (COCO) | 1 (DOTA) | 2 |')
    L.append('| 域变化性质 | 俯视→侧视(视角+域) | 俯视→俯视(纯域) | 双轴 |')
    L.append('| 粒度轴 | vehicle→car/truck/bus/moto | vehicle→large/small-vehicle | 两条 |')
    L.append('')
    L.append('### 4. 说明与注意\n')
    L.append('- DOTA 是旋转框(OBB)，已转 AABB(min/max)，供原型建库足够；定位仍走 GDINO 检测框。')
    L.append('- DOTA 图像最大 3875×5502，需 1024 切块；抽样时优先取 <2000px 小图可省一半算力。')
    L.append('- 未匹配的 DOTA 类（helicopter/roundabout/soccer-ball-field/swimming-pool/'
             'container-crane）：DIOR 无对应源类，不可用。')
    L.append('- **纯域轴(DOTA) vs 视角+域轴(COCO) 双目标域由此成立** → 可写'
             '"few-shot 校准增益集中在视角变化还是域变化" 的双轴诊断。')
    L.append('')

    out_dir = Path(r'C:\Users\张三\Desktop\实验数据')
    (out_dir / 'dota_inventory_report.md').write_text('\n'.join(L), encoding='utf-8')
    summary = {
        'source': 'DOTA_V1.5 local train+val',
        'splits': per_split,
        'avg_tiles_per_image': avg_tiles,
        'dior2dota_pairs': pairs,
        'n_pairs': total_pairs,
        'n_dior_concepts': n_concepts,
        'compute_est': est,
        'total_est_min': round(total_est_min, 1),
    }
    (out_dir / 'dota_inventory.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n'.join(L))
    print(f'\n✅ 报告 → {out_dir / "dota_inventory_report.md"}')


if __name__ == '__main__':
    main()
