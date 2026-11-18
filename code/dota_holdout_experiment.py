# -*- coding: utf-8 -*-
"""dota_holdout_experiment.py — DOTA留出域实验（预登记验证）
验证P1: small vehicle增益 > large vehicle增益
验证P2: 门控开启类增益 > 门控关闭类增益
验证P3: vp>0.5类增益 > vp≤0.5类增益
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import tge_probe as T
from sota_comparison import (
    SOURCE_LOADERS, build_proto_tge, evaluate_with_features,
    SRC_TMPL_OVERHEAD, TGT_TMPL,
)
import unified_eval as U

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
DOTA_ROOT = Path(r'C:\Users\张三\modeling\dota_data\DOTA_V1.5')

# 统一α网格到10.0（与拟合数据完全一致）
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# DOTA 11个类别（预登记第4节）
DOTA_CLASSES = [
    'small-vehicle', 'large-vehicle', 'plane', 'ship', 'storage-tank',
    'bridge', 'harbor', 'baseball-diamond', 'tennis-court',
    'basketball-court', 'ground-track-field'
]

# 预登记的物理属性（第4节）
DOTA_ATTRIBUTES = {
    'small-vehicle': {'size': 4.5, 'vp': 0.7, 'gate': 1},
    'large-vehicle': {'size': 12.0, 'vp': 0.8, 'gate': 0},
    'plane': {'size': 30.0, 'vp': 0.3, 'gate': 0},
    'ship': {'size': 15.0, 'vp': 0.4, 'gate': 0},
    'storage-tank': {'size': 20.0, 'vp': 0.2, 'gate': 0},
    'bridge': {'size': 100.0, 'vp': 0.2, 'gate': 0},
    'harbor': {'size': 500.0, 'vp': 0.2, 'gate': 0},
    'baseball-diamond': {'size': 100.0, 'vp': 0.1, 'gate': 0},
    'tennis-court': {'size': 24.0, 'vp': 0.1, 'gate': 0},
    'basketball-court': {'size': 28.0, 'vp': 0.1, 'gate': 0},
    'ground-track-field': {'size': 200.0, 'vp': 0.1, 'gate': 0},
}

# DOTA源域类别映射（DOTA源域 -> DOTA目标域）
DOTA_SRC_MAP = {
    'small-vehicle': 'small-vehicle',
    'large-vehicle': 'large-vehicle',
    'plane': 'plane',
    'ship': 'ship',
    'storage-tank': 'storage-tank',
    'bridge': 'bridge',
    'harbor': 'harbor',
    'baseball-diamond': 'baseball-diamond',
    'tennis-court': 'tennis-court',
    'basketball-court': 'basketball-court',
    'ground-track-field': 'ground-track-field',
}

MAX_IMG_SIZE = 1024  # DOTA大图resize到这个尺寸再检测


def parse_dota_label(label_path):
    """解析DOTA标注文件，返回[(x1,y1,x2,y2,cls)]，OBB转AABB"""
    boxes = []
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('imagesource') or line.startswith('gsd'):
                continue
            parts = line.split()
            if len(parts) >= 9:
                xs = [float(parts[i]) for i in range(0, 8, 2)]
                ys = [float(parts[i]) for i in range(1, 8, 2)]
                cls = parts[8]
                boxes.append((min(xs), min(ys), max(xs), max(ys), cls))
    return boxes


def load_dota_images(cls_name, n=50, split='val'):
    """加载DOTA数据集中包含指定类别的图片"""
    img_dir = DOTA_ROOT / split / 'images'
    lab_dir = DOTA_ROOT / split / 'labelTxt-v1.5' / f'DOTA-v1.5_{split}'

    images = []
    gts = []
    for img_path in sorted(img_dir.glob('*.png')):
        lab_path = lab_dir / (img_path.stem + '.txt')
        if not lab_path.exists():
            continue
        boxes = parse_dota_label(lab_path)
        cls_boxes = [b[:4] for b in boxes if b[4] == cls_name]
        if not cls_boxes:
            continue
        # 过滤太小的框（<10px）
        cls_boxes = [b for b in cls_boxes if (b[2]-b[0]) > 10 and (b[3]-b[1]) > 10]
        if not cls_boxes:
            continue
        images.append(str(img_path))
        gts.append(cls_boxes)
        if len(images) >= n:
            break
    return images, gts


def resize_image(img, max_size=MAX_IMG_SIZE):
    """resize图片到max_size，保持宽高比，同时缩放GT框"""
    w, h = img.size
    scale = min(max_size / w, max_size / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        scale = 1.0
    return img, scale


def load_target_dota(dota_cls, n=50):
    """加载DOTA目标域数据，运行GDINO检测，提取CLIP特征"""
    images_raw, gts_raw = load_dota_images(dota_cls, n=n)
    if not images_raw:
        print(f'    未找到包含 {dota_cls} 的图片')
        return None

    # resize图片和GT
    images = []
    gts = []
    for img_path, gt_boxes in zip(images_raw, gts_raw):
        img = Image.open(img_path).convert('RGB')
        img, scale = resize_image(img)
        images.append(img)
        scaled_gts = [[b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale] for b in gt_boxes]
        gts.append(scaled_gts)

    # GDINO检测（带缓存）
    cache_p = DATA / f'dota_holdout_dets_{dota_cls}.pkl'
    if cache_p.exists():
        import pickle
        dets = pickle.load(open(cache_p, 'rb'))
        print(f'    加载检测缓存: {cache_p}')
    else:
        print(f'    运行GDINO检测（{len(images)}张图）...')
        dets = U.gdino_detect(images, dota_cls.replace('-', ' '), 0.0)
        import pickle
        pickle.dump(dets, open(cache_p, 'wb'))
        print(f'    检测缓存已保存: {cache_p}')

    # 提取CLIP特征
    per_img = []
    crops_all = []
    PRE_FILTER = 0.05  # 预过滤阈值

    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2])
            sl.append(float(sc))
            ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })

    F = T.encode_clip_batch(crops_all).cpu().numpy() if crops_all else np.zeros((0, 512))
    return {'per_img': per_img, 'features': F, 'n_crops': len(crops_all)}


def process_dota_sample(dota_cls):
    """处理单个DOTA类别的α网格搜索"""
    print(f'\n{"="*60}')
    print(f'处理 DOTA 类别: {dota_cls}')
    print(f'{"="*60}')

    attr = DOTA_ATTRIBUTES[dota_cls]
    src_cls = DOTA_SRC_MAP[dota_cls]

    # 加载DOTA源域原型
    print(f'  加载源域原型: DOTA/{src_cls}')
    src_loader = SOURCE_LOADERS['DOTA'][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        print(f'    源域数据加载失败，跳过')
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']

    # 负原型（源域中最不相似的原型）
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

    # 加载DOTA目标域数据
    print(f'  加载目标域数据: DOTA/{dota_cls}')
    target_data = load_target_dota(dota_cls, n=50)
    if target_data is None:
        print(f'    目标域数据加载失败，跳过')
        return None
    print(f'    图片数: {len(target_data["per_img"])}, 候选框数: {target_data["n_crops"]}')

    # α网格搜索
    print(f'  α网格搜索（{len(ALPHAS)}个值）...')
    ap_curve = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            proto = src_proto
        else:
            proto = build_proto_tge(src_proto, src_cls, dota_cls.replace('-', ' '), beta=alpha)
        ap = evaluate_with_features(target_data, proto, neg_proto=proto_neg)
        ap_curve[str(alpha)] = float(ap)
        print(f'    α={alpha:.1f}: AP={ap:.4f}')

    best_alpha = ALPHAS[0]
    best_ap = ap_curve[str(ALPHAS[0])]
    for alpha in ALPHAS:
        if ap_curve[str(alpha)] > best_ap:
            best_ap = ap_curve[str(alpha)]
            best_alpha = alpha

    baseline_ap = ap_curve['0.0']
    oracle_gain = best_ap - baseline_ap

    print(f'  结果: baseline={baseline_ap:.4f}, best_alpha={best_alpha}, '
          f'best_ap={best_ap:.4f}, oracle_gain={oracle_gain:+.4f}')

    return {
        'dota_cls': dota_cls,
        'src_cls': src_cls,
        'size': attr['size'],
        'vp': attr['vp'],
        'gate': attr['gate'],
        'baseline_ap': float(baseline_ap),
        'best_alpha': float(best_alpha),
        'best_ap': float(best_ap),
        'oracle_gain': float(oracle_gain),
        'ap_curve': ap_curve,
        'n_images': len(target_data['per_img']),
        'n_crops': target_data['n_crops'],
    }


def main():
    print('=' * 80)
    print('DOTA留出域实验（预登记验证）')
    print('=' * 80)
    print(f'类别数: {len(DOTA_CLASSES)}')
    print(f'α网格: {ALPHAS}')
    print(f'门控开启类: {[c for c in DOTA_CLASSES if DOTA_ATTRIBUTES[c]["gate"] == 1]}')
    print(f'门控关闭类: {[c for c in DOTA_CLASSES if DOTA_ATTRIBUTES[c]["gate"] == 0]}')

    results = []
    for dota_cls in DOTA_CLASSES:
        result = process_dota_sample(dota_cls)
        if result is not None:
            results.append(result)

    # 保存结果
    out_file = DATA / 'dota_holdout_experiment_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')

    # 汇总分析
    print('\n' + '=' * 80)
    print('汇总分析')
    print('=' * 80)

    # P1: small vehicle vs large vehicle
    small = [r for r in results if r['dota_cls'] == 'small-vehicle']
    large = [r for r in results if r['dota_cls'] == 'large-vehicle']
    if small and large:
        diff = small[0]['oracle_gain'] - large[0]['oracle_gain']
        print(f'\nP1 (small vs large vehicle):')
        print(f'  small-vehicle: oracle_gain={small[0]["oracle_gain"]:+.4f}')
        print(f'  large-vehicle: oracle_gain={large[0]["oracle_gain"]:+.4f}')
        print(f'  差值: {diff:+.4f}')
        print(f'  P1{"通过" if diff > 0 else "未通过"}')

    # P2: 门控开启类 vs 关闭类
    gate_on = [r for r in results if r['gate'] == 1]
    gate_off = [r for r in results if r['gate'] == 0]
    if gate_on and gate_off:
        mean_on = np.mean([r['oracle_gain'] for r in gate_on])
        mean_off = np.mean([r['oracle_gain'] for r in gate_off])
        print(f'\nP2 (门控开启 vs 关闭):')
        print(f'  开启类 ({len(gate_on)}): 平均oracle_gain={mean_on:+.4f}')
        print(f'  关闭类 ({len(gate_off)}): 平均oracle_gain={mean_off:+.4f}')
        print(f'  差值: {mean_on - mean_off:+.4f}')
        print(f'  P2{"通过" if mean_on > mean_off else "未通过"}')

    # P3: vp>0.5 vs vp≤0.5
    vp_high = [r for r in results if r['vp'] > 0.5]
    vp_low = [r for r in results if r['vp'] <= 0.5]
    if vp_high and vp_low:
        mean_high = np.mean([r['oracle_gain'] for r in vp_high])
        mean_low = np.mean([r['oracle_gain'] for r in vp_low])
        print(f'\nP3 (vp>0.5 vs vp≤0.5):')
        print(f'  vp>0.5类 ({len(vp_high)}): 平均oracle_gain={mean_high:+.4f}')
        print(f'  vp≤0.5类 ({len(vp_low)}): 平均oracle_gain={mean_low:+.4f}')
        print(f'  差值: {mean_high - mean_low:+.4f}')
        print(f'  P3{"通过" if mean_high > mean_low else "未通过"}')

    # 逐类结果表
    print(f'\n逐类结果:')
    print(f'{"类别":<20} {"gate":>5} {"size":>8} {"vp":>5} {"baseline":>10} {"best_alpha":>10} {"oracle_gain":>12}')
    print('-' * 80)
    for r in sorted(results, key=lambda x: -x['oracle_gain']):
        print(f'{r["dota_cls"]:<20} {r["gate"]:>5} {r["size"]:>8.1f} {r["vp"]:>5.1f} '
              f'{r["baseline_ap"]:>10.4f} {r["best_alpha"]:>10.1f} {r["oracle_gain"]:>+12.4f}')


if __name__ == '__main__':
    main()
