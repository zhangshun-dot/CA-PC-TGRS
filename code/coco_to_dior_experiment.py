# -*- coding: utf-8 -*-
"""coco_to_dior_experiment.py — COCO→DIOR（自然→遥感）实验
验证方法在自然图像源域→遥感目标域上的泛化能力。
视角差大、预期正增益、应用价值明确（无需遥感标注）。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import unified_eval as U
import tge_probe as T
from sota_comparison import (
    build_proto_tge, evaluate_with_features,
    SRC_TMPL_OVERHEAD, TGT_TMPL, PRE_FILTER,
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# COCO→DIOR类别对应
# COCO源域类别 → DIOR目标域类别
COCO_TO_DIOR_TARGETS = [
    ('airplane', 'airplane', 'airplane'),
    ('boat', 'ship', 'ship'),
    ('car', 'vehicle', 'vehicle'),
    ('bus', 'vehicle', 'vehicle (from bus)'),
    ('truck', 'vehicle', 'vehicle (from truck)'),
]

# 物理先验门控（专家引导的类级序数属性）
OBJECT_SIZE = {'airplane': 30.0, 'boat': 15.0, 'car': 4.5, 'bus': 12.0,
               'truck': 12.0, 'motorcycle': 2.0, 'bicycle': 1.8, 'person': 1.7}
VIEWPOINT_DIFF = {'airplane': 0.3, 'boat': 0.4, 'car': 0.7, 'bus': 0.8,
                   'truck': 0.75, 'motorcycle': 0.6, 'bicycle': 0.5, 'person': 0.9}

# α网格
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def get_gate(cls):
    """门控决策：size < 10 AND vp > 0.5"""
    return 1 if (OBJECT_SIZE.get(cls, 999) < 10.0 and VIEWPOINT_DIFF.get(cls, 0) > 0.5) else 0


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


def _extract_features(images, gts):
    """从图像和GT中提取CLIP特征"""
    if not images:
        return None
    crops = []
    for img, g in zip(images, gts):
        for gt in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in gt]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
    if not crops:
        return None
    F = T.encode_clip_batch(crops).cpu().numpy()
    proto = F.mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    return {'prototype': proto, 'features': F, 'n_crops': len(crops)}


def load_coco_src(cls_name, n=150):
    """COCO源域加载函数"""
    images, gts = U.load_coco(cls_name, n=n)
    return _extract_features(images, gts)


def load_target_dior(dior_cls, n=200):
    """DIOR目标域加载函数"""
    images, gts = U.load_dior(dior_cls, n=n)
    if not images:
        return None
    cache_p = DATA / f'tge_v2_dets_DIOR-{dior_cls}.pkl'
    if cache_p.exists():
        dets = pickle.load(open(cache_p, 'rb'))
    else:
        dets = U.gdino_detect(images, dior_cls, 0.0)
        pickle.dump(dets, open(cache_p, 'wb'))
    per_img = []
    crops_all = []
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


def run_experiment(coco_cls, dior_cls, display_name):
    """跑单个类别的实验"""
    print(f'\n{"─" * 80}')
    print(f'COCO/{coco_cls} → DIOR/{dior_cls} ({display_name})')
    print(f'{"─" * 80}')

    # 加载COCO源域数据
    print(f'  加载COCO源域数据...')
    src_data = load_coco_src(coco_cls, n=150)
    if src_data is None:
        print(f'  源域数据加载失败，跳过')
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']
    print(f'  源域原型: ||p||={np.linalg.norm(src_proto):.4f}, n_crops={src_data["n_crops"]}')

    # 负原型
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = normalize(proto_neg)

    # 加载DIOR目标域数据
    print(f'  加载DIOR目标域数据...')
    target_data = load_target_dior(dior_cls, n=200)
    if target_data is None:
        print(f'  目标域数据加载失败，跳过')
        return None
    print(f'  目标域: n_imgs={len(target_data["per_img"])}, n_crops={target_data["n_crops"]}')

    # α网格搜索（TGE算子）
    print(f'  α网格搜索（TGE算子）...')
    ap_curve = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            proto = src_proto
        else:
            proto = build_proto_tge(src_proto, coco_cls, dior_cls, beta=alpha)
        ap = evaluate_with_features(target_data, proto, neg_proto=proto_neg)
        ap_curve[str(alpha)] = float(ap)

    # 找最优α
    best_alpha = ALPHAS[0]
    best_ap = ap_curve[str(ALPHAS[0])]
    for alpha in ALPHAS:
        if ap_curve[str(alpha)] > best_ap:
            best_ap = ap_curve[str(alpha)]
            best_alpha = alpha

    baseline_ap = ap_curve['0.0']
    oracle_gain = best_ap - baseline_ap

    # 固定α=2.5的增益
    ap_25 = ap_curve.get('2.5', baseline_ap)
    gain_25 = ap_25 - baseline_ap

    # 门控决策
    gate = get_gate(coco_cls)

    # 门控后的增益（门控关闭则α=0）
    if gate == 1:
        gated_gain = gain_25
        gated_alpha = 2.5
    else:
        gated_gain = 0.0
        gated_alpha = 0.0

    result = {
        'src': 'COCO',
        'src_cls': coco_cls,
        'target_cls': dior_cls,
        'target_domain': 'DIOR',
        'display_name': display_name,
        'gate': gate,
        'baseline_ap': float(baseline_ap),
        'best_alpha': float(best_alpha),
        'best_ap': float(best_ap),
        'oracle_gain': float(oracle_gain),
        'gain_25': float(gain_25),
        'gated_gain': float(gated_gain),
        'gated_alpha': float(gated_alpha),
        'ap_curve': ap_curve,
    }

    print(f'  baseline AP50: {baseline_ap:.4f}')
    print(f'  最优α: {best_alpha}, 最优AP50: {best_ap:.4f}, Oracle增益: {oracle_gain:+.4f}')
    print(f'  固定α=2.5增益: {gain_25:+.4f}')
    print(f'  门控: {"开" if gate == 1 else "关"}, 门控后增益: {gated_gain:+.4f}')

    return result


def main():
    print('=' * 80)
    print('COCO→DIOR（自然→遥感）实验')
    print('验证方法在自然图像源域→遥感目标域上的泛化能力')
    print('=' * 80)

    print(f'\n类别对应:')
    for coco_cls, dior_cls, display_name in COCO_TO_DIOR_TARGETS:
        gate = get_gate(coco_cls)
        print(f'  COCO/{coco_cls} → DIOR/{dior_cls} ({display_name}), gate={"开" if gate == 1 else "关"}')

    # 跑每个类别
    all_results = []
    for coco_cls, dior_cls, display_name in COCO_TO_DIOR_TARGETS:
        result = run_experiment(coco_cls, dior_cls, display_name)
        if result is not None:
            all_results.append(result)

    # 汇总
    print(f'\n{"=" * 80}')
    print('汇总结果')
    print(f'{"=" * 80}')

    print(f'\n{"类别":<25} {"gate":>5} {"baseline":>10} {"best_alpha":>10} {"oracle":>10} {"gain_25":>10} {"gated":>10}')
    print('-' * 90)

    for r in all_results:
        print(f'{r["display_name"]:<25} {r["gate"]:>5} {r["baseline_ap"]:>10.4f} '
              f'{r["best_alpha"]:>10.1f} {r["oracle_gain"]:>+10.4f} '
              f'{r["gain_25"]:>+10.4f} {r["gated_gain"]:>+10.4f}')

    # 统计
    print(f'\n统计:')
    oracle_gains = [r['oracle_gain'] for r in all_results]
    gain_25s = [r['gain_25'] for r in all_results]
    gated_gains = [r['gated_gain'] for r in all_results]

    print(f'  平均Oracle增益: {np.mean(oracle_gains):+.4f}')
    print(f'  平均固定α=2.5增益: {np.mean(gain_25s):+.4f}')
    print(f'  平均门控后增益: {np.mean(gated_gains):+.4f}')
    print(f'  正例比例（固定α=2.5）: {np.mean([1 if g > 0.001 else 0 for g in gain_25s]):.1%}')
    print(f'  正例比例（门控后）: {np.mean([1 if g > 0.001 else 0 for g in gated_gains]):.1%}')

    # 门控开启 vs 关闭
    on_results = [r for r in all_results if r['gate'] == 1]
    off_results = [r for r in all_results if r['gate'] == 0]

    if on_results:
        print(f'\n门控开启类（{len(on_results)}个）:')
        for r in on_results:
            print(f'  {r["display_name"]}: baseline={r["baseline_ap"]:.4f}, '
                  f'oracle={r["oracle_gain"]:+.4f}, gain_25={r["gain_25"]:+.4f}')
        print(f'  平均固定α=2.5增益: {np.mean([r["gain_25"] for r in on_results]):+.4f}')

    if off_results:
        print(f'\n门控关闭类（{len(off_results)}个）:')
        for r in off_results:
            print(f'  {r["display_name"]}: baseline={r["baseline_ap"]:.4f}, '
                  f'oracle={r["oracle_gain"]:+.4f}, gain_25={r["gain_25"]:+.4f}')
        print(f'  平均固定α=2.5增益: {np.mean([r["gain_25"] for r in off_results]):+.4f}')

    # 保存结果
    out_file = DATA / 'coco_to_dior_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
