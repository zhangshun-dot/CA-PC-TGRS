# -*- coding: utf-8 -*-
"""centroid_shrink_experiment.py — 质心收缩方向验证实验
验证：反向（细→粗）时，用视觉质心方向替代文本视角方向，能否让完整校正转正。

根因：A-CSPC的外推本质是"平移"，适用于粗→细（质心→圆周）；
反向细→粗（圆周→质心）需要的是"聚合"，平移表达不了聚合。

升级方案：
- 正向（粗→细）：p̂_c = p_c + β · d_text(c)          # 文本细化方向，向外推
- 反向（细→粗）：p̂_c = p_c + β' · (c_coarse − p_c)   # 视觉质心方向，向内收
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import tge_probe as T
from sota_comparison import evaluate_with_features, PRE_FILTER, LAM
import unified_eval as U

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 多模板集合（用于对比）──
SRC_TEMPLATES_PHOTO = [
    'a photo of {}',
    'a natural image of {}',
    'a ground-level view of {}',
    'a side view of {}',
]

TGT_TEMPLATES_OVERHEAD = [
    'an overhead image of {}',
    'an aerial view of {}',
    'a satellite image of {}',
    'a top-down view of {}',
]

SRC_TEMPLATE_SINGLE = 'a photo of {}'
TGT_TEMPLATE_SINGLE = 'an overhead image of {}'

# COCO→DIOR类别对应
# 注意：vehicle组包含car、bus、truck、motorcycle，需要聚合
COCO_TO_DIOR_TARGETS = [
    ('airplane', 'airplane', 'airplane', 'airplane'),  # (源类, 目标类, 显示名, 组名)
    ('boat', 'ship', 'boat→ship', 'ship'),
    ('car', 'vehicle', 'car→vehicle', 'vehicle'),
    ('bus', 'vehicle', 'bus→vehicle', 'vehicle'),
    ('truck', 'vehicle', 'truck→vehicle', 'vehicle'),
    ('motorcycle', 'vehicle', 'motorcycle→vehicle', 'vehicle'),
]

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


def build_proto_text_direction(src_proto, src_cls, tgt_cls, beta=1.0, multi=True):
    """文本视角方向外推（原A-CSPC）"""
    if multi:
        s_emb = T.text_avg(SRC_TEMPLATES_PHOTO, src_cls).cpu().numpy()
        d_emb = T.text_avg(TGT_TEMPLATES_OVERHEAD, tgt_cls).cpu().numpy()
    else:
        s_emb = T.text_avg([SRC_TEMPLATE_SINGLE], src_cls).cpu().numpy()
        d_emb = T.text_avg([TGT_TEMPLATE_SINGLE], tgt_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def build_proto_centroid_shrink(src_proto, centroid, beta=1.0):
    """质心收缩方向（新增）：p̂ = p + β · (c − p)"""
    delta = centroid - src_proto
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def build_target_text_neg_proto(tgt_cls, all_tgt_classes):
    """构建目标域文本负原型"""
    tgt_emb = T.text_avg(TGT_TEMPLATES_OVERHEAD, tgt_cls).cpu().numpy()
    other_embs = {}
    for c in all_tgt_classes:
        if c == tgt_cls:
            continue
        other_embs[c] = T.text_avg(TGT_TEMPLATES_OVERHEAD, c).cpu().numpy()
    sims = {c: float(np.dot(tgt_emb, e)) for c, e in other_embs.items()}
    top3 = sorted(sims.items(), key=lambda x: -x[1])[:3]
    neg = np.mean([other_embs[c] for c, _ in top3], axis=0)
    return neg / (np.linalg.norm(neg) + 1e-8)


def load_coco_src(cls_name, n=150):
    """加载COCO源域数据"""
    images, gts = U.load_coco(cls_name, n=n)
    if not images:
        return None
    crops_all = []
    for img, g in zip(images, gts):
        for box in g:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            crops_all.append(img.crop((x1, y1, x2, y2)))
    if not crops_all:
        return None
    F = T.encode_clip_batch(crops_all).cpu().numpy()
    proto = F.mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    return {'prototype': proto, 'features': F, 'n_crops': len(crops_all)}


def load_target_dior(dior_cls, n=200):
    """加载DIOR目标域数据"""
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


def main():
    print('=' * 80)
    print('质心收缩方向验证实验（COCO→DIOR反向）')
    print('验证：细→粗迁移时，视觉质心方向能否替代文本视角方向')
    print('=' * 80)

    # ── 第一步：加载所有COCO源域原型，计算各组的质心 ──
    print(f'\n第一步：加载COCO源域原型，计算各组质心')
    print(f'{"─" * 80}')

    src_protos = {}
    src_features = {}
    for coco_cls, _, _, _ in COCO_TO_DIOR_TARGETS:
        if coco_cls in src_protos:
            continue
        print(f'  加载 COCO/{coco_cls}...')
        src_data = load_coco_src(coco_cls, n=150)
        if src_data is not None:
            src_protos[coco_cls] = src_data['prototype']
            src_features[coco_cls] = src_data['features']
            print(f'    ||p||={np.linalg.norm(src_data["prototype"]):.4f}, n_crops={src_data["n_crops"]}')

    # 计算各组质心
    groups = {}
    for coco_cls, dior_cls, display_name, group_name in COCO_TO_DIOR_TARGETS:
        if group_name not in groups:
            groups[group_name] = []
        if coco_cls in src_protos:
            groups[group_name].append(coco_cls)

    centroids = {}
    for group_name, cls_list in groups.items():
        if len(cls_list) >= 2:
            centroid = np.mean([src_protos[c] for c in cls_list], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids[group_name] = centroid
            print(f'\n  组 {group_name} 质心（{len(cls_list)}个类）:')
            for c in cls_list:
                sim = float(np.dot(src_protos[c], centroid))
                print(f'    {c}: cos(centroid)={sim:.4f}')
        else:
            print(f'\n  组 {group_name} 只有1个类，无聚合需求')

    # ── 第二步：对每个类别跑实验，对比三种方向 ──
    print(f'\n\n第二步：对比三种方向（α=2.5）')
    print(f'{"─" * 80}')
    print(f'  A: 单模板文本方向（基线）')
    print(f'  B: 多模板文本方向（方向1）')
    print(f'  C: 质心收缩方向（新增，攻击根因）')
    print(f'  D: 质心收缩 + 目标域负原型（方向2+质心）')

    all_dior_classes = ['airplane', 'ship', 'vehicle', 'storage-tank', 'baseball-diamond',
                         'tennis-court', 'basketball-court', 'ground-track-field',
                         'harbor', 'bridge', 'chimney']

    all_results = []

    for coco_cls, dior_cls, display_name, group_name in COCO_TO_DIOR_TARGETS:
        if coco_cls not in src_protos:
            continue

        print(f'\n{"─" * 80}')
        print(f'COCO/{coco_cls} → DIOR/{dior_cls} ({display_name})')
        print(f'{"─" * 80}')

        src_proto = src_protos[coco_cls]
        src_feat = src_features[coco_cls]

        # 源域负原型
        sims = src_feat @ src_proto
        neg_idx = np.argmin(sims)
        src_neg_proto = src_feat[neg_idx]
        src_neg_proto = normalize(src_neg_proto)

        # 目标域文本负原型
        tgt_neg_proto = build_target_text_neg_proto(dior_cls, all_dior_classes)

        # 质心
        centroid = centroids.get(group_name, None)
        if centroid is not None:
            shrink_dir = centroid - src_proto
            print(f'  质心收缩方向: ||d||={np.linalg.norm(shrink_dir):.4f}, '
                  f'cos(p, d)={float(np.dot(src_proto, normalize(shrink_dir))):.4f}')
        else:
            print(f'  无聚合需求（一对一），质心方向=0')

        # 加载DIOR目标域数据
        print(f'  加载DIOR目标域数据...')
        target_data = load_target_dior(dior_cls, n=200)
        if target_data is None:
            continue
        print(f'  目标域: n_imgs={len(target_data["per_img"])}, n_crops={target_data["n_crops"]}')

        # 对比三种方向
        results = {
            'src_cls': coco_cls,
            'target_cls': dior_cls,
            'display_name': display_name,
            'group_name': group_name,
            'has_centroid': centroid is not None,
            'A_single_text': {},
            'B_multi_text': {},
            'C_centroid_shrink': {},
            'D_centroid_tgt_neg': {},
        }

        print(f'\n  α    A单模板    B多模板    C质心收缩  D质心+目标负')
        print(f'  {"─" * 60}')

        for alpha in ALPHAS:
            # A: 单模板文本方向
            if alpha == 0.0:
                proto_A = src_proto
            else:
                proto_A = build_proto_text_direction(src_proto, coco_cls, dior_cls, beta=alpha, multi=False)

            # B: 多模板文本方向
            if alpha == 0.0:
                proto_B = src_proto
            else:
                proto_B = build_proto_text_direction(src_proto, coco_cls, dior_cls, beta=alpha, multi=True)

            # C: 质心收缩方向
            if alpha == 0.0 or centroid is None:
                proto_C = src_proto
            else:
                proto_C = build_proto_centroid_shrink(src_proto, centroid, beta=alpha)

            # D: 质心收缩 + 目标域负原型
            proto_D = proto_C

            ap_A = evaluate_with_features(target_data, proto_A, neg_proto=src_neg_proto, use_neg=True)
            ap_B = evaluate_with_features(target_data, proto_B, neg_proto=src_neg_proto, use_neg=True)
            ap_C = evaluate_with_features(target_data, proto_C, neg_proto=src_neg_proto, use_neg=True)
            ap_D = evaluate_with_features(target_data, proto_D, neg_proto=tgt_neg_proto, use_neg=True)

            results['A_single_text'][str(alpha)] = float(ap_A)
            results['B_multi_text'][str(alpha)] = float(ap_B)
            results['C_centroid_shrink'][str(alpha)] = float(ap_C)
            results['D_centroid_tgt_neg'][str(alpha)] = float(ap_D)

            print(f'  {alpha:<5.1f} {ap_A:<10.4f} {ap_B:<10.4f} {ap_C:<10.4f} {ap_D:<10.4f}')

        # 找最优α
        baseline = results['A_single_text']['0.0']
        best_alpha_A = max(ALPHAS, key=lambda a: results['A_single_text'][str(a)])
        best_alpha_B = max(ALPHAS, key=lambda a: results['B_multi_text'][str(a)])
        best_alpha_C = max(ALPHAS, key=lambda a: results['C_centroid_shrink'][str(a)])
        best_alpha_D = max(ALPHAS, key=lambda a: results['D_centroid_tgt_neg'][str(a)])

        results['baseline'] = float(baseline)
        results['best_alpha_A'] = float(best_alpha_A)
        results['best_ap_A'] = float(results['A_single_text'][str(best_alpha_A)])
        results['gain_A'] = float(results['best_ap_A'] - baseline)
        results['gain_A_25'] = float(results['A_single_text'].get('2.5', baseline) - baseline)

        results['best_alpha_B'] = float(best_alpha_B)
        results['best_ap_B'] = float(results['B_multi_text'][str(best_alpha_B)])
        results['gain_B'] = float(results['best_ap_B'] - baseline)
        results['gain_B_25'] = float(results['B_multi_text'].get('2.5', baseline) - baseline)

        results['best_alpha_C'] = float(best_alpha_C)
        results['best_ap_C'] = float(results['C_centroid_shrink'][str(best_alpha_C)])
        results['gain_C'] = float(results['best_ap_C'] - baseline)
        results['gain_C_25'] = float(results['C_centroid_shrink'].get('2.5', baseline) - baseline)

        results['best_alpha_D'] = float(best_alpha_D)
        results['best_ap_D'] = float(results['D_centroid_tgt_neg'][str(best_alpha_D)])
        results['gain_D'] = float(results['best_ap_D'] - baseline)
        results['gain_D_25'] = float(results['D_centroid_tgt_neg'].get('2.5', baseline) - baseline)

        print(f'\n  总结:')
        print(f'    baseline (α=0): {baseline:.4f}')
        print(f'    A单模板: 最优α={best_alpha_A}, 增益={results["gain_A"]:+.4f}, α=2.5增益={results["gain_A_25"]:+.4f}')
        print(f'    B多模板: 最优α={best_alpha_B}, 增益={results["gain_B"]:+.4f}, α=2.5增益={results["gain_B_25"]:+.4f}')
        print(f'    C质心收缩: 最优α={best_alpha_C}, 增益={results["gain_C"]:+.4f}, α=2.5增益={results["gain_C_25"]:+.4f}')
        print(f'    D质心+目标负: 最优α={best_alpha_D}, 增益={results["gain_D"]:+.4f}, α=2.5增益={results["gain_D_25"]:+.4f}')

        all_results.append(results)

    # ── 汇总 ──
    print(f'\n\n{"=" * 80}')
    print('汇总（按类别等权平均增益）')
    print(f'{"=" * 80}')

    print(f'\n{"类别":<25} {"baseline":>10} {"A单模板α=2.5":>15} {"B多模板α=2.5":>15} {"C质心α=2.5":>13} {"D质心+负α=2.5":>16}')
    print('-' * 95)

    for r in all_results:
        print(f'{r["display_name"]:<25} {r["baseline"]:>10.4f} '
              f'{r["gain_A_25"]:>+15.4f} {r["gain_B_25"]:>+15.4f} '
              f'{r["gain_C_25"]:>+13.4f} {r["gain_D_25"]:>+16.4f}')

    avg_gain_A_25 = np.mean([r['gain_A_25'] for r in all_results])
    avg_gain_B_25 = np.mean([r['gain_B_25'] for r in all_results])
    avg_gain_C_25 = np.mean([r['gain_C_25'] for r in all_results])
    avg_gain_D_25 = np.mean([r['gain_D_25'] for r in all_results])

    avg_gain_A_best = np.mean([r['gain_A'] for r in all_results])
    avg_gain_B_best = np.mean([r['gain_B'] for r in all_results])
    avg_gain_C_best = np.mean([r['gain_C'] for r in all_results])
    avg_gain_D_best = np.mean([r['gain_D'] for r in all_results])

    print(f'\n平均（α=2.5）:')
    print(f'  A单模板: {avg_gain_A_25:+.4f}')
    print(f'  B多模板: {avg_gain_B_25:+.4f}')
    print(f'  C质心收缩: {avg_gain_C_25:+.4f}')
    print(f'  D质心+目标负: {avg_gain_D_25:+.4f}')

    print(f'\n平均（最优α）:')
    print(f'  A单模板: {avg_gain_A_best:+.4f}')
    print(f'  B多模板: {avg_gain_B_best:+.4f}')
    print(f'  C质心收缩: {avg_gain_C_best:+.4f}')
    print(f'  D质心+目标负: {avg_gain_D_best:+.4f}')

    # 关键判断
    print(f'\n{"=" * 80}')
    print('关键判断')
    print(f'{"=" * 80}')

    best_method = max(['A', 'B', 'C', 'D'], key=lambda m: {
        'A': avg_gain_C_25, 'B': avg_gain_B_25, 'C': avg_gain_C_25, 'D': avg_gain_D_25
    }[m])

    print(f'  α=2.5最佳方法: {best_method}')
    print(f'  C质心收缩 vs A单模板: {avg_gain_C_25 - avg_gain_A_25:+.4f}')
    print(f'  D质心+目标负 vs A单模板: {avg_gain_D_25 - avg_gain_A_25:+.4f}')

    if avg_gain_C_25 > 0 or avg_gain_D_25 > 0:
        print(f'  ✅ 质心收缩方向让反向转正了！')
    else:
        print(f'  ❌ 质心收缩方向仍未转正')

    if avg_gain_C_best > 0 or avg_gain_D_best > 0:
        print(f'  ✅ 最优α下质心收缩方向转正了！')
    else:
        print(f'  ❌ 最优α下质心收缩方向仍未转正')

    # 保存结果
    out_file = DATA / 'centroid_shrink_experiment.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': all_results,
            'centroids': {k: v.tolist() for k, v in centroids.items()},
            'avg_gain_A_25': float(avg_gain_A_25),
            'avg_gain_B_25': float(avg_gain_B_25),
            'avg_gain_C_25': float(avg_gain_C_25),
            'avg_gain_D_25': float(avg_gain_D_25),
            'avg_gain_A_best': float(avg_gain_A_best),
            'avg_gain_B_best': float(avg_gain_B_best),
            'avg_gain_C_best': float(avg_gain_C_best),
            'avg_gain_D_best': float(avg_gain_D_best),
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
