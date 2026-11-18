# -*- coding: utf-8 -*-
"""lambda_interpolation_experiment.py — 弹簧模型λ插值验证实验
验证：0<λ<1的混合校正是否能进一步提升反向COCO→DIOR的效果。

弹簧模型统一公式：
  p̂_c = p_c + k · (a(λ) − p_c)
  a(λ) = (1−λ)·(p_c + d_text) + λ·c_coarse

- λ=0：纯文本外推（正向粗→细适用）
- λ=1：纯质心收缩（反向细→粗适用）
- 0<λ<1：混合校正
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

# ── 多模板集合 ──
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

# COCO→DIOR类别对应
COCO_TO_DIOR_TARGETS = [
    ('airplane', 'airplane', 'airplane', 'airplane'),
    ('boat', 'ship', 'boat→ship', 'ship'),
    ('car', 'vehicle', 'car→vehicle', 'vehicle'),
    ('bus', 'vehicle', 'bus→vehicle', 'vehicle'),
    ('truck', 'vehicle', 'truck→vehicle', 'vehicle'),
    ('motorcycle', 'vehicle', 'motorcycle→vehicle', 'vehicle'),
]

# λ值网格
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# α值网格（外推强度）
ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


def build_proto_spring(src_proto, text_dir, centroid, lam, alpha=1.0):
    """弹簧模型：p̂ = p + α·(a(λ) − p)
    a(λ) = (1−λ)·(p + d_text) + λ·c_coarse
    """
    if centroid is None:
        # 无聚合需求，退化为纯文本外推
        a = src_proto + text_dir
    else:
        # 锚点插值
        text_point = src_proto + text_dir
        a = (1 - lam) * text_point + lam * centroid

    delta = a - src_proto
    p_hat = src_proto + alpha * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def get_text_direction(src_cls, tgt_cls, multi=True):
    """获取文本视角方向 d_text = mean(tgt) - mean(src)"""
    if multi:
        s_emb = T.text_avg(SRC_TEMPLATES_PHOTO, src_cls).cpu().numpy()
        d_emb = T.text_avg(TGT_TEMPLATES_OVERHEAD, tgt_cls).cpu().numpy()
    else:
        s_emb = T.text_avg(['a photo of {}'], src_cls).cpu().numpy()
        d_emb = T.text_avg(['an overhead image of {}'], tgt_cls).cpu().numpy()
    return d_emb - s_emb


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
    print('弹簧模型λ插值验证实验（COCO→DIOR反向）')
    print('验证：0<λ<1的混合校正是否能进一步提升')
    print('=' * 80)

    # ── 第一步：加载所有COCO源域原型，计算各组质心 ──
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
            print(f'\n  组 {group_name} 质心（{len(cls_list)}个类）')

    # ── 第二步：λ×α 网格搜索 ──
    print(f'\n\n第二步：λ×α 网格搜索（每个类别）')
    print(f'{"─" * 80}')

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

        # 文本方向
        text_dir = get_text_direction(coco_cls, dior_cls, multi=True)

        # 质心
        centroid = centroids.get(group_name, None)

        # 加载DIOR目标域数据
        print(f'  加载DIOR目标域数据...')
        target_data = load_target_dior(dior_cls, n=200)
        if target_data is None:
            continue
        print(f'  目标域: n_imgs={len(target_data["per_img"])}, n_crops={target_data["n_crops"]}')

        # λ×α 网格搜索
        baseline = evaluate_with_features(target_data, src_proto, neg_proto=tgt_neg_proto, use_neg=True)

        results = {
            'src_cls': coco_cls,
            'target_cls': dior_cls,
            'display_name': display_name,
            'group_name': group_name,
            'has_centroid': centroid is not None,
            'baseline': float(baseline),
            'lambda_alpha_grid': {},
        }

        print(f'\n  λ×α 网格（增益，baseline={baseline:.4f}）:')
        print(f'  {"λ":<6}', end='')
        for alpha in ALPHAS:
            print(f' {alpha:>7.1f}', end='')
        print()
        print(f'  {"─" * (6 + 8 * len(ALPHAS))}')

        for lam in LAMBDAS:
            print(f'  {lam:<6.1f}', end='')
            for alpha in ALPHAS:
                if alpha == 0.0:
                    proto = src_proto
                else:
                    proto = build_proto_spring(src_proto, text_dir, centroid, lam, alpha=alpha)
                ap = evaluate_with_features(target_data, proto, neg_proto=tgt_neg_proto, use_neg=True)
                gain = ap - baseline
                results['lambda_alpha_grid'][f'{lam:.1f}_{alpha:.1f}'] = float(gain)
                print(f' {gain:>+7.4f}', end='')
            print()

        # 找最优λ和α
        best_key = max(results['lambda_alpha_grid'].keys(),
                       key=lambda k: results['lambda_alpha_grid'][k])
        best_lam, best_alpha = map(float, best_key.split('_'))
        best_gain = results['lambda_alpha_grid'][best_key]

        results['best_lambda'] = float(best_lam)
        results['best_alpha'] = float(best_alpha)
        results['best_gain'] = float(best_gain)

        # λ=0（纯外推）和λ=1（纯聚合）的对比
        results['gain_lam0_alpha25'] = float(results['lambda_alpha_grid'].get('0.0_2.5', 0))
        results['gain_lam1_alpha25'] = float(results['lambda_alpha_grid'].get('1.0_2.5', 0))

        print(f'\n  最优: λ={best_lam:.1f}, α={best_alpha:.1f}, 增益={best_gain:+.4f}')
        print(f'  λ=0 (纯外推) α=2.5: {results["gain_lam0_alpha25"]:+.4f}')
        print(f'  λ=1 (纯聚合) α=2.5: {results["gain_lam1_alpha25"]:+.4f}')

        all_results.append(results)

    # ── 汇总 ──
    print(f'\n\n{"=" * 80}')
    print('汇总（按类别等权平均增益）')
    print(f'{"=" * 80}')

    # 每个λ的平均增益（α=2.5）
    print(f'\nλ扫描（α=2.5）:')
    print(f'  {"λ":<6} {"平均增益":>10}')
    print(f'  {"─" * 20}')

    lambda_gains = {}
    for lam in LAMBDAS:
        gains = []
        for r in all_results:
            key = f'{lam:.1f}_2.5'
            if key in r['lambda_alpha_grid']:
                gains.append(r['lambda_alpha_grid'][key])
        if gains:
            avg_gain = np.mean(gains)
            lambda_gains[lam] = avg_gain
            print(f'  {lam:<6.1f} {avg_gain:>+10.4f}')

    best_lam_overall = max(lambda_gains.keys(), key=lambda l: lambda_gains[l])
    print(f'\n  最优λ: {best_lam_overall:.1f}, 平均增益: {lambda_gains[best_lam_overall]:+.4f}')

    # 每个类别的最优λ
    print(f'\n各类别最优λ:')
    print(f'  {"类别":<25} {"最优λ":>8} {"最优α":>8} {"最优增益":>10} {"λ=0增益":>10} {"λ=1增益":>10}')
    print(f'  {"─" * 75}')

    for r in all_results:
        print(f'  {r["display_name"]:<25} {r["best_lambda"]:>8.1f} {r["best_alpha"]:>8.1f} '
              f'{r["best_gain"]:>+10.4f} {r["gain_lam0_alpha25"]:>+10.4f} {r["gain_lam1_alpha25"]:>+10.4f}')

    avg_best_gain = np.mean([r['best_gain'] for r in all_results])
    avg_lam0_gain = np.mean([r['gain_lam0_alpha25'] for r in all_results])
    avg_lam1_gain = np.mean([r['gain_lam1_alpha25'] for r in all_results])

    print(f'\n平均:')
    print(f'  最优(λ,α): {avg_best_gain:+.4f}')
    print(f'  λ=0 (纯外推) α=2.5: {avg_lam0_gain:+.4f}')
    print(f'  λ=1 (纯聚合) α=2.5: {avg_lam1_gain:+.4f}')
    print(f'  最优λ={best_lam_overall:.1f} α=2.5: {lambda_gains[best_lam_overall]:+.4f}')

    # 关键判断
    print(f'\n{"=" * 80}')
    print('关键判断')
    print(f'{"=" * 80}')

    if best_lam_overall > 0 and best_lam_overall < 1:
        print(f'  ✅ 最优λ={best_lam_overall:.1f} 在(0,1)之间，混合校正优于纯外推和纯聚合！')
        print(f'  弹簧模型的λ插值有效，中间状态有额外增益')
    elif best_lam_overall == 1.0:
        print(f'  ✅ 最优λ=1.0，纯聚合最优（与之前的质心收缩实验一致）')
    else:
        print(f'  ❌ 最优λ={best_lam_overall:.1f}，λ插值没有额外增益')

    if avg_best_gain > avg_lam1_gain:
        print(f'  ✅ 最优(λ,α)比纯聚合(λ=1)提升: {avg_best_gain - avg_lam1_gain:+.4f}')
    else:
        print(f'  ❌ 最优(λ,α)没有超过纯聚合(λ=1)')

    # 保存结果
    out_file = DATA / 'lambda_interpolation_experiment.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': all_results,
            'lambda_gains_alpha25': {str(k): float(v) for k, v in lambda_gains.items()},
            'best_lambda_overall': float(best_lam_overall),
            'avg_best_gain': float(avg_best_gain),
            'avg_lam0_gain': float(avg_lam0_gain),
            'avg_lam1_gain': float(avg_lam1_gain),
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
