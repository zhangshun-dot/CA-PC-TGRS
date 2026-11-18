# -*- coding: utf-8 -*-
"""direction2_3_experiment.py — 方向2（负原型双向处理）+ 方向3（方向门鲁棒化）综合实验
在多模板方向校准的基础上，进一步验证：
- 方向2：反向时负原型回退到目标域文本原型，或关闭负抑制
- 方向3：多模板方向一致性低时，回退到B2（不外推）
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

SRC_TEMPLATE_SINGLE = 'a photo of {}'
TGT_TEMPLATE_SINGLE = 'an overhead image of {}'

COCO_TO_DIOR_TARGETS = [
    ('airplane', 'airplane', 'airplane'),
    ('boat', 'ship', 'boat→ship'),
    ('car', 'vehicle', 'car→vehicle'),
    ('bus', 'vehicle', 'bus→vehicle (提示变体)'),
    ('truck', 'vehicle', 'truck→vehicle (提示变体)'),
]

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


def build_proto_multi_template(src_proto, src_cls, tgt_cls, beta=1.0):
    """多模板方向校准的外推函数"""
    s_emb = T.text_avg(SRC_TEMPLATES_PHOTO, src_cls).cpu().numpy()
    d_emb = T.text_avg(TGT_TEMPLATES_OVERHEAD, tgt_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def build_target_text_neg_proto(tgt_cls, all_tgt_classes):
    """构建目标域文本负原型：与目标类文本原型最接近的top-3其他类均值"""
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


def compute_direction_consistency(src_cls, tgt_cls):
    """计算多模板方向一致性：所有模板对方向的两两余弦相似度均值"""
    src_embs = [T.text_avg([t], src_cls).cpu().numpy() for t in SRC_TEMPLATES_PHOTO]
    tgt_embs = [T.text_avg([t], tgt_cls).cpu().numpy() for t in TGT_TEMPLATES_OVERHEAD]

    # 计算所有16个方向
    directions = []
    for s in src_embs:
        for d in tgt_embs:
            directions.append(normalize(d - s))

    # 计算两两余弦相似度
    sims = []
    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            sims.append(float(np.dot(directions[i], directions[j])))

    return float(np.mean(sims)), float(np.std(sims))


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


def run_experiment(coco_cls, dior_cls, display_name, all_dior_classes):
    """跑单个类别的实验，对比多种配置"""
    print(f'\n{"─" * 80}')
    print(f'COCO/{coco_cls} → DIOR/{dior_cls} ({display_name})')
    print(f'{"─" * 80}')

    # 加载源域数据
    print(f'  加载COCO源域数据...')
    src_data = load_coco_src(coco_cls, n=150)
    if src_data is None:
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']
    print(f'  源域原型: ||p||={np.linalg.norm(src_proto):.4f}, n_crops={src_data["n_crops"]}')

    # 源域负原型（当前做法）
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    src_neg_proto = src_features[neg_idx]
    src_neg_proto = normalize(src_neg_proto)

    # 目标域文本负原型（方向2）
    tgt_neg_proto = build_target_text_neg_proto(dior_cls, all_dior_classes)

    # 方向一致性（方向3）
    dir_consistency, dir_std = compute_direction_consistency(coco_cls, dior_cls)
    print(f'  方向一致性: {dir_consistency:.4f} ± {dir_std:.4f}')

    # 加载目标域数据
    print(f'  加载DIOR目标域数据...')
    target_data = load_target_dior(dior_cls, n=200)
    if target_data is None:
        return None
    print(f'  目标域: n_imgs={len(target_data["per_img"])}, n_crops={target_data["n_crops"]}')

    # 配置列表
    configs = {
        'A_单模板_源域负': {'template': 'single', 'neg': 'src', 'gate': False},
        'B_多模板_源域负': {'template': 'multi', 'neg': 'src', 'gate': False},
        'C_多模板_目标负': {'template': 'multi', 'neg': 'tgt', 'gate': False},
        'D_多模板_无负': {'template': 'multi', 'neg': 'none', 'gate': False},
        'E_多模板_源域负_门控': {'template': 'multi', 'neg': 'src', 'gate': True},
        'F_多模板_目标负_门控': {'template': 'multi', 'neg': 'tgt', 'gate': True},
    }

    results = {
        'src_cls': coco_cls,
        'target_cls': dior_cls,
        'display_name': display_name,
        'dir_consistency': float(dir_consistency),
        'dir_std': float(dir_std),
        'configs': {},
    }

    # 方向门控阈值：一致性低于0.5时回退到B2（不外推）
    GATE_THRESHOLD = 0.5
    use_gate = dir_consistency < GATE_THRESHOLD

    print(f'\n  配置对比（α=2.5）:')
    print(f'  {"配置":<25} {"AP":>10} {"增益":>10}')
    print(f'  {"─" * 50}')

    for config_name, config in configs.items():
        # 门控：如果方向一致性低，回退到α=0
        if config['gate'] and use_gate:
            alpha_eff = 0.0
        else:
            alpha_eff = 2.5

        # 构建正原型
        if alpha_eff == 0.0:
            proto = src_proto
        elif config['template'] == 'single':
            s_emb = T.text_avg([SRC_TEMPLATE_SINGLE], coco_cls).cpu().numpy()
            d_emb = T.text_avg([TGT_TEMPLATE_SINGLE], dior_cls).cpu().numpy()
            delta = d_emb - s_emb
            proto = src_proto + alpha_eff * delta
            proto = proto / (np.linalg.norm(proto) + 1e-8)
        else:  # multi
            proto = build_proto_multi_template(src_proto, coco_cls, dior_cls, beta=alpha_eff)

        # 选择负原型
        if config['neg'] == 'src':
            neg_p = src_neg_proto
            use_neg = True
        elif config['neg'] == 'tgt':
            neg_p = tgt_neg_proto
            use_neg = True
        else:  # none
            neg_p = None
            use_neg = False

        ap = evaluate_with_features(target_data, proto, neg_proto=neg_p, use_neg=use_neg)
        baseline = evaluate_with_features(target_data, src_proto, neg_proto=src_neg_proto, use_neg=True)
        gain = ap - baseline

        results['configs'][config_name] = {
            'ap': float(ap),
            'gain': float(gain),
            'alpha_eff': float(alpha_eff),
        }

        gate_mark = ' [门控→α=0]' if (config['gate'] and use_gate) else ''
        print(f'  {config_name:<25} {ap:>10.4f} {gain:>+10.4f}{gate_mark}')

    return results


def main():
    print('=' * 80)
    print('方向2（负原型双向处理）+ 方向3（方向门鲁棒化）综合实验')
    print('=' * 80)
    print()
    print('配置说明:')
    print('  A: 单模板 + 源域负原型（基线）')
    print('  B: 多模板 + 源域负原型（方向1）')
    print('  C: 多模板 + 目标域文本负原型（方向2）')
    print('  D: 多模板 + 无负抑制（方向2极端）')
    print('  E: 多模板 + 源域负原型 + 方向门控（方向1+3）')
    print('  F: 多模板 + 目标域负原型 + 方向门控（方向1+2+3）')
    print()
    print('方向门控：多模板方向一致性 < 0.5 时回退到α=0（不外推）')

    all_dior_classes = ['airplane', 'ship', 'vehicle', 'storage-tank', 'baseball-diamond',
                         'tennis-court', 'basketball-court', 'ground-track-field',
                         'harbor', 'bridge', 'chimney']

    all_results = []
    for coco_cls, dior_cls, display_name in COCO_TO_DIOR_TARGETS:
        result = run_experiment(coco_cls, dior_cls, display_name, all_dior_classes)
        if result is not None:
            all_results.append(result)

    # 汇总
    print('\n' + '=' * 80)
    print('汇总（α=2.5，按类别等权平均增益）')
    print('=' * 80)

    config_names = ['A_单模板_源域负', 'B_多模板_源域负', 'C_多模板_目标负',
                     'D_多模板_无负', 'E_多模板_源域负_门控', 'F_多模板_目标负_门控']

    print(f'\n{"配置":<30} {"平均增益":>12} {"vs基线A":>12}')
    print('-' * 55)

    baseline_gain = np.mean([r['configs']['A_单模板_源域负']['gain'] for r in all_results])

    for config_name in config_names:
        avg_gain = np.mean([r['configs'][config_name]['gain'] for r in all_results])
        vs_baseline = avg_gain - baseline_gain
        print(f'{config_name:<30} {avg_gain:>+12.4f} {vs_baseline:>+12.4f}')

    # 关键判断
    print(f'\n{"=" * 80}')
    print('关键判断')
    print(f'{"=" * 80}')

    best_config = max(config_names, key=lambda c: np.mean([r['configs'][c]['gain'] for r in all_results]))
    best_gain = np.mean([r['configs'][best_config]['gain'] for r in all_results])

    print(f'  最佳配置: {best_config}')
    print(f'  最佳平均增益: {best_gain:+.4f}')

    if best_gain > 0:
        print(f'  ✅ 反向转正！')
    else:
        print(f'  ❌ 反向仍未转正')

    # 方向2的效果
    b_gain = np.mean([r['configs']['B_多模板_源域负']['gain'] for r in all_results])
    c_gain = np.mean([r['configs']['C_多模板_目标负']['gain'] for r in all_results])
    d_gain = np.mean([r['configs']['D_多模板_无负']['gain'] for r in all_results])
    print(f'\n  方向2（负原型双向处理）效果:')
    print(f'    源域负原型 (B): {b_gain:+.4f}')
    print(f'    目标域负原型 (C): {c_gain:+.4f}, 改进: {c_gain - b_gain:+.4f}')
    print(f'    无负抑制 (D): {d_gain:+.4f}, 改进: {d_gain - b_gain:+.4f}')

    # 方向3的效果
    e_gain = np.mean([r['configs']['E_多模板_源域负_门控']['gain'] for r in all_results])
    f_gain = np.mean([r['configs']['F_多模板_目标负_门控']['gain'] for r in all_results])
    print(f'\n  方向3（方向门鲁棒化）效果:')
    print(f'    源域负+门控 (E): {e_gain:+.4f}, 改进: {e_gain - b_gain:+.4f}')
    print(f'    目标负+门控 (F): {f_gain:+.4f}, 改进: {f_gain - c_gain:+.4f}')

    # 保存结果
    out_file = DATA / 'direction2_3_experiment.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': all_results,
            'config_names': config_names,
            'baseline_gain': float(baseline_gain),
            'best_config': best_config,
            'best_gain': float(best_gain),
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
