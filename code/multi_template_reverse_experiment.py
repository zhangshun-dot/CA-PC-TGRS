# -*- coding: utf-8 -*-
"""multi_template_reverse_experiment.py — 多模板方向校准的反向实验（COCO→DIOR）
验证：用多模板方向校准能否让COCO→DIOR反向的完整外推从-0.0244转正。

这是改进方向1的关键验证实验。
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import tge_probe as T
from sota_comparison import evaluate_with_features, PRE_FILTER
import unified_eval as U

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# ── 多模板集合 ──
# 注意：反向实验中，源域是COCO（自然/侧视），目标域是DIOR（遥感/俯视）
# 所以源域模板集是自然/侧视，目标域模板集是遥感/俯视

# 源域（自然/侧视）模板集
SRC_TEMPLATES_PHOTO = [
    'a photo of {}',
    'a natural image of {}',
    'a ground-level view of {}',
    'a side view of {}',
]

# 目标域（遥感/俯视）模板集
TGT_TEMPLATES_OVERHEAD = [
    'an overhead image of {}',
    'an aerial view of {}',
    'a satellite image of {}',
    'a top-down view of {}',
]

# 单模板（基线）
SRC_TEMPLATE_SINGLE = 'a photo of {}'
TGT_TEMPLATE_SINGLE = 'an overhead image of {}'

# COCO→DIOR类别对应
COCO_TO_DIOR_TARGETS = [
    ('airplane', 'airplane', 'airplane'),
    ('boat', 'ship', 'boat→ship'),
    ('car', 'vehicle', 'car→vehicle'),
    ('bus', 'vehicle', 'bus→vehicle (提示变体)'),
    ('truck', 'vehicle', 'truck→vehicle (提示变体)'),
]

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


def build_proto_multi_template(src_proto, src_cls, tgt_cls, beta=1.0,
                                 src_templates=None, tgt_templates=None):
    """多模板方向校准的外推函数"""
    if src_templates is None:
        src_templates = SRC_TEMPLATES_PHOTO
    if tgt_templates is None:
        tgt_templates = TGT_TEMPLATES_OVERHEAD

    s_emb = T.text_avg(src_templates, src_cls).cpu().numpy()
    d_emb = T.text_avg(tgt_templates, tgt_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def build_proto_single_template(src_proto, src_cls, tgt_cls, beta=1.0):
    """单模板外推（基线）"""
    s_emb = T.text_avg([SRC_TEMPLATE_SINGLE], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TEMPLATE_SINGLE], tgt_cls).cpu().numpy()
    delta = d_emb - s_emb
    p_hat = src_proto + beta * delta
    return p_hat / (np.linalg.norm(p_hat) + 1e-8)


def load_coco_src(cls_name, n=150):
    """加载COCO源域数据（用CLIP提取特征）"""
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
    """加载DIOR目标域数据（用GDINO-tiny检测）"""
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
    """跑单个类别的实验，对比单模板 vs 多模板"""
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

    # 对比单模板 vs 多模板
    results = {
        'src': 'COCO',
        'src_cls': coco_cls,
        'target_cls': dior_cls,
        'display_name': display_name,
        'single_template': {},
        'multi_template': {},
    }

    print(f'\n  α    单模板AP    多模板AP    差值')
    print(f'  {"─" * 50}')

    for alpha in ALPHAS:
        if alpha == 0.0:
            proto_single = src_proto
            proto_multi = src_proto
        else:
            proto_single = build_proto_single_template(src_proto, coco_cls, dior_cls, beta=alpha)
            proto_multi = build_proto_multi_template(src_proto, coco_cls, dior_cls, beta=alpha)

        ap_single = evaluate_with_features(target_data, proto_single, neg_proto=proto_neg)
        ap_multi = evaluate_with_features(target_data, proto_multi, neg_proto=proto_neg)

        results['single_template'][str(alpha)] = float(ap_single)
        results['multi_template'][str(alpha)] = float(ap_multi)

        diff = ap_multi - ap_single
        print(f'  {alpha:<5.1f} {ap_single:<10.4f} {ap_multi:<10.4f} {diff:+.4f}')

    # 找最优α
    best_alpha_single = max(ALPHAS, key=lambda a: results['single_template'][str(a)])
    best_alpha_multi = max(ALPHAS, key=lambda a: results['multi_template'][str(a)])

    baseline = results['single_template']['0.0']
    best_single = results['single_template'][str(best_alpha_single)]
    best_multi = results['multi_template'][str(best_alpha_multi)]

    # 固定α=2.5的增益（关键：完整外推）
    ap_25_single = results['single_template'].get('2.5', baseline)
    ap_25_multi = results['multi_template'].get('2.5', baseline)
    gain_25_single = ap_25_single - baseline
    gain_25_multi = ap_25_multi - baseline

    results['baseline'] = float(baseline)
    results['best_alpha_single'] = float(best_alpha_single)
    results['best_ap_single'] = float(best_single)
    results['gain_single'] = float(best_single - baseline)
    results['best_alpha_multi'] = float(best_alpha_multi)
    results['best_ap_multi'] = float(best_multi)
    results['gain_multi'] = float(best_multi - baseline)
    results['gain_25_single'] = float(gain_25_single)
    results['gain_25_multi'] = float(gain_25_multi)
    results['improvement_25'] = float(gain_25_multi - gain_25_single)
    results['improvement_best'] = float(best_multi - best_single)

    print(f'\n  总结:')
    print(f'    baseline (α=0):           {baseline:.4f}')
    print(f'    单模板最优: α={best_alpha_single}, AP={best_single:.4f}, 增益={best_single - baseline:+.4f}')
    print(f'    多模板最优: α={best_alpha_multi}, AP={best_multi:.4f}, 增益={best_multi - baseline:+.4f}')
    print(f'    单模板α=2.5增益: {gain_25_single:+.4f}')
    print(f'    多模板α=2.5增益: {gain_25_multi:+.4f}')
    print(f'    α=2.5改进: {gain_25_multi - gain_25_single:+.4f}')

    return results


def main():
    print('=' * 80)
    print('多模板方向校准的反向实验（COCO→DIOR）')
    print('验证：多模板能否让反向完整外推从-0.0244转正')
    print('=' * 80)
    print()
    print('源域模板集（自然/侧视）:')
    for t in SRC_TEMPLATES_PHOTO:
        print(f'  - {t}')
    print()
    print('目标域模板集（遥感/俯视）:')
    for t in TGT_TEMPLATES_OVERHEAD:
        print(f'  - {t}')

    # 跑所有类别
    all_results = []
    for coco_cls, dior_cls, display_name in COCO_TO_DIOR_TARGETS:
        result = run_experiment(coco_cls, dior_cls, display_name)
        if result is not None:
            all_results.append(result)

    # 汇总
    print('\n' + '=' * 80)
    print('反向实验汇总（COCO→DIOR）')
    print('=' * 80)

    print(f'\n{"类别":<20} {"baseline":>10} {"单模板α=2.5":>14} {"多模板α=2.5":>14} {"改进":>10}')
    print('-' * 70)

    for r in all_results:
        print(f'{r["display_name"]:<20} {r["baseline"]:>10.4f} '
              f'{r["gain_25_single"]:>+14.4f} {r["gain_25_multi"]:>+14.4f} '
              f'{r["improvement_25"]:>+10.4f}')

    avg_gain_25_single = np.mean([r['gain_25_single'] for r in all_results])
    avg_gain_25_multi = np.mean([r['gain_25_multi'] for r in all_results])
    avg_improvement_25 = np.mean([r['improvement_25'] for r in all_results])

    avg_gain_best_single = np.mean([r['gain_single'] for r in all_results])
    avg_gain_best_multi = np.mean([r['gain_multi'] for r in all_results])

    print(f'\n平均（按类别等权）:')
    print(f'  单模板α=2.5增益: {avg_gain_25_single:+.4f}')
    print(f'  多模板α=2.5增益: {avg_gain_25_multi:+.4f}')
    print(f'  α=2.5改进: {avg_improvement_25:+.4f}')
    print(f'  单模板最优增益: {avg_gain_best_single:+.4f}')
    print(f'  多模板最优增益: {avg_gain_best_multi:+.4f}')

    # 关键判断：反向是否转正
    print(f'\n{"=" * 80}')
    print('关键判断')
    print(f'{"=" * 80}')
    if avg_gain_25_multi > 0:
        print(f'  ✅ 多模板α=2.5平均增益 {avg_gain_25_multi:+.4f} > 0，反向转正！')
        print(f'  单模板是 {avg_gain_25_single:+.4f}，改进 {avg_improvement_25:+.4f}')
    else:
        print(f'  ❌ 多模板α=2.5平均增益 {avg_gain_25_multi:+.4f} < 0，反向仍未转正')
        print(f'  单模板是 {avg_gain_25_single:+.4f}，改进 {avg_improvement_25:+.4f}')

    if avg_gain_best_multi > 0:
        print(f'  ✅ 多模板最优平均增益 {avg_gain_best_multi:+.4f} > 0')
    else:
        print(f'  ❌ 多模板最优平均增益 {avg_gain_best_multi:+.4f} < 0')

    # 保存结果
    out_file = DATA / 'multi_template_reverse_experiment.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': all_results,
            'src_templates': SRC_TEMPLATES_PHOTO,
            'tgt_templates': TGT_TEMPLATES_OVERHEAD,
            'avg_gain_25_single': float(avg_gain_25_single),
            'avg_gain_25_multi': float(avg_gain_25_multi),
            'avg_improvement_25': float(avg_improvement_25),
            'avg_gain_best_single': float(avg_gain_best_single),
            'avg_gain_best_multi': float(avg_gain_best_multi),
        }, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
