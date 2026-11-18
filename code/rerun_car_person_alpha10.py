# -*- coding: utf-8 -*-
"""rerun_car_person_alpha10.py — 重跑所有car和person的样本，α网格扩到10.0
现有的样本只有到6.0，而car和person都顶在上界，需要重跑
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import tge_probe as T
from sota_comparison import (
    SOURCE_LOADERS, build_proto_tge, evaluate_with_features,
    SRC_TMPL_OVERHEAD, TGT_TMPL,
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# α网格扩到10.0
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
ALPHA_MAX = 10.0
ALPHA_MIN = 0.0


def compute_sim_factors(target_data, src_proto):
    features = target_data['features']
    if len(features) == 0:
        return {'sim_pos_mean': 0.0, 'sim_pos_std': 0.0}
    sims = features @ src_proto
    pos_mask = sims > np.percentile(sims, 25)
    pos_sims = sims[pos_mask]
    return {
        'sim_pos_mean': float(np.mean(pos_sims)) if len(pos_sims) > 0 else 0.0,
        'sim_pos_std': float(np.std(pos_sims)) if len(pos_sims) > 0 else 0.0,
    }


def compute_sv(target_data):
    per_img = target_data['per_img']
    if not per_img:
        return 0.0
    sizes = []
    for img_data in per_img:
        boxes = img_data['box']
        if len(boxes) == 0:
            continue
        for box in boxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            sizes.append(w * h)
    if not sizes:
        return 0.0
    return float(np.std(sizes) / (np.mean(sizes) + 1e-8))


def compute_tc(target_cls):
    classes = ['airplane', 'boat', 'car', 'bus', 'motorcycle', 'bicycle', 'truck', 'person']
    t_emb = T.text_avg([TGT_TMPL], target_cls).cpu().numpy()
    sims = []
    for c in classes:
        if c == target_cls:
            continue
        t_c = T.text_avg([TGT_TMPL], c).cpu().numpy()
        sim = float(t_emb @ t_c / (np.linalg.norm(t_emb) * np.linalg.norm(t_c) + 1e-8))
        sims.append(sim)
    return float(max(sims)) if sims else 0.0


def compute_m0(target_data, src_proto):
    features = target_data['features']
    if len(features) == 0:
        return 0.0
    sims = features @ src_proto
    median_sim = float(np.median(sims))
    p25_sim = float(np.percentile(sims, 25))
    m0 = median_sim - p25_sim
    return max(0.0, m0)


def compute_mu(target_data, src_proto, src_cls, target_cls):
    features = target_data['features']
    s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], target_cls).cpu().numpy()
    delta = d_emb - s_emb
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    mu = float(np.mean(features @ delta))
    return abs(mu) + 1e-6


def process_sample(src_name, src_cls, target_cls, target_domain, target_loader_func):
    print(f'\n  处理 {src_name}/{src_cls} -> {target_domain}/{target_cls}...')
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        print(f'    源域数据加载失败，跳过')
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']

    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

    target_data = target_loader_func(target_cls, n=200)
    if target_data is None:
        print(f'    目标域数据加载失败，跳过')
        return None

    sim_factors = compute_sim_factors(target_data, src_proto)
    sv = compute_sv(target_data)
    tc = compute_tc(target_cls)
    m0 = compute_m0(target_data, src_proto)
    mu = compute_mu(target_data, src_proto, src_cls, target_cls)

    # α网格搜索（扩到10.0）
    ap_curve = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            proto = src_proto
        else:
            proto = build_proto_tge(src_proto, src_cls, target_cls, beta=alpha)
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

    return {
        'src': src_name, 'src_cls': src_cls, 'target_cls': target_cls,
        'target_domain': target_domain,
        'sv': sv, 'tc': tc, 'm0': m0, 'mu': mu,
        'sim_pos_mean': sim_factors['sim_pos_mean'],
        'sim_pos_std': sim_factors['sim_pos_std'],
        'best_alpha': float(best_alpha), 'best_ap': float(best_ap),
        'baseline_ap': float(baseline_ap),
        'oracle_gain': float(best_ap - baseline_ap),
        'ap_curve': ap_curve,
    }


def main():
    print('=' * 80)
    print('重跑所有car和person的样本，α网格扩到10.0')
    print('=' * 80)

    # 加载现有数据
    samples_file = DATA / 'zero_shot_gated_samples_extended_alpha.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    print(f'\n现有样本数: {len(all_samples)}')

    # 找出需要重跑的样本（car和person，且ap_curve只有到6.0的）
    samples_to_rerun = []
    samples_to_keep = []
    for s in all_samples:
        target_cls = s['target_cls']
        has_alpha_10 = '10.0' in s['ap_curve']
        if target_cls in ['car', 'person'] and not has_alpha_10:
            samples_to_rerun.append(s)
        else:
            samples_to_keep.append(s)

    print(f'\n需要重跑的样本数: {len(samples_to_rerun)}')
    print(f'保留的样本数: {len(samples_to_keep)}')
    print(f'\n需要重跑的样本:')
    for s in samples_to_rerun:
        print(f'  {s["src"]}/{s["src_cls"]} -> {s["target_domain"]}/{s["target_cls"]}')

    # 重跑
    print(f'\n{"=" * 80}')
    print('开始重跑...')
    print(f'{"=" * 80}')

    from sota_comparison import load_target_coco
    from voc_sota_comparison import load_target_voc

    rerun_results = []
    for i, s in enumerate(samples_to_rerun):
        print(f'\n--- 重跑第 {i+1}/{len(samples_to_rerun)} 个样本 ---')
        target_loader = load_target_coco if s['target_domain'] == 'COCO' else load_target_voc
        result = process_sample(
            s['src'], s['src_cls'], s['target_cls'],
            s['target_domain'], target_loader
        )
        if result:
            rerun_results.append(result)
            print(f'  重跑完成，最优α={result["best_alpha"]}, 增益={result["oracle_gain"]:+.4f}')

    # 合并结果
    final_samples = samples_to_keep + rerun_results
    print(f'\n{"=" * 80}')
    print(f'重跑完成！最终样本数: {len(final_samples)}')
    print(f'{"=" * 80}')

    # 统计car和person的最优α分布
    print(f'\n--- car类最优α分布 ---')
    car_samples = [s for s in final_samples if s['target_cls'] == 'car']
    for s in sorted(car_samples, key=lambda x: x['best_alpha']):
        print(f'  {s["src"]}/{s["src_cls"]} -> {s["target_domain"]}: '
              f'最优α={s["best_alpha"]}, 增益={s["oracle_gain"]:+.4f}')

    print(f'\n--- person类最优α分布 ---')
    person_samples = [s for s in final_samples if s['target_cls'] == 'person']
    for s in sorted(person_samples, key=lambda x: x['best_alpha']):
        print(f'  {s["src"]}/{s["src_cls"]} -> {s["target_domain"]}: '
              f'最优α={s["best_alpha"]}, 增益={s["oracle_gain"]:+.4f}')

    # 保存
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(final_samples, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {samples_file}')


if __name__ == '__main__':
    main()
