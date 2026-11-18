# -*- coding: utf-8 -*-
"""person_extended_alpha_and_sanity.py — person网格延到15 + AP75/框数sanity check + 固定-α表
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
import unified_eval as U

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# 扩展α网格到15
ALPHAS_EXTENDED = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]

# 原始α网格（到10）
ALPHAS_ORIG = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def compute_ap_multi_iou(im_preds, im_gts, iou_thresholds=[0.5, 0.75]):
    """同时计算多个IoU阈值下的AP"""
    n_gt = sum(len(g) for g in im_gts)
    if n_gt == 0:
        return {t: (0.0 if any(len(p[0]) for p in im_preds) else 1.0) for t in iou_thresholds}

    results = {}
    for iou_thresh in iou_thresholds:
        evals = []
        for (boxes, scores), gts in zip(im_preds, im_gts):
            if len(boxes) == 0:
                continue
            matched = [False] * len(gts)
            idx = np.argsort(-np.asarray(scores))
            for i in idx:
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gts):
                    if not matched[j]:
                        v = U.iou(boxes[i], gb)
                        if v > best_iou:
                            best_iou, best_j = v, j
                if best_iou >= iou_thresh and best_j >= 0:
                    evals.append((float(scores[i]), 1))
                    matched[best_j] = True
                else:
                    evals.append((float(scores[i]), 0))
        if not evals:
            results[iou_thresh] = 0.0
            continue
        evals.sort(key=lambda x: -x[0])
        tp = np.array([e[1] for e in evals])
        tc = np.cumsum(tp)
        fc = np.cumsum(1 - tp)
        rec = tc / n_gt
        prec = tc / (tc + fc + 1e-8)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            m = rec >= t
            ap += (float(prec[m].max()) if m.any() else 0) / 11
        results[iou_thresh] = ap
    return results


def evaluate_with_features_multi(target_data, proto, neg_proto=None, lam=0.25):
    """同时返回AP50、AP75和预测框总数"""
    per_img = target_data['per_img']
    F = target_data['features']
    total_boxes = 0
    for p in per_img:
        n = len(p['gd'])
        total_boxes += n
        if n:
            f_boxes = F[p['crop_idx']]
            pos_sim = f_boxes @ proto
            if neg_proto is not None:
                neg_sim = f_boxes @ neg_proto
                vis = np.maximum(0.0, pos_sim - lam * neg_sim)
            else:
                vis = np.maximum(0.0, pos_sim)
            p['fused'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['fused'] = np.zeros(0)
    preds = [(p['box'], p['fused']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    aps = compute_ap_multi_iou(preds, gts, [0.5, 0.75])
    return aps[0.5], aps[0.75], total_boxes


def process_person_extended():
    """person网格延到15 + AP75/框数sanity check"""
    print('=' * 80)
    print('person网格延到15 + AP75/框数sanity check')
    print('=' * 80)

    # 加载原始数据，找到person样本
    with open(DATA / 'zero_shot_gated_samples_extended_alpha.json', 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    person_samples = [s for s in all_samples if s['target_cls'] == 'person']
    print(f'找到 {len(person_samples)} 个person样本')

    results = []
    for idx, sample in enumerate(person_samples):
        print(f'\n--- person样本 {idx+1}/{len(person_samples)} ---')
        print(f'  源域: {sample["src"]}/{sample["src_cls"]}, 目标域: {sample["target_domain"]}')

        # 重新加载源域和目标域数据
        src_loader = SOURCE_LOADERS[sample['src']][0]
        src_data = src_loader(sample['src_cls'], n=150)
        src_proto = src_data['prototype']
        src_features = src_data['features']

        sims = src_features @ src_proto
        neg_idx = np.argmin(sims)
        proto_neg = src_features[neg_idx]
        proto_neg = proto_neg / (np.linalg.norm(proto_neg) + 1e-8)

        # 加载目标域数据（需要重新加载，因为原始数据没有保存per_img）
        # 这里我们用缓存的特征，但是需要per_img来计算AP75和框数
        # 所以需要重新加载目标域数据
        if sample['target_domain'] == 'COCO':
            from sota_comparison import load_target_coco
            target_data = load_target_coco(sample['target_cls'], n=200)
        elif sample['target_domain'] == 'VOC':
            # VOC目标域加载函数可能在别的地方
            print(f'  警告: VOC目标域加载函数未找到，跳过AP75/框数计算')
            continue
        else:
            print(f'  警告: 未知目标域 {sample["target_domain"]}，跳过')
            continue

        if target_data is None:
            print(f'  目标域数据加载失败，跳过')
            continue

        # 扩展α网格搜索
        ap50_curve = {}
        ap75_curve = {}
        box_count_curve = {}
        for alpha in ALPHAS_EXTENDED:
            if alpha == 0.0:
                proto = src_proto
            else:
                proto = build_proto_tge(src_proto, sample['src_cls'], sample['target_cls'], beta=alpha)
            ap50, ap75, n_boxes = evaluate_with_features_multi(target_data, proto, neg_proto=proto_neg)
            ap50_curve[str(alpha)] = float(ap50)
            ap75_curve[str(alpha)] = float(ap75)
            box_count_curve[str(alpha)] = int(n_boxes)
            print(f'    α={alpha:.1f}: AP50={ap50:.4f}, AP75={ap75:.4f}, boxes={n_boxes}')

        best_alpha = ALPHAS_EXTENDED[0]
        best_ap50 = ap50_curve[str(ALPHAS_EXTENDED[0])]
        for alpha in ALPHAS_EXTENDED:
            if ap50_curve[str(alpha)] > best_ap50:
                best_ap50 = ap50_curve[str(alpha)]
                best_alpha = alpha

        baseline_ap50 = ap50_curve['0.0']
        baseline_ap75 = ap75_curve['0.0']
        baseline_boxes = box_count_curve['0.0']

        result = {
            'src': sample['src'],
            'src_cls': sample['src_cls'],
            'target_cls': sample['target_cls'],
            'target_domain': sample['target_domain'],
            'best_alpha': float(best_alpha),
            'best_ap50': float(best_ap50),
            'baseline_ap50': float(baseline_ap50),
            'oracle_gain': float(best_ap50 - baseline_ap50),
            'baseline_ap75': float(baseline_ap75),
            'best_ap75': float(ap75_curve[str(best_alpha)]),
            'baseline_boxes': int(baseline_boxes),
            'best_boxes': int(box_count_curve[str(best_alpha)]),
            'ap50_curve': ap50_curve,
            'ap75_curve': ap75_curve,
            'box_count_curve': box_count_curve,
        }
        results.append(result)

        # 关键α值的sanity check
        print(f'\n  关键α值sanity check:')
        for alpha in [8.0, 10.0, 12.0]:
            if str(alpha) in ap50_curve:
                gain50 = ap50_curve[str(alpha)] - baseline_ap50
                gain75 = ap75_curve[str(alpha)] - baseline_ap75
                box_ratio = box_count_curve[str(alpha)] / (baseline_boxes + 1e-8)
                print(f'    α={alpha:.1f}: AP50增益={gain50:+.4f}, AP75增益={gain75:+.4f}, '
                      f'框数比={box_ratio:.2f}x')

    # 保存结果
    out_file = DATA / 'person_extended_alpha_sanity.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')

    return results


def generate_fixed_alpha_table():
    """生成固定-α版本的完整表（单一全局α，非逐单元argmax）"""
    print('\n' + '=' * 80)
    print('固定-α版本完整表')
    print('=' * 80)

    with open(DATA / 'zero_shot_gated_samples_extended_alpha.json', 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    # 测试几个固定α值
    fixed_alphas = [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0]

    print(f'\n{"α":>6} {"mean_AP50":>10} {"mean_gain":>10} {"median_gain":>12} {"pos_ratio":>10}')
    print('-' * 60)

    results = {}
    for alpha in fixed_alphas:
        gains = []
        aps = []
        for sample in all_samples:
            ap_curve = sample['ap_curve']
            # 插值得到该α的AP
            alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
            ap_values = [ap_curve[str(a)] for a in alphas_sorted]
            ap = float(np.interp(alpha, alphas_sorted, ap_values))
            baseline = ap_curve['0.0']
            gains.append(ap - baseline)
            aps.append(ap)

        mean_ap = np.mean(aps)
        mean_gain = np.mean(gains)
        median_gain = np.median(gains)
        pos_ratio = np.mean([1 if g > 0.001 else 0 for g in gains])

        results[alpha] = {
            'mean_ap50': float(mean_ap),
            'mean_gain': float(mean_gain),
            'median_gain': float(median_gain),
            'pos_ratio': float(pos_ratio),
        }
        print(f'{alpha:>6.1f} {mean_ap:>10.4f} {mean_gain:>+10.4f} {median_gain:>+12.4f} {pos_ratio:>10.1%}')

    # 按类别分组
    print(f'\n按类别分组（α=2.5）:')
    print(f'{"类别":<15} {"n":>4} {"mean_baseline":>14} {"mean_gain":>10} {"pos_ratio":>10}')
    print('-' * 60)

    from collections import defaultdict
    by_cls = defaultdict(list)
    for sample in all_samples:
        by_cls[sample['target_cls']].append(sample)

    alpha = 2.5
    for cls in sorted(by_cls.keys()):
        ss = by_cls[cls]
        gains = []
        baselines = []
        for sample in ss:
            ap_curve = sample['ap_curve']
            alphas_sorted = sorted([float(a) for a in ap_curve.keys()])
            ap_values = [ap_curve[str(a)] for a in alphas_sorted]
            ap = float(np.interp(alpha, alphas_sorted, ap_values))
            baseline = ap_curve['0.0']
            gains.append(ap - baseline)
            baselines.append(baseline)
        print(f'{cls:<15} {len(ss):>4} {np.mean(baselines):>14.4f} {np.mean(gains):>+10.4f} '
              f'{np.mean([1 if g > 0.001 else 0 for g in gains]):>10.1%}')

    # 保存结果
    out_file = DATA / 'fixed_alpha_table.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n固定-α表已保存到: {out_file}')

    return results


def main():
    # 1. person网格延到15 + AP75/框数sanity check
    person_results = process_person_extended()

    # 2. 固定-α版本完整表
    fixed_alpha_results = generate_fixed_alpha_table()

    print('\n' + '=' * 80)
    print('全部完成！')
    print('=' * 80)


if __name__ == '__main__':
    main()
