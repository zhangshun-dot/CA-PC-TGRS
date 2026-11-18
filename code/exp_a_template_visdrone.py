# -*- coding: utf-8 -*-
"""实验A：传感器感知文本模板对VisDrone的影响
对比：'an overhead image of {}' vs 'an aerial view of {}' vs 'a drone view of {}'
只跑VisDrone→COCO，4个类别
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from usc_experiment_v12 import (
    K_PCA, ALPHAS, LAM, USC, build_neg_proto, evaluate, evaluate_baseline,
    load_visdrone_src, load_target_coco, VISDRONE_TARGETS,
)
import tge_probe as T

def run_visdrone_with_template(src_tmpl, alpha_fixed=0.5):
    """用指定源域模板跑VisDrone→COCO"""
    print(f"\n--- 模板: '{src_tmpl}' ---")

    # 加载源域
    source_data = {}
    for eval_name, coco_cls, src_cls in VISDRONE_TARGETS:
        data = load_visdrone_src(src_cls, n=50)
        if data is not None:
            source_data[src_cls] = data

    # 加载目标域
    target_data = {}
    valid_targets = []
    for eval_name, coco_cls, src_cls in VISDRONE_TARGETS:
        if src_cls not in source_data:
            continue
        data = load_target_coco(coco_cls, n=200)
        if data is not None:
            target_data[coco_cls] = data
            valid_targets.append((eval_name, coco_cls, src_cls))

    # 拟合USC
    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
    usc = USC(n_components=K_PCA).fit(all_src, all_tgt)
    all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}

    delta_norm = usc.domain_shift_norm_
    print(f"  delta_norm={delta_norm:.4f}")

    # 计算每个类别的g2和AP
    results = {}
    for eval_name, coco_cls, src_cls in valid_targets:
        # g2
        s_emb = T.text_avg([src_tmpl], src_cls).cpu().numpy()
        d_emb = T.text_avg(['a photo of {}'], coco_cls).cpu().numpy()
        delta_text = d_emb - s_emb
        delta_proj = usc._project(delta_text.reshape(1, -1))[0]
        g2 = float(np.dot(delta_proj, usc.domain_shift_) /
                   (np.linalg.norm(delta_proj) * np.linalg.norm(usc.domain_shift_) + 1e-8))

        # 基线
        baseline_ap = evaluate_baseline(target_data[coco_cls])

        # 扫描alpha找最佳
        best_alpha = 0.0
        best_ap = -1
        alpha_aps = {}
        for alpha in ALPHAS:
            if alpha == 0.0:
                src_corr = source_data[src_cls]['prototype'].copy()
                neg = build_neg_proto(all_protos, src_cls)
            else:
                src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], alpha)
                neg_raw = build_neg_proto(all_protos, src_cls)
                neg = usc.correct_prototype(neg_raw, alpha) if neg_raw is not None else None

            # 用指定模板做文本外推
            s_emb_c = T.text_avg([src_tmpl], src_cls).cpu().numpy()
            d_emb_c = T.text_avg(['a photo of {}'], coco_cls).cpu().numpy()
            delta_c = d_emb_c - s_emb_c
            p_hat = src_corr + 1.0 * delta_c
            p_hat = p_hat / (np.linalg.norm(p_hat) + 1e-8)

            ap = evaluate(target_data[coco_cls], p_hat, neg)
            alpha_aps[alpha] = ap
            if ap > best_ap:
                best_ap = ap
                best_alpha = alpha

        # 固定alpha=0.5的结果
        ap_fixed = alpha_aps.get(0.5, alpha_aps[0.0])

        results[coco_cls] = {
            'g2': g2, 'baseline_ap': baseline_ap,
            'best_alpha': best_alpha, 'best_ap': best_ap,
            'ap_fixed_0.5': ap_fixed,
        }
        print(f"  {coco_cls}: g2={g2:.4f}, baseline={baseline_ap:.4f}, "
              f"bestα={best_alpha}, bestAP={best_ap:.4f}, α=0.5→{ap_fixed:.4f}")

    avg_best = np.mean([r['best_ap'] for r in results.values()])
    avg_fixed = np.mean([r['ap_fixed_0.5'] for r in results.values()])
    avg_g2 = np.mean([r['g2'] for r in results.values()])
    print(f"  平均: g2={avg_g2:.4f}, bestAP={avg_best:.4f}, α=0.5→{avg_fixed:.4f}")

    return results, avg_best, avg_fixed, avg_g2


if __name__ == '__main__':
    templates = [
        'an overhead image of {}',
        'an aerial view of {}',
        'a drone view of {}',
        'a bird-eye view of {}',
    ]

    all_results = {}
    for tmpl in templates:
        results, avg_best, avg_fixed, avg_g2 = run_visdrone_with_template(tmpl)
        all_results[tmpl] = {
            'avg_best_ap': avg_best,
            'avg_fixed_0.5': avg_fixed,
            'avg_g2': avg_g2,
            'per_class': results,
        }

    print("\n" + "=" * 70)
    print("模板对比汇总")
    print("=" * 70)
    print(f"{'模板':<35} {'avg_g2':>8} {'avg_bestAP':>12} {'α=0.5 AP':>10}")
    print("-" * 70)
    for tmpl, r in all_results.items():
        print(f"{tmpl:<35} {r['avg_g2']:>8.4f} {r['avg_best_ap']:>12.4f} {r['avg_fixed_0.5']:>10.4f}")
