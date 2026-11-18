# -*- coding: utf-8 -*-
"""ablation_study.py — 第二篇论文消融实验：证明每个模块的贡献"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
import unified_eval as U

# 直接从sota_comparison.py导入所有需要的函数和常量
from sota_comparison import (
    SOURCE_LOADERS, load_target_coco,
    DIOR_TARGETS, NWPU_TARGETS, DOTA_TARGETS, VISDRONE_TARGETS,
    build_proto_tge, evaluate_with_features, evaluate_baseline,
    compute_box_size_variance, compute_text_confusability,
    load_dual_factor_model, predict_alpha_adaptive, USC,
    BETA, K_PCA, DATA
)

OUT = DATA / 'ablation_study_results.json'

# 类别映射
COCO_MAP = {
    'airplane': 'airplane', 'ship': 'boat', 'vehicle': 'car',
    'plane': 'airplane', 'small-vehicle': 'car', 'large-vehicle': 'truck',
    'bicycle': 'bicycle', 'car': 'car', 'bus': 'bus', 'motor': 'motorcycle',
}

REMOTE_TARGETS = {
    'DIOR': DIOR_TARGETS,
    'NWPU': NWPU_TARGETS,
    'DOTA': DOTA_TARGETS,
    'VisDrone': VISDRONE_TARGETS,
}


def run_ablation():
    print('=' * 80)
    print('第二篇论文消融实验')
    print('=' * 80)

    model_params = load_dual_factor_model()
    if model_params:
        print(f'双因子模型已加载')
    else:
        print('警告: 双因子模型未找到，自适应α将使用默认值1.0')

    results = {}
    # 和SOTA对比完全一致：包含源域类别名+COCO类别名
    all_categories = set()
    for src_name in REMOTE_TARGETS:
        for item in REMOTE_TARGETS[src_name]:
            coco_cls, src_cls, _ = item
            all_categories.add(coco_cls)
            all_categories.add(src_cls)
    all_categories = sorted(list(all_categories))
    print(f"所有类别（含源域类别名）: {all_categories}")

    for src_name in REMOTE_TARGETS:
        print(f'\n{"=" * 60}')
        print(f'源域: {src_name}')
        print(f'{"=" * 60}')

        src_results = {}
        src_cls_list = REMOTE_TARGETS[src_name]  # 格式: [(coco_cls, src_cls, src_cls), ...]
        loader_fn = SOURCE_LOADERS[src_name][0]

        # 加载所有源域数据
        src_data = {}
        for item in src_cls_list:
            coco_cls, src_cls, _ = item
            print(f'  加载源域 {src_name}/{src_cls} -> COCO/{coco_cls}...')
            data = loader_fn(src_cls, n=50)
            if data is not None and 'features' in data and len(data['features']) > 0:
                src_data[src_cls] = data
                print(f'    成功: {len(data["features"])} 个裁剪')
            else:
                print(f'    警告: {src_cls} 无有效特征，跳过')

        for item in src_cls_list:
            coco_cls, src_cls, _ = item
            if src_cls not in src_data:
                continue

            print(f'\n  --- 类别: {src_cls} -> {coco_cls} ---')

            # 加载目标域数据（使用和SOTA对比完全相同的方式和参数）
            try:
                target_data = load_target_coco(coco_cls, n=200)
            except Exception as e:
                print(f'    目标域加载失败: {e}')
                continue

            if target_data is None or len(target_data['per_img']) == 0:
                print(f'    目标域无数据，跳过')
                continue

            src_feat = src_data[src_cls]['features']
            src_proto = src_data[src_cls]['prototype']
            tgt_feat = target_data['features']

            cls_results = {}

            # 配置1: Baseline
            ap_base = evaluate_baseline(target_data)
            cls_results['Baseline'] = ap_base
            print(f'    Baseline: {ap_base:.4f}')

            # 配置2: TGE (文本引导外推，无秩1仿射)
            proto_tge = build_proto_tge(src_proto, src_cls, coco_cls, beta=BETA)
            ap_tge = evaluate_with_features(target_data, proto_tge, use_neg=False)
            cls_results['TGE'] = ap_tge
            print(f'    TGE: {ap_tge:.4f} (增益: {ap_tge-ap_base:+.4f})')

            # 配置3: USC (固定α=1.0，无双因子预测)
            usc = USC(n_components=K_PCA)
            usc.fit(src_feat, tgt_feat)
            proto_usc_fixed = usc.correct_prototype(src_proto, alpha=1.0)
            ap_usc_fixed = evaluate_with_features(target_data, proto_usc_fixed, use_neg=False)
            cls_results['USC_fixed_alpha1.0'] = ap_usc_fixed
            print(f'    USC (α=1.0): {ap_usc_fixed:.4f} (增益: {ap_usc_fixed-ap_base:+.4f})')

            # 配置4: USC (固定α=0.5)
            proto_usc_05 = usc.correct_prototype(src_proto, alpha=0.5)
            ap_usc_05 = evaluate_with_features(target_data, proto_usc_05, use_neg=False)
            cls_results['USC_fixed_alpha0.5'] = ap_usc_05
            print(f'    USC (α=0.5): {ap_usc_05:.4f} (增益: {ap_usc_05-ap_base:+.4f})')

            # 配置5: USC (双因子自适应α)
            box_size_var = compute_box_size_variance(target_data)
            text_conf = compute_text_confusability(coco_cls, all_categories)
            alpha_adaptive = predict_alpha_adaptive(box_size_var, text_conf, model_params)
            proto_usc_adaptive = usc.correct_prototype(src_proto, alpha=alpha_adaptive)
            ap_usc_adaptive = evaluate_with_features(target_data, proto_usc_adaptive, use_neg=False)
            cls_results['USC_adaptive_alpha'] = ap_usc_adaptive
            cls_results['alpha_adaptive'] = alpha_adaptive
            cls_results['box_size_variance'] = box_size_var
            cls_results['text_confusability'] = text_conf
            print(f'    USC (自适应α={alpha_adaptive:.2f}, sv={box_size_var:.0f}, tc={text_conf:.3f}): {ap_usc_adaptive:.4f} (增益: {ap_usc_adaptive-ap_base:+.4f})')

            # 配置6: USC (Oracle最优α，扫描α找最优)
            best_alpha = 0.0
            best_ap = ap_base
            for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
                proto_scan = usc.correct_prototype(src_proto, alpha=alpha)
                ap_scan = evaluate_with_features(target_data, proto_scan, use_neg=False)
                if ap_scan > best_ap:
                    best_ap = ap_scan
                    best_alpha = alpha
            cls_results['USC_oracle_alpha'] = best_ap
            cls_results['oracle_alpha'] = best_alpha
            print(f'    USC (Oracle α={best_alpha:.1f}): {best_ap:.4f} (增益: {best_ap-ap_base:+.4f})')

            src_results[src_cls] = cls_results

        results[src_name] = src_results

    # 保存结果
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印汇总
    print(f'\n\n{"=" * 80}')
    print('消融实验汇总')
    print(f'{"=" * 80}')

    methods = ['Baseline', 'TGE', 'USC_fixed_alpha1.0', 'USC_fixed_alpha0.5',
               'USC_adaptive_alpha', 'USC_oracle_alpha']
    method_names = {
        'Baseline': 'Baseline',
        'TGE': 'TGE (文本外推)',
        'USC_fixed_alpha1.0': 'USC (α=1.0)',
        'USC_fixed_alpha0.5': 'USC (α=0.5)',
        'USC_adaptive_alpha': 'USC (双因子自适应)',
        'USC_oracle_alpha': 'USC (Oracle最优)',
    }

    # 按源域汇总
    for src_name in results:
        print(f'\n--- {src_name} ---')
        for method in methods:
            aps = [results[src_name][cls][method] for cls in results[src_name] if method in results[src_name][cls]]
            if aps:
                mean_ap = np.mean(aps)
                base_aps = [results[src_name][cls]['Baseline'] for cls in results[src_name]]
                mean_base = np.mean(base_aps)
                gain = mean_ap - mean_base
                print(f'  {method_names[method]:25s}: AP50={mean_ap:.4f}, 增益={gain:+.4f}')

    # 总体平均
    total_n = sum(len(results[s]) for s in results)
    print(f'\n--- 总体平均 (N={total_n}) ---')
    for method in methods:
        all_aps = []
        all_base = []
        for src_name in results:
            for cls in results[src_name]:
                if method in results[src_name][cls]:
                    all_aps.append(results[src_name][cls][method])
                    all_base.append(results[src_name][cls]['Baseline'])
        if all_aps:
            mean_ap = np.mean(all_aps)
            mean_base = np.mean(all_base)
            gain = mean_ap - mean_base
            print(f'  {method_names[method]:25s}: AP50={mean_ap:.4f}, 增益={gain:+.4f}')

    # Oracle实现率
    print(f'\n--- Oracle实现率分析 ---')
    adaptive_gains = []
    oracle_gains = []
    for src_name in results:
        for cls in results[src_name]:
            if 'USC_adaptive_alpha' in results[src_name][cls] and 'USC_oracle_alpha' in results[src_name][cls]:
                base = results[src_name][cls]['Baseline']
                adapt_gain = results[src_name][cls]['USC_adaptive_alpha'] - base
                oracle_gain = results[src_name][cls]['USC_oracle_alpha'] - base
                if oracle_gain > 0.001:
                    adaptive_gains.append(adapt_gain)
                    oracle_gains.append(oracle_gain)
    if adaptive_gains and oracle_gains:
        mean_adapt = np.mean(adaptive_gains)
        mean_oracle = np.mean(oracle_gains)
        realization_rate = mean_adapt / mean_oracle * 100 if mean_oracle > 0 else 0
        print(f'  有效类别数: {len(adaptive_gains)}')
        print(f'  自适应平均增益: {mean_adapt:+.4f}')
        print(f'  Oracle平均增益: {mean_oracle:+.4f}')
        print(f'  Oracle实现率: {realization_rate:.1f}%')

    # 模块贡献分析
    print(f'\n--- 模块贡献分析 ---')
    # 秩1仿射的贡献: TGE vs USC_fixed_alpha1.0
    tge_gains = []
    usc_fixed_gains = []
    for src_name in results:
        for cls in results[src_name]:
            base = results[src_name][cls]['Baseline']
            tge_gains.append(results[src_name][cls]['TGE'] - base)
            usc_fixed_gains.append(results[src_name][cls]['USC_fixed_alpha1.0'] - base)
    if tge_gains and usc_fixed_gains:
        rank1_contribution = np.mean(usc_fixed_gains) - np.mean(tge_gains)
        print(f'  秩1仿射的贡献 (USC α=1.0 - TGE): {rank1_contribution:+.4f}')

    # 双因子预测的贡献: USC_fixed_alpha1.0 vs USC_adaptive_alpha
    adaptive_gains_all = []
    for src_name in results:
        for cls in results[src_name]:
            base = results[src_name][cls]['Baseline']
            adaptive_gains_all.append(results[src_name][cls]['USC_adaptive_alpha'] - base)
    if adaptive_gains_all and usc_fixed_gains:
        dual_factor_contribution = np.mean(adaptive_gains_all) - np.mean(usc_fixed_gains)
        print(f'  双因子预测的贡献 (自适应 - α=1.0): {dual_factor_contribution:+.4f}')

    print(f'\n结果已保存: {OUT}')


if __name__ == '__main__':
    run_ablation()
