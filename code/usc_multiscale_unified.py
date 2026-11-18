# -*- coding: utf-8 -*-
"""多尺度校正统一参数实验
验证组合效应：用在开发域对上学到的统一参数，在所有域对上评估
对比：
  1. 原始空间秩2仿射（基线）
  2. 每个域对单独搜索最优参数（oracle上界）
  3. 统一参数（在DIOR上学到的参数，在所有域对上评估）
  4. 统一参数（在DOTA上学到的参数，在所有域对上评估）
  5. 统一参数（在VisDrone上学到的参数，在所有域对上评估）
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

from usc_experiment_v12 import (
    DATA, K_PCA, ALPHAS, LAM, SRC_TMPL_OVERHEAD, TGT_TMPL,
    SOURCE_LOADERS, load_target_coco, build_neg_proto, evaluate,
)
import tge_probe as T
from usc_dct_multiscale import MultiScaleUSC, load_domain_data, evaluate_with_params

OUT = DATA / 'usc_multiscale_unified_results.json'


def search_best_params(domain_data, n_scales=2, param_mode='independent'):
    """在一个域对上搜索最优参数"""
    best_ap = -1
    best_params = None

    if param_mode == 'independent':
        gamma_options = [-1.0, 0.0, 1.0]
        alpha_options = [0.0, 0.5, 1.0]
        theta_options = [-15, 0, 15]
        if n_scales == 2:
            for g0 in gamma_options:
                for g1 in gamma_options:
                    for a0 in alpha_options:
                        for a1 in alpha_options:
                            for t0 in theta_options:
                                for t1 in theta_options:
                                    ap = evaluate_with_params(
                                        domain_data, 0.0, 0.0, 0,
                                        [g0, g1], [a0, a1], [t0, t1]
                                    )
                                    if ap > best_ap:
                                        best_ap = ap
                                        best_params = ([g0, g1], [a0, a1], [t0, t1])
    return best_ap, best_params


def evaluate_with_fixed_params(domain_data, params, n_scales=2, param_mode='independent'):
    """用固定参数评估"""
    if param_mode == 'independent':
        gamma_list, alpha_list, theta_list = params
        ap = evaluate_with_params(
            domain_data, 0.0, 0.0, 0,
            gamma_list, alpha_list, theta_list
        )
        return ap
    return 0.0


def main():
    source_names = list(SOURCE_LOADERS.keys())
    print(f"域对列表: {source_names}")

    # 原始空间秩2仿射的最佳结果（基线）
    original_best = {
        'DIOR': 0.3884, 'NWPU': 0.3910, 'DOTA': 0.3903, 'VisDrone': 0.3443,
    }

    # 2尺度independent的oracle结果（每个域对单独搜索）
    oracle_best = {
        'DIOR': 0.3876, 'NWPU': 0.3866, 'DOTA': 0.3942, 'VisDrone': 0.3505,
    }

    n_scales = 2
    param_mode = 'independent'

    # 第一步：在每个域对上搜索最优参数
    print(f"\n{'='*60}")
    print(f"第一步：在每个域对上搜索最优参数")
    print(f"{'='*60}")

    domain_data_cache = {}
    best_params_per_domain = {}

    for src in source_names:
        print(f"\n--- {src}→COCO ---")
        usc = MultiScaleUSC(n_scales=n_scales, param_mode=param_mode,
                             use_pca=True, n_components=20)
        data = load_domain_data(src, usc)
        if data is None:
            continue
        domain_data_cache[src] = data

        best_ap, best_params = search_best_params(data, n_scales, param_mode)
        best_params_per_domain[src] = best_params
        print(f"  最佳AP: {best_ap:.4f}, 参数: {best_params}")

        del data, usc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 第二步：用每个域对学到的参数，在所有域对上评估（参数迁移矩阵）
    print(f"\n{'='*60}")
    print(f"第二步：参数迁移矩阵（行=参数来源，列=评估域对）")
    print(f"{'='*60}")

    transfer_matrix = {}
    for src_train in source_names:
        if src_train not in best_params_per_domain:
            continue
        params = best_params_per_domain[src_train]
        print(f"\n参数来源: {src_train}, 参数: {params}")
        transfer_matrix[src_train] = {}

        for src_eval in source_names:
            if src_eval not in domain_data_cache:
                # 需要重新加载
                usc = MultiScaleUSC(n_scales=n_scales, param_mode=param_mode,
                                     use_pca=True, n_components=20)
                data = load_domain_data(src_eval, usc)
                if data is None:
                    continue
                domain_data_cache[src_eval] = data

            data = domain_data_cache[src_eval]
            ap = evaluate_with_fixed_params(data, params, n_scales, param_mode)
            transfer_matrix[src_train][src_eval] = ap
            gain = ap - original_best[src_eval]
            print(f"  评估 {src_eval}: AP={ap:.4f}, vs原始={gain:+.4f}")

    # 第三步：汇总对比
    print(f"\n{'='*60}")
    print(f"第三步：汇总对比")
    print(f"{'='*60}")

    print(f"\n参数迁移矩阵：")
    print(f"{'参数来源':<12}", end="")
    for src_eval in source_names:
        print(f"{src_eval:>10}", end="")
    print(f"{'平均':>10}")
    print("-" * 70)

    for src_train in source_names:
        if src_train not in transfer_matrix:
            continue
        print(f"{src_train:<12}", end="")
        aps = []
        for src_eval in source_names:
            ap = transfer_matrix[src_train].get(src_eval, 0)
            aps.append(ap)
            print(f"{ap:>10.4f}", end="")
        mean_ap = np.mean(aps)
        print(f"{mean_ap:>10.4f}")

    print(f"\n对比汇总：")
    print(f"{'方法':<30} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8} {'vs原始':>10}")
    print("-" * 90)

    # 原始空间
    aps = [original_best[s] for s in source_names]
    print(f"{'原始空间秩2仿射（基线）':<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f} {'+0.0000':>10}")

    # Oracle上界（每个域对单独搜索）
    aps = [oracle_best[s] for s in source_names]
    print(f"{'Oracle上界（单独调参）':<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f} {np.mean(aps)-np.mean([original_best[s] for s in source_names]):>+10.4f}")

    # 统一参数（每个来源的平均）
    for src_train in source_names:
        if src_train not in transfer_matrix:
            continue
        aps = [transfer_matrix[src_train].get(s, 0) for s in source_names]
        gain = np.mean(aps) - np.mean([original_best[s] for s in source_names])
        print(f"{'统一参数（来源'+src_train+'）':<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f} {gain:>+10.4f}")

    # 保存结果
    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, tuple):
            return [convert(v) for v in obj]
        return obj

    final_results = {
        'original_baseline': convert(original_best),
        'oracle_best': convert(oracle_best),
        'best_params_per_domain': convert(best_params_per_domain),
        'transfer_matrix': convert(transfer_matrix),
        'n_domains': len(source_names),
        'note': '多尺度校正统一参数实验：验证组合效应',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(final_results), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
