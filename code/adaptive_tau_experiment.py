# -*- coding: utf-8 -*-
"""adaptive_tau_experiment.py — 自适应门控温度τ验证实验
对4个域对分别搜索最优τ，看看不同域对的最优τ差异大不大
如果差异大，说明自适应τ有意义；如果差异小，固定τ就够了
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
from pathlib import Path

from usc_experiment_v14_softgating_fast import (
    DATA, T_OFFSET, TAU_M_RANGE, TAU_D_RANGE,
    SoftGating, load_all_domains, evaluate_domain_with_gate,
)

OUT = DATA / 'adaptive_tau_experiment_results.json'


def main():
    print("=" * 70)
    print("自适应门控温度τ验证实验")
    print("=" * 70)

    # 阶段1：加载所有数据
    all_domains = load_all_domains()
    source_names = list(all_domains.keys())
    print(f"\n共加载 {len(source_names)} 个域对: {source_names}")

    # ============================================================
    # 实验1：对每个域对分别搜索最优τ
    # ============================================================
    print("\n" + "=" * 70)
    print("[实验1] 每个域对分别搜索最优τ (tau_m, tau_d)")
    print("=" * 70)

    domain_optimal_tau = {}
    domain_tau_grid = {}

    for src in source_names:
        print(f"\n--- 搜索 {src}→COCO 的最优τ ---")
        best_ap = -1
        best_tau_m = None
        best_tau_d = None
        grid_results = {}

        for tau_m in TAU_M_RANGE:
            for tau_d in TAU_D_RANGE:
                gate = SoftGating(T_offset=T_OFFSET, tau_m=tau_m, tau_d=tau_d, use_soft=True)
                _, avg_ap, _ = evaluate_domain_with_gate(all_domains[src], gate, 0.5)
                grid_results[f"{tau_m}_{tau_d}"] = float(avg_ap)
                if avg_ap > best_ap:
                    best_ap = avg_ap
                    best_tau_m = tau_m
                    best_tau_d = tau_d

        domain_optimal_tau[src] = {
            'tau_m': best_tau_m,
            'tau_d': best_tau_d,
            'avg_ap': best_ap,
        }
        domain_tau_grid[src] = grid_results
        print(f"  最优τ: tau_m={best_tau_m}, tau_d={best_tau_d}, AP50={best_ap:.4f}")

    # ============================================================
    # 实验2：全局最优τ（所有域对一起搜索）
    # ============================================================
    print("\n" + "=" * 70)
    print("[实验2] 全局最优τ（所有域对平均）")
    print("=" * 70)

    global_best_ap = -1
    global_best_tau_m = None
    global_best_tau_d = None
    global_grid = {}

    for tau_m in TAU_M_RANGE:
        for tau_d in TAU_D_RANGE:
            gate = SoftGating(T_offset=T_OFFSET, tau_m=tau_m, tau_d=tau_d, use_soft=True)
            aps = []
            for src in source_names:
                _, avg_ap, _ = evaluate_domain_with_gate(all_domains[src], gate, 0.5)
                aps.append(avg_ap)
            mean_ap = float(np.mean(aps))
            global_grid[f"{tau_m}_{tau_d}"] = mean_ap
            if mean_ap > global_best_ap:
                global_best_ap = mean_ap
                global_best_tau_m = tau_m
                global_best_tau_d = tau_d

    print(f"  全局最优τ: tau_m={global_best_tau_m}, tau_d={global_best_tau_d}, 平均AP50={global_best_ap:.4f}")

    # ============================================================
    # 实验3：对比 域专属最优τ vs 全局最优τ
    # ============================================================
    print("\n" + "=" * 70)
    print("[实验3] 对比：域专属最优τ vs 全局最优τ")
    print("=" * 70)

    comparison = {}
    domain_specific_aps = []
    global_aps = []

    global_gate = SoftGating(
        T_offset=T_OFFSET, tau_m=global_best_tau_m, tau_d=global_best_tau_d, use_soft=True
    )

    for src in source_names:
        # 域专属最优τ
        opt = domain_optimal_tau[src]
        domain_gate = SoftGating(
            T_offset=T_OFFSET, tau_m=opt['tau_m'], tau_d=opt['tau_d'], use_soft=True
        )
        _, domain_ap, _ = evaluate_domain_with_gate(all_domains[src], domain_gate, 0.5)

        # 全局最优τ
        _, global_ap, _ = evaluate_domain_with_gate(all_domains[src], global_gate, 0.5)

        diff = domain_ap - global_ap
        comparison[src] = {
            'domain_specific_tau': {'tau_m': opt['tau_m'], 'tau_d': opt['tau_d']},
            'domain_specific_ap': domain_ap,
            'global_tau': {'tau_m': global_best_tau_m, 'tau_d': global_best_tau_d},
            'global_ap': global_ap,
            'diff': diff,
        }
        domain_specific_aps.append(domain_ap)
        global_aps.append(global_ap)
        print(f"  {src}:")
        print(f"    域专属τ=({opt['tau_m']},{opt['tau_d']}), AP={domain_ap:.4f}")
        print(f"    全局τ=({global_best_tau_m},{global_best_tau_d}), AP={global_ap:.4f}")
        print(f"    差异: {diff:+.4f}")

    mean_domain = float(np.mean(domain_specific_aps))
    mean_global = float(np.mean(global_aps))
    print(f"\n  总体平均:")
    print(f"    域专属最优τ: {mean_domain:.4f}")
    print(f"    全局最优τ: {mean_global:.4f}")
    print(f"    自适应τ增益: {mean_domain - mean_global:+.4f}")

    # ============================================================
    # 分析：不同域对最优τ的差异
    # ============================================================
    print("\n" + "=" * 70)
    print("[分析] 不同域对最优τ的差异")
    print("=" * 70)

    tau_m_values = [domain_optimal_tau[src]['tau_m'] for src in source_names]
    tau_d_values = [domain_optimal_tau[src]['tau_d'] for src in source_names]

    print(f"  各域对最优tau_m: {tau_m_values}")
    print(f"  各域对最优tau_d: {tau_d_values}")
    print(f"  tau_m 范围: [{min(tau_m_values)}, {max(tau_m_values)}], 差异={max(tau_m_values)-min(tau_m_values)}")
    print(f"  tau_d 范围: [{min(tau_d_values)}, {max(tau_d_values)}], 差异={max(tau_d_values)-min(tau_d_values)}")
    print(f"  tau_m 标准差: {np.std(tau_m_values):.4f}")
    print(f"  tau_d 标准差: {np.std(tau_d_values):.4f}")

    # 判断是否所有域对最优τ都相同
    all_same_m = len(set(tau_m_values)) == 1
    all_same_d = len(set(tau_d_values)) == 1
    print(f"\n  所有域对最优tau_m相同: {all_same_m}")
    print(f"  所有域对最优tau_d相同: {all_same_d}")

    if all_same_m and all_same_d:
        print("\n  ❌ 结论：所有域对最优τ相同，自适应τ没有意义，固定τ就够了")
    else:
        print("\n  ✅ 结论：不同域对最优τ有差异，自适应τ有意义！")
        if mean_domain - mean_global > 0.001:
            print(f"     自适应τ相比全局τ有 {mean_domain-mean_global:+.4f} 的增益")
        else:
            print(f"     但自适应τ增益很小 ({mean_domain-mean_global:+.4f})，实际意义有限")

    # 保存结果
    final_results = {
        'domain_optimal_tau': domain_optimal_tau,
        'global_optimal_tau': {
            'tau_m': global_best_tau_m,
            'tau_d': global_best_tau_d,
            'avg_ap': global_best_ap,
        },
        'comparison': comparison,
        'analysis': {
            'tau_m_values': tau_m_values,
            'tau_d_values': tau_d_values,
            'tau_m_range': [min(tau_m_values), max(tau_m_values)],
            'tau_d_range': [min(tau_d_values), max(tau_d_values)],
            'tau_m_std': float(np.std(tau_m_values)),
            'tau_d_std': float(np.std(tau_d_values)),
            'all_same_m': all_same_m,
            'all_same_d': all_same_d,
            'mean_domain_specific_ap': mean_domain,
            'mean_global_ap': mean_global,
            'adaptive_tau_gain': mean_domain - mean_global,
        },
        'domain_tau_grid': domain_tau_grid,
        'global_tau_grid': global_grid,
    }

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {OUT}")


if __name__ == '__main__':
    main()
