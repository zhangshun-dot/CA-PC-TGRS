# -*- coding: utf-8 -*-
"""usc_experiment_v14_softgating_fast.py — 软门控升级实验（优化版）
优化：数据加载和USC拟合只做一次并缓存，温度扫描只重新评估
实验内容：
  1. 硬门控 vs 软门控 对比（4个域对）
  2. 温度系数网格扫描 (tau_m, tau_d)
  3. 向后兼容验证 (tau→0)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
from pathlib import Path

# 复用v12的数据加载函数
from usc_experiment_v12 import (
    DATA, SRC_TMPL_OVERHEAD, TGT_TMPL, K_PCA, ALPHAS,
    SOURCE_LOADERS, load_target_coco, USC, compute_G,
    build_proto, build_neg_proto, evaluate, evaluate_baseline,
)

OUT = DATA / 'usc_experiment_v14_softgating_results.json'

BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
T_OFFSET = 0.2644

TAU_M_RANGE = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]
TAU_D_RANGE = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]


# ============================================================
# 软门控模块
# ============================================================
class SoftGating:
    def __init__(self, T_offset=0.2644, tau_m=0.08, tau_d=0.08, use_soft=True):
        self.T_offset = T_offset
        self.tau_m = tau_m
        self.tau_d = tau_d
        self.use_soft = use_soft

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, delta_norm, g2):
        if not self.use_soft or self.tau_m < 1e-6:
            s_m = 1.0 if delta_norm > self.T_offset else 0.0
        else:
            s_m = float(self.sigmoid((delta_norm - self.T_offset) / self.tau_m))

        if not self.use_soft or self.tau_d < 1e-6:
            s_d = 1.0 if g2 > 0 else 0.0
        else:
            s_d = float(self.sigmoid(g2 / self.tau_d))

        g_soft = float(np.clip(s_m * s_d, 0.0, 1.0))
        return g_soft, {'s_m': s_m, 's_d': s_d}


def compute_g2(usc, src_cls, tgt_cls, src_tmpl=SRC_TMPL_OVERHEAD):
    import tge_probe as T
    s_emb = T.text_avg([src_tmpl], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], tgt_cls).cpu().numpy()
    delta_text = d_emb - s_emb
    delta_proj = usc._project(delta_text.reshape(1, -1))[0]
    cos_sim = float(np.dot(delta_proj, usc.domain_shift_) /
                    (np.linalg.norm(delta_proj) * np.linalg.norm(usc.domain_shift_) + 1e-8))
    return cos_sim


# ============================================================
# 数据缓存：一次性加载所有域对
# ============================================================
def load_all_domains():
    """一次性加载所有4个域对的数据并缓存"""
    print("=" * 70)
    print("阶段1：加载所有域对数据（一次性）")
    print("=" * 70)

    all_domains = {}
    for source_name, (load_func, targets) in SOURCE_LOADERS.items():
        print(f"\n--- 加载 {source_name}→COCO ---")
        source_data = {}
        for eval_name, coco_cls, src_cls in targets:
            data = load_func(src_cls, n=50)
            if data is not None:
                source_data[src_cls] = data

        target_data = {}
        valid_targets = []
        for eval_name, coco_cls, src_cls in targets:
            if src_cls not in source_data:
                continue
            data = load_target_coco(coco_cls, n=200)
            if data is not None:
                target_data[coco_cls] = data
                valid_targets.append((eval_name, coco_cls, src_cls))

        if len(valid_targets) < 2:
            print(f"  跳过（有效类别不足）")
            continue

        all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
        all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
        usc = USC(n_components=K_PCA).fit(all_src, all_tgt)
        all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}

        # 预计算每个类别的g2和delta_norm
        class_info = {}
        delta_norm = usc.domain_shift_norm_
        for eval_name, coco_cls, src_cls in valid_targets:
            g2 = compute_g2(usc, src_cls, coco_cls)
            class_info[coco_cls] = {
                'src_cls': src_cls, 'g2': g2, 'delta_norm': float(delta_norm),
            }

        all_domains[source_name] = {
            'source_data': source_data,
            'target_data': target_data,
            'usc': usc,
            'all_protos': all_protos,
            'valid_targets': valid_targets,
            'class_info': class_info,
            'delta_norm': float(delta_norm),
        }
        print(f"  有效类别: {len(valid_targets)}, delta_norm={delta_norm:.4f}")

    print(f"\n共加载 {len(all_domains)} 个域对")
    return all_domains


def evaluate_domain_with_gate(domain_data, gate, alpha_fixed=0.5):
    """使用缓存数据和指定门控评估单个域对"""
    usc = domain_data['usc']
    source_data = domain_data['source_data']
    target_data = domain_data['target_data']
    all_protos = domain_data['all_protos']
    valid_targets = domain_data['valid_targets']
    class_info = domain_data['class_info']

    per_class_ap = {}
    gate_info = {}

    for eval_name, coco_cls, src_cls in valid_targets:
        info = class_info[coco_cls]
        delta_norm = info['delta_norm']
        g2 = info['g2']

        G_soft, gi = gate.forward(delta_norm, g2)
        effective_alpha = G_soft * alpha_fixed

        if effective_alpha < 1e-6:
            src_corr = source_data[src_cls]['prototype'].copy()
            neg = build_neg_proto(all_protos, src_cls)
        else:
            src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], effective_alpha)
            neg_raw = build_neg_proto(all_protos, src_cls)
            neg = usc.correct_prototype(neg_raw, effective_alpha) if neg_raw is not None else None

        p_hat = build_proto(src_corr, src_cls, coco_cls, SRC_TMPL_OVERHEAD, use_tge=True)
        ap = evaluate(target_data[coco_cls], p_hat, neg)

        per_class_ap[coco_cls] = float(ap)
        gate_info[coco_cls] = {
            'G_soft': G_soft, 's_m': gi['s_m'], 's_d': gi['s_d'],
            'effective_alpha': effective_alpha, 'g2': g2,
        }

    avg_ap = float(np.mean(list(per_class_ap.values())))
    return per_class_ap, avg_ap, gate_info


def main():
    # 阶段1：加载所有数据
    all_domains = load_all_domains()
    source_names = list(all_domains.keys())

    all_results = {}

    # ============================================================
    # 实验1：硬门控 vs 软门控 对比
    # ============================================================
    print("\n" + "=" * 70)
    print("[实验1] 硬门控 vs 软门控 对比 (固定α=0.5)")
    print("=" * 70)

    # 硬门控
    print("\n--- 硬门控 ---")
    hard_gate = SoftGating(T_offset=T_OFFSET, use_soft=False)
    hard_results = {}
    hard_gate_info_all = {}
    for src in source_names:
        per_class, avg, gate_info = evaluate_domain_with_gate(all_domains[src], hard_gate, 0.5)
        hard_results[src] = {'per_class': per_class, 'avg': avg}
        hard_gate_info_all[src] = gate_info
        print(f"  {src}: 平均AP50={avg:.4f}")

    hard_avg = float(np.mean([v['avg'] for v in hard_results.values()]))
    print(f"  硬门控总体平均AP50: {hard_avg:.4f}")

    # 软门控（默认温度）
    print("\n--- 软门控 (tau_m=0.08, tau_d=0.08) ---")
    soft_gate = SoftGating(T_offset=T_OFFSET, tau_m=0.08, tau_d=0.08, use_soft=True)
    soft_results = {}
    soft_gate_info_all = {}
    for src in source_names:
        per_class, avg, gate_info = evaluate_domain_with_gate(all_domains[src], soft_gate, 0.5)
        soft_results[src] = {'per_class': per_class, 'avg': avg}
        soft_gate_info_all[src] = gate_info
        print(f"  {src}: 平均AP50={avg:.4f}")

    soft_avg = float(np.mean([v['avg'] for v in soft_results.values()]))
    print(f"  软门控总体平均AP50: {soft_avg:.4f}")
    print(f"  软门控 vs 硬门控: {soft_avg - hard_avg:+.4f}")

    all_results['experiment1_hard_vs_soft'] = {
        'hard': {'results': hard_results, 'avg': hard_avg, 'gate_info': hard_gate_info_all},
        'soft': {'results': soft_results, 'avg': soft_avg, 'gate_info': soft_gate_info_all},
        'diff': soft_avg - hard_avg,
        'alpha_fixed': 0.5,
    }

    # ============================================================
    # 实验2：温度系数网格扫描
    # ============================================================
    print("\n" + "=" * 70)
    print("[实验2] 温度系数网格扫描")
    print("=" * 70)

    sweep_results = {}
    best_ap = -1
    best_tau = None

    for tau_m in TAU_M_RANGE:
        for tau_d in TAU_D_RANGE:
            gate = SoftGating(T_offset=T_OFFSET, tau_m=tau_m, tau_d=tau_d, use_soft=True)
            domain_aps = []
            for src in source_names:
                _, avg, _ = evaluate_domain_with_gate(all_domains[src], gate, 0.5)
                domain_aps.append(avg)
            mean_ap = float(np.mean(domain_aps))
            key = f'tau_m={tau_m}_tau_d={tau_d}'
            sweep_results[key] = {'tau_m': tau_m, 'tau_d': tau_d, 'avg_ap': mean_ap}
            print(f"  tau_m={tau_m:.2f}, tau_d={tau_d:.2f}: 平均AP50={mean_ap:.4f}")
            if mean_ap > best_ap:
                best_ap = mean_ap
                best_tau = (tau_m, tau_d)

    print(f"\n  最佳温度: tau_m={best_tau[0]}, tau_d={best_tau[1]}, 平均AP50={best_ap:.4f}")

    all_results['experiment2_temperature_sweep'] = {
        'sweep_results': sweep_results,
        'best_tau_m': best_tau[0],
        'best_tau_d': best_tau[1],
        'best_ap': best_ap,
    }

    # ============================================================
    # 实验3：向后兼容验证 (tau→0)
    # ============================================================
    print("\n" + "=" * 70)
    print("[实验3] 向后兼容验证 (tau→0)")
    print("=" * 70)

    compat_gate = SoftGating(T_offset=T_OFFSET, tau_m=0.001, tau_d=0.001, use_soft=True)
    compat_results = {}
    for src in source_names:
        per_class, avg, gate_info = evaluate_domain_with_gate(all_domains[src], compat_gate, 0.5)
        compat_results[src] = {'per_class': per_class, 'avg': avg, 'gate_info': gate_info}
        print(f"  {src}: 平均AP50={avg:.4f}")

    compat_avg = float(np.mean([v['avg'] for v in compat_results.values()]))
    diff_compat = compat_avg - hard_avg
    print(f"\n  tau→0 软门控平均AP50: {compat_avg:.4f}")
    print(f"  硬门控平均AP50: {hard_avg:.4f}")
    print(f"  差值: {diff_compat:+.6f} (应接近0)")

    # 检查门控值
    print("\n  门控值对比 (tau→0 应接近硬门控的0/1):")
    for src in compat_results:
        for cls, info in compat_results[src]['gate_info'].items():
            g_compat = info['G_soft']
            g_hard = hard_gate_info_all[src][cls]['G_soft']
            print(f"    {src}/{cls}: 软(tau→0)={g_compat:.4f}, 硬={g_hard:.4f}")

    all_results['experiment3_backward_compatibility'] = {
        'tau_near_zero': {'results': compat_results, 'avg': compat_avg},
        'hard_gate': {'avg': hard_avg},
        'diff': diff_compat,
    }

    # ============================================================
    # 保存结果
    # ============================================================
    print(f"\n保存结果到: {OUT}")

    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
    print(f"\n核心结果：")
    print(f"  硬门控平均AP50: {hard_avg:.4f}")
    print(f"  软门控平均AP50: {soft_avg:.4f}")
    print(f"  软门控增益: {soft_avg - hard_avg:+.4f}")
    print(f"  最佳温度: tau_m={best_tau[0]}, tau_d={best_tau[1]}")
    print(f"  向后兼容差值: {diff_compat:+.6f} (应接近0)")


if __name__ == '__main__':
    main()
