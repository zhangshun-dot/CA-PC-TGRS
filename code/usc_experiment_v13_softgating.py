# -*- coding: utf-8 -*-
"""usc_experiment_v13_softgating.py — 软门控升级实验
实验内容：
  1. 硬门控 vs 软门控 对比（4个域对）
  2. 温度系数网格扫描 (tau_m, tau_d)
  3. 向后兼容验证 (tau→0 时软门控退化为硬门控)
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import pickle
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import unified_eval as U
import tge_probe as T
from scipy.stats import spearmanr

# 复用v12的数据加载函数
from usc_experiment_v12 import (
    DATA, SRC_TMPL_OVERHEAD, TGT_TMPL, K_PCA, ALPHAS,
    SOURCE_LOADERS, load_target_coco, USC, compute_G,
    build_proto, build_neg_proto, evaluate, evaluate_baseline,
    REMOTE_TARGETS, DOTA_TARGETS, VISDRONE_TARGETS,
)

OUT = DATA / 'usc_experiment_v13_softgating_results.json'

BETA = 1.0
LAM = 0.25
PRE_FILTER = 0.03
T_OFFSET = 0.2644  # 域偏移阈值（v6实验确定）

# 温度扫描范围
TAU_M_RANGE = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]
TAU_D_RANGE = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]


# ============================================================
# 软门控模块
# ============================================================
class SoftGating:
    """温度可控解析软门控"""
    def __init__(self, T_offset=0.2644, tau_m=0.08, tau_d=0.08, use_soft=True):
        self.T_offset = T_offset
        self.tau_m = tau_m
        self.tau_d = tau_d
        self.use_soft = use_soft

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def compute_magnitude_score(self, delta_norm):
        if not self.use_soft or self.tau_m < 1e-6:
            result = delta_norm > self.T_offset
            return float(result) if np.isscalar(result) else float(result)
        x = (delta_norm - self.T_offset) / self.tau_m
        return float(self.sigmoid(x))

    def compute_direction_score(self, g2):
        if not self.use_soft or self.tau_d < 1e-6:
            result = g2 > 0
            return float(result) if np.isscalar(result) else float(result)
        x = g2 / self.tau_d
        return float(self.sigmoid(x))

    def forward(self, delta_norm, g2):
        s_m = self.compute_magnitude_score(delta_norm)
        s_d = self.compute_direction_score(g2)
        g_soft = float(s_m * s_d)
        return float(np.clip(g_soft, 0.0, 1.0)), {'s_m': s_m, 's_d': s_d}


def compute_g2(usc, src_cls, tgt_cls, src_tmpl=SRC_TMPL_OVERHEAD):
    """计算方向门控 g2 = cos(文本外推方向投影, 域偏移方向)"""
    s_emb = T.text_avg([src_tmpl], src_cls).cpu().numpy()
    d_emb = T.text_avg([TGT_TMPL], tgt_cls).cpu().numpy()
    delta_text = d_emb - s_emb
    # 投影到低秩空间
    delta_proj = usc._project(delta_text.reshape(1, -1))[0]
    # 计算余弦相似度
    cos_sim = float(np.dot(delta_proj, usc.domain_shift_) /
                    (np.linalg.norm(delta_proj) * np.linalg.norm(usc.domain_shift_) + 1e-8))
    return cos_sim


def evaluate_with_gate(target_data, proto, neg_proto, lam=LAM):
    """带门控的评估（和v12的evaluate相同，这里复用）"""
    return evaluate(target_data, proto, neg_proto, lam)


def run_domain_experiment(source_name, gate, alpha_fixed=0.5):
    """
    对单个域对运行实验，使用指定的门控和固定α

    返回：
        per_class_ap: dict, 每个类别的AP
        avg_ap: float, 平均AP
        gate_info: dict, 门控信息
    """
    load_func, targets = SOURCE_LOADERS[source_name]

    # 加载源域数据
    source_data = {}
    for eval_name, coco_cls, src_cls in targets:
        data = load_func(src_cls, n=50)
        if data is not None:
            source_data[src_cls] = data

    # 加载目标域数据
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
        return None, None, None

    # 拟合USC
    all_src = np.vstack([source_data[s]['features'] for _, _, s in valid_targets])
    all_tgt = np.vstack([target_data[c]['features'] for _, c, _ in valid_targets])
    usc = USC(n_components=K_PCA).fit(all_src, all_tgt)
    all_protos = {s: source_data[s]['prototype'] for _, _, s in valid_targets}

    delta_norm = usc.domain_shift_norm_

    per_class_ap = {}
    gate_info = {}

    for eval_name, coco_cls, src_cls in valid_targets:
        # 计算g2
        g2 = compute_g2(usc, src_cls, coco_cls)

        # 计算门控权重
        G_soft, gi = gate.forward(delta_norm, g2)
        effective_alpha = G_soft * alpha_fixed

        # USC校正
        if effective_alpha < 1e-6:
            src_corr = source_data[src_cls]['prototype'].copy()
            neg = build_neg_proto(all_protos, src_cls)
        else:
            src_corr = usc.correct_prototype(source_data[src_cls]['prototype'], effective_alpha)
            neg_raw = build_neg_proto(all_protos, src_cls)
            neg = usc.correct_prototype(neg_raw, effective_alpha) if neg_raw is not None else None

        # 文本外推
        p_hat = build_proto(src_corr, src_cls, coco_cls, SRC_TMPL_OVERHEAD, use_tge=True)

        # 评估
        ap = evaluate(target_data[coco_cls], p_hat, neg)
        per_class_ap[coco_cls] = float(ap)
        gate_info[coco_cls] = {
            'G_soft': G_soft, 's_m': gi['s_m'], 's_d': gi['s_d'],
            'effective_alpha': effective_alpha, 'delta_norm': float(delta_norm), 'g2': float(g2),
        }

    avg_ap = float(np.mean(list(per_class_ap.values())))
    return per_class_ap, avg_ap, gate_info


def main():
    print("=" * 70)
    print("USC 软门控升级实验 v13")
    print("=" * 70)

    source_names = list(SOURCE_LOADERS.keys())
    print(f"域对: {source_names}")

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
        print(f"  运行 {src}→COCO...")
        per_class, avg, gate_info = run_domain_experiment(src, hard_gate, alpha_fixed=0.5)
        if per_class is not None:
            hard_results[src] = {'per_class': per_class, 'avg': avg}
            hard_gate_info_all[src] = gate_info
            print(f"    平均AP50: {avg:.4f}")
        else:
            print(f"    跳过（有效类别不足）")

    hard_avg = float(np.mean([v['avg'] for v in hard_results.values()]))
    print(f"\n  硬门控总体平均AP50: {hard_avg:.4f}")

    # 软门控（默认温度）
    print("\n--- 软门控 (tau_m=0.08, tau_d=0.08) ---")
    soft_gate = SoftGating(T_offset=T_OFFSET, tau_m=0.08, tau_d=0.08, use_soft=True)
    soft_results = {}
    soft_gate_info_all = {}
    for src in source_names:
        print(f"  运行 {src}→COCO...")
        per_class, avg, gate_info = run_domain_experiment(src, soft_gate, alpha_fixed=0.5)
        if per_class is not None:
            soft_results[src] = {'per_class': per_class, 'avg': avg}
            soft_gate_info_all[src] = gate_info
            print(f"    平均AP50: {avg:.4f}")
        else:
            print(f"    跳过（有效类别不足）")

    soft_avg = float(np.mean([v['avg'] for v in soft_results.values()]))
    print(f"\n  软门控总体平均AP50: {soft_avg:.4f}")
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
            print(f"  tau_m={tau_m:.2f}, tau_d={tau_d:.2f}...", end=' ')
            gate = SoftGating(T_offset=T_OFFSET, tau_m=tau_m, tau_d=tau_d, use_soft=True)
            domain_aps = []
            for src in source_names:
                _, avg, _ = run_domain_experiment(src, gate, alpha_fixed=0.5)
                if avg is not None:
                    domain_aps.append(avg)
            mean_ap = float(np.mean(domain_aps)) if domain_aps else 0.0
            sweep_results[f'tau_m={tau_m}_tau_d={tau_d}'] = {
                'tau_m': tau_m, 'tau_d': tau_d, 'avg_ap': mean_ap,
            }
            print(f"平均AP50={mean_ap:.4f}")
            if mean_ap > best_ap:
                best_ap = mean_ap
                best_tau = (tau_m, tau_d)

    print(f"\n  最佳温度: tau_m={best_tau[0]}, tau_d={best_tau[1]}, 平均AP50={best_ap:.4f}")

    all_results['experiment2_temperature_sweep'] = {
        'sweep_results': sweep_results,
        'best_tau_m': best_tau[0],
        'best_tau_d': best_tau[1],
        'best_ap': best_ap,
        'tau_m_range': TAU_M_RANGE,
        'tau_d_range': TAU_D_RANGE,
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
        print(f"  运行 {src}→COCO...")
        per_class, avg, gate_info = run_domain_experiment(src, compat_gate, alpha_fixed=0.5)
        if per_class is not None:
            compat_results[src] = {'per_class': per_class, 'avg': avg, 'gate_info': gate_info}
            print(f"    平均AP50: {avg:.4f}")

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
