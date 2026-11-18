# -*- coding: utf-8 -*-
"""DTDM 对比评估

三路对比：
  1. Static CD-TDM:  每类固定 top-2 OFAT 修饰词
  2. DTDM (uncalibrated): 默认 τ=0.03, CLIP 动态选择
  3. DTDM (calibrated):   校准后的 τ_a(D)
  4. Oracle:              使用 ground-truth δ → 选最优修饰词（性能上界）

评估方式：
  - 模拟评估：基于 OFAT δ 加性假设，计算期望 F1
    F1_expected = baseline_F1 + Σ δ(activated_axes)
  - 对比 Static vs DTDM 的 per-class ΔF1

输出：
  - 汇总对比表
  - Per-class 详细对比
  - DTDM 决策可解释性分析
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from dtdm_axis_descriptions import AXIS_ORDER, AXIS_MODIFIERS
from ofat_knowledge_base import OFATKnowledgeBase

# ── 路径 ───────────────────────────────────────────────
BASE = Path(r'C:\Users\张三\Desktop')
CROSS_VAL_PATH = BASE / '实验数据' / 'cross_domain_validation.json'
CALIB_PATH = BASE / '实验数据' / 'dtdm_calibration.json'
OUTPUT_PATH = BASE / '实验数据' / 'dtdm_evaluation.json'

# ── OFAT 加性假设 ──
# 基于第一篇论文的验证：修饰轴效应近似加性（cos(ε) = -0.77, CV = 0.047）
# F1_expected(base + modifiers) ≈ F1_base + Σ δ(axis)


def expected_f1(baseline_f1, activated_deltas):
    """计算期望 F1（加性假设）。"""
    return baseline_f1 + sum(activated_deltas)


def evaluate_static_tdm(kb, domain, class_name):
    """Static CD-TDM: 每类固定 top-2 OFAT 修饰词。

    Returns:
        (modifiers, expected_delta, expected_f1)
    """
    recs = kb.get_recommended_axes(domain, class_name)
    modifiers = [AXIS_MODIFIERS[a] for a, delta in recs[:2]]
    deltas = [delta for a, delta in recs[:2]]

    # 获取 baseline F1
    if domain == 'natural_scene':
        base_f1 = 0  # COCO 用 DR 度量，用 0 占位
        # 从 OFAT 数据推算
        coco_base = kb.coco['classes'].get(class_name, {}).get('base', 0)
        base_f1 = coco_base
    else:
        base_f1 = kb.dior_baseline.get(class_name, 0)

    exp_f1 = base_f1 + sum(deltas)
    return modifiers, deltas, base_f1, exp_f1


def evaluate_oracle(kb, domain, class_name):
    """Oracle: 用 ground-truth δ 选所有正增益轴（性能上界）。

    Returns:
        (modifiers, expected_delta, expected_f1)
    """
    all_deltas = []
    for axis in AXIS_ORDER:
        delta = kb.get_delta(domain, class_name, axis)
        if delta is not None and delta > 0:
            all_deltas.append((axis, delta))

    all_deltas.sort(key=lambda x: -x[1])
    # Oracle 不受 S-N 约束（理论最优）
    modifiers = [AXIS_MODIFIERS[a] for a, _ in all_deltas[:2]]
    deltas = [d for _, d in all_deltas[:2]]

    if domain == 'natural_scene':
        base_f1 = kb.coco['classes'].get(class_name, {}).get('base', 0)
    else:
        base_f1 = kb.dior_baseline.get(class_name, 0)

    exp_f1 = base_f1 + sum(deltas)
    return modifiers, deltas, base_f1, exp_f1


def evaluate_dtdm(dtdm, kb, domain, class_name, sample_image=None):
    """DTDM 评估（模拟模式，用 CLIP 相似度 + OFAT δ 做决策）。

    如果有 sample_image，用真实 CLIP 匹配；否则用类级统计估算。

    Returns:
        (modifiers, expected_delta, expected_f1, decisions_dict)
    """
    if sample_image is not None:
        # 真实 DTDM 推理
        modifiers, decision = dtdm.select_modifiers(
            sample_image, class_name, domain_hint=domain, return_decision=True
        )
        # 计算期望 F1
        if domain == 'natural_scene':
            base_f1 = kb.coco['classes'].get(class_name, {}).get('base', 0)
        else:
            base_f1 = kb.dior_baseline.get(class_name, 0)

        deltas = []
        for mod in modifiers:
            # 反向查找 modifier → axis
            for axis, mod_text in AXIS_MODIFIERS.items():
                if mod_text == mod:
                    delta = kb.get_delta(domain, class_name, axis) or 0
                    deltas.append(delta)
                    break

        exp_f1 = base_f1 + sum(deltas)
        decisions = decision.decisions
        return modifiers, deltas, base_f1, exp_f1, decisions
    else:
        # 无图像时用统计估算
        # 注意：universal 只保证"跨域稳定"，不保证"始终正向"
        # 修正：直接用 get_recommended_axes（δ > threshold 过滤），
        #       不再优先 universal 轴（它们在 DIOR 上 δ 可能是负的）
        recs = kb.get_recommended_axes(domain, class_name)
        # 按 δ 降序排列（和 Static 行为一致，但后续 CLIP 匹配会做差异化）
        recs.sort(key=lambda x: -x[1])
        modifiers = [AXIS_MODIFIERS[a] for a, _ in recs[:2]]
        deltas = [d for _, d in recs[:2]]

        if domain == 'natural_scene':
            base_f1 = kb.coco['classes'].get(class_name, {}).get('base', 0)
        else:
            base_f1 = kb.dior_baseline.get(class_name, 0)

        exp_f1 = base_f1 + sum(deltas)
        return modifiers, deltas, base_f1, exp_f1, {}


def run_comparison(dtdm, kb, domain, classes):
    """运行完整的三路对比评估。

    Args:
        dtdm: DynamicTDM 实例（可为 None, 此时 DTDM 用统计估算）
        kb: OFATKnowledgeBase
        domain: 域标签
        classes: 类名列表

    Returns:
        dict: 评估结果
    """
    results = {
        'domain': domain,
        'static': {'per_class': {}, 'modifiers': {}, 'base_f1': {}, 'exp_f1': {}},
        'dtdm': {'per_class': {}, 'modifiers': {}, 'base_f1': {}, 'exp_f1': {}},
        'oracle': {'per_class': {}, 'modifiers': {}, 'base_f1': {}, 'exp_f1': {}},
    }

    for cls_name in classes:
        # Static
        s_mods, s_deltas, s_base, s_exp = evaluate_static_tdm(kb, domain, cls_name)
        results['static']['modifiers'][cls_name] = s_mods
        results['static']['base_f1'][cls_name] = s_base
        results['static']['exp_f1'][cls_name] = s_exp
        results['static']['per_class'][cls_name] = {
            'modifiers': s_mods, 'deltas': s_deltas,
            'base_f1': s_base, 'exp_f1': s_exp,
        }

        # DTDM (统计估算)
        d_mods, d_deltas, d_base, d_exp, d_decisions = evaluate_dtdm(
            dtdm, kb, domain, cls_name, sample_image=None
        )
        results['dtdm']['modifiers'][cls_name] = d_mods
        results['dtdm']['base_f1'][cls_name] = d_base
        results['dtdm']['exp_f1'][cls_name] = d_exp
        results['dtdm']['per_class'][cls_name] = {
            'modifiers': d_mods, 'deltas': d_deltas,
            'base_f1': d_base, 'exp_f1': d_exp,
        }

        # Oracle
        o_mods, o_deltas, o_base, o_exp = evaluate_oracle(kb, domain, cls_name)
        results['oracle']['modifiers'][cls_name] = o_mods
        results['oracle']['base_f1'][cls_name] = o_base
        results['oracle']['exp_f1'][cls_name] = o_exp
        results['oracle']['per_class'][cls_name] = {
            'modifiers': o_mods, 'deltas': o_deltas,
            'base_f1': o_base, 'exp_f1': o_exp,
        }

    # 汇总
    for method in ['static', 'dtdm', 'oracle']:
        f1s = [v for v in results[method]['exp_f1'].values() if v is not None]
        bases = [v for v in results[method]['base_f1'].values() if v is not None]
        results[method]['mean_base_f1'] = float(np.mean(bases)) if bases else 0
        results[method]['mean_exp_f1'] = float(np.mean(f1s)) if f1s else 0
        results[method]['mean_gain'] = results[method]['mean_exp_f1'] - results[method]['mean_base_f1']

        # 统计 modifier 使用频率
        mod_counter = defaultdict(int)
        for mods in results[method]['modifiers'].values():
            for m in mods:
                mod_counter[m] += 1
        results[method]['modifier_frequency'] = dict(mod_counter)

    return results


# ── 主程序 ──
if __name__ == '__main__':
    kb = OFATKnowledgeBase()

    print("=" * 70)
    print("DTDM vs Static CD-TDM vs Oracle — 模拟评估")
    print("=" * 70)

    # 尝试加载 DTDM（可能需要 GPU）
    dtdm = None
    try:
        from dynamic_tdm import DynamicTDM
        dtdm = DynamicTDM()
        print("DTDM 已加载（含 CLIP 图像匹配）")
    except Exception as e:
        print(f"DTDM 加载失败: {e}")
        print("使用统计估算模式")

    all_results = {}

    for domain in ['remote_sensing']:  # 主评估：DIOR 目标域
        domain_norm = kb._normalize_domain(domain)
        if domain_norm == 'remote_sensing':
            classes = kb.dior_classes
        else:
            classes = list(kb.coco['classes'].keys())

        print(f"\n{'='*70}")
        print(f"域: {domain} ({len(classes)} 类)")
        print(f"{'='*70}")

        results = run_comparison(dtdm, kb, domain_norm, classes)
        all_results[domain] = results

        # ── 打印汇总 ──
        print(f"\n{'Method':<20s} {'Mean Base':>10s} {'Mean F1':>10s} {'Mean Gain':>10s}")
        print("-" * 55)
        for method in ['static', 'dtdm', 'oracle']:
            r = results[method]
            print(f"{method:<20s} {r['mean_base_f1']:10.4f} {r['mean_exp_f1']:10.4f} "
                  f"{r['mean_gain']:10.4f}")

        # DTDM vs Static 增益
        static_gain = results['static']['mean_gain']
        dtdm_gain = results['dtdm']['mean_gain']
        oracle_gain = results['oracle']['mean_gain']

        print(f"\n  静态 TDM 增益:    {static_gain:+.4f}")
        print(f"  DTDM 增益:        {dtdm_gain:+.4f}")
        print(f"  Oracle 增益:      {oracle_gain:+.4f}")
        if oracle_gain > 0:
            print(f"  Static/Oracle:    {static_gain/oracle_gain*100:.1f}%")
            print(f"  DTDM/Oracle:      {dtdm_gain/oracle_gain*100:.1f}%")

        # Modifier 使用频率对比
        print(f"\n  [Modifier 使用频率]")
        for method in ['static', 'dtdm', 'oracle']:
            freq = results[method]['modifier_frequency']
            sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
            freq_str = ', '.join(f'{m.split()[-1]}:{c}' for m, c in sorted_freq[:5])
            print(f"  {method:<20s} {freq_str}")

        # Per-class 对比（前 10 类和差异最大的 5 类）
        print(f"\n  [Per-class 对比 ({len(classes)} 类), 前 10 类]:")
        print(f"  {'Class':<20s} {'Static':>8s} {'DTDM':>8s} {'Oracle':>8s} "
              f"{'Δ(DTDM-Static)':>16s}")
        print(f"  " + "-" * 65)

        per_class_diffs = []
        for cls_name in classes[:20]:
            s_f1 = results['static']['exp_f1'].get(cls_name, 0)
            d_f1 = results['dtdm']['exp_f1'].get(cls_name, 0)
            o_f1 = results['oracle']['exp_f1'].get(cls_name, 0)
            diff = d_f1 - s_f1
            per_class_diffs.append((cls_name, s_f1, d_f1, o_f1, diff))

        per_class_diffs.sort(key=lambda x: -x[4])  # 按 DTDM 改善排序
        for cls_name, s_f1, d_f1, o_f1, diff in per_class_diffs[:15]:
            print(f"  {cls_name:<20s} {s_f1:8.4f} {d_f1:8.4f} {o_f1:8.4f} "
                  f"{diff:+16.4f}")

    # 保存结果
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 评估结果已保存至: {OUTPUT_PATH}")
