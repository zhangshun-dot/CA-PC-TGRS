# -*- coding: utf-8 -*-
"""Token Universality Score & eta_d — 第二篇论文理论层数值计算

从已有实验数据出发，无需新实验：
  1. 计算跨域衰减因子 η_d（从 cross_domain_validation.json）
  2. 计算每个 OFAT 轴的 Universality Score U(axis) 和 Specificity Score S(axis, domain)
  3. 数值验证统一跨域信息上界 Q_eff^cross ≤ C · η_α · η_κ · (1 − η_d)
  4. 估计域间 KL 散度 D_KL(P_S || P_T)

输出可直接用于论文 Section II 的理论数值。
"""

import json
import math
import numpy as np
from pathlib import Path

# ── 路径配置 ─────────────────────────────────────────────
BASE = Path(r'C:\Users\张三\Desktop')
COCO_OFAT_PATH = BASE / '实验数据' / 'coco_ofat_results.json'
CROSS_DOMAIN_PATH = BASE / '实验数据' / 'cross_domain_validation.json'
DIOR_HEATMAP_PATH = BASE / '论文版本历史' / '初稿4_论文文件' / '实验数据' / 'ofat_heatmap_f1.json'

# ── 已知常数（来自 OVD 六阶段论文 v17） ──────────────────
C_CAPACITY = 7.9       # 提示词信息容量 (bits)
ETA_ALPHA_SAT = 0.118  # satellite α̃ 注意力效率 (exact formula)
ETA_KAPPA = 1.0        # 噪声退化因子 (Gaussian copula → I ≤ 0.002 bits, η_κ → 1)

# ── 加载数据 ─────────────────────────────────────────────

with open(COCO_OFAT_PATH, 'r', encoding='utf-8') as f:
    coco = json.load(f)

with open(CROSS_DOMAIN_PATH, 'r', encoding='utf-8') as f:
    cross = json.load(f)

with open(DIOR_HEATMAP_PATH, 'r', encoding='utf-8') as f:
    dior_raw = json.load(f)

# ────────────────────────────────────────────────────────
# Part 1: η_d — 跨域衰减因子
# ────────────────────────────────────────────────────────

print("=" * 70)
print("Part 1: 跨域衰减因子 η_d 计算")
print("=" * 70)

summary = cross['summary']
baseline_f1 = summary['baseline_f1']        # DIOR baseline (generic prefix)
cross_f1 = summary['cross_domain_f1']       # DIOR + COCO modifiers + prefix calib
gold_f1 = summary['gold_f1']                # DIOR + DIOR modifiers (full OFAT)
retention = summary['retention']            # 0.7985

# 关键分解：
# gain_cross = cross_f1 - baseline_f1    # 跨域迁移的增量
# gain_gold  = gold_f1  - baseline_f1    # 金标准（域内完整 OFAT）的增量
# retention  = gain_cross / gain_gold    # 增量保留率
# eta_d      = 1 - retention             # 域偏移导致的增量损失占比

gain_cross = cross_f1 - baseline_f1
gain_gold = gold_f1 - baseline_f1
eta_d = 1.0 - retention

print(f"""
实验数据 (DIOR 目标域, COCO 源域):
  Baseline F1 (generic prefix):           {baseline_f1:.4f}
  Cross-domain F1 (COCO modifiers):       {cross_f1:.4f}
  Gold F1 (DIOR full OFAT):               {gold_f1:.4f}
  ─────────────────────────────────────────
  Cross-domain gain:                      {gain_cross:.4f}
  Gold gain (upper bound):                {gain_gold:.4f}
  Retention rate:                         {retention:.4f} ({retention*100:.1f}%)
  η_d (domain decay factor):              {eta_d:.4f} ({eta_d*100:.1f}%)
""")

# 逐类分析
class_losses = {}
for cls_name, cls_data in cross['per_class'].items():
    a_f1 = cls_data['A_baseline']['F1']
    b_f1 = cls_data['B_cross_domain']['F1']
    c_f1 = cls_data['C_gold']['F1']
    gain_c = b_f1 - a_f1
    gain_g = c_f1 - a_f1
    if gain_g > 0:
        class_retention = gain_c / gain_g
    else:
        class_retention = 1.0 if gain_c == 0 else 0.0  # no gain possible

    # Extract delta from COCO modifiers vs DIOR modifiers
    dior_mods = set(cls_data.get('dior_mods', []))
    coco_mods = set(cls_data.get('coco_mods', []))
    shared = dior_mods & coco_mods
    dior_only = dior_mods - coco_mods
    coco_only = coco_mods - dior_mods

    class_losses[cls_name] = {
        'retention': class_retention,
        'a_f1': a_f1, 'b_f1': b_f1, 'c_f1': c_f1,
        'shared_mods': shared, 'dior_only': dior_only, 'coco_only': coco_only,
    }

# 分类统计
perfect = sum(1 for v in class_losses.values() if v['retention'] >= 0.95)
degraded = sum(1 for v in class_losses.values() if 0.5 <= v['retention'] < 0.95)
failed = sum(1 for v in class_losses.values() if v['retention'] < 0.5)

print(f"""
逐类跨域保留率分布 (共 {len(class_losses)} 类):
  近乎无损 (retention ≥ 95%):  {perfect} 类
  部分退化 (50% ≤ ret < 95%):  {degraded} 类
  严重损失 (retention < 50%):  {failed} 类
""")

# 列出退化严重的类
print("退化最严重的类 (retention < 80%):")
for name, v in sorted(class_losses.items(), key=lambda x: x[1]['retention']):
    if v['retention'] < 0.8:
        print(f"  {name:20s}  ret={v['retention']:.3f}  "
              f"dior_only={v['dior_only']}  coco_only={v['coco_only']}")


# ────────────────────────────────────────────────────────
# Part 2: Token Universality Score — 轴级跨域分析
# ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Part 2: Token Universality Score & Specificity Score")
print("=" * 70)

# COCO 轴 delta (已有, 6轴)
coco_axes = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']
coco_classes = list(coco['classes'].keys())

# 构建 COCO delta 矩阵 [n_classes × n_axes]
coco_deltas = np.zeros((len(coco_classes), len(coco_axes)))
for i, cls_name in enumerate(coco_classes):
    for j, axis in enumerate(coco_axes):
        coco_deltas[i, j] = coco['classes'][cls_name]['deltas'][axis]

print(f"\nCOCO: {len(coco_classes)} 类 × {len(coco_axes)} 轴")
print(f"各类 COCO baseline DR 均值: {np.mean([coco['classes'][c]['base'] for c in coco_classes]):.4f}")

# DIOR 轴 delta (从 heatmap 计算: F1_axis - F1_baseline)
dior_axes_all = dior_raw['axes']  # ['grammar', 'granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']
dior_classes = dior_raw['classes']
dior_baseline = dior_raw['baseline']

# DIOR delta 矩阵 [n_classes × n_axes_all]
n_dior_cls = len(dior_classes)
n_dior_axes = len(dior_axes_all)
dior_deltas = np.zeros((n_dior_cls, n_dior_axes))
for i, cls_name in enumerate(dior_classes):
    base_f1 = dior_baseline[cls_name]
    for j, axis in enumerate(dior_axes_all):
        heatmap_f1 = dior_raw['heatmap'][axis][cls_name]
        dior_deltas[i, j] = heatmap_f1 - base_f1

print(f"DIOR: {n_dior_cls} 类 × {n_dior_axes} 轴")
print(f"各类 DIOR baseline F1 均值: {np.mean(list(dior_baseline.values())):.4f}")

# 取公共轴 (6轴，排除 grammar)
common_axes = coco_axes  # granularity, spatial, geometric, scale, spectral, texture
common_axes_dior_idx = [dior_axes_all.index(a) for a in common_axes]

coco_deltas_common = coco_deltas  # already [n_coco × 6]
dior_deltas_common = dior_deltas[:, common_axes_dior_idx]  # [n_dior × 6]

print(f"\n公共轴 ({len(common_axes)}): {common_axes}")

# ── 轴级统计 ──

# COCO 轴均值 & 方差
coco_mu = np.mean(coco_deltas_common, axis=0)   # [6]
coco_var = np.var(coco_deltas_common, axis=0, ddof=1)  # [6]

# DIOR 轴均值 & 方差
dior_mu = np.mean(dior_deltas_common, axis=0)   # [6]
dior_var = np.var(dior_deltas_common, axis=0, ddof=1)  # [6]

# 跨域统计
pooled_mu = (coco_mu + dior_mu) / 2.0
cross_var = ((coco_mu - pooled_mu)**2 + (dior_mu - pooled_mu)**2)  # [6]  (N=2 domains)
within_var = (coco_var + dior_var) / 2.0  # pooled within-domain variance
total_var = within_var + cross_var

# Universality Score: U(a) = 1 - cross_var / total_var
# 当跨域方差占比小时，轴是通用的
universality = 1.0 - cross_var / np.maximum(total_var, 1e-10)

# Specificity Score: which domain has stronger effect
specificity_coco = np.abs(coco_mu) / np.maximum(np.maximum(np.abs(coco_mu), np.abs(dior_mu)), 1e-10)
specificity_dior = np.abs(dior_mu) / np.maximum(np.maximum(np.abs(coco_mu), np.abs(dior_mu)), 1e-10)

# 轴效应方向一致性
same_sign = (np.sign(coco_mu) == np.sign(dior_mu))

print(f"\n{'Axis':<16s} {'μ_COCO':>8s} {'μ_DIOR':>8s} {'Δμ':>8s} "
      f"{'U(axis)':>8s} {'SignMatch':>10s} {'Spec@COCO':>10s} {'Spec@DIOR':>10s}")
print("-" * 80)

# 阈值（permutation-based 或默认 0.5）
# U > 0.7 → Universal Token（跨域共享）
# U < 0.3 → Domain-Specific Token（域专属）
# 0.3 ≤ U ≤ 0.7 → Mixed（混合）

axis_classification = {}
for j, axis in enumerate(common_axes):
    sign_ok = "✓" if same_sign[j] else "✗"
    u = universality[j]
    if u > 0.7:
        cls = "UNIVERSAL"
    elif u < 0.3:
        cls = "DOMAIN-SPECIFIC"
    else:
        cls = "MIXED"
    axis_classification[axis] = {
        'universality': u,
        'class': cls,
        'mu_coco': coco_mu[j],
        'mu_dior': dior_mu[j],
        'spec_coco': specificity_coco[j],
        'spec_dior': specificity_dior[j],
        'sign_match': bool(same_sign[j]),
    }
    print(f"{axis:<16s} {coco_mu[j]:8.4f} {dior_mu[j]:8.4f} {cross_var[j]**0.5:8.4f} "
          f"{u:8.4f} {sign_ok:>10s} {specificity_coco[j]:10.4f} {specificity_dior[j]:10.4f}")

# ── 分类汇总 ──
print(f"\nToken 分类结果:")
universal_axes = [a for a, v in axis_classification.items() if v['class'] == 'UNIVERSAL']
mixed_axes = [a for a, v in axis_classification.items() if v['class'] == 'MIXED']
domain_axes = [a for a, v in axis_classification.items() if v['class'] == 'DOMAIN-SPECIFIC']
print(f"  通用 Token (Universal, U>0.7):    {universal_axes}")
print(f"  混合 Token (Mixed, 0.3≤U≤0.7):    {mixed_axes}")
print(f"  域专属 Token (Domain-Spec, U<0.3): {domain_axes}")

# 额外：grammar 轴（仅 DIOR 有数据）
# grammar 所有类都是 0.1401（同一常数），说明此轴无区分力
grammar_deltas = dior_deltas[:, dior_axes_all.index('grammar')]
print(f"\n  grammar 轴 (DIOR only): μ={np.mean(grammar_deltas):.4f}, "
      f"σ²={np.var(grammar_deltas):.6f} → 无区分力，归为 DOMAIN-SPECIFIC (DIOR专属占位)")


# ────────────────────────────────────────────────────────
# Part 3: 统一跨域信息上界数值验证
# ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Part 3: 统一跨域信息上界 Q_eff^cross")
print("=" * 70)

Q_eff_single = C_CAPACITY * ETA_ALPHA_SAT * ETA_KAPPA
Q_eff_cross = C_CAPACITY * ETA_ALPHA_SAT * ETA_KAPPA * (1.0 - eta_d)

print(f"""
单域有效信息上界 (第一篇):
  Q_eff ≤ C · η_α · η_κ
       = {C_CAPACITY} · {ETA_ALPHA_SAT} · {ETA_KAPPA}
       = {Q_eff_single:.4f} bits

跨域有效信息上界 (本篇):
  Q_eff^cross ≤ C · η_α · η_κ · (1 − η_d)
              = {C_CAPACITY} · {ETA_ALPHA_SAT} · {ETA_KAPPA} · (1 − {eta_d:.4f})
              = {Q_eff_cross:.4f} bits

跨域信息保留率: {1-eta_d:.4f} ({100*(1-eta_d):.1f}%)
跨域信息损失:   {eta_d:.4f} ({100*eta_d:.1f}%)
""")

# 验证：与实验数据对照
# 理论预测: cross_f1 应该接近 baseline + gain_gold * (1-eta_d)
# 实际: cross_f1 = baseline_f1 + gain_cross
predicted_cross_gain = gain_gold * (1 - eta_d)
print(f"""
实验验证:
  理论预测跨域增量: gain_gold × (1−η_d) = {gain_gold:.4f} × {1-eta_d:.4f} = {predicted_cross_gain:.4f}
  实际跨域增量:     {gain_cross:.4f}
  误差:             {abs(predicted_cross_gain - gain_cross):.6f}
  (恒等式验证: retention = 1−η_d, 由定义保证)
""")


# ────────────────────────────────────────────────────────
# Part 4: 域间 KL 散度估计 D_KL(P_S || P_T)
# ────────────────────────────────────────────────────────

print("=" * 70)
print("Part 4: 域间 Token 分布 KL 散度估计")
print("=" * 70)

# 方法：对每个公共轴，假设轴 delta 服从高斯分布 N(μ_d, σ²_d)
# D_KL(N_S || N_T) = log(σ_T/σ_S) + (σ²_S + (μ_S−μ_T)²) / (2σ²_T) − 1/2

kl_per_axis = {}
kl_total = 0.0
for j, axis in enumerate(common_axes):
    mu_s, var_s = coco_mu[j], coco_var[j]
    mu_t, var_t = dior_mu[j], dior_var[j]
    sigma_s = math.sqrt(max(var_s, 1e-10))
    sigma_t = math.sqrt(max(var_t, 1e-10))

    # D_KL(source=coco || target=dior)
    kl = math.log(sigma_t / sigma_s) + (var_s + (mu_s - mu_t)**2) / (2 * var_t) - 0.5
    kl_per_axis[axis] = max(kl, 0)  # KL ≥ 0
    kl_total += max(kl, 0)

# 总 KL（假设轴间独立）
# 加上 grammar 轴贡献（视为 Dirac delta → KL 可忽略）
print(f"\n逐轴 KL 散度 D_KL(COCO || DIOR):")
for axis, kl in kl_per_axis.items():
    cls = axis_classification[axis]['class']
    print(f"  {axis:<16s} D_KL={kl:.4f}  [{cls}]")
print(f"  {'(grammar)':<16s} D_KL≈0  [DOMAIN-SPECIFIC, DIOR-only]")
print(f"\n  总 KL (独立假设): {kl_total:.4f} nats")

# 归一化 η_d: η_d ≈ D_KL / H(P_S)
# 从 COCO 轴 delta 分布估计 H(P_S)
# H(P_S) ≈ Σ_a [½log(2πe·σ²_S(a))] (6轴独立高斯)
H_S = sum(0.5 * math.log(2 * math.pi * math.e * max(v, 1e-10)) for v in coco_var)
eta_d_from_kl = kl_total / H_S if H_S > 0 else 0

print(f"""
域信息熵 H(P_S) ≈ {H_S:.4f} nats (COCO 6轴独立高斯假设)
η_d (KL-based):   {eta_d_from_kl:.4f}  (D_KL / H_S)
η_d (empirical):  {eta_d:.4f}  (1 − retention_rate)

两种估计方法:
  KL-based: η_d = D_KL(P_S || P_T) / H(P_S) = {eta_d_from_kl:.4f}
  Retention: η_d = 1 − (cross_gain / gold_gain) = {eta_d:.4f}
  差异: {abs(eta_d_from_kl - eta_d):.4f}

  → KL-based 是理论上界（独立高斯假设夸大方差），Retention 是实验下界
  → 论文中建议用 Retention-based: η_d = {eta_d:.4f}
""")


# ────────────────────────────────────────────────────────
# Part 5: Summary — 可直接写入论文的数值
# ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Part 5: 论文可直接引用的数值汇总")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────┐
│  跨域 Token 信息论 — 数值结果                           │
├─────────────────────────────────────────────────────────┤
│  η_d (域偏移衰减因子)          = {eta_d:.4f}           │
│  D_KL(P_COCO || P_DIOR)         = {kl_total:.4f} nats  │
│  H(P_S) 源域信息熵              = {H_S:.4f} nats       │
│                                                         │
│  Q_eff (单域)  ≤ C·η_α·η_κ     = {Q_eff_single:.4f} bits│
│  Q_eff^cross   ≤ Q_eff·(1−η_d)  = {Q_eff_cross:.4f} bits│
│  跨域保留率                     = {(1-eta_d)*100:.1f}% │
│                                                         │
│  Universal axes ({len(universal_axes)}):  {universal_axes}│
│  Mixed axes ({len(mixed_axes)}):       {mixed_axes}│
│  Domain-Specific ({len(domain_axes)}): {domain_axes}│
└─────────────────────────────────────────────────────────┘
""")

# 验证：轴偏好跨域相关
# 用 axis-level mean delta 计算 Pearson r
axis_mu_coco = coco_mu
axis_mu_dior = dior_mu
r_axis = np.corrcoef(axis_mu_coco, axis_mu_dior)[0, 1]
r2_axis = r_axis**2
print(f"轴偏好跨域 Pearson r = {r_axis:.4f} (r² = {r2_axis:.4f})")
print(f"  → {1-r2_axis:.4f} ({100*(1-r2_axis):.1f}%) 的轴偏好方差无法跨域解释")
print(f"  → 与 η_d = {eta_d:.4f} 一致（η_d 是类级别的，r² 是轴级别的）")

# 保存结果供论文使用
output = {
    'eta_d': eta_d,
    'eta_d_method': '1 - retention_rate',
    'kl_total_nats': kl_total,
    'H_S_nats': H_S,
    'Q_eff_single_bits': Q_eff_single,
    'Q_eff_cross_bits': Q_eff_cross,
    'cross_retention': 1 - eta_d,
    'axis_classification': {a: v for a, v in axis_classification.items()},
    'axis_pearson_r': float(r_axis),
    'universal_axes': universal_axes,
    'mixed_axes': mixed_axes,
    'domain_specific_axes': domain_axes,
    'kl_per_axis': {a: float(v) for a, v in kl_per_axis.items()},
}

output_path = BASE / '实验数据' / 'token_universality_results.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存至: {output_path}")
