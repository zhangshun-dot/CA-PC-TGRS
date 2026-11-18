# -*- coding: utf-8 -*-
"""第二篇跨域 Token Discovery 论文 — 四张核心图表

Fig 1: Token Universality Score — 水平柱状图 (6轴, sequential blue)
Fig 2: Cross-Domain Axis Effect — 哑铃图 (COCO vs DIOR per-axis μ)
Fig 3: Information Bound Decomposition — 阶梯衰减瀑布图
Fig 4: Per-Class Cross-Domain Transfer — 散点图 (baseline vs cross-domain F1)

配色: dataviz reference palette (blue sequential, blue↔red diverging)
输出: 300 DPI PNG, 中文标签
"""

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path

# ── 中文支持 ─────────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 参考调色板 ───────────────────────────────────────────
# 来自 dataviz reference palette (light mode)
SURFACE    = '#fcfcfb'
INK        = '#0b0b0b'
SEC_INK    = '#52514e'
MUTED      = '#898781'
GRID       = '#e1e0d9'
BASELINE   = '#c3c2b7'

# Categorical slots
BLUE       = '#2a78d6'
ORANGE     = '#eb6834'
AQUA       = '#1baf7a'
YELLOW     = '#eda100'
MAGENTA    = '#e87ba4'
GREEN      = '#008300'

# Sequential blue ramp
BLUE_100   = '#cde2fb'
BLUE_250   = '#86b6ef'
BLUE_400   = '#3987e5'
BLUE_500   = '#256abf'
BLUE_600   = '#184f95'
BLUE_700   = '#0d366b'

# Diverging
RED_DIV    = '#d03b3b'

# Universal / Mixed / Domain-Specific classification colors
UNIVERSAL_COLOR = AQUA       # aqua — stable, shared
MIXED_COLOR     = YELLOW     # yellow — needs adaptation
DOMAIN_COLOR    = ORANGE     # orange — domain-specific

# ── 加载数据 ─────────────────────────────────────────────
BASE = Path(r'C:\Users\张三\Desktop')

with open(BASE / '实验数据' / 'coco_ofat_results.json', 'r', encoding='utf-8') as f:
    coco = json.load(f)

with open(BASE / '实验数据' / 'cross_domain_validation.json', 'r', encoding='utf-8') as f:
    cross = json.load(f)

with open(BASE / '论文版本历史' / '初稿4_论文文件' / '实验数据' / 'ofat_heatmap_f1.json', 'r', encoding='utf-8') as f:
    dior_raw = json.load(f)

with open(BASE / '实验数据' / 'token_universality_results.json', 'r', encoding='utf-8') as f:
    universality_data = json.load(f)

# ── 公共计算 ─────────────────────────────────────────────
coco_axes = ['granularity', 'spatial', 'geometric', 'scale', 'spectral', 'texture']
axis_labels_cn = {
    'granularity': '粒度 (Granularity)',
    'spatial': '空间 (Spatial)',
    'geometric': '几何 (Geometric)',
    'scale': '尺度 (Scale)',
    'spectral': '光谱 (Spectral)',
    'texture': '纹理 (Texture)',
}

# Axis statistics
coco_classes = list(coco['classes'].keys())
coco_deltas = np.zeros((len(coco_classes), len(coco_axes)))
for i, cls_name in enumerate(coco_classes):
    for j, axis in enumerate(coco_axes):
        coco_deltas[i, j] = coco['classes'][cls_name]['deltas'][axis]

dior_axes_all = dior_raw['axes']
dior_classes = dior_raw['classes']
dior_baseline = dior_raw['baseline']
common_axes_dior_idx = [dior_axes_all.index(a) for a in coco_axes]
n_dior_cls = len(dior_classes)
n_dior_axes = len(dior_axes_all)
dior_deltas_all = np.zeros((n_dior_cls, n_dior_axes))
for i, cls_name in enumerate(dior_classes):
    base_f1 = dior_baseline[cls_name]
    for j, axis in enumerate(dior_axes_all):
        heatmap_f1 = dior_raw['heatmap'][axis][cls_name]
        dior_deltas_all[i, j] = heatmap_f1 - base_f1
dior_deltas_common = dior_deltas_all[:, common_axes_dior_idx]

coco_mu = np.mean(coco_deltas, axis=0)
dior_mu = np.mean(dior_deltas_common, axis=0)
coco_var = np.var(coco_deltas, axis=0, ddof=1)
dior_var = np.var(dior_deltas_common, axis=0, ddof=1)

pooled_mu = (coco_mu + dior_mu) / 2.0
cross_var = ((coco_mu - pooled_mu)**2 + (dior_mu - pooled_mu)**2)
within_var = (coco_var + dior_var) / 2.0
total_var = within_var + cross_var
universality = 1.0 - cross_var / np.maximum(total_var, 1e-10)

# Sort axes by universality
sort_idx = np.argsort(-universality)
sorted_axes = [coco_axes[i] for i in sort_idx]
sorted_u = universality[sort_idx]
sorted_coco_mu = coco_mu[sort_idx]
sorted_dior_mu = dior_mu[sort_idx]

# Classification
def classify_u(u):
    if u > 0.7: return 'UNIVERSAL', UNIVERSAL_COLOR
    elif u < 0.3: return 'DOMAIN-SPECIFIC', DOMAIN_COLOR
    else: return 'MIXED', MIXED_COLOR

axis_cls = [classify_u(u) for u in sorted_u]

# Summary stats
summary = cross['summary']
baseline_f1 = summary['baseline_f1']
cross_f1_val = summary['cross_domain_f1']
gold_f1 = summary['gold_f1']
C_CAPACITY = 7.9
ETA_ALPHA = 0.118
ETA_KAPPA = 1.0
ETA_D = 1.0 - summary['retention']

Q_eff = C_CAPACITY * ETA_ALPHA * ETA_KAPPA
Q_eff_cross = Q_eff * (1 - ETA_D)

# Per-class data
class_names = []
class_base_f1 = []
class_cross_f1 = []
class_gold_f1 = []
for cls_name, cls_data in cross['per_class'].items():
    class_names.append(cls_name)
    class_base_f1.append(cls_data['A_baseline']['F1'])
    class_cross_f1.append(cls_data['B_cross_domain']['F1'])
    class_gold_f1.append(cls_data['C_gold']['F1'])

class_base_f1 = np.array(class_base_f1)
class_cross_f1 = np.array(class_cross_f1)
class_gold_f1 = np.array(class_gold_f1)

# ── 公共样式设置 ─────────────────────────────────────────
def setup_axis(ax):
    """Apply clean, recessive axis style."""
    ax.set_facecolor(SURFACE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BASELINE)
    ax.spines['bottom'].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

# ────────────────────────────────────────────────────────
# Fig 1: Token Universality Score — 水平柱状图
# ────────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(8, 5))
fig1.patch.set_facecolor(SURFACE)
setup_axis(ax1)

y_pos = range(len(sorted_axes))
bar_labels = [axis_labels_cn[a] for a in sorted_axes]

# Sequential blue ramp: darker = more universal
# Map U ∈ [0.4, 1.0] to blue steps 250→700
u_norm = np.clip((sorted_u - 0.3) / 0.7, 0, 1)
bar_colors = []
for u in u_norm:
    # Interpolate between BLUE_250 and BLUE_700
    if u < 0.3:
        bar_colors.append(RED_DIV)  # domain-specific gets red
    elif u < 0.5:
        bar_colors.append(BLUE_250)
    elif u < 0.7:
        bar_colors.append(BLUE_400)
    elif u < 0.85:
        bar_colors.append(BLUE_500)
    else:
        bar_colors.append(BLUE_600)

bars = ax1.barh(y_pos, sorted_u, height=0.55, color=bar_colors,
                edgecolor='none', zorder=3)

# Threshold lines
ax1.axvline(x=0.7, color=AQUA, linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
ax1.axvline(x=0.3, color=ORANGE, linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)

# Threshold labels
ax1.text(0.71, len(sorted_axes) - 0.4, 'Universal\n(U > 0.7)', fontsize=7.5,
         color=AQUA, va='bottom', fontweight='bold')
ax1.text(0.31, len(sorted_axes) - 0.4, 'Mixed\n(0.3 ≤ U ≤ 0.7)', fontsize=7.5,
         color=ORANGE, va='bottom', fontweight='bold')

# Axis labels
ax1.set_yticks(y_pos)
ax1.set_yticklabels(bar_labels, fontsize=10, color=INK)
ax1.set_xlim(0, 1.05)
ax1.set_xlabel('Universality Score', fontsize=10, color=SEC_INK)

# Value labels on bar ends
for i, (u, cls_info) in enumerate(zip(sorted_u, axis_cls)):
    label = f'{u:.3f}'
    ax1.text(u + 0.015, i, label, va='center', fontsize=8.5,
             color=INK, fontweight='bold')

# Classification tag on right side
for i, (u, (cls_name, cls_color)) in enumerate(zip(sorted_u, axis_cls)):
    tag_map = {'UNIVERSAL': '通用', 'MIXED': '混合', 'DOMAIN-SPECIFIC': '域专属'}
    tag = tag_map.get(cls_name, cls_name)
    ax1.text(1.02, i, tag, va='center', fontsize=7.5, color=cls_color,
             fontweight='bold', transform=ax1.get_yaxis_transform())

ax1.set_title('Token Universality Score by OFAT Axis', fontsize=12,
              color=INK, fontweight='bold', pad=12)

fig1.tight_layout()
fig1_path = BASE / '实验数据' / 'fig1_token_universality.png'
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
print(f"Fig 1 saved: {fig1_path}")


# ────────────────────────────────────────────────────────
# Fig 2: Cross-Domain Axis Effect — 哑铃图
# ────────────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5.5))
fig2.patch.set_facecolor(SURFACE)
setup_axis(ax2)

# Sort by absolute difference between COCO and DIOR
diff_order = np.argsort(-np.abs(sorted_coco_mu - sorted_dior_mu))
plot_axes = [sorted_axes[i] for i in diff_order]
plot_coco = sorted_coco_mu[diff_order]
plot_dior = sorted_dior_mu[diff_order]
plot_labels = [axis_labels_cn[a] for a in plot_axes]

y_positions = range(len(plot_axes))

# Dumbbell: COCO dot (lighter blue) — line — DIOR dot (darker blue)
for i, (y, c_mu, d_mu) in enumerate(zip(y_positions, plot_coco, plot_dior)):
    # Connecting line
    ax2.plot([c_mu, d_mu], [y, y], color=BASELINE, linewidth=1.5, zorder=2)
    # COCO dot
    ax2.scatter(c_mu, y, s=80, color=BLUE_250, edgecolor='white', linewidth=2,
                zorder=3, label='COCO' if i == 0 else '')
    # DIOR dot
    ax2.scatter(d_mu, y, s=80, color=BLUE_700, edgecolor='white', linewidth=2,
                zorder=3, label='DIOR' if i == 0 else '')

# Zero reference line
ax2.axvline(x=0, color=MUTED, linewidth=0.8, linestyle='-', zorder=1)

# Axis labels
ax2.set_yticks(y_positions)
ax2.set_yticklabels(plot_labels, fontsize=10, color=INK)
ax2.set_xlabel('Mean Axis Delta (ΔF1)', fontsize=10, color=SEC_INK)

# Value labels at dots
for i, (y, c_mu, d_mu) in enumerate(zip(y_positions, plot_coco, plot_dior)):
    # COCO value (left or right of dot)
    c_offset = -0.025 if c_mu > 0 else 0.008
    c_ha = 'right' if c_mu > 0 else 'left'
    ax2.text(c_mu + c_offset, y + 0.2, f'{c_mu:+.3f}', fontsize=7.5,
             color=BLUE_400, ha=c_ha, va='bottom')

    # DIOR value
    d_offset = 0.025 if d_mu > 0 else -0.008
    d_ha = 'left' if d_mu > 0 else 'right'
    ax2.text(d_mu + d_offset, y - 0.2, f'{d_mu:+.3f}', fontsize=7.5,
             color=BLUE_700, ha=d_ha, va='top')

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE_250,
           markersize=9, label='COCO (自然场景)', markeredgecolor='white',
           markeredgewidth=2),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE_700,
           markersize=9, label='DIOR (遥感)', markeredgecolor='white',
           markeredgewidth=2),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9,
           frameon=True, facecolor=SURFACE, edgecolor=GRID)

ax2.set_title('Cross-Domain Axis Effect Comparison', fontsize=12,
              color=INK, fontweight='bold', pad=12)

# Annotation: domain conflict axes
for i, (y, c_mu, d_mu) in enumerate(zip(y_positions, plot_coco, plot_dior)):
    if np.sign(c_mu) != np.sign(d_mu):
        ax2.annotate('符号反转', xy=((c_mu + d_mu)/2, y),
                     xytext=(0.08, y + 0.35),
                     fontsize=7, color=RED_DIV, fontweight='bold',
                     ha='center',
                     arrowprops=dict(arrowstyle='->', color=RED_DIV, lw=0.8))

fig2.tight_layout()
fig2_path = BASE / '实验数据' / 'fig2_cross_domain_dumbbell.png'
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
print(f"Fig 2 saved: {fig2_path}")


# ────────────────────────────────────────────────────────
# Fig 3: Information Bound Decomposition — 瀑布图
# ────────────────────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(8, 4.5))
fig3.patch.set_facecolor(SURFACE)
setup_axis(ax3)

steps = [
    ('C', C_CAPACITY, '提示词\n信息容量', BLUE_700),
    ('C·η_α', C_CAPACITY * ETA_ALPHA, f'注意力效率\nη_α = {ETA_ALPHA}', BLUE_500),
    ('C·η_α·η_κ', C_CAPACITY * ETA_ALPHA * ETA_KAPPA, f'噪声退化\nη_κ ≈ {ETA_KAPPA}', BLUE_400),
    ('Q_eff (单域)', Q_eff, '单域有效信息\n(第一篇)', BLUE_250),
    ('Q_eff^cross', Q_eff_cross, f'跨域衰减\nη_d = {ETA_D:.3f}', AQUA),
]

# Draw as connected horizontal bars with annotation
x_positions = [0, 1, 2, 3.5, 5]
bar_width = 0.7

for i, (label, val, desc, color) in enumerate(steps):
    x = x_positions[i]

    # Bar
    bar_height = val / C_CAPACITY * 3.5  # scale to fit
    bottom = 0

    ax3.bar(x, val, bar_width, bottom=0, color=color, edgecolor='none',
            zorder=3, alpha=0.85)

    # Value label on bar
    ax3.text(x, val + 0.15, f'{val:.3f}', fontsize=10, ha='center',
             color=INK, fontweight='bold')

    # Description below
    ax3.text(x, -0.4, desc, fontsize=7.5, ha='center', color=SEC_INK,
             va='top')

    # Step label
    ax3.text(x, -0.05, label, fontsize=8.5, ha='center', color=INK,
             fontweight='bold', va='bottom')

# Down arrows between steps
arrow_y = 0.25
for i in range(len(steps) - 1):
    x1 = x_positions[i] + bar_width/2
    x2 = x_positions[i+1] - bar_width/2
    mid_x = (x1 + x2) / 2
    ax3.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                 arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.2))

    # Loss annotation between steps
    loss = steps[i][1] - steps[i+1][1]
    if loss > 0.01:
        ax3.text(mid_x, arrow_y + 0.35, f'-{loss:.3f}', fontsize=7.5,
                 ha='center', color=RED_DIV, fontweight='bold')

# bits label
ax3.set_ylabel('Effective Information (bits)', fontsize=10, color=SEC_INK)
ax3.set_xticks([])
ax3.set_xlim(-0.8, x_positions[-1] + 0.8)
ax3.set_ylim(-0.8, 8.5)

# Retention annotation
ax3.annotate(f'跨域保留率: {(1-ETA_D)*100:.1f}%',
             xy=(x_positions[-1], Q_eff_cross),
             xytext=(x_positions[-1] + 0.6, Q_eff_cross + 1.5),
             fontsize=9, color=AQUA, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=AQUA, lw=1.0))

ax3.set_title('Cross-Domain Information Bound Decomposition',
              fontsize=12, color=INK, fontweight='bold', pad=12)

fig3.tight_layout()
fig3_path = BASE / '实验数据' / 'fig3_information_bound_waterfall.png'
fig3.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
print(f"Fig 3 saved: {fig3_path}")


# ────────────────────────────────────────────────────────
# Fig 4: Per-Class Cross-Domain Transfer — 散点图
# ────────────────────────────────────────────────────────

fig4, ax4 = plt.subplots(figsize=(7.5, 6.5))
fig4.patch.set_facecolor(SURFACE)
setup_axis(ax4)

# Compute per-class Δ
class_delta = class_cross_f1 - class_base_f1
class_delta_gold = class_gold_f1 - class_base_f1

# Color by outcome: blue = improved, red = degraded, gray = unchanged
point_colors = []
for delta, delta_gold in zip(class_delta, class_delta_gold):
    if delta_gold <= 0:
        point_colors.append(MUTED)  # no possible gain
    elif delta > 0:
        point_colors.append(BLUE)  # cross-domain helped
    elif delta < 0:
        point_colors.append(RED_DIV)  # cross-domain hurt
    else:
        point_colors.append(MUTED)  # no change

# Scatter
scatter = ax4.scatter(class_base_f1, class_cross_f1, c=point_colors,
                      s=60, edgecolors='white', linewidth=1.5, zorder=3,
                      alpha=0.85)

# Diagonal line (y = x, no change)
max_val = max(class_base_f1.max(), class_cross_f1.max(), class_gold_f1.max()) * 1.05
ax4.plot([0, max_val], [0, max_val], color=BASELINE, linewidth=1,
         linestyle='--', zorder=1, label='无变化 (y=x)')

# Gold standard markers (small triangles)
gold_colors = [BLUE_500 if g > b else MUTED for g, b in zip(class_gold_f1, class_base_f1)]
ax4.scatter(class_base_f1, class_gold_f1, c=gold_colors, s=35,
            marker='^', edgecolors='white', linewidth=1, zorder=2,
            alpha=0.7, label='金标准 (DIOR OFAT)')

# Labels for interesting points
for i, name in enumerate(class_names):
    delta = class_delta[i]
    gold_delta = class_delta_gold[i]
    # Label: classes with notable cross-domain effect (positive or negative)
    if abs(delta) > 0.03 or gold_delta > 0.05:
        offset_y = 0.015 if delta >= 0 else -0.02
        ax4.annotate(name, (class_base_f1[i], class_cross_f1[i]),
                     xytext=(class_base_f1[i] + 0.03, class_cross_f1[i] + offset_y),
                     fontsize=7, color=SEC_INK,
                     arrowprops=dict(arrowstyle='->', color=GRID, lw=0.6))

# Mean markers
ax4.scatter([baseline_f1], [cross_f1_val], s=120, marker='D',
            color=BLUE_700, edgecolors='white', linewidth=2, zorder=4,
            label=f'均值 (cross={cross_f1_val:.3f})')
ax4.scatter([baseline_f1], [gold_f1], s=80, marker='s',
            color=AQUA, edgecolors='white', linewidth=2, zorder=4,
            label=f'金标准均值 (gold={gold_f1:.3f})')

# Annotations
ax4.annotate(f'ΔF1 = {cross_f1_val - baseline_f1:+.4f}\n保留率 = {summary["retention"]*100:.1f}%',
             xy=(baseline_f1, cross_f1_val),
             xytext=(baseline_f1 + 0.12, cross_f1_val + 0.05),
             fontsize=8, color=BLUE_700, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=BLUE_700, lw=0.8))

# Axis labels
ax4.set_xlabel('Baseline F1 (generic prefix)', fontsize=10, color=SEC_INK)
ax4.set_ylabel('Cross-Domain F1 (COCO modifiers)', fontsize=10, color=SEC_INK)
ax4.set_xlim(-0.02, max_val)
ax4.set_ylim(-0.02, max_val)

# Legend
ax4.legend(loc='lower right', fontsize=7.5, frameon=True,
           facecolor=SURFACE, edgecolor=GRID, ncol=2)

# Statistics annotation box
stats_text = (
    f'提升: {sum(class_delta > 0)} 类\n'
    f'不变: {sum(class_delta == 0)} 类\n'
    f'退化: {sum(class_delta < 0)} 类'
)
ax4.text(0.97, 0.08, stats_text, transform=ax4.transAxes,
         fontsize=8, color=SEC_INK, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor=SURFACE,
                   edgecolor=GRID, alpha=0.9))

ax4.set_title('Per-Class Cross-Domain Transfer: Baseline → COCO Modifiers on DIOR',
              fontsize=11, color=INK, fontweight='bold', pad=12)

fig4.tight_layout()
fig4_path = BASE / '实验数据' / 'fig4_per_class_scatter.png'
fig4.savefig(fig4_path, dpi=300, bbox_inches='tight', facecolor=SURFACE)
print(f"Fig 4 saved: {fig4_path}")

print("\n✅ All 4 figures generated.")

# ── 补充：生成组合排版预览 ──
# (用于论文初稿的 figure panel)
