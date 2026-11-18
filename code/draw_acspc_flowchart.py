"""
A-CSPC Overall Flowchart - Nature-style academic figure
按照 nature-figure 规范绘制的 A-CSPC 整体流程图
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import numpy as np

# Nature-style rcParams (with Chinese font support)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "mathtext.fontset": "dejavusans",
})

# Color palette (nature-style, restrained)
COLORS = {
    'source': '#4A90D9',      # 蓝色 - 源域
    'source_light': '#EBF2FA',
    'target': '#E8913A',      # 橙色 - 目标域
    'target_light': '#FDF2E6',
    'correction': '#4CAF50',  # 绿色 - 校正融合
    'correction_light': '#E8F5E9',
    'boundary': '#9E9E9E',    # 灰色 - 评估边界
    'boundary_light': '#F5F5F5',
    'text': '#1A1B1C',
    'text_secondary': '#6B7280',
    'arrow': '#555555',
}

def draw_box(ax, x, y, w, h, text, color, text_color=None, fontsize=7, 
              bold=False, boxstyle="round,pad=0.3", alpha=1.0, zorder=2):
    """Draw a rounded rectangle box with text"""
    if text_color is None:
        text_color = COLORS['text']
    
    box = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                          facecolor=color, edgecolor=color,
                          linewidth=1.2, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, weight=weight,
            zorder=zorder+1, wrap=True)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=None, style='->', lw=1.2, 
               connectionstyle="arc3,rad=0", zorder=1):
    """Draw an arrow between two points"""
    if color is None:
        color = COLORS['arrow']
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style, color=color, lw=lw,
                            connectionstyle=connectionstyle, zorder=zorder,
                            mutation_scale=12)
    ax.add_patch(arrow)
    return arrow

def draw_diamond(ax, x, y, w, h, text, color, fontsize=7, zorder=2):
    """Draw a diamond decision box"""
    diamond = plt.Polygon([(x, y+h/2), (x+w/2, y+h), (x+w, y+h/2), (x+w/2, y)],
                           facecolor=color, edgecolor=color, linewidth=1.2, zorder=zorder)
    ax.add_patch(diamond)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color='white', weight='bold', zorder=zorder+1)
    return diamond

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 10), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(-1.5, 10)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(8, 9.6, 'A-CSPC: Asymmetric Cross-Space Prototype Calibration for Cross-Domain Object Detection',
        ha='center', va='center', fontsize=12, weight='bold', color=COLORS['text'])

# ============================================================
# MAIN PIPELINE (top section, y = 6.5 - 9.0)
# ============================================================

# Step 1: Input (left)
draw_box(ax, 0.3, 7.8, 2.2, 1.0, 
         'Source Domain S\n(DIOR, remote sensing)\n+ annotations',
         COLORS['source_light'], COLORS['source'], fontsize=7, bold=True)
draw_box(ax, 0.3, 6.5, 2.2, 1.0,
         'Target Domain T\n(COCO, natural scene)\n+ text prompts',
         COLORS['target_light'], COLORS['target'], fontsize=7, bold=True)

# Frozen modules
draw_box(ax, 0.3, 5.5, 2.2, 0.7,
         'Frozen: GDINO-tiny + CLIP ViT-B/32\n(training-free, zero learnable params)',
         COLORS['boundary_light'], COLORS['text_secondary'], fontsize=6, bold=False)

# Arrow from input to proposal pool
draw_arrow(ax, 2.5, 7.0, 3.2, 7.0)

# Step 2: Proposal pool + features
draw_box(ax, 3.2, 6.5, 2.0, 1.3,
         'Fixed Proposal Pool\nB = {b_k}\n+ detector scores s_det\n+ CLIP box features f_k',
         COLORS['boundary_light'], COLORS['text'], fontsize=7, bold=True)

# Arrow to prototype library
draw_arrow(ax, 5.2, 7.15, 5.8, 7.15)

# Step 3: Prototype library + text direction (split into two boxes)
draw_box(ax, 5.8, 7.5, 2.3, 0.9,
         'Source Prototype Library\np_c (positive) + p_neg (top-k nearest)',
         COLORS['source_light'], COLORS['source'], fontsize=6.5, bold=True)
draw_box(ax, 5.8, 6.4, 2.3, 0.9,
         'Text Direction\nΔ_c = d(x_c) - s(x_s(c))\nD_dom = mean(Δ_c)',
         COLORS['target_light'], COLORS['target'], fontsize=6.5, bold=True)

# Arrow to P1 diagnosis
draw_arrow(ax, 8.1, 7.0, 8.6, 7.0)

# Step 4: P1 Diagnosis
draw_box(ax, 8.6, 6.5, 1.8, 1.3,
         'P1: Layered Failure\nDiagnosis\n\nR_max (recall ceiling)\nAUC (ranking quality)\nΔAUC = AUC_ext - AUC_src',
         COLORS['correction_light'], COLORS['correction'], fontsize=6, bold=True)

# P1 badge
circle = Circle((8.7, 7.65), 0.15, facecolor=COLORS['correction'], edgecolor='white', linewidth=1.5, zorder=5)
ax.add_patch(circle)
ax.text(8.7, 7.65, '1', ha='center', va='center', fontsize=7, color='white', weight='bold', zorder=6)

# Arrow to P3 gate
draw_arrow(ax, 10.4, 7.15, 10.9, 7.15)

# Step 5: P3 Direction Gate (diamond)
draw_diamond(ax, 10.9, 6.5, 1.6, 1.3,
             'P3: g₂(c)\nDirection\nGate',
             COLORS['boundary'], fontsize=6.5)

# P3 badge
circle = Circle((11.0, 7.65), 0.15, facecolor=COLORS['boundary'], edgecolor='white', linewidth=1.5, zorder=5)
ax.add_patch(circle)
ax.text(11.0, 7.65, '3', ha='center', va='center', fontsize=7, color='white', weight='bold', zorder=6)

# Gate labels
ax.text(11.7, 7.8, 'g₂ > 0', fontsize=6.5, color=COLORS['correction'], weight='bold')
ax.text(11.7, 6.3, 'g₂ ≤ 0', fontsize=6.5, color=COLORS['boundary'], weight='bold')

# Arrow from gate to P2 (upper path: extrapolate)
draw_arrow(ax, 12.5, 7.5, 13.0, 7.8, color=COLORS['correction'])
# Arrow from gate to B2 (lower path: conservative)
draw_arrow(ax, 12.5, 6.8, 13.0, 6.2, color=COLORS['boundary'])

# Step 6: P2 Asymmetric Prototype Construction (two boxes)
# Positive prototype extrapolation (green)
draw_box(ax, 13.0, 7.5, 2.7, 0.9,
         'P2a: Positive Prototype Extrapolation\np̂_c = norm(p_c + β·Δ_c)\n(along CLIP text direction)',
         COLORS['correction_light'], COLORS['correction'], fontsize=6, bold=True)

# Negative prototype source retention (blue)
draw_box(ax, 13.0, 6.4, 2.7, 0.9,
         'P2b: Negative Prototype Source-Retention\np_neg = mean(top-k source classes)\n(NOT extrapolated)',
         COLORS['source_light'], COLORS['source'], fontsize=6, bold=True)

# P2 badge
circle = Circle((13.1, 8.25), 0.15, facecolor=COLORS['correction'], edgecolor='white', linewidth=1.5, zorder=5)
ax.add_patch(circle)
ax.text(13.1, 8.25, '2', ha='center', va='center', fontsize=7, color='white', weight='bold', zorder=6)

# Asymmetric annotation
ax.text(14.35, 7.2, 'Asymmetric: + extrapolate, - retain',
        ha='center', va='center', fontsize=6, color=COLORS['text_secondary'], style='italic')

# B2 conservative path
draw_box(ax, 13.0, 5.3, 2.7, 0.7,
         'B2 Conservative: p̂_c = p_c (no extrapolation)\n(negative prototype still source-space)',
         COLORS['boundary_light'], COLORS['boundary'], fontsize=6, bold=False)

# Arrow from P2 to visual scoring
draw_arrow(ax, 15.7, 7.0, 15.7, 4.8, color=COLORS['correction'])
draw_arrow(ax, 14.35, 5.3, 14.35, 4.8, color=COLORS['boundary'])

# ============================================================
# VISUAL SCORING + FUSION (middle section, y = 3.5 - 5.0)
# ============================================================

# Step 7: Visual prototype scoring
draw_box(ax, 4.0, 3.8, 3.5, 1.0,
         'Visual Prototype Scoring\n$s_{vis}(f_k) = \\max(0, \\langle f_k, \\hat{p}_c \\rangle - \\lambda \\langle f_k, p_{neg} \\rangle)$\n(ReLU truncation, non-negative)',
         COLORS['correction_light'], COLORS['correction'], fontsize=6.5, bold=True)

# Step 8: Geometric fusion
draw_box(ax, 8.5, 3.8, 3.5, 1.0,
         'Geometric Mean Fusion\n$s_{fused,k} = \\sqrt{\\mathrm{clip}(s_{det}, \\epsilon, 1) \\cdot \\mathrm{clip}(s_{vis}, \\epsilon, 1)}$\n$\\epsilon = 10^{-6}$ (clip applied separately)',
         COLORS['correction_light'], COLORS['correction'], fontsize=6.5, bold=True)

# Arrow scoring -> fusion
draw_arrow(ax, 7.5, 4.3, 8.5, 4.3)

# Arrow from proposal pool down to scoring
draw_arrow(ax, 4.2, 6.5, 4.2, 4.8, color=COLORS['boundary'])
ax.text(4.35, 5.6, 'f_k', fontsize=7, color=COLORS['boundary'], weight='bold')

# Arrow from fusion to output
draw_arrow(ax, 12.0, 4.3, 12.8, 4.3)

# Output
draw_box(ax, 12.8, 3.8, 2.8, 1.0,
         'Output: Re-ranked Proposals\nAP50 / AP@[0.5:0.95]\n(re-ranking only, no recall change)',
         COLORS['boundary_light'], COLORS['text'], fontsize=6.5, bold=True)

# ============================================================
# PARALLEL BRANCH: Class-level Physical Prior (y = 1.8 - 3.2)
# ============================================================

ax.text(0.5, 3.2, 'Class-level Physical Prior Adaptive Control (zero target-domain info)',
        fontsize=8, weight='bold', color=COLORS['correction'])

# Prior factors
draw_box(ax, 0.3, 2.0, 2.5, 0.9,
         'Physical Prior Factors\nG: granularity mismatch\nS: scale/resolution\nI: intra-class variance\nC: inter-class confusion',
         COLORS['correction_light'], COLORS['correction'], fontsize=6, bold=True)

# Risk score
draw_box(ax, 3.3, 2.0, 2.5, 0.9,
         'Class Risk Score\nR(c) = w_G·G + w_S·S + w_I·I + w_C·C\n(V, M excluded after ablation)',
         COLORS['correction_light'], COLORS['correction'], fontsize=6, bold=True)

# Parameter modulation
draw_box(ax, 6.3, 2.0, 2.8, 0.9,
         'Per-class Parameter Modulation\nβ_eff(c) = β_base·(1 + α·R(c))\nλ_eff(c) = λ_base·(1 + α·R(c))',
         COLORS['correction_light'], COLORS['correction'], fontsize=6, bold=True)

# Result tag
draw_box(ax, 9.6, 2.0, 3.0, 0.9,
         'Result: DIOR→COCO 6 classes\nFixed params: +0.0166\nPrior adaptive: +0.0461 (6/6 positive)',
         COLORS['correction'], 'white', fontsize=6, bold=True)

# Arrows in parallel branch
draw_arrow(ax, 2.8, 2.45, 3.3, 2.45, color=COLORS['correction'])
draw_arrow(ax, 5.8, 2.45, 6.3, 2.45, color=COLORS['correction'])
draw_arrow(ax, 9.1, 2.45, 9.6, 2.45, color=COLORS['correction'])

# Feedback arrow to main pipeline
draw_arrow(ax, 7.7, 2.9, 7.7, 3.8, color=COLORS['correction'], style='->', lw=1.5)
ax.text(7.85, 3.35, 'modulates β, λ', fontsize=6, color=COLORS['correction'], weight='bold')

# ============================================================
# BOTTOM: Three Pillars Validation Band (y = 0.3 - 1.5)
# ============================================================

ax.plot([0.3, 15.7], [1.6, 1.6], color=COLORS['border'] if 'border' in COLORS else COLORS['boundary'], 
        linewidth=0.5, linestyle='--', alpha=0.5)

# P1 Diagnosis pillar
draw_box(ax, 0.3, 0.4, 4.8, 1.0,
         'P1 Diagnosis: Layered Failure Analysis\n• Ranking AUC vs. proposal recall ceiling (separated)\n• ΔAUC connects extrapolation to gain prediction\n• Distinguishes "coverage不足" from "ranking失效"',
         COLORS['correction_light'], COLORS['correction'], fontsize=6, bold=False)

# P2 Repair pillar
draw_box(ax, 5.6, 0.4, 4.8, 1.0,
         'P2 Repair: Asymmetric Prototype Construction\n• + prototype extrapolates along text direction\n• - prototype retains source-space (ablation confirms)\n• Symmetric extrapolation & no-negative as structural controls',
         COLORS['source_light'], COLORS['source'], fontsize=6, bold=False)

# P3 Boundary pillar
draw_box(ax, 10.9, 0.4, 4.8, 1.0,
         'P3 Boundary: Direction Gate & Applicability\n• g₂ controls overhead→photo valid extrapolation\n• Cross-source replication (DIOR/NWPU/DOTA/VisDrone)\n• Weak→strong detector spectrum defines corrector boundary',
         COLORS['boundary_light'], COLORS['boundary'], fontsize=6, bold=False)

# Method positioning tag
draw_box(ax, 5.5, -0.3, 5.0, 0.5,
         'A-CSPC = training-free proposal re-ranking corrector\nfor weakly-calibrated open-vocabulary detectors',
         COLORS['boundary_light'], COLORS['text'], fontsize=6.5, bold=True)

# ============================================================
# LEGEND
# ============================================================
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['source_light'], 
           markeredgecolor=COLORS['source'], markersize=10, label='Source domain / source-space'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['target_light'], 
           markeredgecolor=COLORS['target'], markersize=10, label='Target domain / target-space'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['correction_light'], 
           markeredgecolor=COLORS['correction'], markersize=10, label='Correction & fusion'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['boundary_light'], 
           markeredgecolor=COLORS['boundary'], markersize=10, label='Evaluation & boundary'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02),
          ncol=4, fontsize=7, frameon=False)

plt.tight_layout()
plt.savefig(r'C:\Users\张三\Desktop\实验代码\acspc_flowchart_nature.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(r'C:\Users\张三\Desktop\实验代码\acspc_flowchart_nature.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(r'C:\Users\张三\Desktop\实验代码\acspc_flowchart_nature.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Flowchart saved successfully!")
print("  - PNG: acspc_flowchart_nature.png (300dpi)")
print("  - PDF: acspc_flowchart_nature.pdf")
print("  - SVG: acspc_flowchart_nature.svg")
