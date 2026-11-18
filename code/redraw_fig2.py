"""Redraw Fig.2 (file fig1.png) to make the main inference path bypass
offline diagnosis and the optional gate."""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\scipilot-figure-skill\scripts')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from setup_style import setup_style
from export_figure import export_figure

setup_style(journal='nature', lang='en')

# Canvas size in inches (full IEEE width)
fig_w, fig_h = 7.5, 3.6
fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Colors (light theme)
c_input = (0.95, 0.96, 0.98)
c_calib = (0.98, 0.95, 0.95)
c_score = (0.95, 0.98, 0.95)
c_diag = (0.96, 0.96, 0.94)
c_gate = (0.96, 0.94, 0.98)
c_text = '#222222'
c_muted = '#555555'
c_main_arrow = '#2E5984'
c_dashed = '#888888'

def box(ax, x, y, w, h, text, facecolor, edgecolor='#444444', fontsize=7.5,
        text_color=c_text, bold=False, radius=0.03, line_width=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius*min(w,h)}",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=line_width, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, weight=weight, zorder=3)
    return box

def arrow(ax, x1, y1, x2, y2, color=c_main_arrow, lw=2, style='->', ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls,
                               connectionstyle='arc3,rad=0'),
                zorder=1)

# Title
ax.text(50, 97, 'A-CSPC: main inference path bypasses offline diagnosis and optional gate',
        ha='center', va='top', fontsize=12, weight='bold', color=c_text)

# (a) Inputs
box(ax, 2, 62, 18, 28,
    '(a) Inputs\nSource domain S\nTarget domain T\nText-specified detector B\nProposal pool {b_k}',
    c_input, fontsize=7.5, bold=False)

# Main pipeline arrow: (a) -> (c)
arrow(ax, 20, 76, 27, 76, lw=2.5)
ax.text(23.5, 78.5, 'main path', ha='center', va='bottom', fontsize=7, color=c_main_arrow, weight='bold')

# (c) Asymmetric calibration
box(ax, 27, 62, 22, 28,
    '(c) Asymmetric Prototype Calibration\n\nPositive p_c, text offset\n d(x_c) - s(x_s(c))\nExtrapolate:\n p_hat_c = N(p_c + beta*delta_c)\n\nNegative n_c\n(source hard negatives,\n unchanged)',
    c_calib, fontsize=6.0)

# Main arrow: (c) -> (d)
arrow(ax, 49, 76, 55, 76, lw=2.5)

# (d) Main scoring & re-ranking
box(ax, 55, 62, 23, 28,
    '(d) Score Fusion & Re-ranking\n\nS_vis = max(0,\n s_pos - lambda*s_neg)\nS_fused = sqrt(\n clip(S_vis)*clip(S_det))\n\nRe-rank proposals\nFinal output',
    c_score, fontsize=6.0)

# Main arrow: (d) -> output
arrow(ax, 78, 76, 81, 76, lw=2.5)
box(ax, 81, 68, 17, 16, 'Re-ranked\ndetections', c_score, fontsize=8, bold=True)

# Offline diagnosis (b) below, dashed
box(ax, 22, 18, 50, 26,
    '(b) Offline Failure Diagnosis  (target GT required; not on inference path)\n\nCoverage R_max        Ranking Quality AUC        Failure indicator dAUC\n\nComputed offline for retrospective analysis only;\ndoes not gate or alter the main inference path.',
    c_diag, edgecolor='#999999', fontsize=6.8, line_width=1.2)
# Dashed line from diagnosis up to calibration (informative, not data flow)
arrow(ax, 47, 44, 38, 61, color=c_dashed, lw=1.2, ls='--', style='->')
ax.text(40, 52, 'informs\nunderstanding\nonly', ha='center', va='center', fontsize=6, color=c_muted, style='italic')

# Optional gate branch to the right of (d)
box(ax, 78, 18, 20, 26,
    'Optional direction gate g_2\n(standalone module, not\ninvoked in main results)\n\nIf g_2 > 0: extrapolate\nIf g_2 <= 0: fallback to p_c',
    c_gate, edgecolor='#7A5195', fontsize=6.5, text_color='#4A2C6A')
# Dashed line from calibration to gate
arrow(ax, 60, 62, 82, 44, color='#7A5195', lw=1.2, ls=':', style='->')
# Dashed line from gate to scoring (optional bypass)
arrow(ax, 84, 44, 70, 68, color='#7A5195', lw=1.2, ls=':', style='->')
ax.text(87, 55, 'optional\nbranch', ha='center', va='center', fontsize=6, color='#7A5195', style='italic')

# Bottom legend / note
ax.text(50, 5, 'Main results (Table III) use the raw path (a -> c -> d) with beta=2.0, lambda=0.5 and no gate; g_2 and offline diagnosis are evaluated separately.',
        ha='center', va='bottom', fontsize=7, color=c_muted, style='italic')

plt.tight_layout()
export_figure(fig, basename=r'C:\Users\张三\Desktop\A-CSPC_IEEE_style\figures\fig1',
              formats=['png'], size_inches=(fig_w, fig_h), dpi=300, grayscale_preview=False)
print("Fig.2 redrawn.")
