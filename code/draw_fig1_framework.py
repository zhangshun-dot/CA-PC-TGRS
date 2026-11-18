# -*- coding: utf-8 -*-
"""draw_fig1_framework.py — 重画 CA-PC 方法框架图（figs/fig1_framework.pdf）

rev21 第3条：正文 γ=β⋆·R·g·P_scale 已对，但图仍画 γ=R·g·P_scale 缺 β。
本脚本复刻原图布局，把 γ 公式改为含 β⋆ 的版本。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, RED, GREEN = '#1f77b4', '#e8890c', '#d62728', '#2ca02c'
DARK = '#333333'

fig, ax = plt.subplots(figsize=(11, 5.9), dpi=100)
ax.set_xlim(0, 1100)
ax.set_ylim(590, 0)  # invert y so coordinates match pixel space
ax.axis('off')


def box(x0, y0, x1, y1, color, text, ls='-', fs=13, tcolor='black'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle='round,pad=6', linewidth=1.8,
                                edgecolor=color, facecolor='white', linestyle=ls))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center',
            fontsize=fs, color=tcolor)


def arrow(x0, y0, x1, y1, color='#444444', ls='-'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=16, linewidth=1.6,
                                 color=color, linestyle=ls, shrinkA=2, shrinkB=2))


# ---- 顶部标注 ----
ax.text(155, 40, 'swap $s\\!\\leftrightarrow\\!t$: reverse direction',
        color=BLUE, fontsize=12, style='italic', ha='left')

# ---- 左列（蓝/红）----
box(135, 57, 340, 125, BLUE, 'Condition prompts\n$t_s(y),\\ \\,t_t(y)$')
box(135, 137, 340, 215, BLUE, 'Canonical anchors\n$e_s,\\ e_t$ (CLIP text)')
box(135, 243, 340, 330, RED, '$d_\\xi = e_t - e_s$\n$R(\\xi) = ||d_\\xi||$')

# ---- 中列（橙）----
box(465, 50, 630, 125, ORANGE, 'confidence gate\n$g(q)$')
box(465, 160, 630, 240, ORANGE,
    '$\\gamma = \\beta^{\\star} \\cdot R \\cdot g \\cdot P_{\\mathrm{scale}}$', fs=14)

# ---- 右列（绿）----
box(745, 55, 1010, 125, GREEN, '$P_{\\mathrm{src}},\\ P_{\\mathrm{neg}}$\n(source GT crops)')
box(745, 140, 1010, 215, GREEN,
    '$W = \\mathrm{unit}(P_{\\mathrm{src}} + \\gamma\\hat d_\\xi) - \\lambda P_{\\mathrm{neg}}$')
box(445, 350, 710, 425, GREEN,
    'Prototype rescoring\n$s(x) = \\sqrt{q\\,\\max(0,\\ F(x)\\!\\cdot\\!W)}$')
box(830, 355, 990, 420, GREEN, 'Detections')

# ---- 离线闭环（灰虚线）----
box(70, 440, 375, 510, '#777777',
    'offline source-validation\nAP grid $\\beta^{\\star}$ (once)', ls='--')

# ---- 箭头 ----
arrow(237, 125, 237, 135)            # prompts -> anchors
arrow(237, 215, 237, 241)            # anchors -> d_xi
arrow(342, 286, 463, 206)            # d_xi -> gamma
arrow(547, 125, 547, 158)            # gate -> gamma
arrow(632, 200, 743, 178)            # gamma -> W
arrow(877, 125, 877, 138)            # P_src -> W
arrow(760, 217, 668, 348)            # W -> rescoring
arrow(712, 388, 828, 388)            # rescoring -> detections
arrow(300, 438, 500, 242, ls='--')   # offline grid -> gamma (closed loop)

# ---- 底部标注 ----
ax.text(20, 545, 'closed loop: detection-metric feedback, no target boxes',
        fontsize=11, style='italic', color=DARK, ha='left')

fig.tight_layout(pad=0.5)
fig.savefig(r'C:\Users\张三\WorkBuddy\2026-09-23-16-06-04\figs\fig1_framework.pdf')
fig.savefig(r'C:\Users\张三\WorkBuddy\2026-09-23-16-06-04\figs\fig1_framework.png')
print('saved fig1_framework.pdf/.png')
