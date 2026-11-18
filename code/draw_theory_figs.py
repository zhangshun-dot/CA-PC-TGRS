# -*- coding: utf-8 -*-
"""draw_theory_figs.py — 两层污染理论论文的 Fig-A 研究流程图 + Fig-B 协议模块图

顶刊风格：分面板 (a)(b)(c)、浅色填充+彩边圆角盒、统一箭头、面板标题左上角加粗。
沿用 draw_fig1_framework.py 的四色语言（蓝=文本锚/条件、橙=门控/系数、红=对照/警示、绿=原型/打分）。
输出: figs_theory/figA_workflow.(pdf|png), figs_theory/figB_modules.(pdf|png)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os

BLUE, ORANGE, RED, GREEN = '#1f77b4', '#e8890c', '#d62728', '#2ca02c'
DARK, GREY = '#333333', '#777777'
FILL = {BLUE: '#eaf2fb', ORANGE: '#fdf1e3', RED: '#fdecec', GREEN: '#eaf6ea',
        GREY: '#f2f2f2', '#8e44ad': '#f3eaf8'}
OUT = r'C:\Users\张三\Desktop\实验数据\figs_theory'
os.makedirs(OUT, exist_ok=True)


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 1200 * h / w)
    ax.axis('off')
    return fig, ax


def box(ax, x0, y0, x1, y1, color, text, ls='-', fs=11, tcolor='black', bold=False,
        fill=True, lw=1.8):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle='round,pad=5', linewidth=lw,
                                edgecolor=color, facecolor=FILL.get(color, 'white') if fill else 'white',
                                linestyle=ls))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center',
            fontsize=fs, color=tcolor, fontweight='bold' if bold else 'normal')


def arrow(ax, x0, y0, x1, y1, color='#444444', ls='-', lw=1.6, ms=15):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                 mutation_scale=ms, linewidth=lw,
                                 color=color, linestyle=ls, shrinkA=3, shrinkB=3))


def panel_label(ax, x, y, letter, title):
    ax.add_patch(Circle((x, y), 17, facecolor=DARK, edgecolor='none'))
    ax.text(x, y, letter, color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(x + 28, y, title, ha='left', va='center', fontsize=13.5, fontweight='bold', color=DARK)


# ══════════════════════ Fig A: 研究流程图 ══════════════════════
fig, ax = new_ax(13.5, 6.4)
H = 1200 * 6.4 / 13.5  # ~569

panel_label(ax, 30, H - 28, 'a', 'Two-level contamination chain (framework)')
panel_label(ax, 470, H - 28, 'b', 'Controlled experiment (three axes + controls)')
panel_label(ax, 880, H - 28, 'c', 'Mechanism attribution')

# (a) 两级污染链：竖直管线
chain = [('Scene $z$', GREY), ('Observation $X=\\mathcal{O}(z,c,\\varepsilon)$', BLUE),
         ('Feature $f(X)$', GREEN), ('Prototype scoring $\\hat y$', GREEN)]
yy = 60
prev = None
for i, (t, c) in enumerate(chain):
    y0 = yy + i * 108
    box(ax, 40, y0, 300, y0 + 74, c, t, fs=11.5)
    if prev is not None:
        arrow(ax, 170, y0 - 34, 170, y0 - 2, lw=1.8)
    prev = y0
# 两级标注（右侧红括号）
ax.annotate('', xy=(312, 168 + 74), xytext=(312, 168),
            arrowprops=dict(arrowstyle='-', color=RED, lw=2.2))
ax.text(322, 168 + 37, 'Level 1\nobservational\n(irreversible,\nscoring-side\ncannot reach)', color=RED,
        fontsize=9.5, va='center')
ax.annotate('', xy=(312, 384 + 74), xytext=(312, 384),
            arrowprops=dict(arrowstyle='-', color=ORANGE, lw=2.2))
ax.text(322, 384 + 37, 'Level 2\nrepresentational\n(correctable,\nwhere prior work\noperates)', color=ORANGE,
        fontsize=9.5, va='center')

# (b) 受控实验
box(ax, 480, 60, 700, 130, GREY, 'Clean COCO GT\n(6 classes)', fs=10.5)
box(ax, 480, 170, 700, 300, BLUE, 'Degradation engine\nfog / blur / brightness\nsev 1·3·5  (deterministic)', fs=10.5)
arrow(ax, 590, 132, 590, 166)

# 两条修复分支
box(ax, 740, 330, 960, 420, GREEN, 'Image-side restoration\nDCP dehaze / unsharp /\nautocontrast  (blind)', fs=10)
box(ax, 740, 170, 960, 260, ORANGE, 'Prototype-side correction\n$W=\\mathrm{unit}(P_{src}+\\beta d_c)-\\lambda P_{neg}$'
                                    '\nbase / TGE / CA-PC', fs=9.5)
arrow(ax, 640, 302, 780, 326, lw=1.6)
arrow(ax, 660, 302, 800, 264, lw=1.6)
# 汇合到重检测/打分
box(ax, 740, 460, 960, 530, GREY, 'GDINO-tiny re-detect\n+ CLIP-b32 rescoring', fs=10)
arrow(ax, 850, 424, 850, 456)
arrow(ax, 850, 264, 850, 456, ls=(0, (4, 3)), color=ORANGE)

# 对照组（红虚线）
box(ax, 480, 340, 700, 500, RED, 'Controls (mechanism\nattribution)\n• wrong-condition anchor\n• 49 random directions\n'
                                 '• preprocess pathway\n• gate-only ablation', fs=9.5, ls='--')
arrow(ax, 700, 420, 736, 420, color=RED, ls='--')

# (c) 归属判决
box(ax, 1000, 330, 1170, 420, BLUE, '$H_{ext}$ = AP$_{clean}$−AP$_{deg}$\n(independent headroom)', fs=9.5)
box(ax, 1000, 200, 1170, 300, GREEN, 'P1: $R_{rest}$\nimage-side recovery\nshare of $H_{ext}$', fs=9.5)
box(ax, 1000, 60, 1170, 170, ORANGE, 'P2: $\\Delta g_{CA}$\ngain persistence\nafter restoration', fs=9.5)
arrow(ax, 964, 495, 1040, 424)
arrow(ax, 964, 215, 996, 250)
arrow(ax, 964, 215, 996, 120, ls=(0, (4, 3)))
# 结论
box(ax, 480, 560, 1170, H - 8, DARK, 'Refuted: scoring-side sufficiency · gate-only as standalone · '
    'anchor direction specificity        Retained: two-level structure (L1 dominant) '
    '→ image-side compensation as research direction', fs=10.5, tcolor='white', bold=True, fill=True)

fig.savefig(os.path.join(OUT, 'figA_workflow.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'figA_workflow.png'), bbox_inches='tight')
plt.close(fig)

# ══════════════════════ Fig B: 协议模块图 ══════════════════════
fig, ax = new_ax(13.5, 7.2)
H = 1200 * 7.2 / 13.5

panel_label(ax, 30, H - 28, 'a', 'Shared inputs (per class)')
panel_label(ax, 470, H - 28, 'b', 'Scoring variants (identical detections per condition)')
panel_label(ax, 880, H - 28, 'c', 'Condition axes & evaluation')

# (a) 输入模块
box(ax, 40, 470, 260, 560, GREEN, 'Source GT crops\n→ $P_{src}$', fs=10.5)
box(ax, 40, 330, 260, 420, GREEN, 'Other-class crops\n→ $P_{neg}$ (top-k)', fs=10.5)
box(ax, 40, 190, 260, 280, BLUE, 'Condition prompt bank\n$a$ clear / foggy / sharp /\noverexposed / overhead', fs=9.5)
box(ax, 40, 60, 260, 150, BLUE, 'Text anchors $e_s,e_t$\n→ $d_c$, $R(c)$ (CLIP-b32)', fs=10)
arrow(ax, 150, 466, 150, 424)
arrow(ax, 150, 186, 150, 154)

# (b) 打分变体（同一检测结果上只换 W）
box(ax, 470, 380, 700, 470, GREY, 'base  $W=P_{src}-\\lambda P_{neg}$', fs=10)
box(ax, 470, 290, 700, 360, ORANGE, 'TGE  $W=\\mathrm{unit}(P_{src}+d_c)-\\lambda P_{neg}$', fs=9.5)
box(ax, 470, 200, 700, 270, GREEN, 'CA-PC  $W=\\mathrm{unit}(P_{src}+\\beta d_c)-\\lambda P_{neg}$'
                                   '\n$\\beta$=2 fixed; $\\beta^\\star$ grid (oracle-tuned ref.)', fs=9.5)
box(ax, 470, 110, 700, 180, RED, 'Controls: ca\\_wrong (other condition)\nca\\_rand (49 norm-matched dirs)', fs=9.5, ls='--')
arrow(ax, 266, 105, 466, 320, color=BLUE, lw=1.4)
arrow(ax, 266, 515, 466, 425, color=GREEN, lw=1.4)
arrow(ax, 266, 375, 466, 150, color=RED, lw=1.4, ls='--')

# (c) 条件轴与评测
box(ax, 880, 420, 1160, 530, BLUE, 'Degradation axis\nfog / blur / brightness × sev 1·3·5', fs=10)
box(ax, 880, 290, 1160, 390, GREEN, 'Restoration branch\nblind operators → re-detect', fs=10)
box(ax, 880, 160, 1160, 260, GREY, 'AP50 (11-pt) + bootstrap CI\nidentical GT, identical detections\nper condition across all $W$', fs=9.5)
box(ax, 880, 40, 1160, 130, RED, 'Pathway control\npreprocess-only branch (ENH vs RAW)', fs=9.5, ls='--')
arrow(ax, 706, 240, 876, 215, lw=1.5)
arrow(ax, 1020, 416, 1020, 394)
arrow(ax, 1020, 286, 1020, 264)
arrow(ax, 1020, 156, 1020, 134, ls='--', color=RED)

# 底部注释
ax.text(600, 18, 'Detection cache shared across all scoring variants within a condition — '
        'mechanism attribution is not confounded by detection noise.',
        ha='center', fontsize=10, style='italic', color=DARK)

fig.savefig(os.path.join(OUT, 'figB_modules.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'figB_modules.png'), bbox_inches='tight')
plt.close(fig)
print('saved to', OUT)
