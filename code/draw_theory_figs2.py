# -*- coding: utf-8 -*-
"""draw_theory_figs2.py — 顶刊风格重画：Fig-A 研究流程图 + Fig-B 协议模块图

规范：正交折线箭头（总线式）、muted 色盲安全配色、面板 (a)(b)(c) 互不侵入、
盒内文字 9-10pt 不溢出、面板间留白 20%。倒置 y 坐标（自上而下布局）。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os

BLUE, ORANGE, GREEN, RED = '#4C72B0', '#DD8452', '#55A868', '#C44E52'
GREY, DARK = '#8C8C8C', '#2b2b2b'
FILL = {BLUE: '#e9eef7', ORANGE: '#faf0e6', GREEN: '#e9f3ea', RED: '#f9e9e9', GREY: '#f0f0f0',
        DARK: '#2b2b2b'}
OUT = r'C:\Users\张三\Desktop\实验数据\figs_theory'
os.makedirs(OUT, exist_ok=True)


def new_ax(w, h, W=1320):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, W)
    ax.set_ylim(W * h / w, 0)          # 倒置 y：自上而下
    ax.axis('off')
    return fig, ax, W * h / w


def box(ax, x0, y0, x1, y1, color, text, ls='-', fs=10, bold=False, lw=1.6, tcolor='#1a1a1a'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=4',
                                lw=lw, edgecolor=color,
                                facecolor=FILL.get(color, 'white'), linestyle=ls))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center', fontsize=fs,
            color=tcolor, fontweight='bold' if bold else 'normal')


def earrow(ax, x0, y, x1, y2=None, color='#555555', ls='-', lw=1.5, ms=13):
    """正交箭头：给 (x0,y)→(x1,y2)，y2=None 时水平；x0==x1 时垂直。"""
    y2 = y if y2 is None else y2
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y2), arrowstyle='-|>', mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls, shrinkA=2, shrinkB=2,
                                 connectionstyle='angle,angleA=0,angleB=90,rad=6'
                                 if abs(x0 - x1) > 5 and abs(y - y2) > 5 else 'arc3,rad=0'))


def panel(ax, x, letter, title, fs=13):
    ax.add_patch(Circle((x, 34), 13, facecolor=DARK, edgecolor='none'))
    ax.text(x, 34, letter, color='white', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(x + 22, 34, title, ha='left', va='center', fontsize=fs, fontweight='bold', color=DARK)


# ═══════════ Fig A: 研究流程图 ═══════════
fig, ax, H = new_ax(13.2, 6.8)

# --- 面板 a：框架（两级污染链，自上而下） ---
panel(ax, 24, 'a', 'Two-level contamination chain')
chain = [('Scene $z$', GREY, 80, 128), ('Observation $X=\\mathcal{O}(z,c,\\varepsilon)$', BLUE, 158, 226),
         ('Feature $f(X)$', GREEN, 256, 304), ('Prototype scoring $\\hat y$', GREEN, 334, 382)]
for t, c, y0, y1 in chain:
    box(ax, 30, y0, 280, y1, c, t, fs=10.5)
for i in range(len(chain) - 1):
    earrow(ax, 155, chain[i][3] + 4, 155, chain[i + 1][2] - 4, lw=1.7)
# 两级标注
box(ax, 292, 150, 396, 234, RED, 'Level 1\nobservational\nscoring-side\ncannot reach', fs=9, tcolor=RED, lw=1.3)
box(ax, 292, 250, 396, 388, ORANGE, 'Level 2\nrepresentational\nprior work\noperates here', fs=9, tcolor='#b35a1f', lw=1.3)
ax.text(213, 430, 'reframed claim: the two-level structure is retained;\n'
        'what changes is how Level 1 is addressed', ha='center', fontsize=8.8,
        style='italic', color=DARK)

# --- 面板 b：受控实验 ---
panel(ax, 470, 'b', 'Controlled experiment')
box(ax, 480, 80, 660, 126, GREY, 'Clean COCO GT (6 cls)', fs=9.5)
earrow(ax, 570, 130, 570, 152)
box(ax, 480, 156, 660, 232, BLUE, 'Degradation engine\nfog / blur / brightness × sev 1·3·5', fs=9.5)
earrow(ax, 540, 236, 540, 262)
earrow(ax, 600, 236, 760, 262)
box(ax, 460, 266, 650, 350, GREEN, 'Image-side restoration\nDCP dehaze · unsharp ·\nautocontrast (blind)', fs=9.5)
box(ax, 690, 266, 880, 350, ORANGE, 'Prototype-side correction\nbase / TGE / CA-PC', fs=9.5)
ax.text(760, 372, 'controls: wrong-anchor · random · gate-only', ha='center',
        fontsize=8.2, color=RED, style='italic')
earrow(ax, 555, 354, 555, 388)
earrow(ax, 785, 380, 785, 388)
box(ax, 560, 392, 790, 442, GREY, 'GDINO-tiny re-detect + CLIP-b32 rescoring\n(identical per condition across all $W$)', fs=9.2)

# --- 面板 c：归属端点（总线式正交走线） ---
panel(ax, 950, 'c', 'Attribution endpoints')
box(ax, 960, 74, 1300, 140, BLUE, '$H_{ext} = \\mathrm{AP}_{clean} - \\mathrm{AP}_{deg}$'
                                    '\nindependent headroom (base only)', fs=9.2)
box(ax, 960, 168, 1300, 238, GREEN, 'P1  $R_{rest}$ — image-side recovery share\n'
                                    'does restoration eat what scoring cannot?', fs=9.2)
box(ax, 960, 272, 1300, 342, ORANGE, 'P2  $\\Delta g_{CA}$ — gain persistence\n'
                                     'does anchored gain survive repair?', fs=9.2)
# 总线：从 re-detect 右侧出，垂直总线 x=915，分三路
earrow(ax, 794, 417, 915, 417, ls='-', lw=1.5)
ax.plot([915, 915], [107, 417], color='#555555', lw=1.5)
earrow(ax, 915, 107, 956, 107)
earrow(ax, 915, 203, 956, 203)
earrow(ax, 915, 307, 956, 307)

# --- 底部结论条 ---
box(ax, 24, 560, 1300, 646, DARK, 'Refuted by controls:  scoring-side sufficiency · gate-only standalone · anchor direction specificity\n'
    'Retained with first signals:  two-level structure, dominant Level-1 loss → image-side compensation', fs=9.8, tcolor='white', bold=True)

fig.savefig(os.path.join(OUT, 'figA_workflow.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'figA_workflow.png'), bbox_inches='tight')
plt.close(fig)

# ═══════════ Fig B: 协议模块图 ═══════════
fig, ax, H = new_ax(13.2, 7.0)

# --- 面板 a：共享输入 ---
panel(ax, 24, 'a', 'Shared inputs (per class)')
box(ax, 30, 80, 300, 150, GREEN, 'Source GT crops\n→ $P_{src}$ (CLIP-b32)', fs=9.5)
box(ax, 30, 180, 300, 250, GREEN, 'Other-class crops\n→ $P_{neg}$ (top-k)', fs=9.5)
box(ax, 30, 300, 300, 390, BLUE, 'Condition prompt bank\n"clear / foggy / sharp /\noverexposed / overhead …"', fs=9.3)
box(ax, 30, 420, 300, 495, BLUE, 'Text anchors $e_s, e_t$\n→ $d_c$, $R(c)$', fs=9.5)
earrow(ax, 165, 394, 165, 416)

# --- 面板 b：打分变体 ---
panel(ax, 420, 'b', 'Scoring variants')
box(ax, 430, 80, 800, 148, GREY, 'base    $W = P_{src} - \\lambda P_{neg}$', fs=9.8)
box(ax, 430, 175, 800, 243, ORANGE, 'TGE      $W = \\mathrm{unit}(P_{src} + d_c) - \\lambda P_{neg}$', fs=9.8)
box(ax, 430, 270, 800, 352, GREEN, 'CA-PC   $W = \\mathrm{unit}(P_{src} + \\beta d_c) - \\lambda P_{neg}$\n'
                                   'β = 2 fixed;  β★ grid → oracle-tuned reference', fs=9.8)
box(ax, 430, 380, 800, 470, RED, 'Controls\nwrong-condition anchor · 49 norm-matched\ndirections · gate-only ablation', fs=9.5, ls='--')
# 输入总线：绿（原型）→ base/TGE；蓝（锚）→ CA-PC；红虚（锚）→ Controls
ax.plot([360, 360], [115, 215], color=GREEN, lw=1.6)
earrow(ax, 305, 115, 360, 115, color=GREEN)
earrow(ax, 305, 215, 360, 215, color=GREEN)
earrow(ax, 360, 115, 426, 115, color=GREEN)
earrow(ax, 360, 215, 426, 215, color=GREEN)
ax.plot([360, 360], [311, 457], color=BLUE, lw=1.6)
earrow(ax, 305, 457, 360, 457, color=BLUE)
earrow(ax, 360, 311, 426, 311, color=BLUE)
earrow(ax, 360, 425, 426, 425, color=RED, ls='--')

# --- 面板 c：条件轴与评测 ---
panel(ax, 900, 'c', 'Condition axes & evaluation')
box(ax, 910, 76, 1300, 150, BLUE, 'Degradation axis:  fog / blur / brightness × sev 1·3·5\n'
                                  '(deterministic seeds, bitwise-identical images)', fs=8.8)
earrow(ax, 1105, 154, 1105, 174)
box(ax, 910, 178, 1300, 252, GREEN, 'Restoration branch:  blind operators\n→ re-detect on repaired images', fs=9.3)
earrow(ax, 1105, 256, 1105, 278)
box(ax, 910, 282, 1300, 372, GREY, 'Evaluation:  AP50 (11-pt) + bootstrap CI\nidentical GT & detections per condition\n→ mechanism attribution not confounded', fs=8.8)
box(ax, 910, 400, 1300, 470, RED, 'Pathway control:  preprocess-only branch\n(ENH vs RAW — does repair absorb the gain?)', fs=9.3, ls='--')
earrow(ax, 1105, 396, 1105, 376, color=RED, ls='--')
earrow(ax, 804, 320, 906, 320, lw=1.6)

ax.text(660, 540, 'All scoring variants share one detection cache per condition —\n'
        'only the scoring weight $W$ differs across modes.',
        ha='center', fontsize=9.5, style='italic', color=DARK)

fig.savefig(os.path.join(OUT, 'figB_modules.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'figB_modules.png'), bbox_inches='tight')
plt.close(fig)
print('saved to', OUT)
