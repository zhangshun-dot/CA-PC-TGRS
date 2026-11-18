# -*- coding: utf-8 -*-
"""Generate Figure 4: Cross-Model TDM Transfer Framework.
Clean flat design matching Figure 1's style."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')

# Colors matching Figure 1
BLUE = '#D6E4F0'
BLUE_DARK = '#2B579A'
GREEN = '#E2EFDA'
GREEN_DARK = '#548235'
ORANGE = '#FCE4D6'
ORANGE_DARK = '#ED7D31'
GRAY = '#F2F2F2'
WHITE = '#FFFFFF'
BLACK = '#333333'

def draw_box(ax, x, y, w, h, color, text='', text_color=BLACK, lw=2, fontsize=9):
    """Draw a rounded rectangle with text."""
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=BLACK, linewidth=lw)
    ax.add_patch(rect)
    if text:
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold' if 'GDINO' in text or 'BERT' in text else 'normal')

def draw_arrow(ax, x1, y1, x2, y2, color=BLUE_DARK, lw=2):
    """Draw a straight arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

def draw_label(ax, x, y, text, color=BLACK, fontsize=8, ha='center', fontweight='normal'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, color=color, fontweight=fontweight)

# ═══════════════════════════════════════
# Column headers (label bars)
# ═══════════════════════════════════════
col_w = 2.4; gap = 0.4; start_x = 0.5; y_top = 4.2
cols = [
    (start_x, BLUE, 'OFAT Measurement'),
    (start_x + col_w + gap, GREEN, 'Same Encoder Transfer'),
    (start_x + 2*(col_w + gap), ORANGE, 'Cross-Encoder Transfer'),
    (start_x + 3*(col_w + gap), BLUE, 'Performance'),
    (start_x + 4*(col_w + gap), GRAY, 'Key Insight'),
]
for x, color, label in cols:
    rect = FancyBboxPatch((x, y_top), col_w, 0.4, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor=BLACK, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + col_w/2, y_top + 0.2, label, ha='center', va='center', fontsize=9, fontweight='bold')

# ═══════════════════════════════════════
# Column 1: OFAT Source
# ═══════════════════════════════════════
x1 = start_x; cx1 = x1 + col_w/2
# Mini table (3x3 grid)
for r in range(3):
    for c in range(3):
        rx = x1 + 0.4 + c * 0.5
        ry = 3.1 - r * 0.4
        color = BLUE if c == 1 and r == 1 else WHITE
        rect = Rectangle((rx, ry), 0.45, 0.35, facecolor=color, edgecolor=BLACK, linewidth=0.5)
        ax.add_patch(rect)
draw_label(ax, cx1, 2.5, '7 axes x 20 classes', fontsize=8)
draw_label(ax, cx1, 2.2, 'BERT Encoder', fontsize=8)
draw_label(ax, cx1, 1.9, '140 experiments', fontsize=7, color='#666')

# ═══════════════════════════════════════
# Column 2: Same Encoder
# ═══════════════════════════════════════
x2 = start_x + col_w + gap; cx2 = x2 + col_w/2
draw_box(ax, cx2 - 0.7, 3.4, 1.4, 0.35, WHITE, 'GDINO-tiny', lw=1.5)
draw_box(ax, cx2 - 0.7, 2.8, 1.4, 0.35, WHITE, 'GDINO-base', lw=1.5)
draw_box(ax, cx2 - 0.7, 2.2, 1.4, 0.35, WHITE, 'YOLO-World', lw=1.5)
# Green arrows between
draw_arrow(ax, cx2, 3.35, cx2, 3.2, GREEN_DARK)
draw_arrow(ax, cx2, 2.75, cx2, 2.6, GREEN_DARK)
# Checkmark
ax.text(cx2 + 0.6, 2.85, '✓', fontsize=18, color=GREEN_DARK, fontweight='bold')
draw_label(ax, cx2, 1.85, 'BERT → BERT', fontsize=8, color=GREEN_DARK)
draw_label(ax, cx2, 1.6, 'Zero-cost transfer', fontsize=8)
draw_label(ax, cx2, 1.35, 'ΔF1 ≤ 0.005', fontsize=7, color='#666')

# ═══════════════════════════════════════
# Column 3: Cross-Encoder
# ═══════════════════════════════════════
x3 = start_x + 2*(col_w + gap); cx3 = x3 + col_w/2
draw_box(ax, cx3 - 0.7, 3.6, 1.4, 0.35, WHITE, 'GDINO-tiny BERT', lw=1.5, fontsize=8)
draw_arrow(ax, cx3, 3.55, cx3, 3.2, ORANGE_DARK)
draw_box(ax, cx3 - 0.7, 2.85, 1.4, 0.35, WHITE, 'OWLv2 CLIP', lw=1.5, fontsize=8)

# 5 prefix blocks
prefixes = ['bare', 'satellite', 'aerial', 'overhead', 'rs']
pfx_y = 2.2
for i, (pfx, highlight) in enumerate(zip(prefixes, [False, False, False, True, False])):
    px = x3 + 0.15 + i * 0.44
    lw = 3 if highlight else 1.5
    fc = BLUE if highlight else WHITE
    rect = Rectangle((px, pfx_y), 0.4, 0.35, facecolor=fc, edgecolor=BLACK, linewidth=lw)
    ax.add_patch(rect)
    ax.text(px + 0.2, pfx_y + 0.17, pfx, ha='center', va='center', fontsize=7)

draw_label(ax, cx3, 1.8, '5 prefix scans only', fontsize=8, color=ORANGE_DARK)
draw_label(ax, cx3, 1.55, 'F1: 0.277 → 0.338', fontsize=8)
draw_label(ax, cx3, 1.3, '(+0.061, 22% gain)', fontsize=7, color='#666')

# ═══════════════════════════════════════
# Column 4: Performance Comparison
# ═══════════════════════════════════════
x4 = start_x + 3*(col_w + gap); cx4 = x4 + col_w/2
# Mini bar chart
bar_data = [('sat', 0.071, '#7B68EE'), ('oh', 0.138, BLUE_DARK), ('TDM', 0.142, ORANGE_DARK)]
bar_w = 0.4; bar_start = x4 + 0.4
for i, (label, val, color) in enumerate(bar_data):
    bx = bar_start + i * 0.6
    bh = val * 12  # scale for visibility
    rect = Rectangle((bx, 2.8), bar_w, bh, facecolor=color, edgecolor=BLACK, linewidth=1)
    ax.add_patch(rect)
    ax.text(bx + bar_w/2, 2.8 + bh + 0.05, f'{val:.3f}', ha='center', fontsize=7)
    ax.text(bx + bar_w/2, 2.6, label, ha='center', fontsize=8)
draw_label(ax, cx4, 2.2, 'F1 on DIOR (20-class)', fontsize=8)
draw_label(ax, cx4, 1.9, 'mAP: TDM +18.6% vs oh', fontsize=7, color='#666')

# ═══════════════════════════════════════
# Column 5: Key Insight
# ═══════════════════════════════════════
x5 = start_x + 4*(col_w + gap); cx5 = x5 + col_w/2
# Lock icon (simplified)
lock = mpatches.Circle((cx5, 3.2), 0.3, facecolor=BLUE_DARK, edgecolor=BLACK, linewidth=2)
ax.add_patch(lock)
ax.text(cx5, 3.2, 'K', ha='center', va='center', fontsize=18, fontweight='bold', color='white')
draw_label(ax, cx5, 2.7, 'Token distribution', fontsize=8, fontweight='bold')
draw_label(ax, cx5, 2.4, 'determined by', fontsize=8)
draw_label(ax, cx5, 2.15, 'text encoder only', fontsize=8, fontweight='bold')
draw_label(ax, cx5, 1.85, 'Decoupled from', fontsize=8)
draw_label(ax, cx5, 1.6, 'visual backbone', fontsize=8, color='#666')

# ═══════════════════════════════════════
# Arrows between columns
# ═══════════════════════════════════════
# Col 1 → Col 2
draw_arrow(ax, x1 + col_w, 3.8, x2, 3.8, GREEN_DARK)
draw_label(ax, (x1+col_w + x2)/2, 4.0, 'Reuse', fontsize=7, color=GREEN_DARK)
# Col 1 → Col 3
draw_arrow(ax, x1 + col_w, 3.3, x3, 3.3, ORANGE_DARK)
draw_label(ax, (x1+col_w + x3)/2, 3.5, 'Calibrate', fontsize=7, color=ORANGE_DARK)
# Both → Col 4
draw_arrow(ax, x2 + col_w, 3.0, x4, 3.0, BLUE_DARK)
draw_arrow(ax, x3 + col_w, 3.0, x4, 3.0, BLUE_DARK)
# Col 4 → Col 5
draw_arrow(ax, x4 + col_w, 3.0, x5, 3.0, BLUE_DARK)

# ═══════════════════════════════════════
# Legend bar at bottom
# ═══════════════════════════════════════
legend_items = [
    (0.5, 'OFAT Measurement', BLUE),
    (3.5, 'Same Encoder (BERT→BERT)', GREEN),
    (6.5, 'Cross-Encoder (BERT→CLIP)', ORANGE),
    (9.5, 'Performance Summary', BLUE),
    (11.5, 'Key Insight', GRAY),
]
for lx, label, color in legend_items:
    rect = Rectangle((lx, 0.3), 0.25, 0.25, facecolor=color, edgecolor=BLACK, linewidth=1)
    ax.add_patch(rect)
    ax.text(lx + 0.35, 0.42, label, fontsize=7, va='center')

plt.tight_layout(pad=0.5)
out = r'C:\Users\张三\Desktop\fig4_framework.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {out}')
plt.close()
