"""Redraw Fig.14 (paper fig9.png): the gamma-probe grouped bar chart.

Data is identical to the original (figure8_realized_oracle arrays in
redraw_remaining_figures.py). Only the TEXT LABELS change, to close the
reviewer's "visible bar heights do not match the main protocol" complaint:
  - legend "A-CSPC (gamma=0)" -> "No-prior ref. (gamma=0)"  (this is the
    probe's own no-prior reference, not a re-tabulation of the main table)
  - title/tick labels consistent with the paper caption
No value is altered; nothing is re-estimated.
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\scipilot-figure-skill\scripts')

import matplotlib.pyplot as plt
import numpy as np
from setup_style import setup_style
from export_figure import export_figure

setup_style(journal='nature', lang='en')

categories_coco = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
categories_voc = ['aeroplane', 'boat', 'car', 'bus', 'motorbike']

acspc_coco =   [0.012, 0.025, 0.081, 0.077, 0.074, 0.119]
oracle_coco =  [0.012, 0.025, 0.136, 0.079, 0.085, 0.119]
realized_coco =[0.012, 0.025, 0.136, 0.079, 0.085, 0.119]

acspc_voc =   [0.002, -0.022, 0.086, 0.066, -0.061]
oracle_voc =  [0.002, -0.022, 0.088, 0.069, -0.061]
realized_voc =[0.002, -0.022, 0.088, 0.066, -0.061]

x_coco = np.arange(len(categories_coco))
x_voc = np.arange(len(categories_voc))
width = 0.25

GRAY, BLUE, GREEN = '#888888', '#56B4E9', '#009E73'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))

ax1.bar(x_coco - width, acspc_coco, width, label=r'No-prior ref. ($\gamma{=}0$)',
        color=GRAY, edgecolor='white')
ax1.bar(x_coco, oracle_coco, width, label=r'Oracle $\gamma$', color=BLUE, edgecolor='white')
ax1.bar(x_coco + width, realized_coco, width, label=r'Realized $\gamma$', color=GREEN, edgecolor='white')
ax1.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8)
ax1.set_ylabel('Gain (AP50)', fontsize=10)
ax1.set_title(r'(a) DIOR$\rightarrow$COCO', fontsize=11, fontweight='bold')
ax1.set_xticks(x_coco)
ax1.set_xticklabels(categories_coco, rotation=30, ha='right', fontsize=8.5)
ax1.set_ylim(0, 0.145)
ax1.legend(fontsize=8, loc='upper left')

ax2.bar(x_voc - width, acspc_voc, width, label=r'No-prior ref. ($\gamma{=}0$)',
        color=GRAY, edgecolor='white')
ax2.bar(x_voc, oracle_voc, width, label=r'Oracle $\gamma$', color=BLUE, edgecolor='white')
ax2.bar(x_voc + width, realized_voc, width, label=r'Realized $\gamma$', color=GREEN, edgecolor='white')
ax2.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8)
ax2.set_title(r'(b) DIOR$\rightarrow$VOC', fontsize=11, fontweight='bold')
ax2.set_xticks(x_voc)
ax2.set_xticklabels(categories_voc, rotation=30, ha='right', fontsize=8.5)
ax2.set_ylim(-0.07, 0.10)
ax2.legend(fontsize=8, loc='upper left')

fig.suptitle(r'Prior as a Multiplicative Term $p^{\gamma}$: Per-class Gain (Probe)',
             fontsize=11.5, fontweight='bold', y=1.02)

plt.tight_layout()
export_figure(
    fig,
    basename=r'C:\Users\张三\Desktop\A-CSPC_IEEE_style\figures\fig9',
    formats=['png'],
    size_inches=(7.2, 3.8),
    dpi=300,
    grayscale_preview=True,
)
print('Fig.14 (fig9.png) regenerated: probe reference relabelled, data unchanged.')
