# -*- coding: utf-8 -*-
"""Figure 5: Cross-Model Main Experiment Comparison — 深色学术风"""

import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_dir = r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据'
with open(f'{data_dir}/map_full20_all.json') as f:
    d = json.load(f)

BLUE   = '#6B8EAD'
GREEN  = '#7A9A7E'
ORANGE = '#C4825A'

models_data = {
    'GDINO\ntiny': [
        ('sat', d['gdino_tiny']['satellite']['f1'], BLUE),
        ('overhead', d['gdino_tiny']['overhead']['f1'], GREEN),
        ('TDM', d['gdino_tiny']['overhead+TDM']['f1'], ORANGE),
    ],
    'GDINO\nbase': [
        ('sat', d['gdino_base']['satellite']['f1'], BLUE),
        ('overhead', d['gdino_base']['overhead']['f1'], GREEN),
        ('TDM', d['gdino_base']['tdm']['f1'], ORANGE),
    ],
    'YOLO\nWorld': [
        ('sat', d['yolo_world']['satellite']['f1'], BLUE),
        ('overhead', d['yolo_world']['overhead']['f1'], GREEN),
        ('TDM', d['yolo_world']['tdm']['f1'], ORANGE),
    ],
    'OWLv2': [
        ('sat', d['owlv2']['satellite']['f1'], BLUE),
        ('overhead', d['owlv2']['overhead']['f1'], GREEN),
        ('TDM\n(cal)', d['owlv2']['tdm_cal']['f1'], ORANGE),
    ],
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#CCCCCC',
    'text.color': '#333333',
})

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('white')

n_models = len(models_data)
n_methods = 3
bar_width = 0.20
group_gap = 0.18

all_vals = []
group_centers = []
for i, (mname, methods) in enumerate(models_data.items()):
    base_x = i * (n_methods * bar_width + group_gap)
    for j, (label, val, color) in enumerate(methods):
        x = base_x + j * bar_width
        bar = ax.bar(x, val, width=bar_width, color=color, edgecolor='white', linewidth=0.5)
        all_vals.append(val)
        # value label
        ax.text(x + bar_width/2, val + max(all_vals)*0.008,
                f'{val:.4f}', ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', color=color)
    group_centers.append(base_x + (n_methods - 1) * bar_width / 2)

ax.set_xticks(group_centers)
ax.set_xticklabels(list(models_data.keys()), fontsize=10)

ax.set_ylabel('F1', fontsize=11, color='#666666')
ax.grid(axis='y', alpha=0.2, linewidth=0.5, color='#CCCCCC')
ax.set_axisbelow(True)
ax.set_ylim(0, max(all_vals) * 1.1)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')
ax.tick_params(axis='y', labelsize=9, colors='#999999')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=BLUE, label='Satellite prefix'),
    Patch(facecolor=GREEN, label='Overhead prefix'),
    Patch(facecolor=ORANGE, label='TDM (Ours)'),
]
ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=9)

plt.tight_layout(pad=1.5)
out = r'C:\Users\张三\Desktop\fig5_current_data.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}')
