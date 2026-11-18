# -*- coding: utf-8 -*-
"""Generate Figures 5-8 for 初稿5: charts with clean academic style."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.edgecolor'] = '#333333'

data_dir = r'C:\Users\张三\Desktop\初稿4_论文文件\实验数据'
def load(f):
    with open(f'{data_dir}/{f}') as fh: return json.load(fh)

BLUE, ORANGE, GREEN, RED, PURPLE, GRAY = '#2B579A','#ED7D31','#548235','#C00000','#7030A0','#888888'

# ═══════════════════════════════════════
# Figure 5: Cross-Model Bar Chart
# ═══════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
models = [
    ('GDINO-tiny', [('sat',0.0903,BLUE),('oh',0.1738,GREEN),('TDM',0.1745,ORANGE)]),
    ('GDINO-base', [('sat',0.1777,BLUE),('oh',0.2057,GREEN),('TDM',0.2012,ORANGE)]),
    ('YOLO-World', [('sat',0.0719,BLUE),('oh',0.1058,GREEN),('TDM',0.1115,ORANGE)]),
    ('OWLv2', [('sat',0.4174,BLUE),('oh',0.4141,GREEN),('TDM(cal)',0.4070,ORANGE)]),
]
for ax, (name, data) in zip(axes.flat, models):
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [d[2] for d in data]
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('F1')
    ax.set_ylim(0, max(values)*1.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}',
                ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(r'C:\Users\张三\Desktop\fig5_cross_model.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 5 done.')

# ═══════════════════════════════════════
# Figure 7: Modifier Count Ablation (Line chart)
# ═══════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
x = [0,1,2,3,4,5]
gdino = [0.1375,0.1203,0.0977,0.0995,0.0896,0.0654]
yolo  = [0.0843,0.1104,0.1192,0.1125,0.1113,0.1074]
owl   = [0.3258,0.2171,0.2203,0.2203,0.2203,0.2203]
ax.plot(x, gdino, 'o-', color=BLUE, label='GDINO-tiny', linewidth=2, markersize=6)
ax.plot(x, yolo, 's-', color=ORANGE, label='YOLO-World', linewidth=2, markersize=6)
ax.plot(x, owl, 'D-', color=GREEN, label='OWLv2', linewidth=2, markersize=6)
# Mark optima
ax.annotate('optimal', xy=(0,0.1375), xytext=(1,0.16), arrowprops=dict(arrowstyle='->',color=BLUE), fontsize=9, color=BLUE)
ax.annotate('optimal', xy=(2,0.1192), xytext=(3,0.14), arrowprops=dict(arrowstyle='->',color=ORANGE), fontsize=9, color=ORANGE)
ax.set_xlabel('Number of Modifiers')
ax.set_ylabel('F1')
ax.legend(loc='upper right')
ax.set_xticks(x)
plt.tight_layout()
plt.savefig(r'C:\Users\张三\Desktop\fig7_ablation.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 7 done.')

# ═══════════════════════════════════════
# Figure 8: Few-shot Comparison
# ═══════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
methods = ['Random','1-shot','3-shot','5-shot','10-shot','TDM\n(zero-shot)','Oracle']
f1s = [0.1319, 0.1226, 0.1591, 0.2033, 0.2137, 0.2096, 0.2586]
errors = [0.0277, 0.0406, 0.0145, 0.0123, 0.0000, 0, 0]
colors = [GRAY, '#A0C0E0', '#6090C0', BLUE, BLUE, ORANGE, GREEN]
bars = ax.bar(methods, f1s, color=colors, edgecolor='white', linewidth=0.5, yerr=errors, capsize=4)
ax.axhline(y=0.2096, color=ORANGE, linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(5.5, 0.212, 'TDM zero-shot level', fontsize=8, color=ORANGE)
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008, f'{val:.4f}', ha='center', fontsize=8)
ax.set_ylabel('Detection F1')
plt.tight_layout()
plt.savefig(r'C:\Users\张三\Desktop\fig8_fewshot.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 8 done.')

# ═══════════════════════════════════════
# Figure 6: OFAT Heatmap (from existing data)
# ═══════════════════════════════════════
heatmap_data = load('ofat_heatmap_f1.json')
hmap_dict = heatmap_data['heatmap']  # {axis_name: {class_name: value}}
classes = heatmap_data['classes']  # 20 classes
axes_names = heatmap_data['axes']  # 7 axes
# Build 2D array: rows=classes, cols=axes
hmap = np.zeros((len(classes), len(axes_names)))
for j, ax_name in enumerate(axes_names):
    if ax_name in hmap_dict:
        for i, cls_name in enumerate(classes):
            hmap[i, j] = hmap_dict[ax_name].get(cls_name, 0)

fig, ax = plt.subplots(figsize=(10, 8))
vmax = max(abs(hmap.min()), abs(hmap.max()), 0.01)
im = ax.imshow(hmap, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(axes_names)))
ax.set_xticklabels(axes_names, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes, fontsize=8)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('F1 change (vs overhead baseline)', fontsize=9)
plt.tight_layout()
plt.savefig(r'C:\Users\张三\Desktop\fig6_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('Fig 6 done.')

print('All figures saved to Desktop.')
