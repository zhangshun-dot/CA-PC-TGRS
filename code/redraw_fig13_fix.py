"""Regenerate Fig.13 (factor ablation) with corrected y-axis label and title."""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\scipilot-figure-skill\scripts')

import matplotlib.pyplot as plt
import numpy as np
from setup_style import setup_style
from export_figure import export_figure

setup_style(journal='nature', lang='en')

output_dir = r'C:\Users\张三\Desktop\实验代码\figures_redrawn'

fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))

factors = ['Full\n(V+M+G+S+I+C)', 'w/o V', 'w/o M', 'w/o G', 'w/o S', 'w/o I', 'w/o C']
rho_values = [0.623, 0.688, 0.699, 0.506, 0.581, 0.601, 0.579]
color_main = '#3366CC'
color_drop = '#2255AA'
colors = [color_main] * 7
colors[3] = color_drop  # slightly darker for the largest drop, but not red

bars = ax.bar(factors, rho_values, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

# value labels
for bar, val in zip(bars, rho_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'+{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='#111111')

# annotation for core factor, neutral dark colour, placed in open space
ax.annotate('Δ = −0.117\n(core factor)', xy=(3, 0.506), xytext=(5.35, 0.62),
            fontsize=9, color='#111111', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

ax.set_ylabel('Spearman ρ (37 class-domain-pair cells, 15 classes)', fontsize=10)
ax.set_title('Factor ablation of the development-time six-factor model',
             fontsize=11, fontweight='bold')
ax.set_ylim(0.45, 0.75)
ax.tick_params(axis='x', labelsize=8.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
export_figure(fig, basename=f'{output_dir}/figure6_factor_ablation_fixed',
              formats=['png'], size_inches=(6.5, 4.0), dpi=300, grayscale_preview=False)
plt.close()
print("Fig.13 regenerated with fixed labels.")
