"""
Redraw Figure 2: Layered Failure Diagnosis (AUC + Recall)
Using scipilot-figure-skill conventions
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\scipilot-figure-skill\scripts')

import matplotlib.pyplot as plt
import numpy as np
from setup_style import setup_style

# Setup style for IEEE/Nature journal, English
setup_style(journal='nature', lang='en')

# Data from original figure
categories = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
auc_values = [0.635, 0.526, 0.783, 0.565, 0.200, 0.292]
recall_values = [0.97, 0.41, 0.58, 0.89, 0.71, 0.85]

# Colorblind-safe palette
colors_auc = ['#3366CC' if v >= 0.5 else '#CC3311' for v in auc_values]
colors_recall = ['#009988' if v >= 0.7 else '#EE7733' for v in recall_values]

# Create figure with 2 panels (Nature double-column width: 7.2 in)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

# Panel (a): AUC
x = np.arange(len(categories))
bars1 = ax1.bar(x, auc_values, color=colors_auc, width=0.6, edgecolor='white', linewidth=0.5)
ax1.axhline(y=0.5, color='#888888', linestyle='--', linewidth=1, label='Random (0.5)')
ax1.set_ylabel('Ranking AUC (within proposal pool)', fontsize=9)
ax1.set_title('(a) Source Prototype Failure Diagnosis (AUC)', fontsize=10, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=30, ha='right', fontsize=8)
ax1.set_ylim(0, 0.9)
ax1.legend(fontsize=7, loc='upper right')

# Add value labels
for bar, val in zip(bars1, auc_values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.015,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

# Panel (b): Recall
bars2 = ax2.bar(x, recall_values, color=colors_recall, width=0.6, edgecolor='white', linewidth=0.5)
ax2.set_ylabel('Proposal GT Coverage Rate (N=500)', fontsize=9)
ax2.set_title('(b) Proposal Pool Coverage Stratification (Recall)', fontsize=10, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, rotation=30, ha='right', fontsize=8)
ax2.set_ylim(0, 1.05)

# Add value labels
for bar, val in zip(bars2, recall_values):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)

# Overall title
fig.suptitle('Fig. 2  Layered Failure Diagnosis: Ranking AUC and Proposal Pool Recall Reported Separately',
             fontsize=11, fontweight='bold', y=1.02)

# Adjust layout
plt.tight_layout()

# Export using scipilot convention
from export_figure import export_figure
export_figure(
    fig,
    basename=r'C:\Users\张三\Desktop\实验代码\figures_redrawn\figure2_failure_diagnosis',
    formats=['pdf', 'svg', 'png'],
    size_inches=(7.2, 3.2),
    dpi=300,
    grayscale_preview=True
)

print("Figure 2 redrawn successfully!")
print(f"Output: figures_redrawn/figure2_failure_diagnosis.{pdf,svg,png}")
