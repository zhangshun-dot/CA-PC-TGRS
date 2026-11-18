"""
Redraw Figure 3: Beta Collapse Point Alignment (AUC + Round-trip Residual)
Using scipilot-figure-skill conventions
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\scipilot-figure-skill\scripts')

import matplotlib.pyplot as plt
import numpy as np
from setup_style import setup_style

setup_style(journal='nature', lang='en')

# Beta values
beta = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])

# Panel (a): Target domain AUC vs beta (joint variant, exact from tge_probe_12.json)
# 6 categories: airplane, boat, car, truck, bus, motorcycle
auc_data = {
    'airplane':   [0.4017, 0.4072, 0.443, 0.6356, 0.6916, 0.5242],
    'boat':       [0.4257, 0.4358, 0.4722, 0.598, 0.5352, 0.4828],
    'car':        [0.6222, 0.6447, 0.696, 0.8172, 0.5458, 0.526],
    'truck':      [0.3736, 0.3933, 0.4421, 0.6068, 0.5231, 0.5057],
    'bus':        [0.2568, 0.3029, 0.4485, 0.8488, 0.5659, 0.4473],
    'motorcycle': [0.3165, 0.3537, 0.4674, 0.7908, 0.4944, 0.4498],
}

# Panel (b): Round-trip residual vs beta (source domain autonomous, joint variant,
# exact from tge_probe_12.json). Previously airplane was mis-entered as 0.30/0.83;
# true values are 0.0636/0.5826, matching the text "spread 0.06 to 0.30, mean 0.20".
residual_data = {
    'airplane':   [0.0, 0.0, 0.0012, 0.0636, 0.5826, 0.8831],
    'boat':       [0.0, 0.0001, 0.005, 0.1723, 0.7516, 0.9507],
    'car':        [0.0, 0.0001, 0.0057, 0.1859, 0.7684, 0.9619],
    'truck':      [0.0, 0.0002, 0.0081, 0.2373, 0.7956, 0.9599],
    'bus':        [0.0, 0.0002, 0.0088, 0.2503, 0.8016, 0.9597],
    'motorcycle': [0.0, 0.0002, 0.0118, 0.2959, 0.8269, 0.9687],
}

# Colorblind-safe palette with distinct markers
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
markers = ['o', 's', '^', 'D', 'v', 'p']
categories = list(auc_data.keys())

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

# Panel (a): AUC vs beta
for i, cat in enumerate(categories):
    ax1.plot(beta, auc_data[cat], color=colors[i], marker=markers[i],
             markersize=5, linewidth=1.5, label=cat, markeredgecolor='white', markeredgewidth=0.5)

ax1.axhline(y=0.5, color='#888888', linestyle='--', linewidth=1, alpha=0.7)
ax1.axvline(x=2.0, color='#009E73', linestyle=':', linewidth=1.5, alpha=0.8, label='β=2 (safe)')
ax1.axvline(x=4.0, color='#D55E00', linestyle=':', linewidth=1.5, alpha=0.8, label='β=4 (collapse)')
ax1.set_xlabel('β (extrapolation step)', fontsize=9)
ax1.set_ylabel('Extrapolated Prototype Ranking AUC', fontsize=9)
ax1.set_title('(a) Target Domain AUC vs. β', fontsize=10, fontweight='bold')
ax1.set_xscale('log')
ax1.set_xticks(beta)
ax1.set_xticklabels([str(b) for b in beta])
ax1.set_ylim(0.25, 0.9)
ax1.legend(fontsize=6.5, ncol=2, loc='lower right')

# Panel (b): Round-trip residual vs beta
for i, cat in enumerate(categories):
    ax2.plot(beta, residual_data[cat], color=colors[i], marker=markers[i],
             markersize=5, linewidth=1.5, label=cat, markeredgecolor='white', markeredgewidth=0.5)

ax2.axhline(y=0.15, color='#009E73', linestyle='--', linewidth=1.5, alpha=0.8, label='Round-trip threshold 0.15')
ax2.axvline(x=2.0, color='#009E73', linestyle=':', linewidth=1.5, alpha=0.8)
ax2.axvline(x=4.0, color='#D55E00', linestyle=':', linewidth=1.5, alpha=0.8)
ax2.set_xlabel('β (extrapolation step)', fontsize=9)
ax2.set_ylabel('Round-trip Residual\n(source domain autonomous, zero target data)', fontsize=9)
ax2.set_title('(b) Round-trip Residual vs. β', fontsize=10, fontweight='bold')
ax2.set_xscale('log')
ax2.set_xticks(beta)
ax2.set_xticklabels([str(b) for b in beta])
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=6.5, loc='lower right')

fig.suptitle('Fig. 3  β Collapse Point Aligns with Source-Domain Autonomous Round-trip Residual Inflection (β=2 safe, β=4 collapse)',
             fontsize=11, fontweight='bold', y=1.02)

plt.tight_layout()

from export_figure import export_figure
export_figure(
    fig,
    basename=r'C:\Users\张三\Desktop\A-CSPC_IEEE_style\figures\fig3',
    formats=['png'],
    size_inches=(7.2, 3.2),
    dpi=300,
    grayscale_preview=True
)

print("Figure 3 redrawn successfully!")
