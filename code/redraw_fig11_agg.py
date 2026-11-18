"""Redraw Fig.11 (paper fig3.png): add the aggregate min_c residual curve (the
selection curve the round-trip rule reads, Table XII bottom row) to panel (b),
and the mean target-domain AUC curve (the one the text quotes as peaking at
beta=2.0) to panel (a). Keeps everything else identical to redraw_figure3_fix.py.
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\scipilot-figure-skill\scripts')

import matplotlib.pyplot as plt
import numpy as np
from setup_style import setup_style
from export_figure import export_figure

setup_style(journal='nature', lang='en')

beta = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])

auc_data = {
    'airplane':   [0.4017, 0.4072, 0.443, 0.6356, 0.6916, 0.5242],
    'boat':       [0.4257, 0.4358, 0.4722, 0.598, 0.5352, 0.4828],
    'car':        [0.6222, 0.6447, 0.696, 0.8172, 0.5458, 0.526],
    'truck':      [0.3736, 0.3933, 0.4421, 0.6068, 0.5231, 0.5057],
    'bus':        [0.2568, 0.3029, 0.4485, 0.8488, 0.5659, 0.4473],
    'motorcycle': [0.3165, 0.3537, 0.4674, 0.7908, 0.4944, 0.4498],
}

residual_data = {
    'airplane':   [0.0, 0.0, 0.0012, 0.0636, 0.5826, 0.8831],
    'boat':       [0.0, 0.0001, 0.005, 0.1723, 0.7516, 0.9507],
    'car':        [0.0, 0.0001, 0.0057, 0.1859, 0.7684, 0.9619],
    'truck':      [0.0, 0.0002, 0.0081, 0.2373, 0.7956, 0.9599],
    'bus':        [0.0, 0.0002, 0.0088, 0.2503, 0.8016, 0.9597],
    'motorcycle': [0.0, 0.0002, 0.0118, 0.2959, 0.8269, 0.9687],
}

colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
markers = ['o', 's', '^', 'D', 'v', 'p']
categories = list(auc_data.keys())

A = np.array([auc_data[c] for c in categories])          # (6 classes, 6 betas)
mean_auc = A.mean(axis=0)
agg_res = np.array([residual_data[c] for c in categories]).min(axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.6))

# Panel (a): per-class AUC + mean curve
for i, cat in enumerate(categories):
    ax1.plot(beta, auc_data[cat], color=colors[i], marker=markers[i],
             markersize=5.5, linewidth=1.6, label=cat, markeredgecolor='white',
             markeredgewidth=0.5, alpha=0.85)
ax1.plot(beta, mean_auc, color='black', marker='X', markersize=9, linewidth=3.0,
         linestyle='--', label=r'mean AUC (6 classes)', zorder=10)
ax1.axhline(y=0.5, color='#888888', linestyle='--', linewidth=1, alpha=0.6)
ax1.axvline(x=2.0, color='#009E73', linestyle=':', linewidth=1.5, alpha=0.8)
ax1.axvline(x=4.0, color='#D55E00', linestyle=':', linewidth=1.5, alpha=0.8)
ax1.set_xlabel(r'$\beta$ (extrapolation step)', fontsize=11)
ax1.set_ylabel('Extrapolated Prototype Ranking AUC', fontsize=11)
ax1.set_title('(a) Target Domain AUC vs. $\\beta$', fontsize=12, fontweight='bold')
ax1.set_xscale('log')
ax1.set_xticks(beta)
ax1.set_xticklabels(['0.25', '0.5', '1', '2', '4', '8'])
ax1.tick_params(axis='x', which='minor', labelbottom=False)
ax1.set_ylim(0.25, 0.9)
ax1.legend(fontsize=7.5, ncol=2, loc='lower right')
ax1.tick_params(axis='both', labelsize=10)
# annotate the mean peak
ax1.annotate(r'mean max at $\beta{=}2$', xy=(2.0, mean_auc[3]), xytext=(2.6, 0.86),
             fontsize=8, color='black',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

# Panel (b): per-class residual + aggregate min_c selection curve
for i, cat in enumerate(categories):
    ax2.plot(beta, residual_data[cat], color=colors[i], marker=markers[i],
             markersize=5.5, linewidth=1.6, label=cat, markeredgecolor='white',
             markeredgewidth=0.5, alpha=0.85)
ax2.plot(beta, agg_res, color='black', marker='X', markersize=9, linewidth=3.0,
         linestyle='--',
         label=r'aggregate $\bar{\mathrm{Res}}=\min_c\,\mathrm{Res}_c$ (selection curve)',
         zorder=10)
ax2.axhline(y=0.15, color='#009E73', linestyle='--', linewidth=1.5, alpha=0.8,
            label='Round-trip threshold 0.15')
ax2.axvline(x=2.0, color='#009E73', linestyle=':', linewidth=1.5, alpha=0.8)
ax2.axvline(x=4.0, color='#D55E00', linestyle=':', linewidth=1.5, alpha=0.8)
ax2.set_xlabel(r'$\beta$ (extrapolation step)', fontsize=11)
ax2.set_ylabel('Round-trip Residual\n(source-domain, zero target data)', fontsize=10.5)
ax2.set_title('(b) Round-trip Residual vs. $\\beta$', fontsize=12, fontweight='bold')
ax2.set_xscale('log')
ax2.set_xticks(beta)
ax2.set_xticklabels(['0.25', '0.5', '1', '2', '4', '8'])
ax2.tick_params(axis='x', which='minor', labelbottom=False)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=7.5, loc='upper left')
ax2.tick_params(axis='both', labelsize=10)

fig.suptitle(r'$\beta$ Collapse Aligns with the Source-domain Autonomous Round-trip Residual Inflection',
             fontsize=11.5, fontweight='bold', y=1.02)

plt.tight_layout()
export_figure(
    fig,
    basename=r'C:\Users\张三\Desktop\A-CSPC_IEEE_style\figures\fig3',
    formats=['png'],
    size_inches=(7.2, 3.6),
    dpi=300,
    grayscale_preview=True,
)
print('Fig.11 (fig3.png) regenerated with aggregate min_c curve and mean AUC.')
print('mean AUC per beta:', np.round(mean_auc, 4).tolist())
print('agg min_c residual:', np.round(agg_res, 4).tolist())
