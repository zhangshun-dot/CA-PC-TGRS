"""
Batch redraw remaining figures: Figure 4, 6, 7, 8
Using scipilot-figure-skill conventions
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\scipilot-figure-skill\scripts')

import matplotlib.pyplot as plt
import numpy as np
from setup_style import setup_style
from export_figure import export_figure

setup_style(journal='nature', lang='en')

output_dir = r'C:\Users\张三\Desktop\实验代码\figures_redrawn'

# ============================================================
# FIGURE 4: Detector Spectrum Scatter Plot
# ============================================================
def draw_figure4():
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))

    # GDINO-tiny data (blue)
    gdino_cats = ['car', 'boat', 'truck', 'motorcycle', 'bus', 'airplane']
    gdino_baseline = [0.09, 0.18, 0.27, 0.45, 0.58, 0.75]
    gdino_gain = [0.09, 0.04, 0.072, 0.124, 0.059, 0.012]

    # OWLv2 data (orange)
    owl_cats = ['boat', 'car', 'truck', 'motorcycle', 'bus', 'airplane']
    owl_baseline = [0.55, 0.60, 0.58, 0.78, 0.90, 0.90]
    owl_gain = [-0.024, -0.022, -0.056, -0.060, -0.035, 0.001]

    # Plot GDINO-tiny
    ax.scatter(gdino_baseline, gdino_gain, color='#0072B2', s=120, zorder=5,
               edgecolor='white', linewidth=1.5, label='GDINO-tiny (N=100)')
    for i, cat in enumerate(gdino_cats):
        ax.annotate(cat, (gdino_baseline[i], gdino_gain[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8, color='#0072B2')

    # Plot OWLv2
    ax.scatter(owl_baseline, owl_gain, color='#D55E00', s=120, zorder=5,
               edgecolor='white', linewidth=1.5, label='OWLv2 (N=100)')
    for i, cat in enumerate(owl_cats):
        ax.annotate(cat, (owl_baseline[i], owl_gain[i]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8, color='#D55E00')

    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1)
    ax.set_xlabel('Detector Baseline AP50 (weaker → left)', fontsize=10)
    ax.set_ylabel('A-CSPC Per-class Gain (AP50)', fontsize=10)
    ax.set_title('Fig. 4  Gain vs. Detector Baseline and Proposal Pool Saturation:\nWeak Calibration Yields Gain',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.07, 0.14)

    plt.tight_layout()
    export_figure(fig, basename=f'{output_dir}/figure4_detector_spectrum',
                  formats=['pdf', 'svg', 'png'], size_inches=(5.5, 4.0), dpi=300, grayscale_preview=True)
    plt.close()
    print("Figure 4 redrawn!")

# ============================================================
# FIGURE 6: Physical Prior Factor Ablation
# ============================================================
def draw_figure6():
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))

    factors = ['Full\n(V+M+G+S+I+C)', 'w/o V', 'w/o M', 'w/o G', 'w/o S', 'w/o I', 'w/o C']
    rho_values = [0.623, 0.688, 0.699, 0.506, 0.581, 0.601, 0.579]
    colors = ['#3366CC'] * 7
    colors[3] = '#CC3311'  # Highlight w/o G as core factor

    bars = ax.bar(factors, rho_values, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, rho_values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'+{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add annotation for core factor
    ax.annotate('Δ=-0.117\n(core factor)', xy=(3, 0.506), xytext=(4.2, 0.42),
                fontsize=9, color='#CC3311', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#CC3311', lw=1.5))

    ax.set_ylabel('Spearman ρ (37 classes, 6 domain pairs)', fontsize=10)
    ax.set_title('Fig. 6  Factor Ablation: Contribution of Each Physical Prior Factor',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0.45, 0.75)
    ax.tick_params(axis='x', labelsize=8.5)

    plt.tight_layout()
    export_figure(fig, basename=f'{output_dir}/figure6_factor_ablation',
                  formats=['pdf', 'svg', 'png'], size_inches=(6.5, 4.0), dpi=300, grayscale_preview=True)
    plt.close()
    print("Figure 6 redrawn!")

# ============================================================
# FIGURE 7: Fine-grained Prototype Gain vs. Gamma
# ============================================================
def draw_figure7():
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    gamma = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])

    # Data from original figure
    car_gain =       [0.080, 0.087, 0.098, 0.108, 0.120, 0.135, 0.107]
    truck_gain =     [0.078, 0.079, 0.079, 0.067, 0.039, -0.025, -0.032]
    bus_gain =       [0.073, 0.082, 0.085, 0.081, 0.069, 0.039, -0.015]
    motorcycle_gain = [0.118, 0.109, 0.093, 0.061, 0.014, -0.085, -0.120]

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    markers = ['o', 's', '^', 'D']
    labels = ['car (oracle γ=1.5)', 'truck (oracle γ=0.25)',
              'bus (oracle γ=0.5)', 'motorcycle (oracle γ=0.0)']
    data_list = [car_gain, truck_gain, bus_gain, motorcycle_gain]

    for i, (data, color, marker, label) in enumerate(zip(data_list, colors, markers, labels)):
        ax.plot(gamma, data, color=color, marker=marker, markersize=8, linewidth=2,
                label=label, markeredgecolor='white', markeredgewidth=1)

    # Mark realized gamma positions (stars)
    realized_gamma = {'car': 1.5, 'truck': 0.5, 'bus': 0.5, 'motorcycle': 0.5}
    realized_gain = {'car': 0.135, 'truck': 0.079, 'bus': 0.085, 'motorcycle': 0.093}
    for cat, color in zip(['car', 'truck', 'bus', 'motorcycle'], colors):
        ax.scatter(realized_gamma[cat], realized_gain[cat], color=color, marker='*',
                   s=300, zorder=10, edgecolor='white', linewidth=1.5)

    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1)
    ax.set_xlabel('γ (fine prototype offset strength)', fontsize=11)
    ax.set_ylabel('Gain (AP50)', fontsize=11)
    ax.set_title('Fig. 7  Fine-grained Source Prototype: Gain vs. γ (DIOR→COCO)\n★ = realized γ (text-similarity × domain saturation)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.14, 0.15)

    plt.tight_layout()
    export_figure(fig, basename=f'{output_dir}/figure7_gamma_gain',
                  formats=['pdf', 'svg', 'png'], size_inches=(6.5, 4.5), dpi=300, grayscale_preview=True)
    plt.close()
    print("Figure 7 redrawn!")

# ============================================================
# FIGURE 8: Realized vs. Oracle Gamma Per-class Gain
# ============================================================
def draw_figure8():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.8))

    categories_coco = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
    categories_voc = ['aeroplane', 'boat', 'car', 'bus', 'motorbike']

    # DIOR→COCO data
    acspc_coco =   [0.012, 0.025, 0.081, 0.077, 0.074, 0.119]
    oracle_coco =  [0.012, 0.025, 0.136, 0.079, 0.085, 0.119]
    realized_coco =[0.012, 0.025, 0.136, 0.079, 0.085, 0.119]

    # DIOR→VOC data
    acspc_voc =   [0.002, -0.022, 0.086, 0.066, -0.061]
    oracle_voc =  [0.002, -0.022, 0.088, 0.069, -0.061]
    realized_voc =[0.002, -0.022, 0.088, 0.066, -0.061]

    x_coco = np.arange(len(categories_coco))
    x_voc = np.arange(len(categories_voc))
    width = 0.25

    # Panel (a): DIOR→COCO
    ax1.bar(x_coco - width, acspc_coco, width, label='A-CSPC (γ=0)', color='#888888', edgecolor='white')
    ax1.bar(x_coco, oracle_coco, width, label='Oracle γ', color='#56B4E9', edgecolor='white')
    ax1.bar(x_coco + width, realized_coco, width, label='Realized γ', color='#009E73', edgecolor='white')
    ax1.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8)
    ax1.set_ylabel('Gain (AP50)', fontsize=10)
    ax1.set_title('(a) DIOR→COCO', fontsize=11, fontweight='bold')
    ax1.set_xticks(x_coco)
    ax1.set_xticklabels(categories_coco, rotation=30, ha='right', fontsize=8.5)
    ax1.set_ylim(0, 0.145)
    ax1.legend(fontsize=8, loc='upper left')

    # Panel (b): DIOR→VOC
    ax2.bar(x_voc - width, acspc_voc, width, label='A-CSPC (γ=0)', color='#888888', edgecolor='white')
    ax2.bar(x_voc, oracle_voc, width, label='Oracle γ', color='#56B4E9', edgecolor='white')
    ax2.bar(x_voc + width, realized_voc, width, label='Realized γ', color='#009E73', edgecolor='white')
    ax2.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8)
    ax2.set_title('(b) DIOR→VOC', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_voc)
    ax2.set_xticklabels(categories_voc, rotation=30, ha='right', fontsize=8.5)
    ax2.set_ylim(-0.07, 0.10)
    ax2.legend(fontsize=8, loc='upper left')

    fig.suptitle('Fig. 8  Realized γ vs. Oracle γ vs. A-CSPC: Per-class Gain',
                 fontsize=12, fontweight='bold', y=1.02)

    plt.tight_layout()
    export_figure(fig, basename=f'{output_dir}/figure8_realized_oracle',
                  formats=['pdf', 'svg', 'png'], size_inches=(7.2, 3.8), dpi=300, grayscale_preview=True)
    plt.close()
    print("Figure 8 redrawn!")

# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    draw_figure4()
    draw_figure6()
    draw_figure7()
    draw_figure8()
    print("\nAll remaining figures redrawn successfully!")
