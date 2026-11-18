"""
Figure 5: Direction Gate g₂ Strategy Selection (REAL DATA)
Data from tge_probe_gate.json: 24 scene classes
Left: Full extrapolation / Full B2 / Gate three-strategy mean comparison
Right: Per-class extrapolation - B2 gain, blue=gate correct, red=misclassified
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os

sys.path.insert(0, r"C:\Users\张三\Desktop\scipilot-figure-skill\scripts")
from setup_style import setup_style
from export_figure import export_figure

setup_style(journal="nature", lang="zh", use_sciplots=True)
mpl.rcParams['font.size'] = 9

COLORS = {
    'blue': '#0072B2', 'orange': '#D55E00', 'green': '#009E73',
    'red': '#CC79A7', 'gray': '#999999', 'dark_gray': '#555555',
}

# Load real data
with open(r'C:\Users\张三\Desktop\实验数据\tge_probe_gate.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rows = data['rows']
n = len(rows)
print(f"Loaded {n} scene classes")

# Extract data
class_names = [r['name'] for r in rows]
gains = np.array([r['gain'] for r in rows])  # ext - b2
g2_scores = np.array([r['g2'] for r in rows])
b2_aps = np.array([r['b2'] for r in rows])
ext_aps = np.array([r['ext'] for r in rows])
base_aps = np.array([r['base'] for r in rows])

# Gate decisions: g2 > 0 -> extrap, else B2
gate_extrap = g2_scores > 0
truth_extrap = gains > 0  # extrap better than B2
correct = (gate_extrap == truth_extrap)

print(f"Gate extrapolate: {gate_extrap.sum()}/{n}")
print(f"Truth extrapolate: {truth_extrap.sum()}/{n}")
print(f"Correct: {correct.sum()}/{n} = {correct.sum()/n:.3f}")
print(f"Rule: {data['rule']}")
print(f"Reported trigger_acc: {data['trigger_acc']}, hits: {data['hits']}, n: {data['n']}")

# Three-strategy mean AP50 (relative to baseline)
full_extrap_gain = np.mean(ext_aps - base_aps)
full_b2_gain = np.mean(b2_aps - base_aps)
gate_gain = np.mean(np.where(gate_extrap, ext_aps, b2_aps) - base_aps)

# Or use gain (ext - b2) directly for the right panel
# Left panel: mean AP50 (absolute)
full_extrap_mean = np.mean(ext_aps)
full_b2_mean = np.mean(b2_aps)
gate_mean = np.mean(np.where(gate_extrap, ext_aps, b2_aps))

# Use gain relative to baseline
full_extrap_gain = full_extrap_mean - np.mean(base_aps)
full_b2_gain = full_b2_mean - np.mean(base_aps)
gate_gain = gate_mean - np.mean(base_aps)

# Standard errors
full_extrap_se = np.std(ext_aps - base_aps) / np.sqrt(n)
full_b2_se = np.std(b2_aps - base_aps) / np.sqrt(n)
gate_values = np.where(gate_extrap, ext_aps, b2_aps) - base_aps
gate_se = np.std(gate_values) / np.sqrt(n)

print(f"\nThree-strategy mean gain (vs baseline):")
print(f"  Full Extrapolation: {full_extrap_gain:.4f} ± {full_extrap_se:.4f}")
print(f"  Full B2:            {full_b2_gain:.4f} ± {full_b2_se:.4f}")
print(f"  Gate (g2):          {gate_gain:.4f} ± {gate_se:.4f}")

# Sort by gain for better visualization
sort_idx = np.argsort(gains)[::-1]  # descending
class_names_sorted = [class_names[i] for i in sort_idx]
gains_sorted = gains[sort_idx]
correct_sorted = correct[sort_idx]
gate_extrap_sorted = gate_extrap[sort_idx]

bar_colors = [COLORS['blue'] if c else COLORS['red'] for c in correct_sorted]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left panel: Three-strategy comparison ---
strategies = ['Full\nExtrapolation', 'Full B2\n(Conservative)', 'Gate\n(g2)']
strategy_gains = [full_extrap_gain, full_b2_gain, gate_gain]
strategy_errors = [full_extrap_se, full_b2_se, gate_se]

x = np.arange(len(strategies))
ax1.bar(x, strategy_gains, yerr=strategy_errors,
        color=[COLORS['orange'], COLORS['gray'], COLORS['blue']],
        edgecolor='white', linewidth=0.8, capsize=3, width=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=9)
ax1.set_ylabel('Mean AP50 Gain (vs baseline)', fontsize=10)
ax1.set_title(f'(a) Three-Strategy Comparison ({n} scene classes)', fontsize=11, fontweight='bold')
ax1.axhline(y=0, color=COLORS['dark_gray'], linewidth=0.8)

for i, (v, e) in enumerate(zip(strategy_gains, strategy_errors)):
    label = f'+{v:.4f}' if v >= 0 else f'{v:.4f}'
    ax1.text(i, v + e + 0.002, label, ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Right panel: Per-class gain ---
x2 = np.arange(len(class_names_sorted))
ax2.bar(x2, gains_sorted, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.7)
ax2.set_xticks(x2)
ax2.set_xticklabels(class_names_sorted, fontsize=6.5, rotation=45, ha='right')
ax2.set_ylabel('Extrapolation - B2 (AP50)', fontsize=10)
ax2.set_title(f'(b) Per-Class Gain: Gate Decision Accuracy {correct.sum()}/{n}', fontsize=11, fontweight='bold')
ax2.axhline(y=0, color=COLORS['dark_gray'], linewidth=0.8)

from matplotlib.patches import Patch
n_correct = correct.sum()
n_wrong = n - n_correct
ax2.legend(handles=[
    Patch(facecolor=COLORS['blue'], edgecolor='white', label=f'Gate correct ({n_correct}/{n})'),
    Patch(facecolor=COLORS['red'], edgecolor='white', label=f'Misclassified ({n_wrong}/{n})')
], loc='upper right', fontsize=8, frameon=False)

# Mark misclassified
for i, m in enumerate(~correct_sorted):
    if m:
        ax2.annotate('x', (i, gains_sorted[i]), textcoords="offset points",
                    xytext=(0, 10 if gains_sorted[i] > 0 else -15), ha='center',
                    fontsize=10, color=COLORS['red'], fontweight='bold')

plt.tight_layout()

# Export
output_dir = r"C:\Users\张三\Desktop\实验代码\figures_redrawn"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "figure5_gate_strategy_real")
export_figure(fig, output_path, dpi=300, formats=['png', 'pdf', 'svg'], grayscale_preview=True)
print(f"\nFigure 5 (real data) exported to {output_path}")
plt.close()
