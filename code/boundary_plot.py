# -*- coding: utf-8 -*-
"""boundary_plot.py — 边界图 (论文 Figure 1 候选): 失效可测 ⇔ 测试时可救

横轴: 各类源-目标分离度 AUC (route_proxy_probe.json, GT-oracle, 0.5=随机 1=完美)
纵轴: 相对 baseline 的 ΔAP50 (gain_decomposition.json, 与主表逐位对账)

两个系列:
  zero-shot (CDVM_zs − baseline)   — 纯源域原型 + 负原型
  few-shot  (CDVM_fs − baseline)   — 测试时校准 (α=0.5, K=10)

预期叙事:
  * AUC < ~0.7 (boat 0.503, airplane 0.635): zero-shot 为负 (越帮越忙)
  * AUC > ~0.75 (car 0.771, DIOR 各 0.78-0.91): zero-shot 为正
  * few-shot 在所有类都 ≥0, 且把失败区两类从负翻正 → "校准把边界往下推"

输出: 实验数据/fig_boundary.png + 控制台表格
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
OUT_PNG = DATA / 'fig_boundary.png'

# ── 数据源 ──
# 优先扩类版(12 类), 否则退回 6 类版
gain_path = DATA / 'final_results_12class.json'
if not gain_path.exists():
    gain_path = DATA / 'gain_decomposition.json'
    print('[boundary_plot] 未找到 final_results_12class.json，退回 6 类 gain_decomposition.json')
gain = json.load(open(gain_path, encoding='utf-8'))
# AUC 数据源: 优先 12 类全量(N=500), 否则退回旧 6 类(N=100)
auc_path = DATA / 'route_proxy_probe_12.json'
if not auc_path.exists():
    auc_path = DATA / 'route_proxy_probe.json'
    print('[boundary_plot] 未找到 route_proxy_probe_12.json，退回旧 6 类 AUC')
aucs = json.load(open(auc_path, encoding='utf-8'))

# 类键规范化为 'COCO-x' / 'DIOR-x'
def norm_key(k):
    return k.replace('storage tank', 'storage-tank')

gain_pc = {norm_key(k): v for k, v in gain['per_class'].items()}

# 12 类 (跨域 COCO 6 + 同域 DIOR 6)
CLASSES = ['COCO-airplane', 'COCO-boat', 'COCO-car',
           'COCO-truck', 'COCO-bus', 'COCO-motorcycle',
           'DIOR-airplane', 'DIOR-ship', 'DIOR-vehicle',
           'DIOR-harbor', 'DIOR-storage-tank', 'DIOR-bridge']
SHORT = {'COCO-airplane': 'airplane', 'COCO-boat': 'boat', 'COCO-car': 'car',
         'COCO-truck': 'truck', 'COCO-bus': 'bus', 'COCO-motorcycle': 'motorcycle',
         'DIOR-airplane': 'd-air', 'DIOR-ship': 'd-ship', 'DIOR-vehicle': 'd-veh',
         'DIOR-harbor': 'd-harbor', 'DIOR-storage-tank': 'd-tank', 'DIOR-bridge': 'd-bridge'}
DOMAIN = {c: ('cross' if c.startswith('COCO') else 'same') for c in CLASSES}

rows = []
for c in CLASSES:
    pc = gain_pc.get(c)
    if pc is None:
        print(f'  !! 缺 {c} 数据，跳过')
        continue
    base = pc['baseline']
    zs = pc['zero']['cdvm']
    fs = pc['few']['cdvm']
    rows.append({
        'cls': c, 'short': SHORT[c], 'domain': DOMAIN[c],
        'auc': aucs[c]['auc'],
        'd_zs': zs - base, 'd_fs': fs - base,
    })

# 表格打印
print('类(域)         AUC    Δzero   Δfew    零样本状态')
print('-' * 58)
for r in rows:
    tag = '❌负' if r['d_zs'] < 0 else ('≈0' if r['d_zs'] < 0.01 else '✅正')
    print(f"{r['short']:<14s} {r['auc']:.3f}  {r['d_zs']:+7.4f} {r['d_fs']:+7.4f}   {tag}")

# ── 相关性检验 (12 点, 边界图的统计支撑) ──
import scipy.stats as st
if len(rows) >= 6:
    aucs_x = np.array([r['auc'] for r in rows])
    zs_y = np.array([r['d_zs'] for r in rows])
    fs_y = np.array([r['d_fs'] for r in rows])
    print('\nSpearman 相关 (AUC vs ΔAP50):')
    for tag, y in [('zero-shot', zs_y), ('few-shot', fs_y)]:
        rho, p = st.spearmanr(aucs_x, y)
        print(f'  {tag:<10s} ρ={rho:+.3f}  p={p:.4f}  ({"显著" if p < 0.05 else "不显著"})')

# ── 画图 ──
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)

for r in rows:
    marker = 'o' if r['domain'] == 'cross' else 's'
    color = '#1f77b4' if r['domain'] == 'cross' else '#7f7f7f'
    ax.plot(r['auc'], r['d_zs'], marker, color=color, ms=9, mec='k', mew=0.8,
            zorder=3, label='zero-shot' if r['short'] == 'airplane' else None)
    ax.plot(r['auc'], r['d_fs'], '^', color=color, ms=9, mec='k', mew=0.8,
            zorder=4, label='few-shot' if r['short'] == 'airplane' else None)
    ax.plot([r['auc'], r['auc']], [r['d_zs'], r['d_fs']], color=color, lw=1.0, ls=':', zorder=2)

    label = r['short']
    ax.annotate(label, (r['auc'], r['d_zs']), textcoords='offset points',
                xytext=(8, 6), fontsize=8, color=color)
    ax.annotate(label, (r['auc'], r['d_fs']), textcoords='offset points',
                xytext=(8, -10), fontsize=8, color=color, fontstyle='italic')

# 零线 + 经验边界 (zero-shot 由负转正的 AUC 阈值)
ax.axhline(0.0, color='k', lw=1.2)
ax.axvspan(0.5, 0.70, color='#ffd9d9', alpha=0.5, zorder=1)
ax.axvline(0.70, color='#c00000', lw=1.4, ls='--')
ax.text(0.60, 0.005, 'zero-shot 失效区\n(AUC < 0.70)', ha='center', va='bottom',
        fontsize=9, color='#c00000')
ax.text(0.78, 0.005, 'few-shot 把边界\n向下推', ha='center', va='bottom',
        fontsize=9, color='#2a7f2a')

ax.set_xlabel('source–target separation AUC (oracle)', fontsize=11)
ax.set_ylabel('ΔAP50 vs baseline', fontsize=11)
ax.set_title('Cross-domain OVD: failure is measurable, and measurable is fixable',
             fontsize=12)
ax.set_xlim(0.48, 0.93)
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f'\n✅ 图已保存 -> {OUT_PNG}')

