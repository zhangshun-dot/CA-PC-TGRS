# -*- coding: utf-8 -*-
"""tge_probe_corr.py — 解拥挤相关性: 类间余弦变化 × AP 增益

把"解拥挤"从几何解释升级为相关性证据:
  每类: Δcos = 外推前(与其余类源原型均值余弦) − 外推后(与其余类外推原型均值余弦)
       ΔAP  = joint2 − baseline (N=500 口径)
  Spearman(Δcos, ΔAP) 显著 → 解拥挤与判别改善有统计关联

数据源 (无需重跑实验):
  tge_probe_entangle.json — hat_cos 6×6 (外推后类间余弦)
  route_proxy_probe_12.json — B1 (dinov2 AUC, 近似"外推前"判别)
  tge_probe_ap_st.json — baseline / B3_joint2 (AP 增益)
注: 外推前类间余弦用 DIOR 源原型 (3 类: airplane/ship/vehicle) 近似, 外推后 6 类矩阵已有。
输出: 控制台表 + 实验数据/tge_probe_corr.json
"""
import json
import sys
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
ENT = json.load(open(DATA / 'tge_probe_entangle.json', encoding='utf-8'))
AP = json.load(open(DATA / 'tge_probe_ap_st.json', encoding='utf-8'))['classes']
AUC = json.load(open(DATA / 'route_proxy_probe_12.json', encoding='utf-8'))

CLASSES = ['COCO-airplane', 'COCO-boat', 'COCO-car', 'COCO-truck', 'COCO-bus',
           'COCO-motorcycle']
SHORT = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']

# 外推后: 每类与其余 5 类均值余弦 (从 hat_cos)
hat_cos = ENT['hat_cos']
post_mean = {}
for i, c in enumerate(SHORT):
    vals = [hat_cos[f'{c}|{d}'] for d in SHORT if d != c]
    post_mean[c] = float(np.mean(vals))

# 外推前: 源原型类间余弦 (DIOR 3 类, src_cos) — 每类按源类取"与其余源类的均值"
src_cos = ENT['src_cos']
src_pairs = {'airplane': ['airplane|ship', 'airplane|vehicle'],
             'boat': ['ship|airplane', 'ship|vehicle'],
             'car': ['vehicle|airplane', 'vehicle|ship'],
             'truck': ['vehicle|airplane', 'vehicle|ship'],
             'bus': ['vehicle|airplane', 'vehicle|ship'],
             'motorcycle': ['vehicle|airplane', 'vehicle|ship']}
pre_mean = {c: float(np.mean([src_cos[k] for k in src_pairs[c]])) for c in SHORT}

rows = []
for i, c in enumerate(SHORT):
    name = CLASSES[i]
    dcos = pre_mean[c] - post_mean[c]            # 余弦降幅 (正 = 更分散)
    dap = AP[name]['B3_joint2'] - AP[name]['baseline']
    auc = AUC[name]['auc']
    rows.append((c, pre_mean[c], post_mean[c], dcos, dap, auc))

print(f'{"类":<12s} {"源余弦":>8s} {"外推余弦":>8s} {"Δcos":>8s} {"ΔAP":>8s} {"AUC":>7s}')
for c, pre, post, dcos, dap, auc in rows:
    print(f'{c:<12s} {pre:>8.3f} {post:>8.3f} {dcos:>+8.3f} {dap:>+8.4f} {auc:>7.3f}')

# 相关性
from scipy.stats import spearmanr
dcos_arr = np.array([r[3] for r in rows])
dap_arr = np.array([r[4] for r in rows])
rho1, p1 = spearmanr(dcos_arr, dap_arr)
rho2, p2 = spearmanr(dcos_arr, np.array([r[5] for r in rows]))
print(f'\nSpearman(Δcos, ΔAP):   ρ={rho1:+.3f}  p={p1:.4f}  ({"显著" if p1 < 0.05 else "不显著"})')
print(f'Spearman(Δcos, AUC):    ρ={rho2:+.3f}  p={p2:.4f}  ({"显著" if p2 < 0.05 else "不显著"})')

with open(DATA / 'tge_probe_corr.json', 'w', encoding='utf-8') as f:
    json.dump({'rows': [{'cls': c, 'pre_cos': pre, 'post_cos': post, 'dcos': dcos,
                         'dap': dap, 'auc': auc} for c, pre, post, dcos, dap, auc in rows],
               'spearman_dcos_dap': [round(rho1, 4), round(p1, 4)],
               'spearman_dcos_auc': [round(rho2, 4), round(p2, 4)]},
              f, indent=2, ensure_ascii=False)
print(f'\n✅ -> {DATA / "tge_probe_corr.json"}')
