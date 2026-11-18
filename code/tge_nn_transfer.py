# -*- coding: utf-8 -*-
r"""tge_nn_transfer.py — 选择器探索协议·实验 2: 最近邻域迁移 (S2)

问题: 域相似度 (无标签域特征) 是否携带选参信息?
方法: 对留出域 d*, 用无标签域特征 φ(d*) 找最近开发域 d_nn, 用 d_nn 的增益网格最优配置。

φ(d) = 域内类均值 (agree, sep_src, gd_hi_frac)  —— 全部来自已有 proxy JSON, 零新检测。
       (small_frac/gd_mean_conf 因 VOC 缺失而排除, 保证 4 域可比较)
配置池: β∈{0.5,1,2,4} × λ∈{0,0.25,0.5,0.75} (排除 λ=1.0 退化格)。

评价: LODO 4 域, 报告 NN 选出配置、留出域增益, 对公平基准 Fixed(1.0,0.25)=0.0064
      与 target-tuned Fixed(2.0,0.5)=0.0302 对比。n=4 仅作诊断, 不宣称显著性。

运行: python tge_nn_transfer.py   (~秒级)
输出: 实验数据/tge_nn_transfer.json
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy.stats import zscore

DATA = Path(r'C:\Users\张三\Desktop\实验数据')
BETAS = [0.5, 1.0, 2.0, 4.0]
LAMBDAS = [0.0, 0.25, 0.5, 0.75]   # 排除 λ=1.0 (退化到基线)
GRID_KEYS = ['b%g_l%g' % (b, l) for b in BETAS for l in LAMBDAS]

# ── 域特征 φ(d) = 类均值 (agree, sep_src, gd_hi_frac) ──
p24 = json.load(open(DATA / 'tge_proxy_selector_24.json', encoding='utf-8'))['proxy']
pvd = json.load(open(DATA / 'tge_proxy_selector_visdrone.json', encoding='utf-8'))['proxy']
grid = json.load(open(DATA / 'tge_dccs_grid.json', encoding='utf-8'))
lodo = json.load(open(DATA / 'tge_dccs_lodo.json', encoding='utf-8'))['summary']


def domain_phi(prefix):
    rows = [v for k, v in p24.items() if k.startswith(prefix)]
    if prefix == 'VD-':
        rows = list(pvd.values())
    if not rows:
        return None
    return {key: float(np.mean([r[key] for r in rows]))
            for key in ['agree_spearman', 'sep_src', 'gd_hi_frac']}


DOMAINS = {
    'COCO': ('COCO-', domain_phi('COCO-')),
    'VOC': ('VOC-', domain_phi('VOC-')),
    'DOTA': ('DOTA-', domain_phi('DOTA-')),
    'VisDrone': ('VD-', domain_phi('VD-')),
}
FEAT_KEYS = ['agree_spearman', 'sep_src', 'gd_hi_frac']

# ── 每域增益网格 (类均值) ──
def canon(b, l):
    return 'b%g_l%g' % (b, l)


def domain_grid(name):
    if name == 'COCO':
        return {k: grid['coco']['grid'][k] for k in GRID_KEYS}
    if name == 'DOTA':
        return {k: grid['dota']['grid'][k] for k in GRID_KEYS}
    if name == 'VisDrone':
        return {k: grid['visdrone']['grid'][k] for k in GRID_KEYS}
    if name == 'VOC':
        # VOC 网格 key 格式不同 (b0.5_l0.0), 数值翻译回 canon
        out = {}
        for raw_key, val in grid['voc']['grid'].items():
            import re
            m = re.match(r'b([0-9.]+)_l([0-9.]+)', raw_key)
            if m:
                out[canon(float(m.group(1)), float(m.group(2)))] = val
        return {k: out[k] for k in GRID_KEYS}
    return None


def best_config(g):
    return max(g, key=g.get)


print('域特征 φ (类均值, 无标签):')
print('%-10s %10s %10s %10s' % ('domain', 'agree', 'sep_src', 'gd_hi'))
phivec = {}
for name, (prefix, phi) in DOMAINS.items():
    phivec[name] = phi
    print('%-10s %10.4f %10.4f %10.4f' % (name, phi['agree_spearman'], phi['sep_src'],
                                          phi['gd_hi_frac']))

# z-score 特征 (用全部 4 域)
Z = np.array([[phivec[n][k] for k in FEAT_KEYS] for n in DOMAINS])
Zm = Z.mean(axis=0); Zs = Z.std(axis=0) + 1e-9
Zz = (Z - Zm) / Zs

print('\nLODO 最近邻迁移:')
print('%-10s %-10s %-14s %9s %9s' % ('held-out', 'NN dev', 'cfg', 'gain', 'vs Fixed(1.0,.25)'))
per_fold = {}
for i, (held, (prefix, phi)) in enumerate(DOMAINS.items()):
    # 最近邻 = 特征距离最小的开发域
    dists = {}
    for j, (dev, _) in enumerate(DOMAINS.items()):
        if dev == held:
            continue
        dists[dev] = float(np.linalg.norm(Zz[i] - Zz[j]))
    nn = min(dists, key=dists.get)
    cfg = best_config(domain_grid(nn))
    g = domain_grid(held)[cfg]
    f125 = lodo['Fixed(1.0,0.25)']['per_fold'][held.lower()]
    per_fold[held] = {'nn_dev': nn, 'cfg': cfg, 'gain': round(g, 4),
                      'gain_fixed125': f125, 'delta': round(g - f125, 4)}
    print('%-10s %-10s %-14s %+9.4f %+9.4f' % (held, nn, cfg, g, g - f125))

mean_g = float(np.mean([v['gain'] for v in per_fold.values()]))
mean_d = float(np.mean([v['delta'] for v in per_fold.values()]))
pos = sum(1 for v in per_fold.values() if v['delta'] > 0)
print('\nNN 迁移均值: %+.4f  vs Fixed(1.0,.25)=0.0064 → %+ .4f  (%d/4 折为正)'
      % (mean_g, mean_d, pos))

out = {'config': {'protocol': 'S2 NN transfer, φ=(agree,sep_src,gd_hi) 类均值, λ=1.0 排除, '
                               'z-score 欧氏最近邻'},
       'phi': {n: phivec[n] for n in DOMAINS}, 'per_fold': per_fold,
       'mean_gain': round(mean_g, 4), 'mean_delta_vs_fixed125': round(mean_d, 4),
       'n_pos_folds': pos}
with open(DATA / 'tge_nn_transfer.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print('\n→ tge_nn_transfer.json')
