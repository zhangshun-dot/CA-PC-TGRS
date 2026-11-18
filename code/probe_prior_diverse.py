# -*- coding: utf-8 -*-
"""probe_prior_diverse.py — 方向1 自动化先验在非车辆类上的区分力诊断（零GT，纯文本几何）

目的: 在 DIOR 全类（含非车辆/基础设施类 bridge/harbor/storage-tank 等）上验证
自动化先验 R(c) = ‖E_photo(c) − E_overhead(c)‖（dom_gap，即人工 OBJECT_SIZE/VIEWPOINT_DIFF 的零GT替代）
是否真有区分力——即"非车辆类域差更大"的直觉是否成立。

背景: run_closed_loop 的 9 单元全为车辆类，dom_gap 仅 0.496–0.550（flat），与增益不相关(ρ=+0.38, ns)。
但那只是子集同质性的假象，还是自动化先验本身就没区分力？本脚本在 DIOR 全类上给出答案。

做法: transformers CLIP 文本编码（离线），对每类算 photo↔overhead 文本间隙 dom_gap；
分组(交通/载具 vs 基础设施/非载具)做两样本检验，看分离是否显著。

用法 (CPU):
  HF_HUB_OFFLINE=1 "C:/Users/张三/vg/Scripts/python.exe" -u probe_prior_diverse.py
"""
import os, sys, json, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
from scipy import stats
import run_bidirectional as RB

RB.DEVICE = 'cpu'
DEVICE = 'cpu'

# DIOR 全类（含非车辆/基础设施类）——用以压方向1在非车辆类的区分力
DIOR_CLASSES = [
    'airplane', 'ship', 'vehicle', 'bridge', 'harbor', 'storage tank', 'dam', 'overpass',
    'chimney', 'stadium', 'wind turbine', 'airport', 'train station', 'expressway toll station',
    'baseball diamond', 'basketball court', 'tennis court', 'ground track field',
]
# 分组：交通/载具（人工先验预期域差小） vs 基础设施/非载具（预期域差大）
TRANSPORT = {'airplane', 'ship', 'vehicle', 'train station', 'airport'}


def gap(name):
    e_ph = np.asarray(RB.text_embs([f'a photo of {name}'])[0])
    e_ov = np.asarray(RB.text_embs([f'an overhead image of {name}'])[0])
    return float(np.linalg.norm(e_ph - e_ov))


def main():
    t0 = time.time()
    rows = [(c, gap(c)) for c in DIOR_CLASSES]
    rows.sort(key=lambda x: x[1], reverse=True)

    veh = [g for c, g in rows if c in TRANSPORT]
    inf = [g for c, g in rows if c not in TRANSPORT]

    # 两样本检验：基础设施 vs 交通
    u, pu = stats.mannwhitneyu(inf, veh, alternative='greater')
    t, pt = stats.ttest_ind(inf, veh, equal_var=False)

    print('=' * 78)
    print('方向1 自动化先验 R(c)=dom_gap 在 DIOR 全类的区分力（零GT 文本几何）')
    print('=' * 78)
    print(f'{"类":<22} {"dom_gap":>8}   分组')
    for c, g in rows:
        tag = '交通/载具' if c in TRANSPORT else '基础设施/非载具'
        star = '  <-- 高域差' if g >= np.mean(inf) else ''
        print(f'{c:<22} {g:>8.3f}   {tag}{star}')
    print('-' * 78)
    print(f'交通/载具 均值 dom_gap = {np.mean(veh):.3f}  (n={len(veh)})')
    print(f'基础设施/非载具 均值 dom_gap = {np.mean(inf):.3f}  (n={len(inf)})')
    print(f'组间差 = {np.mean(inf)-np.mean(veh):+.3f}')
    print(f'Mann-Whitney U(inf>veh) p = {pu:.4f}    Welch t p = {pt:.4f}')
    print(f'全类 dom_gap 极差 = {max(g for _,g in rows)-min(g for _,g in rows):.3f}'
          f'（9 单元车辆子集仅 0.054，此处应显著更大）')

    sep = (np.mean(inf) - np.mean(veh)) > 0.05 and pu < 0.05
    print(f'\n结论: 自动化先验 R(c)=dom_gap 在 DIOR 全类上'
          + ('**有显著区分力**（基础设施类域差显著大于交通类，方向1 先验有效、非子集假象）'
             if sep else
             '**区分力有限**（基础设施与交通类未显著分离，需换更细粒度先验）'))

    out = {
        'config': {'method': 'zero-GT text geometry', 'prior': 'dom_gap=||E_photo(c)-E_overhead(c)||'},
        'rows': [{'class': c, 'dom_gap': round(g, 4), 'group': ('transport' if c in TRANSPORT else 'infra')} for c, g in rows],
        'mean_transport': round(float(np.mean(veh)), 4), 'n_transport': len(veh),
        'mean_infra': round(float(np.mean(inf)), 4), 'n_infra': len(inf),
        'group_diff': round(float(np.mean(inf) - np.mean(veh)), 4),
        'mannwhitney_p': float(pu), 'welch_t_p': float(pt),
        'range': round(float(max(g for _, g in rows) - min(g for _, g in rows)), 4),
        'prior_discriminative': bool(sep),
        'elapsed_s': time.time() - t0,
    }
    p = RB.OUT.parent / 'probe_prior_diverse.json'
    json.dump(out, open(p, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {p}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
