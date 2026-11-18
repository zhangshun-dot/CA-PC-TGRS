# -*- coding: utf-8 -*-
r"""confirm_full_grid_dump.py — 全量 1291 图上的 (beta,lam) 完整网格面

目的：核实「主配置 (2.0,0.5) 是六类均值增益在 1291 图上的网格 argmax」这一说法
是否成立，以及它与 (1.0,0.75) 的差距。复用 confirm_devtest_global_split 的
数据缓存与逐位等价 AP。
"""
import os, sys, json, pickle, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import numpy as np
from pathlib import Path

import confirm_devtest_global_split as C

DATA = C.DATA
OUT = DATA / 'confirm_full_grid_1291.json'


def main():
    t0 = time.time()
    protos = C.build_protos()
    blobs = {cls: C.build_or_load(cls, protos) for cls, _ in C.DEV_CLASSES}

    grid = {}
    for be in C.BETA_GRID:
        for la in C.LAM_GRID:
            per_cls = {}
            for cls, blob in blobs.items():
                idx = list(range(C.K_CALIB, len(blob['per_img'])))
                n_gt = sum(len(blob['per_img'][i]['gt']) for i in idx)
                b = C.ap50_fast(C.base_contribs(blob, idx), n_gt)
                W = C.make_W(blob, be, la)
                a = C.ap50_fast([C.image_contrib(blob['per_img'][i]['box'],
                                                 blob['per_img'][i]['gd'],
                                                 blob['per_img'][i]['gt'],
                                                 blob['per_img'][i]['crop_idx'],
                                                 blob['F'], W) for i in idx], n_gt)
                per_cls[cls] = {'base_ap': float(b), 'ap': float(a), 'gain': float(a - b)}
            grid[(be, la)] = per_cls
            gm = np.mean([v['gain'] for v in per_cls.values()])
            print(f'  beta={be:<4} lam={la:<5} mean_gain={gm:+.4f}  '
                  + ' '.join(f'{c[:4]}={per_cls[c]["gain"]:+.4f}' for c in per_cls))
            sys.stdout.flush()

    means = {f'{b}_{l}': float(np.mean([v['gain'] for v in pc.values()]))
             for (b, l), pc in grid.items()}
    order = sorted(means, key=lambda k: -means[k])
    print('\n=== 全量 1291 网格按均值增益排序 (top 8) ===')
    for k in order[:8]:
        print(f'  {k:>8}  {means[k]:+.4f}')
    print('  (2.0,0.5) 排名 =', order.index('2.0_0.5') + 1, '/', len(order),
          f'  mean={means["2.0_0.5"]:+.4f}')
    print('  (1.0,0.75) mean =', f'{means.get("1.0_0.75", float("nan")):+.4f}')

    res = {'config': {'K_CALIB': C.K_CALIB, 'pre_filter': C.PRE_FILTER,
                      'beta_grid': C.BETA_GRID, 'lam_grid': C.LAM_GRID},
           'mean_gain_grid': means,
           'argmax': order[0],
           'rank_2.0_0.5': order.index('2.0_0.5') + 1,
           'per_class_grid': {f'{b}_{l}': pc for (b, l), pc in grid.items()},
           'elapsed_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n-> {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
