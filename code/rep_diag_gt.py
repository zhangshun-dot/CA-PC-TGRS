# -*- coding: utf-8 -*-
"""rep_diag_gt.py — 纯 GT 的"物体/场景特征"因子(零 GPU, CPU 即可)

补充 factor_importance 的缺失列(☆): object_size / scene_difficulty 的 GT 代理。
  object_rel_size : 每框面积 / 图像面积 的中位数(尺度不变, 小目标→近0)
  scene_density   : 每图 GT 框数均值(拥挤度, 遥感常高)
  small_frac      : 相对面积<1% 的小目标占比
注意: 这些是"物体/场景客观特征", 不是"表示质量"。表示质量(z_c_probe/effective_rank/
z_k_leakage)需要逐框 DINOv2 嵌入 + CondEncoder 标量 → 需 GPU 重跑 prep(另脚本)。
输出 factor_matrix_extra.json, 顺序与 weight_probe_combined_cv.json 场景顺序一致。
"""
import os, sys, json
import numpy as np
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from unified_eval import load_coco, load_dior

BASE = r'C:\Users\张三\Desktop\实验数据'
N_COCO, N_DIOR = 80, 50

# 顺序必须 = weight_probe_combined_cv.json 的 scenarios 插入顺序
SCEN = [
    ('COCO-airplane', 'coco', 'airplane', 'airplane'),
    ('COCO-boat',     'coco', 'boat',     'ship'),
    ('COCO-car',      'coco', 'car',      'vehicle'),
    ('DIOR-airplane', 'dior', 'airplane', 'airplane'),
    ('DIOR-ship',     'dior', 'ship',     'ship'),
    ('DIOR-vehicle',  'dior', 'vehicle',  'vehicle'),
    ('DIOR-harbor',   'dior', 'harbor',   'harbor'),
    ('DIOR-storage',  'dior', 'storage tank', 'storage tank'),
    ('DIOR-tennis',   'dior', 'tennis court', 'tennis court'),
    ('DIOR-baseball', 'dior', 'baseball field', 'baseball field'),
    ('DIOR-bridge',   'dior', 'bridge',   'bridge'),
]


def main():
    rows = {'object_rel_size': [], 'scene_density': [], 'small_frac': []}
    for name, dset, cls, dior in SCEN:
        if dset == 'coco':
            images, gts = load_coco(cls, n=N_COCO)
        else:
            images, gts = load_dior(cls, n=N_DIOR)
        rels, nbox, nsmall = [], 0, 0
        for im, g in zip(images, gts):
            Iarea = im.width * im.height
            for b in g:
                w = b[2] - b[0]; h = b[3] - b[1]; a = w * h
                rel = a / Iarea if Iarea > 0 else 0.0
                rels.append(rel); nbox += 1
                if rel < 0.01:
                    nsmall += 1
        rows['object_rel_size'].append(round(float(np.median(rels)), 5))
        rows['scene_density'].append(round(len(gts) / max(1, len(images)), 4))
        rows['small_frac'].append(round(nsmall / max(1, nbox), 4))
        print(f'  {name}: n_img={len(images)} rel_size_med={rows["object_rel_size"][-1]:.4f} '
              f'density={rows["scene_density"][-1]:.3f} small_frac={rows["small_frac"][-1]:.3f}', flush=True)
    out = {'columns': list(rows.keys()), 'data': rows}
    with open(os.path.join(BASE, 'factor_matrix_extra.json'), 'w', encoding='utf-8') as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print('\n写出 factor_matrix_extra.json:', rows, flush=True)


if __name__ == '__main__':
    main()
