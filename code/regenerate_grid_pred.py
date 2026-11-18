# -*- coding: utf-8 -*-
"""regenerate_grid_pred.py — 重新生成grid.csv和pred.csv，用(src, dataset, class)作为单元"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')


def compute_alpha_closed_form(m0, mu, rho=0.13, mu_scale=0.01):
    mu_eff = mu * mu_scale
    alpha = max(0.0, (rho - m0) / max(mu_eff, 1e-8))
    return float(np.clip(alpha, 0.0, 2.0))


def main():
    print('=' * 80)
    print('重新生成grid.csv和pred.csv，用(src, dataset, class)作为单元')
    print('=' * 80)

    samples_file = DATA / 'zero_shot_gated_samples.json'
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    print(f'\n总样本数: {len(samples)}')

    # 生成grid.csv，用(src, dataset, class)作为单元
    grid_csv = DATA / 'grid.csv'
    grid_units = set()
    with open(grid_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('dataset,class,alpha,ap50\n')
        for i, s in enumerate(samples):
            # 用(src_dataset_targetcls)作为唯一单元标识
            unit_id = f"{s['src']}_{s['target_domain']}_{s['target_cls']}"
            grid_units.add(unit_id)
            for alpha_str, ap in s['ap_curve'].items():
                f.write(f'{unit_id},{s["target_cls"]},{alpha_str},{ap:.6f}\n')

    print(f'\ngrid.csv单元数: {len(grid_units)}')
    assert len(grid_units) == 27, f'应该是27个单元，实际是{len(grid_units)}'

    # 生成pred.csv，用同样的单元标识
    pred_csv = DATA / 'pred.csv'
    pred_units = set()
    with open(pred_csv, 'w', encoding='utf-8', newline='') as f:
        f.write('dataset,class,alpha_hat,setting\n')
        for i, s in enumerate(samples):
            unit_id = f"{s['src']}_{s['target_domain']}_{s['target_cls']}"
            pred_units.add(unit_id)

            # 最优参数的闭式解
            alpha_opt = compute_alpha_closed_form(s['m0'], s['mu'], rho=0.13, mu_scale=0.01)
            f.write(f'{unit_id},{s["target_cls"]},{alpha_opt:.6f},closed_form_opt\n')

            # 旧参数的闭式解
            alpha_old = compute_alpha_closed_form(s['m0'], s['mu'], rho=0.2, mu_scale=0.1)
            f.write(f'{unit_id},{s["target_cls"]},{alpha_old:.6f},closed_form_old\n')

            # Oracle
            f.write(f'{unit_id},{s["target_cls"]},{s["best_alpha"]:.6f},oracle\n')

    print(f'pred.csv单元数: {len(pred_units)}')
    assert len(pred_units) == 27, f'应该是27个单元，实际是{len(pred_units)}'

    print(f'\n✅ 两个文件都有27个单元，口径一致')
    print(f'  grid.csv: {grid_csv}')
    print(f'  pred.csv: {pred_csv}')


if __name__ == '__main__':
    main()
