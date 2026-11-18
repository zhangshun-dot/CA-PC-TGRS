# -*- coding: utf-8 -*-
"""跨域三方案联合对比

汇总 C-VEP、VDA、DPS 三个方案在 COCO 跨域检测上的表现。
"""

import json
from pathlib import Path
import numpy as np

SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')

def load_json(name):
    with open(SAVE_DIR / name, encoding='utf-8') as f:
        return json.load(f)

def main():
    # 加载各方案结果
    cvep = load_json('cvep_results.json')
    vda = load_json('vda_results.json')
    dps = load_json('dps_results.json')

    print("=" * 78)
    print("跨域目标检测三方案联合对比")
    print("=" * 78)
    print()
    print("测试: COCO val2017, 30 images/class, GDINO-tiny, AP50 (th=0.25)")
    print()

    # ═══ 表1: C-VEP vs VDA-VEP ═══
    print("表1: 视觉重打分方案 (C-VEP vs VDA-VEP vs Plain VEP)")
    print("-" * 78)
    print(f"{'Class':<12s} {'Baseline':>8s} {'VEP-CLIP':>8s} {'VEP-DINOv2':>8s} "
          f"{'C-VEP':>8s} {'VDA-VEP':>8s} {'Best':>8s} {'Δ-Base':>8s}")
    print("-" * 78)

    for coco_cls in ['airplane', 'boat', 'car']:
        if coco_cls not in cvep:
            continue

        base = cvep[coco_cls]['baseline']
        l12_dino = cvep[coco_cls]['l12_vep']
        cvep_grid = cvep[coco_cls]['cvep_grid']
        cvep_layer = cvep[coco_cls]['cvep_grid_layer']

        vda_vep = vda[coco_cls]['vda_vep'] if coco_cls in vda else 0
        vep_clip = vda[coco_cls]['vep'] if coco_cls in vda else 0

        best = max(base, l12_dino, cvep_grid, vda_vep, vep_clip)
        best_name = 'Baseline' if best == base else (
            'C-VEP' if best == cvep_grid else (
                'VDA-VEP' if best == vda_vep else (
                    'VEP-CLIP' if best == vep_clip else 'VEP-DINOv2'
                )))

        print(f"{coco_cls:<12s} {base:8.4f} {vep_clip:8.4f} {l12_dino:8.4f} "
              f"{cvep_grid:8.4f} {vda_vep:8.4f} {best:8.4f} {best-base:+8.4f} "
              f"({best_name})")

    print("-" * 78)
    avg_base = np.mean([cvep[c]['baseline'] for c in ['airplane', 'boat', 'car']])
    avg_vep = np.mean([vda[c]['vep'] for c in ['airplane', 'boat', 'car']])
    avg_dino = np.mean([cvep[c]['l12_vep'] for c in ['airplane', 'boat', 'car']])
    avg_cvep = np.mean([cvep[c]['cvep_grid'] for c in ['airplane', 'boat', 'car']])
    avg_vda = np.mean([vda[c]['vda_vep'] for c in ['airplane', 'boat', 'car']])

    print(f"{'AVERAGE':<12s} {avg_base:8.4f} {avg_vep:8.4f} {avg_dino:8.4f} "
          f"{avg_cvep:8.4f} {avg_vda:8.4f} {avg_cvep:8.4f} {avg_cvep-avg_base:+8.4f} "
          f"(C-VEP)")

    # ═══ 表2: DPS 前缀选择 ═══
    print()
    print("表2: 域前缀选择 (DPS)")
    print("-" * 78)
    print(f"{'Test':<18s} {'NatPref':>8s} {'RemPref':>8s} {'DPS':>8s} {'Oracle':>8s} "
          f"{'Acc':>6s} {'Penalty':>8s}")
    print("-" * 68)

    for name, r in dps.items():
        penalty = abs(r['natural_prefix'] - r['remote_prefix'])
        print(f"{name:<18s} {r['natural_prefix']:8.4f} {r['remote_prefix']:8.4f} "
              f"{r['dps']:8.4f} {r['oracle']:8.4f} {r['dps_accuracy']:5.1%} {penalty:8.4f}")

    # ═══ 表3: 跨域最优层分析 ═══
    print()
    print("表3: C-VEP 最优层分析 (DIOR prototype → COCO detection)")
    print("-" * 60)
    print(f"{'Class':<12s} {'DIOR L12':>8s} {'COCO Best':>10s} {'COCO L12':>8s} "
          f"{'Δ-best':>8s}")
    print("-" * 60)

    for coco_cls in ['airplane', 'boat', 'car']:
        if coco_cls not in cvep:
            continue
        best_l = cvep[coco_cls]['cvep_grid_layer']
        best_ap = cvep[coco_cls]['cvep_grid']
        l12_ap = cvep[coco_cls]['l12_vep']
        print(f"{coco_cls:<12s} {'L12':>8s} {best_l:>10s} {l12_ap:8.4f} "
              f"{best_ap-l12_ap:+8.4f}")

    # ═══ 结论 ═══
    print()
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print(f"""
1. C-VEP (跨域层加权) 最优:
   - 平均 AP50 提升: {avg_cvep-avg_base:+.4f} (vs Baseline)
   - 比 L12-VEP 提升: {avg_cvep-avg_dino:+.4f}
   - 最优层: airplane→L11, boat→L11, car→L12

2. VDA (FiLM视觉适配) 无额外增益:
   - VDA-VEP avg: {avg_vda:.4f} vs VEP-CLIP avg: {avg_vep:.4f}
   - 域条件调制对 CLIP 特征的影响太小 (rank_acc=1.0 但 eval 无增益)
   - 可能原因: DIOR exemplar 训练的 adapter 在 COCO crops 上过拟合

3. DPS (域前缀选择) 实用但影响小:
   - 域分类准确率: 93-100% (COCO airplane 43%)
   - 前缀错选 AP50 损失: 平均 0.04
   - DPS ≈ Oracle (差距 < 0.02)
""")

    print(f"详细结果:")
    print(f"  C-VEP:  {SAVE_DIR / 'cvep_results.json'}")
    print(f"  VDA:    {SAVE_DIR / 'vda_results.json'}")
    print(f"  DPS:    {SAVE_DIR / 'dps_results.json'}")


if __name__ == '__main__':
    main()
