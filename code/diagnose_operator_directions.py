# -*- coding: utf-8 -*-
"""diagnose_operator_directions.py — 诊断三个算子的方向向量模长
检查DDC的-0.333是不是因为方向向量模长过大导致步长失控
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import tge_probe as T
from sota_comparison import (
    SOURCE_LOADERS, build_proto_tge, evaluate_with_features,
    SRC_TMPL_OVERHEAD, TGT_TMPL,
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

MAIN_CLASSES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'truck']


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


def get_direction_vectors(src_name, src_cls, target_cls, target_domain):
    """计算三个算子的方向向量，并返回它们的模长"""
    # 加载源域数据
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']

    # 加载目标域数据
    if target_domain == 'COCO':
        from sota_comparison import load_target_coco
        target_data = load_target_coco(target_cls, n=200)
    elif target_domain == 'VOC':
        from voc_loader import load_target_voc
        voc_cls_map = {'airplane': 'aeroplane', 'motorcycle': 'motorbike',
                       'boat': 'boat', 'car': 'car', 'bus': 'bus',
                       'bicycle': 'bicycle', 'person': 'person'}
        voc_cls = voc_cls_map.get(target_cls, target_cls)
        target_data = load_target_voc(voc_cls, n=200)
    else:
        return None

    if target_data is None:
        return None
    tgt_features = target_data['features']

    # 算子1: TGE方向向量
    s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls)
    d_emb = T.text_avg([TGT_TMPL], target_cls)
    # 转到CPU并转为numpy
    if hasattr(s_emb, 'cpu'):
        s_emb = s_emb.cpu().numpy()
    if hasattr(d_emb, 'cpu'):
        d_emb = d_emb.cpu().numpy()
    d_tge = d_emb - s_emb
    norm_tge = np.linalg.norm(d_tge)

    # 算子2: FMS方向向量
    tgt_mean = np.mean(tgt_features, axis=0)
    d_fms = tgt_mean - src_proto
    norm_fms = np.linalg.norm(d_fms)

    # 算子3: DDC方向向量
    src_mean = np.mean(src_features, axis=0)
    tgt_mean_all = np.mean(tgt_features, axis=0)
    d_ddc = tgt_mean_all - src_mean
    norm_ddc = np.linalg.norm(d_ddc)

    # 源域原型的模长（作为参考）
    norm_proto = np.linalg.norm(src_proto)

    # α=2.5时的实际位移
    displacement_tge = 2.5 * norm_tge
    displacement_fms = 2.5 * norm_fms
    displacement_ddc = 2.5 * norm_ddc

    # 位移相对于原型模长的比例
    ratio_tge = displacement_tge / norm_proto
    ratio_fms = displacement_fms / norm_proto
    ratio_ddc = displacement_ddc / norm_proto

    return {
        'src': src_name,
        'src_cls': src_cls,
        'target_cls': target_cls,
        'target_domain': target_domain,
        'norm_proto': float(norm_proto),
        'norm_tge': float(norm_tge),
        'norm_fms': float(norm_fms),
        'norm_ddc': float(norm_ddc),
        'displacement_tge': float(displacement_tge),
        'displacement_fms': float(displacement_fms),
        'displacement_ddc': float(displacement_ddc),
        'ratio_tge': float(ratio_tge),
        'ratio_fms': float(ratio_fms),
        'ratio_ddc': float(ratio_ddc),
    }


def main():
    print('=' * 100)
    print('诊断三个算子的方向向量模长')
    print('=' * 100)

    # 加载原始样本
    with open(DATA / 'zero_shot_gated_samples_extended_alpha.json', 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    # 筛选主分析7类的样本
    main_samples = [s for s in all_samples if s['target_cls'] in MAIN_CLASSES]
    print(f'主分析样本数: {len(main_samples)}')

    # 对每个样本计算方向向量模长
    results = []
    for idx, sample in enumerate(main_samples[:10]):  # 先看前10个
        print(f'\n[{idx+1}/{min(10, len(main_samples))}] {sample["src"]}/{sample["src_cls"]} -> '
              f'{sample["target_domain"]}/{sample["target_cls"]}')
        result = get_direction_vectors(
            sample['src'], sample['src_cls'], sample['target_cls'], sample['target_domain']
        )
        if result is not None:
            results.append(result)
            print(f'  原型模长: {result["norm_proto"]:.4f}')
            print(f'  TGE方向向量模长: {result["norm_tge"]:.4f}, α=2.5位移: {result["displacement_tge"]:.4f}, '
                  f'相对原型: {result["ratio_tge"]:.1%}')
            print(f'  FMS方向向量模长: {result["norm_fms"]:.4f}, α=2.5位移: {result["displacement_fms"]:.4f}, '
                  f'相对原型: {result["ratio_fms"]:.1%}')
            print(f'  DDC方向向量模长: {result["norm_ddc"]:.4f}, α=2.5位移: {result["displacement_ddc"]:.4f}, '
                  f'相对原型: {result["ratio_ddc"]:.1%}')

    # 汇总统计
    print('\n' + '=' * 100)
    print('汇总统计（前10个样本）')
    print('=' * 100)

    if results:
        print(f'\n{"算子":<8} {"平均模长":>10} {"平均位移(α=2.5)":>18} {"平均相对原型":>14} {"最大相对原型":>14}')
        print('-' * 70)

        for op_name, norm_key, disp_key, ratio_key in [
            ('TGE', 'norm_tge', 'displacement_tge', 'ratio_tge'),
            ('FMS', 'norm_fms', 'displacement_fms', 'ratio_fms'),
            ('DDC', 'norm_ddc', 'displacement_ddc', 'ratio_ddc'),
        ]:
            mean_norm = np.mean([r[norm_key] for r in results])
            mean_disp = np.mean([r[disp_key] for r in results])
            mean_ratio = np.mean([r[ratio_key] for r in results])
            max_ratio = np.max([r[ratio_key] for r in results])
            print(f'{op_name:<8} {mean_norm:>10.4f} {mean_disp:>18.4f} {mean_ratio:>14.1%} {max_ratio:>14.1%}')

        # 关键发现
        print('\n' + '=' * 100)
        print('关键发现')
        print('=' * 100)

        mean_ratio_tge = np.mean([r['ratio_tge'] for r in results])
        mean_ratio_fms = np.mean([r['ratio_fms'] for r in results])
        mean_ratio_ddc = np.mean([r['ratio_ddc'] for r in results])

        print(f'\nTGE的α=2.5位移平均是原型模长的 {mean_ratio_tge:.1%}')
        print(f'FMS的α=2.5位移平均是原型模长的 {mean_ratio_fms:.1%}')
        print(f'DDC的α=2.5位移平均是原型模长的 {mean_ratio_ddc:.1%}')

        if mean_ratio_ddc > mean_ratio_tge * 2:
            print(f'\n⚠️ DDC的位移是TGE的 {mean_ratio_ddc/mean_ratio_tge:.1f} 倍！')
            print('   这证实了DDC的-0.333很可能是步长失控导致的，不是真实结论。')
            print('   修复方案：对三个算子的方向向量做模长归一化，或者每个算子独立扫α网格。')
        else:
            print(f'\nDDC的位移与TGE相当，-0.333可能是真实结论。')

    # 保存结果
    out_file = DATA / 'operator_direction_diagnosis.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n结果已保存到: {out_file}')


if __name__ == '__main__':
    main()
