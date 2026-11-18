# -*- coding: utf-8 -*-
"""generate_grid_csv.py — 生成α网格搜索的CSV长表
格式：dataset,class,alpha,ap50,m0,mu,sv,tc
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import csv
from pathlib import Path
import numpy as np

DATA = Path(r'C:\Users\张三\Desktop\实验数据')


def load_and_convert():
    """加载COCO和VOC的α网格搜索结果，转换为CSV长表"""
    # 加载COCO结果
    with open(DATA / 'coco_alpha_grid_scan_results.json', 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 加载VOC结果
    with open(DATA / 'voc_alpha_grid_scan_results.json', 'r', encoding='utf-8') as f:
        voc_data = json.load(f)

    all_rows = []

    for item in coco_data:
        dataset = 'COCO'
        cls_name = f"{item['src']}_{item['target_cls']}"
        m0 = item['m_0']
        mu = item['mu_c']
        ap_curve = item.get('ap_curve', {})

        for alpha_str, ap50 in ap_curve.items():
            try:
                alpha = float(alpha_str)
            except:
                continue
            all_rows.append({
                'dataset': dataset,
                'class': cls_name,
                'alpha': alpha,
                'ap50': ap50,
                'm0': m0,
                'mu': mu,
                'sv': 0,  # 暂时留空
                'tc': 0,  # 暂时留空
            })

    for item in voc_data:
        dataset = 'VOC'
        cls_name = f"{item['src']}_{item['target_cls']}"
        m0 = item['m_0']
        mu = item['mu_c']
        ap_curve = item.get('ap_curve', {})

        for alpha_str, ap50 in ap_curve.items():
            try:
                alpha = float(alpha_str)
            except:
                continue
            all_rows.append({
                'dataset': dataset,
                'class': cls_name,
                'alpha': alpha,
                'ap50': ap50,
                'm0': m0,
                'mu': mu,
                'sv': 0,
                'tc': 0,
            })

    # 写入CSV
    csv_file = DATA / 'alpha_grid_search.csv'
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'class', 'alpha', 'ap50', 'm0', 'mu', 'sv', 'tc'])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'CSV已保存到: {csv_file}')
    print(f'总行数: {len(all_rows)}')
    print(f'COCO行数: {sum(1 for r in all_rows if r["dataset"] == "COCO")}')
    print(f'VOC行数: {sum(1 for r in all_rows if r["dataset"] == "VOC")}')
    print(f'唯一类别数: {len(set(r["class"] for r in all_rows))}')

    # 同时生成alpha_star汇总表
    alpha_star_rows = []
    for item in coco_data + voc_data:
        dataset = 'COCO' if item in coco_data else 'VOC'
        cls_name = f"{item['src']}_{item['target_cls']}"
        alpha_star_rows.append({
            'dataset': dataset,
            'class': cls_name,
            'alpha_star': item['best_alpha'],
            'ap50_star': item['best_ap'],
            'ap50_alpha0': item['baseline_ap'],
            'm0': item['m_0'],
            'mu': item['mu_c'],
        })

    csv_file2 = DATA / 'alpha_star_summary.csv'
    with open(csv_file2, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'class', 'alpha_star', 'ap50_star', 'ap50_alpha0', 'm0', 'mu'])
        writer.writeheader()
        writer.writerows(alpha_star_rows)

    print(f'\nalpha_star汇总表已保存到: {csv_file2}')
    print(f'类别数: {len(alpha_star_rows)}')

    # 打印α*=0的类别
    print(f'\n=== α*=0的类别 ===')
    for row in alpha_star_rows:
        if row['alpha_star'] == 0.0:
            print(f'  {row["dataset"]}/{row["class"]}: AP(α=0)={row["ap50_alpha0"]:.4f}')

    return csv_file, csv_file2


if __name__ == '__main__':
    load_and_convert()
