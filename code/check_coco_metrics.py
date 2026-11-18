# -*- coding: utf-8 -*-
import json

with open(r'C:\Users\张三\Desktop\实验数据\coco_standard_metrics.json', 'r') as f:
    data = json.load(f)

print('=== 逐类标准COCO指标 ===')
for cls, metrics in data['per_class'].items():
    print(f'\n{cls}:')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

print('\n=== 汇总 ===')
print(f'AP50均值增益: {data["mean_gain50"]:.4f}')
print(f'AP@[.5:.95]均值增益: {data["mean_gain_coco"]:.4f}')
print(f'AP50正增益类别数: {data["n_positive_50"]}/6')
print(f'AP@[.5:.95]正增益类别数: {data["n_positive_coco"]}/6')
