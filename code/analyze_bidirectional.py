import json
import numpy as np

# 读取双向实验结果
with open(r'C:\Users\张三\Desktop\实验数据\tge_v2_auto_tune_bidirectional_results.json', 'r', encoding='utf-8') as f:
    bidir = json.load(f)

print('=== 双向实验配置 ===')
config = bidir['config']
print('开发集:', config.get('dev_set'))
print('测试集:', config.get('test_set'))
print('模式:', config.get('modes'))
print('最佳tau:', config.get('best_tau'))
print('delta门控阈值:', config.get('delta_gate_threshold'))
print()

print('=== 双向实验结果 ===')
results = bidir['results']
for mode, data in results.items():
    if isinstance(data, dict):
        mean_gain = data.get('mean_gain', data.get('gain', 'N/A'))
        per_class = data.get('per_class', data.get('class_gains', []))
        n_pos = sum(1 for g in per_class if isinstance(g, (int, float)) and g > 0) if per_class else 'N/A'
        print(f'\n{mode}:')
        print(f'  平均增益: {mean_gain}')
        if isinstance(per_class, list) and len(per_class) > 0:
            print(f'  逐类增益: {[f"{g:.4f}" if isinstance(g, (int,float)) else g for g in per_class[:10]]}')
            print(f'  正增益类别数: {n_pos}/{len(per_class)}')
    elif isinstance(data, (int, float)):
        print(f'{mode}: {data}')

print()
print('=== 最佳非oracle结果 ===')
best = bidir.get('best_non_oracle', {})
print('最佳配置key:', best.get('key'))
print('平均增益:', best.get('mean_gain'))
print()
print('基线正向DIOR:', bidir.get('baseline_forward_dior'))

# 看看有没有域偏移方向的数据
print()
print('=== 检查其他相关文件 ===')
import os
data_dir = r'C:\Users\张三\Desktop\实验数据'
keywords = ['domain', 'shift', 'offset', 'direction', 'align', 'coral', 'mcd']
for f in os.listdir(data_dir):
    if f.endswith('.json') and any(k in f.lower() for k in keywords):
        print(f'  {f}')
