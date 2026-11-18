import json
import numpy as np

# 读取PCA分析结果
with open(r'C:\Users\张三\Desktop\实验数据\pca_analysis_results.json', 'r', encoding='utf-8') as f:
    pca = json.load(f)

print('=== PCA分析关键指标 ===')
print('总tokens:', pca.get('total_tokens'))
print('隐藏维度:', pca.get('hidden_dim'))
print('类别数:', pca.get('n_classes'))
print()
print('解释方差比（前10个PC）:')
for i, v in enumerate(pca['explained_variance_ratio'][:10]):
    print(f'  PC{i+1}: {v*100:.2f}%')
print()
print(f'前3PC累计: {pca["cumulative_variance"][2]*100:.2f}%')
print(f'前5PC累计: {pca["cumulative_variance"][4]*100:.2f}%')
print(f'前10PC累计: {pca["cumulative_variance"][9]*100:.2f}%')
print()

# 查看所有key
print('=== 文件中的所有key ===')
for k in pca.keys():
    if k not in ['explained_variance_ratio', 'cumulative_variance', 'pca_components']:
        v = pca[k]
        if isinstance(v, (int, float, str, bool)):
            print(f'  {k}: {v}')
        elif isinstance(v, list):
            print(f'  {k}: list of {len(v)} items')
        elif isinstance(v, dict):
            print(f'  {k}: dict with keys {list(v.keys())[:8]}')

print()
print('=== 双向实验结果 ===')
with open(r'C:\Users\张三\Desktop\实验数据\tge_v2_auto_tune_bidirectional_results.json', 'r', encoding='utf-8') as f:
    bidir = json.load(f)

for k in bidir.keys():
    v = bidir[k]
    if isinstance(v, (int, float, str, bool)):
        print(f'  {k}: {v}')
    elif isinstance(v, list):
        print(f'  {k}: list of {len(v)} items')
        if len(v) > 0 and isinstance(v[0], dict):
            print(f'    first item keys: {list(v[0].keys())[:8]}')
    elif isinstance(v, dict):
        print(f'  {k}: dict with keys {list(v.keys())[:8]}')
