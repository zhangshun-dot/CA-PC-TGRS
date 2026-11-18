# -*- coding: utf-8 -*-
import json, os
os.chdir(r'C:\Users\张三\Desktop\实验数据')

# 检查可能包含21类/3域结果的文件
candidate_files = [
    'expanded_eval_results.json',
    'expanded_eval_summary.json',
    'final_results_12class.json',
    'final_results.json',
    'full_pipeline_results.json',
    'cross_domain_validation.json',
    'unified_eval_results.json',
]

for f in candidate_files:
    if os.path.exists(f):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            print(f'\n=== {f} ===')
            print(f'  keys: {list(data.keys())[:10]}')
            # 搜索airplane和motorcycle的增益
            def find_values(obj, path=''):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if 'airplane' in str(k).lower() or 'motorcycle' in str(k).lower():
                            if isinstance(v, (int, float)):
                                print(f'  {path}.{k} = {v}')
                            elif isinstance(v, dict):
                                for k2, v2 in v.items():
                                    if 'gain' in str(k2).lower() or 'ap' in str(k2).lower() or 'delta' in str(k2).lower():
                                        print(f'  {path}.{k}.{k2} = {v2}')
                        find_values(v, f'{path}.{k}')
                elif isinstance(obj, list):
                    for i, item in enumerate(obj[:5]):
                        find_values(item, f'{path}[{i}]')
            find_values(data)
        except Exception as e:
            print(f'{f}: 读取失败 {e}')
