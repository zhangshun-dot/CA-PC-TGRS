# -*- coding: utf-8 -*-
"""check_target_data_format.py — 检查target_data的格式"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
from sota_comparison import load_target_coco

print('检查target_data格式...')

# 加载一个样本
target_data = load_target_coco('airplane', n=10)

print(f'\ntarget_data keys: {list(target_data.keys())}')
print(f'features shape: {target_data["features"].shape}')
print(f'per_img length: {len(target_data["per_img"])}')

if len(target_data['per_img']) > 0:
    p = target_data['per_img'][0]
    print(f'\nper_img[0] keys: {list(p.keys())}')

    # 检查boxes字段
    if 'boxes' in p:
        boxes = p['boxes']
        print(f'\nboxes type: {type(boxes)}')
        print(f'boxes length: {len(boxes)}')
        if len(boxes) > 0:
            print(f'boxes[0]: {boxes[0]}')
            print(f'boxes[0] type: {type(boxes[0])}')
            print(f'boxes[0] length: {len(boxes[0])}')
    else:
        print(f'\n没有boxes字段！')
        print(f'可用字段: {list(p.keys())}')

    # 检查其他可能的字段
    for key in p.keys():
        val = p[key]
        if isinstance(val, (list, np.ndarray)) and len(val) > 0:
            print(f'\n{key}: type={type(val)}, length={len(val)}')
            if isinstance(val[0], (list, np.ndarray)):
                print(f'  {key}[0]: {val[0]}')
            else:
                print(f'  {key}[0]: {val[0]}')
