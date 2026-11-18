# -*- coding: utf-8 -*-
import json, os
os.chdir(r'C:\Users\张三\Desktop\实验数据')
for f in sorted(os.listdir('.')):
    if f.endswith('.json') and ('appg' in f or 'alpha' in f or 'auto_prior' in f or 'automated' in f):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if 'classes' in data or 'per_class' in data or 'class_results' in data:
                print(f'{f}: 有逐类结果')
                if 'classes' in data:
                    cls = data['classes']
                    print(f'  classes类型: {type(cls)}, 长度: {len(cls) if isinstance(cls, list) else "N/A"}')
                    if isinstance(cls, list) and len(cls) > 0:
                        print(f'  第一个类: {json.dumps(cls[0], ensure_ascii=False)[:200]}')
                if 'mean' in data:
                    print(f'  mean: {json.dumps(data["mean"], ensure_ascii=False)[:200]}')
            else:
                print(f'{f}: 无逐类结果, keys={list(data.keys())[:8]}')
        except Exception as e:
            print(f'{f}: 读取失败 {e}')
