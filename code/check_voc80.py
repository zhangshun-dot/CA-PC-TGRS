# -*- coding: utf-8 -*-
import json
with open(r'C:\Users\张三\Desktop\实验数据\tge_v2_appg_voc80_alpha1.5.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print('VOC 80类扩展结果:')
print('config:', json.dumps(data.get('config', {}), ensure_ascii=False))
print('mean:', json.dumps(data.get('mean', {}), ensure_ascii=False))
classes = data.get('classes', {})
print(f'类别数: {len(classes)}')
print('前15类:')
for i, (k, v) in enumerate(sorted(classes.items())[:15]):
    base = v.get('base50', 0)
    fixed = v.get('fixed50', 0)
    adapt = v.get('adapt50', 0)
    gain = v.get('gain_adapt', 0)
    print(f'  {k}: base={base:.4f}, fixed={fixed:.4f}, adapt={adapt:.4f}, gain={gain:+.4f}')
