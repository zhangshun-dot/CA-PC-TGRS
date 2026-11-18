# -*- coding: utf-8 -*-
import json
with open(r'C:\Users\张三\Desktop\实验数据\tge_v2_appg_alpha1.5.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
classes = data['classes']
print('人工先验 α=1.5 逐类结果:')
header = f'{"类别":<20} {"基线":>8} {"固定参数":>8} {"自适应":>8} {"固定增益":>8} {"自适应增益":>8}'
print(header)
print('-' * 70)
gains_fixed = []
gains_adapt = []
for cls_name, cls_data in sorted(classes.items()):
    base = cls_data.get('base50', 0)
    fixed = cls_data.get('fixed50', 0)
    adapt = cls_data.get('adapt50', 0)
    gain_fixed = cls_data.get('gain_fixed', 0)
    gain_adapt = cls_data.get('gain_adapt', 0)
    gains_fixed.append(gain_fixed)
    gains_adapt.append(gain_adapt)
    print(f'{cls_name:<20} {base:>8.4f} {fixed:>8.4f} {adapt:>8.4f} {gain_fixed:>+8.4f} {gain_adapt:>+8.4f}')
print('-' * 70)
avg_base = sum([c['base50'] for c in classes.values()]) / 6
avg_fixed = sum([c['fixed50'] for c in classes.values()]) / 6
avg_adapt = sum([c['adapt50'] for c in classes.values()]) / 6
print(f'{"6类平均":<20} {avg_base:>8.4f} {avg_fixed:>8.4f} {avg_adapt:>8.4f} {sum(gains_fixed)/6:>+8.4f} {sum(gains_adapt)/6:>+8.4f}')
