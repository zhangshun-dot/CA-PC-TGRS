# -*- coding: utf-8 -*-
import json
with open(r'C:\Users\张三\Desktop\实验数据\tge_v2_appg_voc80_alpha1.5.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
classes = data.get('classes', {})
print(f'VOC→COCO 完整20类结果 (α=1.5):')
print(f'{"类别":<20} {"基线":>8} {"固定参数":>8} {"自适应":>8} {"固定增益":>8} {"自适应增益":>8}')
print('-' * 80)
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
print('-' * 80)
avg_base = sum([c['base50'] for c in classes.values()]) / len(classes)
avg_fixed = sum([c['fixed50'] for c in classes.values()]) / len(classes)
avg_adapt = sum([c['adapt50'] for c in classes.values()]) / len(classes)
print(f'{"20类平均":<20} {avg_base:>8.4f} {avg_fixed:>8.4f} {avg_adapt:>8.4f} {sum(gains_fixed)/len(gains_fixed):>+8.4f} {sum(gains_adapt)/len(gains_adapt):>+8.4f}')
print(f'\n正增益类别数: 固定参数 {sum(1 for g in gains_fixed if g > 0)}/20, 自适应 {sum(1 for g in gains_adapt if g > 0)}/20')
