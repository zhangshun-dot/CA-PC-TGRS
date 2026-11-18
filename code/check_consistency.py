# -*- coding: utf-8 -*-
import json

with open(r'C:\Users\张三\Desktop\实验数据\tge_v2_appg_sweep.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=== Alpha扫描结果 ===')
print('配置: beta_base={}, lambda_base={}'.format(data.get('beta_base'), data.get('lambda_base')))
print()
for result in data.get('results', []):
    alpha = result.get('alpha')
    mean_gain = result.get('mean_gain')
    pos_count = result.get('positive_count', '?')
    print('  alpha={}: 均值增益={:+.4f}, 正增益类数={}/6'.format(alpha, mean_gain, pos_count))

print()
alpha0 = [r for r in data.get('results', []) if r.get('alpha') == 0.0]
if alpha0:
    gain = alpha0[0].get('mean_gain')
    print('alpha=0（固定参数）增益: {:+.4f}'.format(gain))
    print('论文中目标域无关配置: +0.0166')
    if abs(gain - 0.0166) < 0.001:
        print('✅ 完全一致！两套实验可比，不需要重跑主实验')
    else:
        print('⚠️  差异={:.4f}，需要确认'.format(abs(gain-0.0166)))
