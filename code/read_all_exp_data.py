# -*- coding: utf-8 -*-
import json

# 1. Bootstrap置信区间
print('=' * 60)
print('1. Bootstrap置信区间 (tge_bootstrap_ci.json)')
print('=' * 60)
with open(r'C:\Users\张三\Desktop\实验数据\tge_bootstrap_ci.json', 'r') as f:
    boot_data = json.load(f)
print(f'配置: {boot_data["config"]}')
print(f'\n6类均值:')
for k, v in boot_data['mean6'].items():
    print(f'  {k}: {v}')
print(f'\n逐类结果:')
for cls, data in boot_data['classes'].items():
    boot = data['boot']
    print(f'  {cls}: mean_delta={boot["mean_delta"]:.4f}, CI=[{boot["ci95_low"]:.4f}, {boot["ci95_high"]:.4f}], p={boot["p_delta_le_0"]:.4f}, n_pos={boot["n_pos_rounds"]}/1000')

# 2. 效率分析
print('\n' + '=' * 60)
print('2. 效率分析 (tge_cost_analysis.json)')
print('=' * 60)
with open(r'C:\Users\张三\Desktop\实验数据\tge_cost_analysis.json', 'r') as f:
    cost_data = json.load(f)
print(json.dumps(cost_data, indent=2, ensure_ascii=False))

# 3. 预过滤阈值敏感性
print('\n' + '=' * 60)
print('3. 预过滤阈值敏感性 (tge_probe_prefilter.json)')
print('=' * 60)
with open(r'C:\Users\张三\Desktop\实验数据\tge_probe_prefilter.json', 'r') as f:
    prefilter_data = json.load(f)
print(json.dumps(prefilter_data, indent=2, ensure_ascii=False)[:1500])

# 4. 源域样本量敏感性
print('\n' + '=' * 60)
print('4. 源域样本量敏感性 (tge_probe_srcsize.json)')
print('=' * 60)
with open(r'C:\Users\张三\Desktop\实验数据\tge_probe_srcsize.json', 'r') as f:
    srcsize_data = json.load(f)
print(json.dumps(srcsize_data, indent=2, ensure_ascii=False)[:1500])

# 5. NWPU Bootstrap
print('\n' + '=' * 60)
print('5. NWPU Bootstrap (tge_bootstrap_ci_nwpu.json)')
print('=' * 60)
with open(r'C:\Users\张三\Desktop\实验数据\tge_bootstrap_ci_nwpu.json', 'r') as f:
    boot_nwpu_data = json.load(f)
print(f'6类均值:')
for k, v in boot_nwpu_data['mean6'].items():
    print(f'  {k}: {v}')
