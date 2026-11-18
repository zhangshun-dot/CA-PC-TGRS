import json
r = json.load(open(r'C:\Users\张三\Desktop\实验数据\usc_experiment_v8_taskc_results.json', 'r', encoding='utf-8'))
print('=' * 100)
print('任务C：完整门控模块测试（4个域对）')
print('=' * 100)
print(f'{"源域":<8} {"类别":<5} {"域偏移":<8} {"偏移门":<6} {"g2>0":<6} {"基线":<8} {"TGE":<8} {"USC+TGE":<10} {"完整门控":<10} {"门控增益"}')
print('-' * 100)
for k, v in r.items():
    cfg = v['configs']
    gate_gain = cfg['full_gating']['gain_over_usc']
    print(f'{k:<8} {v["n_classes"]:<5} {v["domain_shift_norm"]:<8.4f} '
          f'{"启用" if v["domain_shift_enabled"] else "关闭":<6} '
          f'{v["g2_enabled_count"]}/{v["g2_enabled_count"]+v["g2_disabled_count"]:<6} '
          f'{cfg["baseline"]["mean"]:<8.4f} {cfg["tge_only"]["mean"]:<8.4f} '
          f'{cfg["usc_tge"]["mean"]:<10.4f} {cfg["full_gating"]["mean"]:<10.4f} '
          f'{gate_gain:+.4f}')

print()
print('各域对门控决策详情:')
for k, v in r.items():
    print(f'  {k}:')
    print(f'    启用USC: {v["enabled_classes"]}')
    print(f'    关闭USC: {v["disabled_classes"]}')
    g2_vals = list(v['g2_values'].values())
    print(f'    g2范围: [{min(g2_vals):.4f}, {max(g2_vals):.4f}]')
