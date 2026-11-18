import json
with open(r'C:\Users\张三\Desktop\实验数据\usc_experiment_v8_results.json', 'r', encoding='utf-8') as f:
    r = json.load(f)

print('=' * 60)
print('任务B：物理先验因子20类验证（VOC→COCO）')
print('=' * 60)
tb = r['task_b_physical_priors_20class']
print(f'有效类别数: {tb["n_classes"]}')
print(f'域偏移范数: {tb["domain_shift_norm"]:.4f}')
print()
print('各因子 vs oracle最佳α 相关性:')
for factor, corr in tb['correlations'].items():
    sig = '✅显著' if corr['p'] < 0.05 else '❌不显著'
    print(f'  {factor}: ρ={corr["rho"]:+.3f}, p={corr["p"]:.3f} {sig}')
print(f'  等权G/S/I/C: ρ={tb["correlation_gsic"]["rho"]:+.3f}, p={tb["correlation_gsic"]["p"]:.3f}')
print(f'  域偏移投影D: ρ={tb["correlation_d"]["rho"]:+.3f}, p={tb["correlation_d"]["p"]:.3f}')

print()
print('oracle最佳α分布:')
alpha_counts = {}
for cls, v in tb['oracle_best'].items():
    a = v['best_alpha']
    alpha_counts[a] = alpha_counts.get(a, 0) + 1
for a in sorted(alpha_counts.keys()):
    print(f'  α={a}: {alpha_counts[a]}类')

print()
print('=' * 60)
print('任务C：完整门控模块测试（4个域对）')
print('=' * 60)
tc = r['task_c_full_gating']
print(f'{"源域":<10} {"类别":<5} {"域偏移":<8} {"偏移门":<6} {"g2>0":<6} {"基线":<8} {"TGE":<8} {"USC+TGE":<10} {"完整门控":<10} {"门控增益"}')
print('-' * 100)
for source, res in tc.items():
    cfg = res['configs']
    gate_gain = cfg['full_gating']['gain_over_usc']
    print(f'{source:<10} {res["n_classes"]:<5} {res["domain_shift_norm"]:<8.4f} '
          f'{"启用" if res["domain_shift_enabled"] else "关闭":<6} '
          f'{res["g2_enabled_count"]}/{res["g2_enabled_count"]+res["g2_disabled_count"]:<6} '
          f'{cfg["baseline"]["mean"]:<8.4f} {cfg["tge_only"]["mean"]:<8.4f} '
          f'{cfg["usc_tge"]["mean"]:<10.4f} {cfg["full_gating"]["mean"]:<10.4f} '
          f'{gate_gain:+.4f}')

print()
print('各域对门控决策详情:')
for source, res in tc.items():
    print(f'  {source}:')
    print(f'    启用USC: {res["enabled_classes"]}')
    print(f'    关闭USC: {res["disabled_classes"]}')
    print(f'    g2范围: [{min(res["g2_values"].values()):.4f}, {max(res["g2_values"].values()):.4f}]')
