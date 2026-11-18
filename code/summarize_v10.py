import json
r = json.load(open(r'C:\Users\张三\Desktop\实验数据\usc_experiment_v10_results.json', 'r', encoding='utf-8'))

print('=' * 60)
print('因子相关性分析（合并所有类别，n=18）')
print('=' * 60)
for factor, corr in r['factor_correlations'].items():
    sig = '✅显著' if corr['p'] < 0.05 else '❌不显著'
    print(f'  {factor} vs best_alpha: ρ={corr["rho"]:+.4f}, p={corr["p"]:.4f} {sig}')

print()
print('=' * 60)
print('LOO验证汇总（三个域对的平均）')
print('=' * 60)
summary = r['summary']
config_labels = {
    'alpha_0': 'α=0（无USC）',
    'fixed_best': '固定最佳α',
    'G_only_linear': 'G-only线性映射',
    'linear_multi': '多因子线性映射',
    'sigmoid_multi': '多因子sigmoid映射',
    'oracle': 'Oracle逐类最佳α',
}
for cfg, label in config_labels.items():
    print(f'  {label}: {summary[cfg]:.4f}')

print()
print('=' * 60)
print('各留出域对的详细结果')
print('=' * 60)
for source, res in r['loo_results'].items():
    print(f'\n  留出 {source}（固定最佳α={res["best_fixed_alpha"]}）:')
    for cfg, label in config_labels.items():
        print(f'    {label}: {res["configs"][cfg]:.4f}')
    w = res['linear_weights']
    print(f'    线性权重: G={w["G"]:.4f}, baseAP={w["baseline_ap"]:.4f}, D={w["D"]:.4f}, bias={w["bias"]:.4f}')

print()
print('=' * 60)
print('关键对比')
print('=' * 60)
fixed = summary['fixed_best']
linear = summary['linear_multi']
sigmoid = summary['sigmoid_multi']
oracle = summary['oracle']
g_only = summary['G_only_linear']
print(f'  固定最佳α: {fixed:.4f}')
print(f'  G-only线性: {g_only:.4f} (vs固定: {g_only-fixed:+.4f})')
print(f'  多因子线性: {linear:.4f} (vs固定: {linear-fixed:+.4f})')
print(f'  多因子sigmoid: {sigmoid:.4f} (vs固定: {sigmoid-fixed:+.4f})')
print(f'  Oracle逐类: {oracle:.4f} (vs固定: {oracle-fixed:+.4f})')
print(f'  多因子线性达到oracle: {linear/oracle*100:.1f}%')
print(f'  多因子sigmoid达到oracle: {sigmoid/oracle*100:.1f}%')
