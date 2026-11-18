import json
r = json.load(open(r'C:\Users\张三\Desktop\实验数据\usc_experiment_v11_results.json', 'r', encoding='utf-8'))

print('=' * 60)
print('所有因子相关性分析（合并所有类别，n=18）')
print('=' * 60)
for factor, corr in r['factor_correlations'].items():
    sig = '✅显著' if corr['p'] < 0.05 else '❌'
    print(f'  {factor}: ρ={corr["rho"]:+.4f}, p={corr["p"]:.4f} {sig}')

print()
print('=' * 60)
print('因子组合定义')
print('=' * 60)
for combo, factors in r['factor_combos_used'].items():
    print(f'  {combo}: {factors}')

print()
print('=' * 60)
print('LOO汇总（三个域对的平均）')
print('=' * 60)
summary = r['summary']
fixed = summary['fixed_best']
oracle = summary['oracle']
print(f'  固定最佳α: {fixed:.4f}')
print(f'  Oracle逐类: {oracle:.4f} (vs固定: +{oracle-fixed:.4f})')
print()
for combo in r['factor_combos_used']:
    if combo in summary:
        ap = summary[combo]
        print(f'  {combo}: {ap:.4f} (vs固定: {ap-fixed:+.4f}, 达oracle: {ap/oracle*100:.1f}%)')

print()
print('=' * 60)
print('各留出域对的详细结果')
print('=' * 60)
for source in r['fixed_results']:
    print(f'\n  留出 {source}:')
    print(f'    固定最佳α: {r["fixed_results"][source]:.4f}')
    print(f'    Oracle: {r["oracle_results"][source]:.4f}')
    for combo in r['factor_combos_used']:
        if source in r['loo_results'][combo]:
            ap = r['loo_results'][combo][source]
            print(f'    {combo}: {ap:.4f} (vs固定: {ap-r["fixed_results"][source]:+.4f})')
