import json
r = json.load(open(r'C:\Users\张三\Desktop\实验数据\usc_experiment_v12_results.json', 'r', encoding='utf-8'))

print(f'域对数: {r["n_domains"]}, 类别数: {r["n_classes"]}')
print()
print('=' * 60)
print('因子相关性分析')
print('=' * 60)
for factor, corr in r['factor_correlations'].items():
    sig = '✅显著' if corr['p'] < 0.05 else '❌'
    print(f'  {factor}: ρ={corr["rho"]:+.4f}, p={corr["p"]:.4f} {sig}')

print()
print('=' * 60)
print('LOO汇总（4个域对的平均）')
print('=' * 60)
s = r['summary']
print(f'  α=0: {s["alpha_0"]:.4f}')
print(f'  固定最佳α: {s["fixed_best"]:.4f}')
print(f'  det_stats线性: {s["det_stats_linear"]:.4f} (vs固定: {s["det_stats_linear"]-s["fixed_best"]:+.4f}, 达oracle: {s["det_stats_linear"]/s["oracle"]*100:.1f}%)')
print(f'  det_stats MLP: {s["det_stats_mlp"]:.4f} (vs固定: {s["det_stats_mlp"]-s["fixed_best"]:+.4f}, 达oracle: {s["det_stats_mlp"]/s["oracle"]*100:.1f}%)')
print(f'  Oracle逐类: {s["oracle"]:.4f}')

print()
print('=' * 60)
print('各留出域对的详细结果')
print('=' * 60)
loo = r['loo_results']
for source in loo['fixed_best']:
    print(f'\n  留出 {source}:')
    print(f'    α=0: {loo["alpha_0"][source]:.4f}')
    print(f'    固定最佳α: {loo["fixed_best"][source]:.4f}')
    print(f'    det_stats线性: {loo["det_stats_linear"][source]:.4f} (vs固定: {loo["det_stats_linear"][source]-loo["fixed_best"][source]:+.4f})')
    print(f'    det_stats MLP: {loo["det_stats_mlp"][source]:.4f} (vs固定: {loo["det_stats_mlp"][source]-loo["fixed_best"][source]:+.4f})')
    print(f'    Oracle: {loo["oracle"][source]:.4f}')
