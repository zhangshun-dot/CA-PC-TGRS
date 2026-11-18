import json
with open(r'C:\Users\张三\Desktop\实验数据\usc_experiment_v4_results.json', 'r', encoding='utf-8') as f:
    r = json.load(f)

print('=== 核心指标 ===')
print('有效类别数:', r['config']['n_classes'])
print(f'基线均值AP: {r["baseline_mean"]:.4f}')
print(f'TGE-only均值AP: {r["tge_only_mean"]:.4f} (gain={r["tge_only_gain"]:+.4f})')
print(f'Oracle均值AP: {r["oracle_mean"]:.4f} (gain={r["oracle_gain"]:+.4f})')
print('最佳固定α:', r['best_fixed_alpha'])
print(f'最佳固定α均值AP: {r["best_fixed_mean"]:.4f} (gain={r["best_fixed_gain"]:+.4f})')

print()
print('=== 逐类oracle最佳α ===')
for c, v in r['oracle_best'].items():
    bl = r['per_class_results'][c]['baseline']
    print(f'  {c:15s}: a={v["best_alpha"]:.1f}, baseline={bl:.4f}, oracleAP={v["best_ap"]:.4f}, gain={v["best_ap"]-bl:+.4f}')

print()
print('=== 固定α扫描 ===')
for alpha, v in sorted(r['fixed_alpha_results'].items(), key=lambda x: float(x[0])):
    print(f'  a={float(alpha):.1f}: AP={v["mean"]:.4f}, gain={v["gain"]:+.4f}')

print()
print('=== 低α诊断 ===')
for group, stats in r['low_alpha_diagnosis'].items():
    print(f'  {group}:')
    print('    类别:', stats['classes'])
    print(f'    平均oraclea: {stats["avg_oracle_alpha"]:.3f}')
    print(f'    平均基线AP: {stats["avg_baseline_ap"]:.4f}')
    print(f'    平均G: {stats["avg_G"]:.3f}, S: {stats["avg_S"]:.3f}, I: {stats["avg_I"]:.3f}, C: {stats["avg_C"]:.3f}')
    print(f'    平均域偏移投影: {stats["avg_domain_shift_proj"]:.4f}')
