import json

# 读取文件，只解析任务B部分（到task_c_full_gating之前）
with open(r'C:\Users\张三\Desktop\实验数据\usc_experiment_v8_results.json', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到task_c_full_gating的位置，截断
idx = content.find('"task_c_full_gating"')
if idx > 0:
    # 去掉最后一个逗号和后面的内容
    task_b_content = content[:idx].rstrip().rstrip(',') + '}'
    # 需要补上外层括号
    task_b_content = '{' + task_b_content
    tb = json.loads(task_b_content)['task_b_physical_priors_20class']
else:
    print('未找到task_c_full_gating')
    exit()

print('=' * 60)
print('任务B：物理先验因子20类验证（VOC→COCO）')
print('=' * 60)
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
print('各因子取值范围:')
for factor in ['G', 'S', 'I', 'C', 'D']:
    vals = [v[factor] for v in tb['factors'].values()]
    print(f'  {factor}: [{min(vals):.4f}, {max(vals):.4f}], mean={sum(vals)/len(vals):.4f}')
