# -*- coding: utf-8 -*-
"""test_factor_combos.py — 测试不同因子组合的LODO跨目标域泛化能力
COCO训练，VOC测试，比较哪个因子组合的跨目标域泛化最好
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats
from itertools import combinations

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

def load_coco_data():
    """从新因子验证结果中加载COCO目标域数据"""
    coco_file = DATA / 'new_factor_validation_results.json'
    with open(coco_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data.get('all_factor_data', []):
        samples.append({
            'src': item.get('domain', 'unknown'),
            'cls': item.get('src_cls', 'unknown'),
            'target': 'COCO',
            'sv': item['box_size_variance'],
            'tc': item['text_confusability'],
            'n_boxes': item.get('n_boxes', 0),
            'oracle_alpha': item['optimal_alpha'],
        })
    return samples


def load_voc_data():
    """从VOC目标域SOTA对比结果中加载VOC目标域数据"""
    voc_file = DATA / 'voc_sota_comparison_results.json'
    with open(voc_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for src_name in data:
        for cls in data[src_name]:
            item = data[src_name][cls]
            samples.append({
                'src': src_name,
                'cls': cls,
                'target': 'VOC',
                'sv': item['box_size_variance'],
                'tc': item['text_confusability'],
                'n_boxes': None,  # VOC样本没有n_boxes，需要估算
                'oracle_alpha': item['oracle_alpha'],
            })
    return samples


def train_and_test(train_samples, test_samples, factors):
    """用指定因子训练，在测试集上测试"""
    # 过滤有效样本
    train_valid = [s for s in train_samples if all(f in s and s[f] is not None for f in factors)]
    test_valid = [s for s in test_samples if all(f in s and s[f] is not None for f in factors)]

    if len(train_valid) < 5 or len(test_valid) < 3:
        return None

    # 训练集
    X_train = np.column_stack([[s[f] for s in train_valid] for f in factors])
    y_train = np.array([s['oracle_alpha'] for s in train_valid])

    # 标准化（用训练集的均值和标准差）
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_train_norm = np.column_stack([X_train_norm, np.ones(len(train_valid))])

    # OLS回归
    w, _, _, _ = np.linalg.lstsq(X_train_norm, y_train, rcond=None)
    y_train_pred = X_train_norm @ w
    y_train_pred = np.clip(y_train_pred, 0, 2)

    rho_train, p_train = stats.spearmanr(y_train, y_train_pred)
    mae_train = np.mean(np.abs(y_train - y_train_pred))

    # 测试集
    X_test = np.column_stack([[s[f] for s in test_valid] for f in factors])
    y_test = np.array([s['oracle_alpha'] for s in test_valid])
    X_test_norm = (X_test - X_mean) / X_std
    X_test_norm = np.column_stack([X_test_norm, np.ones(len(test_valid))])
    y_test_pred = X_test_norm @ w
    y_test_pred = np.clip(y_test_pred, 0, 2)

    rho_test, p_test = stats.spearmanr(y_test, y_test_pred)
    mae_test = np.mean(np.abs(y_test - y_test_pred))

    # 逐类预测
    predictions = []
    for i, s in enumerate(test_valid):
        predictions.append({
            'src': s['src'],
            'cls': s['cls'],
            'true_alpha': y_test[i],
            'pred_alpha': y_test_pred[i],
            'error': y_test_pred[i] - y_test[i],
        })

    return {
        'factors': factors,
        'n_train': len(train_valid),
        'n_test': len(test_valid),
        'rho_train': rho_train,
        'p_train': p_train,
        'mae_train': mae_train,
        'rho_test': rho_test,
        'p_test': p_test,
        'mae_test': mae_test,
        'weights': w,
        'X_mean': X_mean,
        'X_std': X_std,
        'predictions': predictions,
    }


def estimate_n_boxes_for_voc(voc_samples, coco_samples):
    """为VOC样本估算n_boxes（用候选框尺寸方差和文本混淆度回归）"""
    # 用COCO样本训练n_boxes的回归模型
    coco_valid = [s for s in coco_samples if s['n_boxes'] is not None and s['n_boxes'] > 0]
    X_train = np.column_stack([[s['sv'] for s in coco_valid], [s['tc'] for s in coco_valid]])
    y_train = np.array([s['n_boxes'] for s in coco_valid])
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_train_norm = np.column_stack([X_train_norm, np.ones(len(coco_valid))])
    w, _, _, _ = np.linalg.lstsq(X_train_norm, y_train, rcond=None)

    # 为VOC样本估算n_boxes
    for s in voc_samples:
        X_test = np.array([[s['sv'], s['tc']]])
        X_test_norm = (X_test - X_mean) / X_std
        X_test_norm = np.column_stack([X_test_norm, np.ones(1)])
        s['n_boxes'] = float(np.clip(X_test_norm @ w, 100, 10000)[0])

    return voc_samples


def main():
    print('=' * 80)
    print('测试不同因子组合的LODO跨目标域泛化能力')
    print('COCO训练，VOC测试')
    print('=' * 80)

    # 加载数据
    coco_samples = load_coco_data()
    voc_samples = load_voc_data()

    print(f'\nCOCO样本数: {len(coco_samples)}')
    print(f'VOC样本数: {len(voc_samples)}')

    # 为VOC样本估算n_boxes
    voc_samples = estimate_n_boxes_for_voc(voc_samples, coco_samples)
    print(f'\n已为VOC样本估算n_boxes')

    # 要测试的因子组合
    all_factors = ['sv', 'tc', 'n_boxes']
    factor_names = {
        'sv': '候选框尺寸方差',
        'tc': '文本混淆度',
        'n_boxes': '候选框数量',
    }

    # 生成所有可能的因子组合（1-3个因子）
    combos = []
    for k in range(1, len(all_factors) + 1):
        for combo in combinations(all_factors, k):
            combos.append(combo)

    print(f'\n要测试的因子组合数: {len(combos)}')
    print(f'\n{"组合":<35s} {"N_train":>7s} {"N_test":>7s} {"ρ_train":>9s} {"ρ_test":>9s} {"MAE_test":>9s} {"测试显著":>8s}')
    print('-' * 95)

    results = []
    for combo in combos:
        result = train_and_test(coco_samples, voc_samples, combo)
        if result:
            combo_name = ' + '.join([factor_names[f] for f in combo])
            sig = '***' if result['p_test'] < 0.001 else ('**' if result['p_test'] < 0.01 else ('*' if result['p_test'] < 0.05 else ''))
            print(f'{combo_name:<35s} {result["n_train"]:>7d} {result["n_test"]:>7d} {result["rho_train"]:>9.4f} {result["rho_test"]:>9.4f} {result["mae_test"]:>9.4f} {sig:>8s}')
            results.append(result)

    # 按ρ_test排序
    results.sort(key=lambda x: x['rho_test'], reverse=True)

    print(f'\n{"=" * 80}')
    print('按VOC测试ρ排序（跨目标域泛化能力）')
    print(f'{"=" * 80}')
    for i, r in enumerate(results):
        combo_name = ' + '.join([factor_names[f] for f in r['factors']])
        sig = '***' if r['p_test'] < 0.001 else ('**' if r['p_test'] < 0.01 else ('*' if r['p_test'] < 0.05 else ''))
        print(f'{i+1}. {combo_name:<35s} ρ_test={r["rho_test"]:.4f} (p={r["p_test"]:.4f}) {sig}, MAE={r["mae_test"]:.4f}')

    # 最佳组合的详细结果
    if results:
        best = results[0]
        print(f'\n{"=" * 80}')
        print(f'最佳组合: {" + ".join([factor_names[f] for f in best["factors"]])}')
        print(f'{"=" * 80}')
        print(f'COCO训练: N={best["n_train"]}, ρ={best["rho_train"]:.4f}, MAE={best["mae_train"]:.4f}')
        print(f'VOC测试:  N={best["n_test"]}, ρ={best["rho_test"]:.4f}, MAE={best["mae_test"]:.4f}')
        print(f'权重: {best["weights"]}')
        print(f'\nVOC逐类预测:')
        for p in best['predictions']:
            print(f'  {p["src"]}/{p["cls"]}: 真实α={p["true_alpha"]:.2f}, 预测α={p["pred_alpha"]:.2f}, 误差={p["error"]:+.2f}')

        # 保存最佳模型参数
        best_model = {
            'factors': list(best['factors']),
            'weights': best['weights'].tolist(),
            'X_mean': best['X_mean'].tolist(),
            'X_std': best['X_std'].tolist(),
            'rho_train': float(best['rho_train']),
            'rho_test': float(best['rho_test']),
            'mae_test': float(best['mae_test']),
            'n_train': best['n_train'],
            'n_test': best['n_test'],
        }
        with open(DATA / 'best_factor_model_lodo.json', 'w', encoding='utf-8') as f:
            json.dump(best_model, f, indent=2, ensure_ascii=False)
        print(f'\n最佳模型参数已保存到: {DATA / "best_factor_model_lodo.json"}')

    # 对比：当前的双因子模型（sv + tc）
    print(f'\n{"=" * 80}')
    print('对比：当前双因子模型（sv + tc）vs 最佳组合')
    print(f'{"=" * 80}')
    for r in results:
        if r['factors'] == ('sv', 'tc'):
            print(f'当前双因子（sv+tc）: ρ_test={r["rho_test"]:.4f}, MAE={r["mae_test"]:.4f}')
        if r == results[0]:
            print(f'最佳组合: ρ_test={r["rho_test"]:.4f}, MAE={r["mae_test"]:.4f}')
            improvement = r['rho_test'] - [x['rho_test'] for x in results if x['factors'] == ('sv', 'tc')][0]
            print(f'提升: {improvement:+.4f}')


if __name__ == '__main__':
    main()
