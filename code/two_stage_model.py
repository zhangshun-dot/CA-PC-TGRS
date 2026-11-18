# -*- coding: utf-8 -*-
"""two_stage_model.py — 两阶段模型：先判断是否需要校正，再预测校正强度
阶段1：二分类（α=0 vs α>0）
阶段2：回归（只对α>0的类别预测具体值）
LODO验证：COCO训练，VOC测试
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats

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
                'n_boxes': None,
                'oracle_alpha': item['oracle_alpha'],
            })
    return samples


def estimate_n_boxes_for_voc(voc_samples, coco_samples):
    """为VOC样本估算n_boxes"""
    coco_valid = [s for s in coco_samples if s['n_boxes'] is not None and s['n_boxes'] > 0]
    X_train = np.column_stack([[s['sv'] for s in coco_valid], [s['tc'] for s in coco_valid]])
    y_train = np.array([s['n_boxes'] for s in coco_valid])
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_train_norm = np.column_stack([X_train_norm, np.ones(len(coco_valid))])
    w, _, _, _ = np.linalg.lstsq(X_train_norm, y_train, rcond=None)

    for s in voc_samples:
        X_test = np.array([[s['sv'], s['tc']]])
        X_test_norm = (X_test - X_mean) / X_std
        X_test_norm = np.column_stack([X_test_norm, np.ones(1)])
        s['n_boxes'] = float(np.clip(X_test_norm @ w, 100, 10000)[0])

    return voc_samples


def sigmoid(x):
    """sigmoid函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def train_logistic_regression(X, y, lr=0.1, n_iter=1000):
    """训练逻辑回归（梯度下降）"""
    n_samples, n_features = X.shape
    X_bias = np.column_stack([X, np.ones(n_samples)])
    w = np.zeros(n_features + 1)

    for _ in range(n_iter):
        z = X_bias @ w
        y_pred = sigmoid(z)
        gradient = X_bias.T @ (y_pred - y) / n_samples
        w -= lr * gradient

    return w


def predict_logistic_regression(X, w):
    """逻辑回归预测"""
    n_samples = X.shape[0]
    X_bias = np.column_stack([X, np.ones(n_samples)])
    z = X_bias @ w
    return sigmoid(z)


def train_linear_regression(X, y):
    """训练线性回归（OLS）"""
    n_samples = X.shape[0]
    X_bias = np.column_stack([X, np.ones(n_samples)])
    w, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
    return w


def predict_linear_regression(X, w):
    """线性回归预测"""
    n_samples = X.shape[0]
    X_bias = np.column_stack([X, np.ones(n_samples)])
    return X_bias @ w


def two_stage_model(train_samples, test_samples, factors, threshold=0.5):
    """两阶段模型：先分类（是否需要校正），再回归（校正强度）"""
    # 过滤有效样本
    train_valid = [s for s in train_samples if all(f in s and s[f] is not None for f in factors)]
    test_valid = [s for s in test_samples if all(f in s and s[f] is not None for f in factors)]

    if len(train_valid) < 5 or len(test_valid) < 3:
        return None

    # 准备特征
    X_train = np.column_stack([[s[f] for s in train_valid] for f in factors])
    y_train = np.array([s['oracle_alpha'] for s in train_valid])
    X_test = np.column_stack([[s[f] for s in test_valid] for f in factors])
    y_test = np.array([s['oracle_alpha'] for s in test_valid])

    # 标准化（用训练集的均值和标准差）
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # 阶段1：二分类（α=0 vs α>0）
    y_train_binary = (y_train > 0.05).astype(int)  # α>0.05视为需要校正
    y_test_binary = (y_test > 0.05).astype(int)

    w_clf = train_logistic_regression(X_train_norm, y_train_binary, lr=0.5, n_iter=2000)
    prob_train = predict_logistic_regression(X_train_norm, w_clf)
    prob_test = predict_logistic_regression(X_test_norm, w_clf)
    pred_binary_train = (prob_train > threshold).astype(int)
    pred_binary_test = (prob_test > threshold).astype(int)

    # 分类准确率
    acc_train = np.mean(pred_binary_train == y_train_binary)
    acc_test = np.mean(pred_binary_test == y_test_binary)

    # 阶段2：回归（只对α>0的训练样本）
    train_positive_idx = y_train > 0.05
    if train_positive_idx.sum() < 3:
        # 如果正样本太少，用所有样本训练回归
        w_reg = train_linear_regression(X_train_norm, y_train)
    else:
        w_reg = train_linear_regression(X_train_norm[train_positive_idx], y_train[train_positive_idx])

    # 预测
    y_train_pred_reg = predict_linear_regression(X_train_norm, w_reg)
    y_test_pred_reg = predict_linear_regression(X_test_norm, w_reg)

    # 两阶段结合：分类为"不需要校正"的设为0，否则用回归预测值
    y_train_pred = np.where(pred_binary_train == 1, np.clip(y_train_pred_reg, 0, 2), 0)
    y_test_pred = np.where(pred_binary_test == 1, np.clip(y_test_pred_reg, 0, 2), 0)

    # 评估
    rho_train, p_train = stats.spearmanr(y_train, y_train_pred)
    rho_test, p_test = stats.spearmanr(y_test, y_test_pred)
    mae_train = np.mean(np.abs(y_train - y_train_pred))
    mae_test = np.mean(np.abs(y_test - y_test_pred))

    # 逐类预测
    predictions = []
    for i, s in enumerate(test_valid):
        predictions.append({
            'src': s['src'],
            'cls': s['cls'],
            'true_alpha': y_test[i],
            'pred_alpha': y_test_pred[i],
            'prob_need_correction': prob_test[i],
            'pred_binary': pred_binary_test[i],
            'true_binary': y_test_binary[i],
            'error': y_test_pred[i] - y_test[i],
        })

    return {
        'factors': factors,
        'threshold': threshold,
        'n_train': len(train_valid),
        'n_test': len(test_valid),
        'n_train_positive': int(train_positive_idx.sum()),
        'n_test_positive': int(y_test_binary.sum()),
        'clf_acc_train': acc_train,
        'clf_acc_test': acc_test,
        'rho_train': rho_train,
        'p_train': p_train,
        'mae_train': mae_train,
        'rho_test': rho_test,
        'p_test': p_test,
        'mae_test': mae_test,
        'w_clf': w_clf,
        'w_reg': w_reg,
        'X_mean': X_mean,
        'X_std': X_std,
        'predictions': predictions,
    }


def single_stage_model(train_samples, test_samples, factors):
    """单阶段回归模型（作为对比）"""
    train_valid = [s for s in train_samples if all(f in s and s[f] is not None for f in factors)]
    test_valid = [s for s in test_samples if all(f in s and s[f] is not None for f in factors)]

    if len(train_valid) < 5 or len(test_valid) < 3:
        return None

    X_train = np.column_stack([[s[f] for s in train_valid] for f in factors])
    y_train = np.array([s['oracle_alpha'] for s in train_valid])
    X_test = np.column_stack([[s[f] for s in test_valid] for f in factors])
    y_test = np.array([s['oracle_alpha'] for s in test_valid])

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    w = train_linear_regression(X_train_norm, y_train)
    y_train_pred = np.clip(predict_linear_regression(X_train_norm, w), 0, 2)
    y_test_pred = np.clip(predict_linear_regression(X_test_norm, w), 0, 2)

    rho_train, _ = stats.spearmanr(y_train, y_train_pred)
    rho_test, p_test = stats.spearmanr(y_test, y_test_pred)
    mae_train = np.mean(np.abs(y_train - y_train_pred))
    mae_test = np.mean(np.abs(y_test - y_test_pred))

    return {
        'factors': factors,
        'rho_train': rho_train,
        'rho_test': rho_test,
        'p_test': p_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
    }


def main():
    print('=' * 80)
    print('两阶段模型 vs 单阶段模型：LODO跨目标域泛化对比')
    print('COCO训练，VOC测试')
    print('=' * 80)

    # 加载数据
    coco_samples = load_coco_data()
    voc_samples = load_voc_data()
    voc_samples = estimate_n_boxes_for_voc(voc_samples, coco_samples)

    print(f'\nCOCO样本数: {len(coco_samples)}')
    print(f'VOC样本数: {len(voc_samples)}')
    print(f'COCO中α>0的样本数: {sum(1 for s in coco_samples if s["oracle_alpha"] > 0.05)}')
    print(f'VOC中α>0的样本数: {sum(1 for s in voc_samples if s["oracle_alpha"] > 0.05)}')

    # 要测试的因子组合
    factor_combos = [
        ('sv',),
        ('tc',),
        ('n_boxes',),
        ('sv', 'tc'),
        ('sv', 'n_boxes'),
        ('tc', 'n_boxes'),
        ('sv', 'tc', 'n_boxes'),
    ]
    factor_names = {
        'sv': '候选框尺寸方差',
        'tc': '文本混淆度',
        'n_boxes': '候选框数量',
    }

    # 测试不同的分类阈值
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print(f'\n{"=" * 100}')
    print('单阶段模型（对比）')
    print(f'{"=" * 100}')
    print(f'{"因子组合":<35s} {"ρ_train":>9s} {"ρ_test":>9s} {"MAE_test":>9s} {"p_test":>8s}')
    print('-' * 80)

    single_results = []
    for combo in factor_combos:
        result = single_stage_model(coco_samples, voc_samples, combo)
        if result:
            combo_name = ' + '.join([factor_names[f] for f in combo])
            sig = '***' if result['p_test'] < 0.001 else ('**' if result['p_test'] < 0.01 else ('*' if result['p_test'] < 0.05 else ''))
            print(f'{combo_name:<35s} {result["rho_train"]:>9.4f} {result["rho_test"]:>9.4f} {result["mae_test"]:>9.4f} {sig:>8s}')
            single_results.append(result)

    print(f'\n{"=" * 100}')
    print('两阶段模型')
    print(f'{"=" * 100}')
    print(f'{"因子组合":<30s} {"阈值":>5s} {"分类准确率":>10s} {"ρ_train":>9s} {"ρ_test":>9s} {"MAE_test":>9s} {"p_test":>8s}')
    print('-' * 100)

    two_stage_results = []
    for combo in factor_combos:
        for threshold in thresholds:
            result = two_stage_model(coco_samples, voc_samples, combo, threshold)
            if result:
                combo_name = ' + '.join([factor_names[f] for f in combo])
                sig = '***' if result['p_test'] < 0.001 else ('**' if result['p_test'] < 0.01 else ('*' if result['p_test'] < 0.05 else ''))
                print(f'{combo_name:<30s} {threshold:>5.1f} {result["clf_acc_test"]:>10.2%} {result["rho_train"]:>9.4f} {result["rho_test"]:>9.4f} {result["mae_test"]:>9.4f} {sig:>8s}')
                two_stage_results.append(result)

    # 按MAE_test排序（越低越好）
    two_stage_results.sort(key=lambda x: x['mae_test'])

    print(f'\n{"=" * 100}')
    print('按VOC测试MAE排序（越低越好）')
    print(f'{"=" * 100}')
    for i, r in enumerate(two_stage_results[:10]):
        combo_name = ' + '.join([factor_names[f] for f in r['factors']])
        print(f'{i+1}. {combo_name:<30s} 阈值={r["threshold"]:.1f} 分类准确率={r["clf_acc_test"]:.2%} ρ_test={r["rho_test"]:.4f} MAE={r["mae_test"]:.4f}')

    # 最佳两阶段模型的详细结果
    if two_stage_results:
        best = two_stage_results[0]
        print(f'\n{"=" * 100}')
        print(f'最佳两阶段模型: {" + ".join([factor_names[f] for f in best["factors"]])}, 阈值={best["threshold"]}')
        print(f'{"=" * 100}')
        print(f'COCO训练: N={best["n_train"]}, 正样本={best["n_train_positive"]}, 分类准确率={best["clf_acc_train"]:.2%}')
        print(f'         ρ={best["rho_train"]:.4f}, MAE={best["mae_train"]:.4f}')
        print(f'VOC测试:  N={best["n_test"]}, 正样本={best["n_test_positive"]}, 分类准确率={best["clf_acc_test"]:.2%}')
        print(f'         ρ={best["rho_test"]:.4f} (p={best["p_test"]:.4f}), MAE={best["mae_test"]:.4f}')
        print(f'\nVOC逐类预测:')
        for p in best['predictions']:
            binary_str = '需要校正' if p['pred_binary'] == 1 else '不需要校正'
            true_binary_str = '需要' if p['true_binary'] == 1 else '不需要'
            print(f'  {p["src"]}/{p["cls"]}: 真实α={p["true_alpha"]:.2f}({true_binary_str}), 预测α={p["pred_alpha"]:.2f}({binary_str}), 概率={p["prob_need_correction"]:.2f}, 误差={p["error"]:+.2f}')

        # 保存最佳模型
        best_model = {
            'factors': list(best['factors']),
            'threshold': best['threshold'],
            'w_clf': best['w_clf'].tolist(),
            'w_reg': best['w_reg'].tolist(),
            'X_mean': best['X_mean'].tolist(),
            'X_std': best['X_std'].tolist(),
            'clf_acc_test': float(best['clf_acc_test']),
            'rho_test': float(best['rho_test']),
            'mae_test': float(best['mae_test']),
        }
        with open(DATA / 'best_two_stage_model.json', 'w', encoding='utf-8') as f:
            json.dump(best_model, f, indent=2, ensure_ascii=False)
        print(f'\n最佳两阶段模型已保存到: {DATA / "best_two_stage_model.json"}')

    # 对比：最佳两阶段 vs 最佳单阶段
    print(f'\n{"=" * 100}')
    print('对比：最佳两阶段模型 vs 最佳单阶段模型')
    print(f'{"=" * 100}')
    if two_stage_results and single_results:
        best_two = two_stage_results[0]
        best_single = min(single_results, key=lambda x: x['mae_test'])
        print(f'最佳两阶段: ρ_test={best_two["rho_test"]:.4f}, MAE={best_two["mae_test"]:.4f}, 分类准确率={best_two["clf_acc_test"]:.2%}')
        print(f'最佳单阶段: ρ_test={best_single["rho_test"]:.4f}, MAE={best_single["mae_test"]:.4f}')
        print(f'MAE提升: {best_single["mae_test"] - best_two["mae_test"]:+.4f} ({(best_single["mae_test"] - best_two["mae_test"]) / best_single["mae_test"] * 100:+.1f}%)')


if __name__ == '__main__':
    main()
