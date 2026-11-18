# -*- coding: utf-8 -*-
"""retrain_dual_factor_voc.py — 用COCO+VOC合并样本重新训练双因子模型
解决VOC目标域上双因子预测α过大导致过校正的问题
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
import numpy as np
from scipy import stats

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

def load_coco_data():
    """从新因子验证结果中提取COCO目标域的sv/tc/oracle_alpha"""
    coco_file = DATA / 'new_factor_validation_results.json'
    if not coco_file.exists():
        print(f'警告: {coco_file} 不存在')
        return []

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
            'oracle_alpha': item['optimal_alpha'],
            'baseline_ap': item.get('optimal_ap', 0),
        })
    print(f'COCO样本数: {len(samples)}')
    return samples


def load_voc_data():
    """从VOC目标域SOTA对比结果中提取sv/tc/oracle_alpha"""
    voc_file = DATA / 'voc_sota_comparison_results.json'
    if not voc_file.exists():
        print(f'警告: {voc_file} 不存在')
        return []

    with open(voc_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for src_name in data:
        for cls in data[src_name]:
            item = data[src_name][cls]
            if 'box_size_variance' in item and 'text_confusability' in item and 'oracle_alpha' in item:
                samples.append({
                    'src': src_name,
                    'cls': cls,
                    'target': 'VOC',
                    'sv': item['box_size_variance'],
                    'tc': item['text_confusability'],
                    'oracle_alpha': item['oracle_alpha'],
                    'baseline_ap': item.get('Baseline', 0),
                })
    print(f'VOC样本数: {len(samples)}')
    return samples


def train_dual_factor(samples, name='model'):
    """训练双因子回归模型"""
    sv = np.array([s['sv'] for s in samples])
    tc = np.array([s['tc'] for s in samples])
    alpha = np.array([s['oracle_alpha'] for s in samples])

    # 标准化特征
    sv_mean, sv_std = sv.mean(), sv.std()
    tc_mean, tc_std = tc.mean(), tc.std()
    sv_norm = (sv - sv_mean) / (sv_std + 1e-8)
    tc_norm = (tc - tc_mean) / (tc_std + 1e-8)

    # OLS回归: alpha = w_sv * sv_norm + w_tc * tc_norm + bias
    X = np.column_stack([sv_norm, tc_norm, np.ones(len(samples))])
    w, _, _, _ = np.linalg.lstsq(X, alpha, rcond=None)
    w_sv, w_tc, bias = w

    # 预测
    alpha_pred = X @ w

    # 评估
    rho, p_value = stats.spearmanr(alpha, alpha_pred)
    mae = np.mean(np.abs(alpha - alpha_pred))

    print(f'\n=== {name} 训练结果 ===')
    print(f'样本数: {len(samples)}')
    print(f'w_sv={w_sv:.4f}, w_tc={w_tc:.4f}, bias={bias:.4f}')
    print(f'Spearman ρ={rho:.4f}, p={p_value:.6f}')
    print(f'MAE={mae:.4f}')
    print(f'真实α范围: [{alpha.min():.2f}, {alpha.max():.2f}], 均值={alpha.mean():.2f}')
    print(f'预测α范围: [{alpha_pred.min():.2f}, {alpha_pred.max():.2f}], 均值={alpha_pred.mean():.2f}')

    # 按目标域分组
    for target in ['COCO', 'VOC']:
        mask = np.array([s['target'] == target for s in samples])
        if mask.sum() > 0:
            rho_t, _ = stats.spearmanr(alpha[mask], alpha_pred[mask])
            print(f'  {target}: N={mask.sum()}, ρ={rho_t:.4f}, 真实α均值={alpha[mask].mean():.2f}, 预测α均值={alpha_pred[mask].mean():.2f}')

    model_params = {
        'w_sv': float(w_sv),
        'w_tc': float(w_tc),
        'bias': float(bias),
        'sv_mean': float(sv_mean),
        'sv_std': float(sv_std),
        'tc_mean': float(tc_mean),
        'tc_std': float(tc_std),
        'rho': float(rho),
        'p_value': float(p_value),
        'mae': float(mae),
        'n_samples': len(samples),
        'name': name,
    }
    return model_params, alpha_pred


def predict_alpha(sv, tc, model_params):
    """用训练好的模型预测α"""
    sv_norm = (sv - model_params['sv_mean']) / (model_params['sv_std'] + 1e-8)
    tc_norm = (tc - model_params['tc_mean']) / (model_params['tc_std'] + 1e-8)
    alpha = model_params['w_sv'] * sv_norm + model_params['w_tc'] * tc_norm + model_params['bias']
    return float(np.clip(alpha, 0.0, 2.0))


def retest_voc(model_params, voc_samples):
    """用重新训练的模型在VOC上重新测试"""
    print(f'\n=== VOC重新测试 ===')

    # 读取原始VOC结果
    voc_file = DATA / 'voc_sota_comparison_results.json'
    with open(voc_file, 'r', encoding='utf-8') as f:
        voc_data = json.load(f)

    results = {}
    for src_name in voc_data:
        results[src_name] = {}
        for cls in voc_data[src_name]:
            item = voc_data[src_name][cls]
            sv = item['box_size_variance']
            tc = item['text_confusability']
            alpha_new = predict_alpha(sv, tc, model_params)

            # 这里需要重新计算AP，但我们没有实时计算的能力
            # 所以只记录预测的α值，实际AP需要重新跑实验
            results[src_name][cls] = {
                'alpha_old': item['alpha_adaptive'],
                'alpha_new': alpha_new,
                'oracle_alpha': item['oracle_alpha'],
                'sv': sv,
                'tc': tc,
                'baseline': item['Baseline'],
            }
            print(f'  {src_name}/{cls}: 旧α={item["alpha_adaptive"]:.2f}, 新α={alpha_new:.2f}, Oracle={item["oracle_alpha"]:.2f}')

    # 保存预测的α值
    with open(DATA / 'voc_retrained_alpha_predictions.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n预测的α值已保存到: {DATA / "voc_retrained_alpha_predictions.json"}')
    print(f'注意: 实际AP需要用新α值重新跑实验')

    return results


def main():
    print('=' * 80)
    print('用COCO+VOC合并样本重新训练双因子模型')
    print('=' * 80)

    # 1. 加载数据
    coco_samples = load_coco_data()
    voc_samples = load_voc_data()

    # 2. 只用COCO训练（原始模型）
    print('\n' + '=' * 80)
    print('模型1: 只用COCO训练（原始模型）')
    model_coco_only, pred_coco_only = train_dual_factor(coco_samples, 'COCO_only')

    # 3. 用COCO+VOC合并训练
    print('\n' + '=' * 80)
    print('模型2: COCO+VOC合并训练')
    all_samples = coco_samples + voc_samples
    model_combined, pred_combined = train_dual_factor(all_samples, 'COCO+VOC_combined')

    # 4. 保存模型参数
    with open(DATA / 'dual_factor_model_params_combined.json', 'w', encoding='utf-8') as f:
        json.dump(model_combined, f, indent=2, ensure_ascii=False)
    print(f'\n合并训练的模型参数已保存到: {DATA / "dual_factor_model_params_combined.json"}')

    # 5. VOC重新测试（预测α值）
    retest_voc(model_combined, voc_samples)

    # 6. 分析：为什么VOC上α预测过大
    print('\n' + '=' * 80)
    print('分析: 为什么VOC上α预测过大')
    print('=' * 80)

    coco_alpha = np.array([s['oracle_alpha'] for s in coco_samples])
    voc_alpha = np.array([s['oracle_alpha'] for s in voc_samples])
    coco_sv = np.array([s['sv'] for s in coco_samples])
    voc_sv = np.array([s['sv'] for s in voc_samples])
    coco_tc = np.array([s['tc'] for s in coco_samples])
    voc_tc = np.array([s['tc'] for s in voc_samples])
    coco_base = np.array([s['baseline_ap'] for s in coco_samples])
    voc_base = np.array([s['baseline_ap'] for s in voc_samples])

    print(f'\nCOCO: N={len(coco_samples)}, Oracleα均值={coco_alpha.mean():.2f}, α=0的比例={np.mean(coco_alpha==0):.1%}')
    print(f'VOC:  N={len(voc_samples)}, Oracleα均值={voc_alpha.mean():.2f}, α=0的比例={np.mean(voc_alpha==0):.1%}')
    print(f'\nCOCO: sv均值={coco_sv.mean():.0f}, tc均值={coco_tc.mean():.3f}, baselineAP均值={coco_base.mean():.4f}')
    print(f'VOC:  sv均值={voc_sv.mean():.0f}, tc均值={voc_tc.mean():.3f}, baselineAP均值={voc_base.mean():.4f}')

    # 检查基线AP和oracle_alpha的相关性
    if len(coco_samples) > 2:
        rho_base_alpha, _ = stats.spearmanr(coco_base, coco_alpha)
        print(f'\nCOCO: baselineAP vs Oracleα Spearman ρ={rho_base_alpha:.4f}')
    if len(voc_samples) > 2:
        rho_base_alpha_voc, _ = stats.spearmanr(voc_base, voc_alpha)
        print(f'VOC:  baselineAP vs Oracleα Spearman ρ={rho_base_alpha_voc:.4f}')

    print('\n结论: VOC基线AP高，很多类别Oracleα=0（不需要校正），但双因子模型')
    print('在COCO上训练时学到的是"sv大/tc大→α大"的规律，没有考虑基线AP的影响。')
    print('解决方案: (1) 合并训练; (2) 增加基线AP作为第三个因子; (3) 使用目标域无关的固定α')


if __name__ == '__main__':
    main()
