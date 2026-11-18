# -*- coding: utf-8 -*-
"""x1_cross_operator_replication.py — X1跨算子复现实验
验证门控规则是否在3种完全不同的零训练校正算子上都成立。
如果成立，claim升格为"类级适用边界与具体算子无关"（L2，可冲一区）。

3种算子的校正方向完全不同：
1. TGE（文本引导外推）：文本嵌入差异驱动（我们现在的方法）
2. FMS（特征均值偏移）：目标域特征均值驱动
3. DDC（域方向校正）：源域-目标域特征均值差驱动
"""
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import tge_probe as T
from sota_comparison import (
    SOURCE_LOADERS, build_proto_tge, evaluate_with_features,
    SRC_TMPL_OVERHEAD, TGT_TMPL,
)

DATA = Path(r'C:\Users\张三\Desktop\实验数据')

# 统一α网格（与主实验一致，到10）
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# 主分析7类（剔除person）
MAIN_CLASSES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'truck']

# 门控规则（专家引导的序数属性）
OBJECT_SIZE = {'airplane': 30.0, 'bicycle': 1.8, 'boat': 15.0, 'bus': 12.0,
               'car': 4.5, 'motorcycle': 2.0, 'truck': 12.0, 'person': 1.7}
VIEWPOINT_DIFF = {'airplane': 0.3, 'bicycle': 0.5, 'boat': 0.4, 'bus': 0.8,
                   'car': 0.7, 'motorcycle': 0.6, 'truck': 0.75, 'person': 0.9}


def get_gate(cls):
    """门控决策：size < 10 AND vp > 0.5"""
    return 1 if (OBJECT_SIZE.get(cls, 999) < 10.0 and VIEWPOINT_DIFF.get(cls, 0) > 0.5) else 0


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


# ========== 3种不同的校正算子 ==========

def operator_tge(src_proto, src_cls, tgt_cls, src_features, tgt_features, alpha):
    """算子1: TGE（文本引导外推）——文本嵌入差异驱动"""
    if alpha == 0.0:
        return src_proto
    return build_proto_tge(src_proto, src_cls, tgt_cls, beta=alpha)


def operator_fms(src_proto, src_cls, tgt_cls, src_features, tgt_features, alpha):
    """算子2: FMS（特征均值偏移）——目标域特征均值驱动"""
    if alpha == 0.0:
        return src_proto
    tgt_mean = np.mean(tgt_features, axis=0)
    delta = tgt_mean - src_proto
    proto = src_proto + alpha * delta
    return normalize(proto)


def operator_ddc(src_proto, src_cls, tgt_cls, src_features, tgt_features, alpha):
    """算子3: DDC（域方向校正）——源域-目标域特征均值差驱动"""
    if alpha == 0.0:
        return src_proto
    src_mean = np.mean(src_features, axis=0)
    tgt_mean = np.mean(tgt_features, axis=0)
    delta = tgt_mean - src_mean
    delta = normalize(delta)
    proto = src_proto + alpha * delta
    return normalize(proto)


OPERATORS = {
    'TGE': operator_tge,
    'FMS': operator_fms,
    'DDC': operator_ddc,
}


def load_target_data(target_domain, target_cls, n=200):
    """加载目标域数据"""
    if target_domain == 'COCO':
        from sota_comparison import load_target_coco
        return load_target_coco(target_cls, n=n)
    elif target_domain == 'VOC':
        # VOC目标域加载
        from voc_loader import load_target_voc
        voc_cls_map = {'airplane': 'aeroplane', 'motorcycle': 'motorbike',
                       'boat': 'boat', 'car': 'car', 'bus': 'bus',
                       'bicycle': 'bicycle', 'person': 'person'}
        voc_cls = voc_cls_map.get(target_cls, target_cls)
        return load_target_voc(voc_cls, n=n)
    else:
        print(f'    未知目标域: {target_domain}')
        return None


def process_sample(sample, operator_name, operator_func):
    """处理单个样本，跑α网格搜索"""
    src_name = sample['src']
    src_cls = sample['src_cls']
    target_cls = sample['target_cls']
    target_domain = sample['target_domain']

    # 加载源域数据
    src_loader = SOURCE_LOADERS[src_name][0]
    src_data = src_loader(src_cls, n=150)
    if src_data is None:
        return None
    src_proto = src_data['prototype']
    src_features = src_data['features']

    # 负原型
    sims = src_features @ src_proto
    neg_idx = np.argmin(sims)
    proto_neg = src_features[neg_idx]
    proto_neg = normalize(proto_neg)

    # 加载目标域数据
    target_data = load_target_data(target_domain, target_cls, n=200)
    if target_data is None:
        return None
    tgt_features = target_data['features']

    # α网格搜索
    ap_curve = {}
    for alpha in ALPHAS:
        proto = operator_func(src_proto, src_cls, target_cls, src_features, tgt_features, alpha)
        ap = evaluate_with_features(target_data, proto, neg_proto=proto_neg)
        ap_curve[str(alpha)] = float(ap)

    best_alpha = ALPHAS[0]
    best_ap = ap_curve[str(ALPHAS[0])]
    for alpha in ALPHAS:
        if ap_curve[str(alpha)] > best_ap:
            best_ap = ap_curve[str(alpha)]
            best_alpha = alpha

    baseline_ap = ap_curve['0.0']
    oracle_gain = best_ap - baseline_ap

    # 固定α=2.5的增益
    ap_25 = float(np.interp(2.5, [float(a) for a in sorted(ap_curve.keys(), key=float)],
                             [ap_curve[a] for a in sorted(ap_curve.keys(), key=float)]))
    gain_25 = ap_25 - baseline_ap

    return {
        'operator': operator_name,
        'src': src_name,
        'src_cls': src_cls,
        'target_cls': target_cls,
        'target_domain': target_domain,
        'gate': get_gate(target_cls),
        'baseline_ap': float(baseline_ap),
        'best_alpha': float(best_alpha),
        'best_ap': float(best_ap),
        'oracle_gain': float(oracle_gain),
        'gain_25': float(gain_25),
        'ap_curve': ap_curve,
    }


def class_level_permutation_test(gains, gates, n_perm=10000):
    """类级置换检验：开启类平均增益是否显著高于关闭类"""
    classes = sorted(gains.keys())
    on_classes = [c for c in classes if gates[c] == 1]
    off_classes = [c for c in classes if gates[c] == 0]

    if not on_classes or not off_classes:
        return 1.0, 0.0

    obs_diff = np.mean([gains[c] for c in on_classes]) - np.mean([gains[c] for c in off_classes])

    count = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perm):
        perm_gates = rng.permutation(list(gates.values()))
        perm_on = [classes[i] for i in range(len(classes)) if perm_gates[i] == 1]
        perm_off = [classes[i] for i in range(len(classes)) if perm_gates[i] == 0]
        if perm_on and perm_off:
            perm_diff = np.mean([gains[c] for c in perm_on]) - np.mean([gains[c] for c in perm_off])
            if perm_diff >= obs_diff:
                count += 1

    p_value = count / n_perm
    return p_value, obs_diff


def main():
    print('=' * 80)
    print('X1跨算子复现实验')
    print('=' * 80)
    print(f'算子: {list(OPERATORS.keys())}')
    print(f'主分析类别: {MAIN_CLASSES}')
    print(f'α网格: {ALPHAS}')
    print()

    # 加载原始样本
    with open(DATA / 'zero_shot_gated_samples_extended_alpha.json', 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    # 筛选主分析7类的样本
    main_samples = [s for s in all_samples if s['target_cls'] in MAIN_CLASSES]
    print(f'主分析样本数: {len(main_samples)}')
    for cls in MAIN_CLASSES:
        n = len([s for s in main_samples if s['target_cls'] == cls])
        print(f'  {cls}: {n}个样本, gate={get_gate(cls)}')

    # 对每种算子跑实验
    all_results = {}
    for op_name, op_func in OPERATORS.items():
        print(f'\n{"="*80}')
        print(f'算子: {op_name}')
        print(f'{"="*80}')

        results = []
        for idx, sample in enumerate(main_samples):
            print(f'\n  [{idx+1}/{len(main_samples)}] {sample["src"]}/{sample["src_cls"]} -> '
                  f'{sample["target_domain"]}/{sample["target_cls"]} (gate={get_gate(sample["target_cls"])})')
            result = process_sample(sample, op_name, op_func)
            if result is not None:
                results.append(result)
                print(f'    baseline={result["baseline_ap"]:.4f}, best_alpha={result["best_alpha"]}, '
                      f'oracle_gain={result["oracle_gain"]:+.4f}, gain_25={result["gain_25"]:+.4f}')

        all_results[op_name] = results

        # 按类别汇总
        print(f'\n  {op_name} 按类别汇总（固定α=2.5）:')
        print(f'  {"类别":<12} {"n":>4} {"gate":>5} {"mean_baseline":>14} {"mean_gain_25":>12} {"pos_ratio":>10}')
        print('  ' + '-' * 70)

        by_cls = defaultdict(list)
        for r in results:
            by_cls[r['target_cls']].append(r)

        class_gains = {}
        class_gates = {}
        for cls in MAIN_CLASSES:
            if cls in by_cls:
                rs = by_cls[cls]
                mean_baseline = np.mean([r['baseline_ap'] for r in rs])
                mean_gain = np.mean([r['gain_25'] for r in rs])
                pos_ratio = np.mean([1 if r['gain_25'] > 0.001 else 0 for r in rs])
                gate = get_gate(cls)
                class_gains[cls] = mean_gain
                class_gates[cls] = gate
                print(f'  {cls:<12} {len(rs):>4} {gate:>5} {mean_baseline:>14.4f} '
                      f'{mean_gain:>+12.4f} {pos_ratio:>10.1%}')

        # 门控分离度
        on_gains = [class_gains[c] for c in class_gains if class_gates[c] == 1]
        off_gains = [class_gains[c] for c in class_gains if class_gates[c] == 0]
        if on_gains and off_gains:
            separation = np.mean(on_gains) / (np.abs(np.mean(off_gains)) + 1e-8)
            print(f'\n  门控分离度: 开启类均值={np.mean(on_gains):+.4f}, '
                  f'关闭类均值={np.mean(off_gains):+.4f}, 分离比={separation:.1f}x')

            # 类级置换检验
            p_value, obs_diff = class_level_permutation_test(class_gains, class_gates)
            print(f'  类级置换检验: p={p_value:.4f}, 观测差值={obs_diff:+.4f}')

    # 保存全部结果
    out_file = DATA / 'x1_cross_operator_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n全部结果已保存到: {out_file}')

    # 跨算子一致性分析
    print('\n' + '=' * 80)
    print('跨算子一致性分析')
    print('=' * 80)

    print(f'\n{"算子":<8} {"开启类均值增益":>14} {"关闭类均值增益":>14} {"分离比":>8} {"p值":>8} {"门控成立?"}')
    print('-' * 70)

    consistency_count = 0
    for op_name in OPERATORS:
        results = all_results[op_name]
        by_cls = defaultdict(list)
        for r in results:
            by_cls[r['target_cls']].append(r)

        class_gains = {}
        class_gates = {}
        for cls in MAIN_CLASSES:
            if cls in by_cls:
                class_gains[cls] = np.mean([r['gain_25'] for r in by_cls[cls]])
                class_gates[cls] = get_gate(cls)

        on_gains = [class_gains[c] for c in class_gains if class_gates[c] == 1]
        off_gains = [class_gains[c] for c in class_gains if class_gates[c] == 0]

        if on_gains and off_gains:
            mean_on = np.mean(on_gains)
            mean_off = np.mean(off_gains)
            separation = mean_on / (np.abs(mean_off) + 1e-8)
            p_value, _ = class_level_permutation_test(class_gains, class_gates)
            holds = mean_on > mean_off and p_value < 0.05
            if holds:
                consistency_count += 1
            print(f'{op_name:<8} {mean_on:>+14.4f} {mean_off:>+14.4f} {separation:>8.1f}x '
                  f'{p_value:>8.4f} {"✅" if holds else "❌"}')

    print(f'\n跨算子一致性: {consistency_count}/{len(OPERATORS)} 种算子上门控规则成立')

    if consistency_count == len(OPERATORS):
        print('\n🎉 结论: 门控规则在所有3种算子上都成立！')
        print('   Claim可以升格为: "CLIP空间中零训练原型校正存在类级适用边界，与具体算子无关"')
        print('   这是关于问题的发现，不是关于方法的发现 → L2，可冲一区')
    elif consistency_count >= 2:
        print(f'\n⚠️ 结论: 门控规则在{consistency_count}/{len(OPERATORS)}种算子上成立')
        print('   部分跨算子一致性，claim可以写"在多种零训练校正算子上观察到类级适用边界"')
    else:
        print(f'\n❌ 结论: 门控规则只在{consistency_count}/{len(OPERATORS)}种算子上成立')
        print('   跨算子一致性不足，claim限定为"在TGE算子上观察到类级适用边界"')


if __name__ == '__main__':
    main()
