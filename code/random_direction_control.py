# -*- coding: utf-8 -*-
"""random_direction_control.py — 随机方向对照实验（R1/R2/R3）
验证TGE的增益是来自语义方向，还是纯位移效应。

三组对照，固定相同位移s，只换方向：
- R1: 随机单位向量，每类采样m=20
- R2: 其他类的真实d（打乱类-方向配对）
- R3: 反向-d
- 真实d: TGE的真实方向（对照基准）

如果真实d明显优于R1且R2 → "语义方向有用"成立
如果真实d ≈ R1 → 纯位移效应，主方法机理失效
如果真实d > R1但≈R2 → 只有"是真实类条件方向"有效，无类-算子匹配信息
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

# 主分析7类
MAIN_CLASSES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'truck']

# 共同位移网格（对数均匀28点，s ∈ [0.02, 4.0]）
# 额外在s ∈ [0.02, 0.5]上做线性细化（步长0.02）
DISPLACEMENTS = sorted(set(
    list(np.logspace(np.log10(0.02), np.log10(4.0), 28)) +
    list(np.arange(0.02, 0.52, 0.02))
))

# 随机方向采样数
N_RANDOM = 20

# 随机种子（保证可复现）
RANDOM_SEED = 42


def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8) if norm > 1e-8 else v


def get_tge_direction(src_cls, target_cls):
    """获取TGE的真实方向向量"""
    s_emb = T.text_avg([SRC_TMPL_OVERHEAD], src_cls)
    d_emb = T.text_avg([TGT_TMPL], target_cls)
    # 转到CPU并转为numpy
    if hasattr(s_emb, 'cpu'):
        s_emb = s_emb.cpu().numpy()
    if hasattr(d_emb, 'cpu'):
        d_emb = d_emb.cpu().numpy()
    d = d_emb - s_emb
    return normalize(d)


def evaluate_with_direction(target_data, src_proto, d_direction, displacement, neg_proto):
    """用指定方向和位移评估AP50
    proto = normalize(src_proto + displacement * d_direction)
    """
    proto = normalize(src_proto + displacement * d_direction)
    return evaluate_with_features(target_data, proto, neg_proto=neg_proto)


def load_sample_data(sample):
    """加载单个样本的源域和目标域数据"""
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
    if target_domain == 'COCO':
        from sota_comparison import load_target_coco
        target_data = load_target_coco(target_cls, n=200)
    elif target_domain == 'VOC':
        from voc_loader import load_target_voc
        voc_cls_map = {'airplane': 'aeroplane', 'motorcycle': 'motorbike',
                       'boat': 'boat', 'car': 'car', 'bus': 'bus',
                       'bicycle': 'bicycle', 'person': 'person'}
        voc_cls = voc_cls_map.get(target_cls, target_cls)
        target_data = load_target_voc(voc_cls, n=200)
    else:
        return None

    if target_data is None:
        return None

    return {
        'src_proto': src_proto,
        'src_features': src_features,
        'proto_neg': proto_neg,
        'target_data': target_data,
        'src_cls': src_cls,
        'target_cls': target_cls,
        'target_domain': target_domain,
    }


def run_random_direction_experiment(sample, sample_data, all_directions):
    """对单个样本跑随机方向对照实验"""
    src_proto = sample_data['src_proto']
    proto_neg = sample_data['proto_neg']
    target_data = sample_data['target_data']
    target_cls = sample_data['target_cls']

    # baseline (displacement=0)
    baseline_ap = evaluate_with_features(target_data, src_proto, neg_proto=proto_neg)

    results = {
        'baseline_ap': float(baseline_ap),
        'real_direction': {},  # {displacement: ap}
        'random_directions': [],  # [{displacement: ap}, ...]
        'shuffled_direction': {},  # {displacement: ap}
        'reverse_direction': {},  # {displacement: ap}
    }

    # 真实方向
    real_d = get_tge_direction(sample_data['src_cls'], target_cls)
    for s in DISPLACEMENTS:
        ap = evaluate_with_direction(target_data, src_proto, real_d, s, proto_neg)
        results['real_direction'][str(s)] = float(ap)

    # R1: 随机方向（N_RANDOM个）
    rng = np.random.RandomState(RANDOM_SEED)
    for i in range(N_RANDOM):
        random_d = rng.randn(len(src_proto))
        random_d = normalize(random_d)
        random_result = {}
        for s in DISPLACEMENTS:
            ap = evaluate_with_direction(target_data, src_proto, random_d, s, proto_neg)
            random_result[str(s)] = float(ap)
        results['random_directions'].append(random_result)

    # R2: 打乱类-方向配对（用其他类的真实方向）
    # 从all_directions中随机选一个不是当前类的方向
    other_directions = [d for cls, d in all_directions.items() if cls != target_cls]
    if other_directions:
        shuffled_d = other_directions[rng.randint(len(other_directions))]
        for s in DISPLACEMENTS:
            ap = evaluate_with_direction(target_data, src_proto, shuffled_d, s, proto_neg)
            results['shuffled_direction'][str(s)] = float(ap)

    # R3: 反向
    reverse_d = -real_d
    for s in DISPLACEMENTS:
        ap = evaluate_with_direction(target_data, src_proto, reverse_d, s, proto_neg)
        results['reverse_direction'][str(s)] = float(ap)

    return results


def main():
    print('=' * 100)
    print('随机方向对照实验（R1/R2/R3）')
    print('验证TGE的增益是来自语义方向，还是纯位移效应')
    print('=' * 100)

    # 加载原始样本
    with open(DATA / 'zero_shot_gated_samples_extended_alpha.json', 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    # 筛选主分析7类的样本，每个类选一个代表性样本（COCO目标域）
    main_samples = []
    for cls in MAIN_CLASSES:
        cls_samples = [s for s in all_samples if s['target_cls'] == cls and s['target_domain'] == 'COCO']
        if cls_samples:
            main_samples.append(cls_samples[0])  # 选第一个

    print(f'\n代表性样本数: {len(main_samples)}')
    for s in main_samples:
        print(f'  {s["src"]}/{s["src_cls"]} -> {s["target_domain"]}/{s["target_cls"]}')

    # 预计算所有类的真实方向（用于R2打乱配对）
    print(f'\n预计算所有类的真实方向...')
    all_directions = {}
    for cls in MAIN_CLASSES:
        all_directions[cls] = get_tge_direction(cls, cls)  # 用类名本身作为源和目标
        print(f'  {cls}: ||d||={np.linalg.norm(all_directions[cls]):.4f}')

    # 对每个样本跑实验
    all_results = {}
    for idx, sample in enumerate(main_samples):
        print(f'\n{"─" * 100}')
        print(f'[{idx+1}/{len(main_samples)}] {sample["src"]}/{sample["src_cls"]} -> '
              f'{sample["target_domain"]}/{sample["target_cls"]}')
        print(f'{"─" * 100}')

        # 加载数据
        sample_data = load_sample_data(sample)
        if sample_data is None:
            print(f'  数据加载失败，跳过')
            continue

        print(f'  baseline AP50: {sample_data["target_data"]["per_img"][0]["gd"].shape if "gd" in sample_data["target_data"]["per_img"][0] else "N/A"}')

        # 跑实验
        results = run_random_direction_experiment(sample, sample_data, all_directions)
        all_results[sample['target_cls']] = results

        # 打印关键结果
        print(f'\n  关键结果（最优位移处的AP50）:')

        # 真实方向的最优位移和AP
        real_aps = [results['real_direction'][str(s)] for s in DISPLACEMENTS]
        best_real_idx = np.argmax(real_aps)
        best_real_s = DISPLACEMENTS[best_real_idx]
        best_real_ap = real_aps[best_real_idx]
        print(f'    真实方向: 最优位移s={best_real_s:.3f}, AP50={best_real_ap:.4f}, '
              f'增益={best_real_ap-results["baseline_ap"]:+.4f}')

        # 随机方向的平均最优位移和AP
        random_best_aps = []
        random_best_ss = []
        for i in range(N_RANDOM):
            random_aps = [results['random_directions'][i][str(s)] for s in DISPLACEMENTS]
            best_idx = np.argmax(random_aps)
            random_best_aps.append(random_aps[best_idx])
            random_best_ss.append(DISPLACEMENTS[best_idx])
        mean_random_ap = np.mean(random_best_aps)
        mean_random_s = np.mean(random_best_ss)
        print(f'    随机方向(R1): 平均最优位移s={mean_random_s:.3f}, 平均AP50={mean_random_ap:.4f}, '
              f'平均增益={mean_random_ap-results["baseline_ap"]:+.4f}')

        # 打乱方向的最优位移和AP
        if results['shuffled_direction']:
            shuffled_aps = [results['shuffled_direction'][str(s)] for s in DISPLACEMENTS]
            best_shuffled_idx = np.argmax(shuffled_aps)
            best_shuffled_s = DISPLACEMENTS[best_shuffled_idx]
            best_shuffled_ap = shuffled_aps[best_shuffled_idx]
            print(f'    打乱方向(R2): 最优位移s={best_shuffled_s:.3f}, AP50={best_shuffled_ap:.4f}, '
                  f'增益={best_shuffled_ap-results["baseline_ap"]:+.4f}')

        # 反向的最优位移和AP
        reverse_aps = [results['reverse_direction'][str(s)] for s in DISPLACEMENTS]
        best_reverse_idx = np.argmax(reverse_aps)
        best_reverse_s = DISPLACEMENTS[best_reverse_idx]
        best_reverse_ap = reverse_aps[best_reverse_idx]
        print(f'    反向(R3): 最优位移s={best_reverse_s:.3f}, AP50={best_reverse_ap:.4f}, '
              f'增益={best_reverse_ap-results["baseline_ap"]:+.4f}')

        # 关键对比
        print(f'\n  关键对比:')
        real_gain = best_real_ap - results['baseline_ap']
        random_gain = mean_random_ap - results['baseline_ap']
        print(f'    真实方向增益 - 随机方向增益 = {real_gain - random_gain:+.4f}')
        if real_gain - random_gain > 0.01:
            print(f'    → 真实方向明显优于随机方向，语义方向有用 ✅')
        elif real_gain - random_gain > 0:
            print(f'    → 真实方向略优于随机方向，但差异不大 ⚠️')
        else:
            print(f'    → 真实方向不优于随机方向，可能是纯位移效应 ❌')

    # 保存全部结果
    out_file = DATA / 'random_direction_control_results.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n全部结果已保存到: {out_file}')

    # 汇总统计
    print(f'\n{"=" * 100}')
    print('汇总统计（所有类）')
    print(f'{"=" * 100}')

    real_gains = []
    random_gains = []
    shuffled_gains = []
    reverse_gains = []

    for cls, results in all_results.items():
        real_aps = [results['real_direction'][str(s)] for s in DISPLACEMENTS]
        best_real_ap = np.max(real_aps)
        real_gains.append(best_real_ap - results['baseline_ap'])

        random_best_aps = []
        for i in range(N_RANDOM):
            random_aps = [results['random_directions'][i][str(s)] for s in DISPLACEMENTS]
            random_best_aps.append(np.max(random_aps))
        random_gains.append(np.mean(random_best_aps) - results['baseline_ap'])

        if results['shuffled_direction']:
            shuffled_aps = [results['shuffled_direction'][str(s)] for s in DISPLACEMENTS]
            shuffled_gains.append(np.max(shuffled_aps) - results['baseline_ap'])

        reverse_aps = [results['reverse_direction'][str(s)] for s in DISPLACEMENTS]
        reverse_gains.append(np.max(reverse_aps) - results['baseline_ap'])

    print(f'\n{"方向类型":<15} {"平均增益":>10} {"中位数":>10} {"最小":>10} {"最大":>10}')
    print('-' * 60)
    print(f'{"真实方向":<15} {np.mean(real_gains):>+10.4f} {np.median(real_gains):>+10.4f} '
          f'{np.min(real_gains):>+10.4f} {np.max(real_gains):>+10.4f}')
    print(f'{"随机方向(R1)":<15} {np.mean(random_gains):>+10.4f} {np.median(random_gains):>+10.4f} '
          f'{np.min(random_gains):>+10.4f} {np.max(random_gains):>+10.4f}')
    if shuffled_gains:
        print(f'{"打乱方向(R2)":<15} {np.mean(shuffled_gains):>+10.4f} {np.median(shuffled_gains):>+10.4f} '
              f'{np.min(shuffled_gains):>+10.4f} {np.max(shuffled_gains):>+10.4f}')
    print(f'{"反向(R3)":<15} {np.mean(reverse_gains):>+10.4f} {np.median(reverse_gains):>+10.4f} '
          f'{np.min(reverse_gains):>+10.4f} {np.max(reverse_gains):>+10.4f}')

    # 最终结论
    print(f'\n{"=" * 100}')
    print('最终结论')
    print(f'{"=" * 100}')

    mean_real = np.mean(real_gains)
    mean_random = np.mean(random_gains)
    diff = mean_real - mean_random

    print(f'\n真实方向平均增益: {mean_real:+.4f}')
    print(f'随机方向平均增益: {mean_random:+.4f}')
    print(f'差异: {diff:+.4f}')

    if diff > 0.01:
        print(f'\n✅ 结论：真实方向明显优于随机方向（差异>{0.01}），语义方向有用。')
        print(f'   主方法的机制解释（"文本侧嵌入差异提供了有效校正方向"）成立。')
    elif diff > 0:
        print(f'\n⚠️ 结论：真实方向略优于随机方向，但差异不大（{diff:.4f}）。')
        print(f'   语义方向可能有一定作用，但位移效应也贡献了相当一部分。')
        print(f'   需要更多样本或更精细的分析来确认。')
    else:
        print(f'\n❌ 结论：真实方向不优于随机方向（差异={diff:.4f}）。')
        print(f'   TGE的增益可能主要来自纯位移效应（把原型推离原位），')
        print(f'   而不是语义方向本身。主方法的机制解释需要重新思考。')

    if shuffled_gains:
        mean_shuffled = np.mean(shuffled_gains)
        print(f'\n真实方向 vs 打乱方向: {mean_real - mean_shuffled:+.4f}')
        if mean_real - mean_shuffled > 0.01:
            print(f'  → 真实方向明显优于打乱方向，存在类-方向匹配信息 ✅')
        else:
            print(f'  → 真实方向不优于打乱方向，只有"是真实类条件方向"有效，无类-算子匹配信息 ⚠️')

    mean_reverse = np.mean(reverse_gains)
    print(f'\n真实方向 vs 反向: {mean_real - mean_reverse:+.4f}')
    if mean_real - mean_reverse > 0.01:
        print(f'  → 真实方向明显优于反向，方向信息有用 ✅')
    else:
        print(f'  → 真实方向不优于反向，方向信息可能为零 ❌')


if __name__ == '__main__':
    main()
