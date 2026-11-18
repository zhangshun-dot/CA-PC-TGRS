# -*- coding: utf-8 -*-
"""实验数据分析脚本：读取所有结果JSON，生成汇总分析报告"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
REPORT_FILE = DATA_DIR / '实验结果分析报告.txt'

DOMAINS = ['DIOR', 'NWPU', 'DOTA', 'VisDrone']


def load_json(filepath):
    """安全加载JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None


def analyze_rank2_spectral():
    """分析秩2版本谱归一化结果"""
    filepath = DATA_DIR / 'usc_spectral_renormalization_results.json'
    data = load_json(filepath)
    if not data:
        return None

    lines = []
    lines.append("=" * 70)
    lines.append("一、秩2版本谱归一化结果")
    lines.append("=" * 70)

    original = data.get('original_baseline', {})
    all_results = data.get('all_results', {})

    lines.append(f"\n{'变换类型':<25} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8} {'vs原始':>10}")
    lines.append("-" * 85)

    baseline_aps = [original.get(s, 0) for s in DOMAINS]
    baseline_mean = np.mean(baseline_aps)

    for transform_type, results in all_results.items():
        name = data.get('transform_names', {}).get(transform_type, transform_type)
        aps = [results.get(s, {}).get('ap', 0) for s in DOMAINS]
        mean_ap = np.mean(aps)
        gain = mean_ap - baseline_mean
        lines.append(f"{name:<25} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {mean_ap:>8.4f} {gain:>+10.4f}")

    lines.append(f"\n关键发现：")
    lines.append(f"  - 原始空间基线平均AP: {baseline_mean:.4f}")
    spectral = all_results.get('spectral_norm', {})
    if spectral:
        visdrone_gain = spectral.get('VisDrone', {}).get('gain_vs_original', 0)
        lines.append(f"  - 谱归一化在VisDrone上提升: {visdrone_gain:+.4f}（域偏移最大的场景）")
        lines.append(f"  - 谱归一化平均提升: {np.mean([spectral.get(s, {}).get('gain_vs_original', 0) for s in DOMAINS]):+.4f}")
    variance = all_results.get('variance_reorder', {})
    if variance:
        lines.append(f"  - 按方差重新排序平均提升: {np.mean([variance.get(s, {}).get('gain_vs_original', 0) for s in DOMAINS]):+.4f}（效果不好）")

    return '\n'.join(lines)


def analyze_rank1_spectral():
    """分析秩1版本谱归一化结果"""
    filepath = DATA_DIR / 'usc_spectral_renormalization_rank1_results.json'
    data = load_json(filepath)
    if not data:
        return None

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("二、秩1版本谱归一化结果（与第二篇论文核心方法统一）")
    lines.append("=" * 70)

    original = data.get('original_baseline', {})
    all_results = data.get('all_results', {})

    lines.append(f"\n{'变换类型':<25} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8} {'vs原始':>10}")
    lines.append("-" * 85)

    baseline_aps = [original.get(s, 0) for s in DOMAINS]
    baseline_mean = np.mean(baseline_aps)

    for transform_type, results in all_results.items():
        name = data.get('transform_names', {}).get(transform_type, transform_type)
        aps = [results.get(s, {}).get('ap', 0) for s in DOMAINS]
        mean_ap = np.mean(aps)
        gain = mean_ap - baseline_mean
        lines.append(f"{name:<25} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {mean_ap:>8.4f} {gain:>+10.4f}")

    lines.append(f"\n关键发现：")
    lines.append(f"  - 秩1原始空间基线平均AP: {baseline_mean:.4f}（比秩2低，因为参数少）")
    spectral = all_results.get('spectral_norm', {})
    if spectral:
        gains = [spectral.get(s, {}).get('gain_vs_original', 0) for s in DOMAINS]
        lines.append(f"  - 谱归一化平均提升: {np.mean(gains):+.4f}")
        lines.append(f"  - 各域对提升: DIOR {gains[0]:+.4f}, NWPU {gains[1]:+.4f}, DOTA {gains[2]:+.4f}, VisDrone {gains[3]:+.4f}")

    return '\n'.join(lines)


def analyze_adaptive_spectral():
    """分析自适应谱归一化结果"""
    filepath = DATA_DIR / 'usc_adaptive_spectral_norm_results.json'
    data = load_json(filepath)
    if not data:
        return None

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("三、自适应谱归一化结果")
    lines.append("=" * 70)

    domain_shifts = data.get('domain_shifts', {})
    lines.append(f"\n域偏移大小：")
    for domain, shift in domain_shifts.items():
        lines.append(f"  - {domain}: {shift:.4f}")
    lines.append(f"  - 最大域偏移: {data.get('delta_max', 0):.4f}")

    global_results = data.get('global_results', {})
    adaptive1_results = data.get('adaptive1_results', {})

    lines.append(f"\n{'方法':<30} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8}")
    lines.append("-" * 80)

    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w_results = global_results.get(str(w), {})
        aps = [w_results.get(s, {}).get('ap', 0) for s in DOMAINS]
        lines.append(f"{'全局权重 w=' + str(w):<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f}")

    aps = [adaptive1_results.get(s, {}).get('ap', 0) for s in DOMAINS]
    lines.append(f"{'自适应 w=δ/δ_max':<30} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {np.mean(aps):>8.4f}")

    # 找最优方法
    all_methods = []
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w_results = global_results.get(str(w), {})
        mean_ap = np.mean([w_results.get(s, {}).get('ap', 0) for s in DOMAINS])
        all_methods.append((f'全局w={w}', mean_ap))
    mean_ap = np.mean([adaptive1_results.get(s, {}).get('ap', 0) for s in DOMAINS])
    all_methods.append(('自适应w=δ/δ_max', mean_ap))

    best_method = max(all_methods, key=lambda x: x[1])
    lines.append(f"\n最优方法: {best_method[0]} (平均AP={best_method[1]:.4f})")

    return '\n'.join(lines)


def analyze_physical_prior():
    """分析物理先验 + 频域校正结果"""
    filepath = DATA_DIR / 'usc_physical_prior_spectral_results.json'
    data = load_json(filepath)
    if not data:
        return None

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("四、物理先验 + 频域校正结果")
    lines.append("=" * 70)

    category_scale = data.get('category_scale_prior', {})
    lines.append(f"\n类别尺度先验：")
    for cat, scale in sorted(category_scale.items(), key=lambda x: -x[1]):
        lines.append(f"  - {cat}: {scale:.2f}")

    all_results = data.get('all_results', {})
    mode_names = data.get('mode_names', {})

    lines.append(f"\n{'模式':<35} {'DIOR':>8} {'NWPU':>8} {'DOTA':>8} {'VisDrone':>10} {'平均':>8} {'vs基线':>10}")
    lines.append("-" * 95)

    baseline_aps = [all_results.get('none', {}).get(s, {}).get('ap', 0) for s in DOMAINS]
    baseline_mean = np.mean(baseline_aps)

    for mode, results in all_results.items():
        name = mode_names.get(mode, mode)
        aps = [results.get(s, {}).get('ap', 0) for s in DOMAINS]
        mean_ap = np.mean(aps)
        gain = mean_ap - baseline_mean
        lines.append(f"{name:<35} {aps[0]:>8.4f} {aps[1]:>8.4f} {aps[2]:>8.4f} {aps[3]:>10.4f} {mean_ap:>8.4f} {gain:>+10.4f}")

    # 找最优模式
    all_modes = []
    for mode, results in all_results.items():
        name = mode_names.get(mode, mode)
        mean_ap = np.mean([results.get(s, {}).get('ap', 0) for s in DOMAINS])
        all_modes.append((name, mean_ap, mean_ap - baseline_mean))

    best_mode = max(all_modes, key=lambda x: x[1])
    lines.append(f"\n最优模式: {best_mode[0]} (平均AP={best_mode[1]:.4f}, 提升={best_mode[2]:+.4f})")

    return '\n'.join(lines)


def generate_summary():
    """生成总体汇总"""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("五、总体汇总与结论")
    lines.append("=" * 70)

    # 收集所有方法的最优结果
    all_method_results = []

    # 秩2谱归一化
    data = load_json(DATA_DIR / 'usc_spectral_renormalization_results.json')
    if data:
        spectral = data.get('all_results', {}).get('spectral_norm', {})
        if spectral:
            mean_ap = np.mean([spectral.get(s, {}).get('ap', 0) for s in DOMAINS])
            all_method_results.append(('秩2-谱归一化', mean_ap))

    # 秩1谱归一化
    data = load_json(DATA_DIR / 'usc_spectral_renormalization_rank1_results.json')
    if data:
        spectral = data.get('all_results', {}).get('spectral_norm', {})
        if spectral:
            mean_ap = np.mean([spectral.get(s, {}).get('ap', 0) for s in DOMAINS])
            all_method_results.append(('秩1-谱归一化', mean_ap))
        baseline = data.get('original_baseline', {})
        if baseline:
            mean_ap = np.mean([baseline.get(s, 0) for s in DOMAINS])
            all_method_results.append(('秩1-无归一化(基线)', mean_ap))

    # 自适应谱归一化
    data = load_json(DATA_DIR / 'usc_adaptive_spectral_norm_results.json')
    if data:
        adaptive = data.get('adaptive1_results', {})
        if adaptive:
            mean_ap = np.mean([adaptive.get(s, {}).get('ap', 0) for s in DOMAINS])
            all_method_results.append(('自适应谱归一化', mean_ap))

    # 物理先验
    data = load_json(DATA_DIR / 'usc_physical_prior_spectral_results.json')
    if data:
        all_results = data.get('all_results', {})
        for mode, results in all_results.items():
            name = data.get('mode_names', {}).get(mode, mode)
            mean_ap = np.mean([results.get(s, {}).get('ap', 0) for s in DOMAINS])
            all_method_results.append((f'物理先验-{name}', mean_ap))

    if all_method_results:
        lines.append(f"\n所有方法平均AP对比（4域对平均）：")
        lines.append(f"\n{'排名':<6} {'方法':<40} {'平均AP':>10}")
        lines.append("-" * 60)
        sorted_results = sorted(all_method_results, key=lambda x: -x[1])
        for i, (name, ap) in enumerate(sorted_results, 1):
            marker = " ⭐" if i == 1 else ""
            lines.append(f"{i:<6} {name:<40} {ap:>10.4f}{marker}")

        best = sorted_results[0]
        lines.append(f"\n最优方法: {best[0]} (平均AP={best[1]:.4f})")

    lines.append(f"\n实验完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"结果文件保存在: {DATA_DIR}")

    return '\n'.join(lines)


def main():
    print("开始分析实验数据...")

    report_parts = []
    report_parts.append("=" * 70)
    report_parts.append("频域/傅里叶方向实验结果分析报告")
    report_parts.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_parts.append("=" * 70)

    # 分析各个实验
    analyses = [
        analyze_rank2_spectral,
        analyze_rank1_spectral,
        analyze_adaptive_spectral,
        analyze_physical_prior,
    ]

    for analyze_func in analyses:
        result = analyze_func()
        if result:
            report_parts.append(result)
            print(f"  ✅ {analyze_func.__name__} 分析完成")
        else:
            print(f"  ⚠️ {analyze_func.__name__} 结果文件未生成，跳过")

    # 总体汇总
    report_parts.append(generate_summary())

    # 保存报告
    full_report = '\n'.join(report_parts)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_report)

    print(f"\n分析报告已保存: {REPORT_FILE}")
    print(f"报告长度: {len(full_report)} 字符")


if __name__ == '__main__':
    main()
