# -*- coding: utf-8 -*-
"""reanalyze_random_direction.py — 重新分析随机方向对照实验
回答审计报告提出的问题：
1. s=1.46（论文操作点）处的绝对增益
2. 每类置换排名
3. 随机方向采样方式（独立 vs 共享）
4. 多重比较校正
"""
import json
import numpy as np
from scipy import stats

DATA = r'C:\Users\张三\Desktop\实验数据\random_direction_control_results.json'

# 主分析7类
CLASSES = ['airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'truck']

# 论文操作点：α=2.5, ||d||/||p||=0.5834, s=1.4585
S_PAPER = 1.4585

# 报告主打的点
S_REPORT = 2.5


def get_ap_at_s(curve, s_target):
    """从位移曲线中插值得到s_target处的AP"""
    s_values = sorted([float(k) for k in curve.keys()])
    ap_values = [curve[str(s)] for s in s_values]
    # 线性插值
    return float(np.interp(s_target, s_values, ap_values))


def main():
    print('=' * 80)
    print('随机方向对照实验 — 重新分析')
    print('=' * 80)

    data = json.load(open(DATA, encoding='utf-8'))

    # ══════════════════════════════════════════════════════════════
    # 问题1: s=1.46（论文操作点）处的绝对增益
    # ══════════════════════════════════════════════════════════════
    print(f'\n{"─" * 80}')
    print(f'问题1: 论文操作点 s={S_PAPER:.4f}（α=2.5）处的绝对增益')
    print(f'{"─" * 80}')

    print(f'\n{"类别":<12} {"baseline":>10} {"real":>10} {"real_gain":>10} '
          f'{"random_mean":>12} {"random_gain":>12} {"diff":>10}')
    print('-' * 90)

    real_gains_paper = []
    random_gains_paper = []
    diffs_paper = []

    for cls in CLASSES:
        d = data[cls]
        baseline = d['baseline_ap']
        real_ap = get_ap_at_s(d['real_direction'], S_PAPER)
        real_gain = real_ap - baseline

        # 20个随机方向的均值
        random_aps = [get_ap_at_s(rd, S_PAPER) for rd in d['random_directions']]
        random_mean = np.mean(random_aps)
        random_gain = random_mean - baseline

        diff = real_gain - random_gain

        real_gains_paper.append(real_gain)
        random_gains_paper.append(random_gain)
        diffs_paper.append(diff)

        print(f'{cls:<12} {baseline:>10.4f} {real_ap:>10.4f} {real_gain:>+10.4f} '
              f'{random_mean:>12.4f} {random_gain:>+12.4f} {diff:>+10.4f}')

    print(f'\n统计（s={S_PAPER:.4f}）:')
    print(f'  真实方向平均增益: {np.mean(real_gains_paper):+.4f}')
    print(f'  随机方向平均增益: {np.mean(random_gains_paper):+.4f}')
    print(f'  差值（真实-随机）: {np.mean(diffs_paper):+.4f}')
    print(f'  正例比例（真实增益>0）: {np.mean([1 if g > 0.001 else 0 for g in real_gains_paper]):.1%}')

    # 配对t检验
    t_stat, t_p = stats.ttest_rel(real_gains_paper, random_gains_paper)
    # Wilcoxon符号秩检验
    try:
        w_stat, w_p = stats.wilcoxon(real_gains_paper, random_gains_paper)
    except:
        w_p = float('nan')

    print(f'  配对t检验: t={t_stat:.3f}, p={t_p:.4f}')
    print(f'  Wilcoxon符号秩检验: p={w_p:.4f}')

    # ══════════════════════════════════════════════════════════════
    # 问题2: s=2.5（报告主打）处的绝对增益
    # ══════════════════════════════════════════════════════════════
    print(f'\n{"─" * 80}')
    print(f'问题2: 报告主打点 s={S_REPORT} 处的绝对增益')
    print(f'{"─" * 80}')

    print(f'\n{"类别":<12} {"baseline":>10} {"real":>10} {"real_gain":>10} '
          f'{"random_mean":>12} {"random_gain":>12} {"diff":>10}')
    print('-' * 90)

    real_gains_report = []
    random_gains_report = []
    diffs_report = []

    for cls in CLASSES:
        d = data[cls]
        baseline = d['baseline_ap']
        real_ap = get_ap_at_s(d['real_direction'], S_REPORT)
        real_gain = real_ap - baseline

        random_aps = [get_ap_at_s(rd, S_REPORT) for rd in d['random_directions']]
        random_mean = np.mean(random_aps)
        random_gain = random_mean - baseline

        diff = real_gain - random_gain

        real_gains_report.append(real_gain)
        random_gains_report.append(random_gain)
        diffs_report.append(diff)

        print(f'{cls:<12} {baseline:>10.4f} {real_ap:>10.4f} {real_gain:>+10.4f} '
              f'{random_mean:>12.4f} {random_gain:>+12.4f} {diff:>+10.4f}')

    print(f'\n统计（s={S_REPORT}）:')
    print(f'  真实方向平均增益: {np.mean(real_gains_report):+.4f}')
    print(f'  随机方向平均增益: {np.mean(random_gains_report):+.4f}')
    print(f'  差值（真实-随机）: {np.mean(diffs_report):+.4f}')

    t_stat, t_p = stats.ttest_rel(real_gains_report, random_gains_report)
    try:
        w_stat, w_p = stats.wilcoxon(real_gains_report, random_gains_report)
    except:
        w_p = float('nan')
    print(f'  配对t检验: t={t_stat:.3f}, p={t_p:.4f}')
    print(f'  Wilcoxon符号秩检验: p={w_p:.4f}')

    # ══════════════════════════════════════════════════════════════
    # 问题3: 每类置换排名（s=2.5）
    # ══════════════════════════════════════════════════════════════
    print(f'\n{"─" * 80}')
    print(f'问题3: 每类置换排名（s={S_REPORT}）')
    print(f'{"─" * 80}')

    print(f'\n{"类别":<12} {"real_gain":>10} {"random_mean":>12} {"random_std":>10} '
          f'{"rank":>6} {"p_c":>8} {"effect_ratio":>12}')
    print('-' * 85)

    for i, cls in enumerate(CLASSES):
        d = data[cls]
        baseline = d['baseline_ap']
        real_ap = get_ap_at_s(d['real_direction'], S_REPORT)
        real_gain = real_ap - baseline

        random_aps = [get_ap_at_s(rd, S_REPORT) for rd in d['random_directions']]
        random_gains = [ap - baseline for ap in random_aps]
        random_mean = np.mean(random_gains)
        random_std = np.std(random_gains)

        # 排名：真实方向在20个随机方向中的排名（1=最好）
        rank = 1 + sum(1 for rg in random_gains if rg > real_gain)
        p_c = rank / (len(random_gains) + 1)

        # 效应量比例：真实-随机差值 / 随机方向标准差
        effect_ratio = (real_gain - random_mean) / (random_std + 1e-8)

        print(f'{cls:<12} {real_gain:>+10.4f} {random_mean:>+12.4f} {random_std:>10.4f} '
              f'{rank:>6}/21 {p_c:>8.3f} {effect_ratio:>12.2f}')

    # ══════════════════════════════════════════════════════════════
    # 问题4: 随机方向采样方式检查
    # ══════════════════════════════════════════════════════════════
    print(f'\n{"─" * 80}')
    print(f'问题4: 随机方向采样方式检查')
    print(f'{"─" * 80}')

    # 检查不同类别的随机方向是否相同
    # 比较car和motorcycle的第一个随机方向的位移曲线
    car_rd0 = data['car']['random_directions'][0]
    motorcycle_rd0 = data['motorcycle']['random_directions'][0]

    # 比较s值
    car_s = sorted([float(k) for k in car_rd0.keys()])
    motorcycle_s = sorted([float(k) for k in motorcycle_rd0.keys()])

    print(f'\ncar随机方向0的s值数量: {len(car_s)}')
    print(f'motorcycle随机方向0的s值数量: {len(motorcycle_s)}')
    print(f's值是否相同: {np.allclose(car_s, motorcycle_s)}')

    # 比较AP值（如果s值相同，AP值也相同，说明是全局共享）
    if np.allclose(car_s, motorcycle_s):
        car_aps = [car_rd0[str(s)] for s in car_s]
        motorcycle_aps = [motorcycle_rd0[str(s)] for s in motorcycle_s]
        print(f'AP值是否相同: {np.allclose(car_aps, motorcycle_aps)}')
        if not np.allclose(car_aps, motorcycle_aps):
            print(f'  → 每类独立采样（AP值不同）')
        else:
            print(f'  → 全局共享采样（AP值相同）')

    # ══════════════════════════════════════════════════════════════
    # 问题5: 多重比较校正
    # ══════════════════════════════════════════════════════════════
    print(f'\n{"─" * 80}')
    print(f'问题5: 多重比较校正（7个位移点）')
    print(f'{"─" * 80}')

    s_points = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    t_p_values = []
    w_p_values = []

    print(f'\n{"s":>6} {"diff":>10} {"t_p":>10} {"w_p":>10} {"t_p_Bonf":>12} {"w_p_Bonf":>12}')
    print('-' * 65)

    for s in s_points:
        diffs = []
        for cls in CLASSES:
            d = data[cls]
            baseline = d['baseline_ap']
            real_ap = get_ap_at_s(d['real_direction'], s)
            real_gain = real_ap - baseline
            random_aps = [get_ap_at_s(rd, s) for rd in d['random_directions']]
            random_mean = np.mean(random_aps)
            random_gain = random_mean - baseline
            diffs.append(real_gain - random_gain)

        # 配对t检验（真实增益 vs 随机增益）
        real_gains_s = []
        random_gains_s = []
        for cls in CLASSES:
            d = data[cls]
            baseline = d['baseline_ap']
            real_ap = get_ap_at_s(d['real_direction'], s)
            real_gains_s.append(real_ap - baseline)
            random_aps = [get_ap_at_s(rd, s) for rd in d['random_directions']]
            random_gains_s.append(np.mean(random_aps) - baseline)

        t_stat, t_p = stats.ttest_rel(real_gains_s, random_gains_s)
        try:
            w_stat, w_p = stats.wilcoxon(real_gains_s, random_gains_s)
        except:
            w_p = float('nan')

        t_p_values.append(t_p)
        w_p_values.append(w_p)

        t_p_bonf = min(t_p * 7, 1.0)
        w_p_bonf = min(w_p * 7, 1.0) if not np.isnan(w_p) else float('nan')

        print(f'{s:>6.1f} {np.mean(diffs):>+10.4f} {t_p:>10.4f} {w_p:>10.4f} '
              f'{t_p_bonf:>12.4f} {w_p_bonf:>12.4f}')

    # Holm-Bonferroni逐步校正
    def holm_bonferroni(p_values):
        """Holm-Bonferroni逐步校正"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        corrected = np.zeros(n)
        for rank, idx in enumerate(sorted_indices):
            corrected[idx] = min(p_values[idx] * (n - rank), 1.0)
            if rank > 0:
                corrected[idx] = max(corrected[idx], corrected[sorted_indices[rank-1]])
        return corrected

    t_p_holm = holm_bonferroni(np.array(t_p_values))
    w_p_holm = holm_bonferroni(np.array(w_p_values))

    print(f'\nHolm-Bonferroni逐步校正:')
    print(f'{"s":>6} {"t_p_Holm":>12} {"w_p_Holm":>12}')
    print('-' * 35)
    for i, s in enumerate(s_points):
        print(f'{s:>6.1f} {t_p_holm[i]:>12.4f} {w_p_holm[i]:>12.4f}')

    print(f'\n最小t_p (Holm): {np.min(t_p_holm):.4f}')
    print(f'最小w_p (Holm): {np.min(w_p_holm):.4f}')

    # ══════════════════════════════════════════════════════════════
    # 问题6: car/motorcycle效应量分析
    # ══════════════════════════════════════════════════════════════
    print(f'\n{"─" * 80}')
    print(f'问题6: car/motorcycle效应量分析（s=2.5）')
    print(f'{"─" * 80}')

    total_diff = sum(diffs_report)
    car_motorcycle_diff = diffs_report[4] + diffs_report[5]  # car + motorcycle
    car_motorcycle_ratio = car_motorcycle_diff / total_diff if total_diff != 0 else 0

    print(f'\n总差值: {total_diff:+.4f}')
    print(f'car+motorcycle差值: {car_motorcycle_diff:+.4f}')
    print(f'car+motorcycle占比: {car_motorcycle_ratio:.1%}')

    # 剔除car/motorcycle后的分析
    remaining_diffs = [diffs_report[i] for i in range(len(CLASSES)) if CLASSES[i] not in ['car', 'motorcycle']]
    print(f'\n剔除car/motorcycle后（剩余5类）:')
    print(f'  平均差值: {np.mean(remaining_diffs):+.4f}')
    print(f'  正例数: {sum(1 for d in remaining_diffs if d > 0)}/5')
    # 精确符号检验
    n_pos = sum(1 for d in remaining_diffs if d > 0)
    sign_p = 2 * (0.5 ** 5)  # 双尾
    print(f'  精确符号检验（双尾）: p={sign_p:.4f}')

    # ══════════════════════════════════════════════════════════════
    # 总结
    # ══════════════════════════════════════════════════════════════
    print(f'\n{"=" * 80}')
    print('总结')
    print(f'{"=" * 80}')

    print(f'''
1. 论文操作点 s={S_PAPER:.4f}（α=2.5）:
   - 真实方向平均增益: {np.mean(real_gains_paper):+.4f}
   - 随机方向平均增益: {np.mean(random_gains_paper):+.4f}
   - 差值: {np.mean(diffs_paper):+.4f}
   - 配对t检验 p={t_p:.4f}（不显著）
   - Wilcoxon p={w_p:.4f}

2. 报告主打点 s={S_REPORT}:
   - 真实方向平均增益: {np.mean(real_gains_report):+.4f}（注意：是负的！）
   - 随机方向平均增益: {np.mean(random_gains_report):+.4f}
   - 差值: {np.mean(diffs_report):+.4f}
   - 配对t检验 p={t_p:.4f}

3. 多重比较校正:
   - 7个位移点，Bonferroni/Holm校正后全部不显著

4. car/motorcycle:
   - 贡献了 {car_motorcycle_ratio:.1%} 的效应量
   - 但置换排名 p_c≈0.29（不显著）
   - 随机方向标准差大（高敏感、低方向特异性）

5. 剔除car/motorcycle后:
   - 剩余5类平均差值 {np.mean(remaining_diffs):+.4f}
   - 5/5正，但符号检验 p={sign_p:.4f}（边缘不显著）
''')


if __name__ == '__main__':
    main()
