# -*- coding: utf-8 -*-
"""summarize_expanded.py — 跨域扩展评估汇总（分层报告 + concept-stratified bootstrap CI）

读 实验数据/expanded_eval_results.json，输出分层报告（markdown + JSON）。

分层（用户拍板，跑前锁定的论文身份）：
  6 目标类 = 3 源语义概念；vehicle 原型用于 4 类 → bootstrap 按源概念分层，
  不能把 6 类当 6 个独立观测独立重采样（会因 vehicle 重复而人为缩窄 CI）。

报告区块：
  1. 严格/近严格跨域组（airplane, boat, car）
  2. vehicle 子类扩展压力测试组（car, truck, bus, motorcycle）
  3. 6 类宏平均 + 3 源概念均值 + vehicle 4 子类均值（并列给出，不混）
  4. shot 曲线（α=0.5）与 α 灵敏度（K=10）
  5. concept-stratified bootstrap 95% CI（仅对主表 6 类；vehicle 层内重采样）
  6. 语义错配负对照（独立区块，不进主表）
  7. 内部一致性检查：K>0 α=1（源原型、排除校准图）vs zero_shot（全池）
"""
import json
import numpy as np
from pathlib import Path

SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
IN = SAVE_DIR / 'expanded_eval_results.json'
OUT_MD = SAVE_DIR / 'expanded_eval_report.md'
OUT_JSON = SAVE_DIR / 'expanded_eval_summary.json'

ALPHA = 0.5
K_FS = 10
K_CURVE = [1, 2, 3, 5, 10]
N_BOOT = 10000
SEED = 42

# 源概念 → 目标类
CONCEPTS = {
    'airplane': ['COCO-airplane'],
    'ship':     ['COCO-boat'],
    'vehicle':  ['COCO-car', 'COCO-truck', 'COCO-bus', 'COCO-motorcycle'],
}
STRICT = ['COCO-airplane', 'COCO-boat', 'COCO-car']
VEHICLE = ['COCO-car', 'COCO-truck', 'COCO-bus', 'COCO-motorcycle']


def split_stats(row, key):
    """跨 10 split 的均值±std。"""
    vals = [row['splits'][s][key] for s in row['splits']]
    return float(np.mean(vals)), float(np.std(vals))


def concept_stratified_bootstrap(aps_by_class, n_boot=N_BOOT, seed=SEED):
    """按 3 源概念分层的 bootstrap 均值。vehicle 层内重采样 4 个（有放回）。"""
    rng = np.random.default_rng(seed)
    airplane = aps_by_class['COCO-airplane']
    boat = aps_by_class['COCO-boat']
    v = np.array([aps_by_class[c] for c in CONCEPTS['vehicle']])
    out = np.empty(n_boot)
    for b in range(n_boot):
        vs = v[rng.integers(0, len(v), size=len(v))]
        out[b] = (airplane + boat + vs.mean()) / 3.0
    lo, hi = np.percentile(out, [2.5, 97.5])
    return lo, hi, float(np.mean(out))


def fmt_ci(lo, hi):
    return f'[{lo:.4f}, {hi:.4f}]'


def main():
    data = json.loads(IN.read_text(encoding='utf-8'))
    M, N = data['main'], data['neg_controls']
    L = []
    P = L.append

    # ── 每类主结果 ──
    P('## 1. 逐类结果（baseline / zero-shot / few-shot K=10 α=0.5）\n')
    P('| 类 | 组 | baseline | zero-shot(Δ) | few-shot K10 α.5 (Δ) |')
    P('|---|---|---|---|---|')
    per_class = {}
    for name, r in M.items():
        base = r['baseline']
        zs = r['zero_shot']
        fs, fs_std = split_stats(r, f'K{K_FS}_a{ALPHA}')
        per_class[name] = {'baseline': base, 'zero': zs,
                           f'fewshot_K{K_FS}_a{ALPHA}': fs, 'std': fs_std}
        P(f"| {name} | {r['group']} | {base:.4f} | {zs:.4f} ({zs-base:+.4f}) | "
          f"{fs:.4f}±{fs_std:.4f} ({fs-base:+.4f}) |")
    P('')

    # ── 分组均值 ──
    P('## 2. 分组均值（并列报告，不混用）\n')
    P('| 口径 | baseline | zero-shot | few-shot K10 α.5 |')
    P('|---|---|---|---|')
    groups = {
        '严格跨域组 (airplane/boat/car)': STRICT,
        'vehicle 压力测试组 (car/truck/bus/moto)': VEHICLE,
        '6 类宏平均': list(M),
    }
    group_mean = {}
    for label, sel in groups.items():
        base = np.mean([M[n]['baseline'] for n in sel])
        zs = np.mean([M[n]['zero_shot'] for n in sel])
        fs = np.mean([split_stats(M[n], f'K{K_FS}_a{ALPHA}')[0] for n in sel])
        group_mean[label] = {'baseline': base, 'zero': zs, 'fewshot': fs}
        P(f'| {label} | {base:.4f} | {zs:.4f} ({zs-base:+.4f}) | {fs:.4f} ({fs-base:+.4f}) |')
    # 3 源概念均值（每概念内类平均，再平均）
    src_means = {}
    for concept, clist in CONCEPTS.items():
        base = np.mean([M[n]['baseline'] for n in clist])
        fs = np.mean([split_stats(M[n], f'K{K_FS}_a{ALPHA}')[0] for n in clist])
        src_means[concept] = {'baseline': base, 'fewshot': fs}
    three_src = np.mean([v['baseline'] for v in src_means.values()])
    three_src_fs = np.mean([v['fewshot'] for v in src_means.values()])
    P(f'| 3 源概念均值 (airplane/ship/vehicle 各取类内平均) | {three_src:.4f} | — | '
      f'{three_src_fs:.4f} ({three_src_fs-three_src:+.4f}) |')
    P('')

    # ── shot 曲线 + α 灵敏度 ──
    P('## 3. shot 曲线（α=0.5）与 α 灵敏度（K=10）\n')
    P('shot 曲线：6 类宏平均（few-shot 跨 split 均值）\n')
    P('| K | 1 | 2 | 3 | 5 | 10 |')
    P('|---|---|---|---|---|---|')
    curve = {}
    for k in K_CURVE:
        fs = np.mean([split_stats(M[n], f'K{k}_a{ALPHA}')[0] for n in M])
        curve[k] = fs
    P('| 6类宏平均 few-shot | ' + ' | '.join(f'{curve[k]:.4f}' for k in K_CURVE) + ' |')
    P('')
    P('α 灵敏度（K=10，6 类宏平均 few-shot）\n')
    P('| α | 0 | 0.25 | 0.5 | 0.75 | 1 |')
    P('|---|---|---|---|---|---|')
    alpha_sens = {}
    for a in data['config']['alpha_list']:
        fs = np.mean([split_stats(M[n], f'K{K_FS}_a{a}')[0] for n in M])
        alpha_sens[a] = fs
    P('| 6类宏平均 | ' + ' | '.join(f'{alpha_sens[a]:.4f}' for a in data['config']['alpha_list']) + ' |')
    P('')

    # ── concept-stratified bootstrap CI ──
    P('## 4. concept-stratified bootstrap 95% CI（主表 6 类，3 源概念分层）\n')
    P('vehicle 层内重采样 4 类（有放回）；airplane/ship 层各 1 类取原值。'
      '不等价于 6 个独立类独立重采样。\n')
    P('| 指标 | 点估计 | 95% CI |')
    P('|---|---|---|')
    for metric, key in [('few-shot AP (K=10, α=0.5)', 'fewshot'),
                        ('Δ vs baseline (few-shot−baseline)', 'delta')]:
        if metric.startswith('Δ'):
            aps = {n: split_stats(M[n], f'K{K_FS}_a{ALPHA}')[0] - M[n]['baseline']
                   for n in M}
        else:
            aps = {n: split_stats(M[n], f'K{K_FS}_a{ALPHA}')[0] for n in M}
        lo, hi, mu = concept_stratified_bootstrap(aps)
        P(f'| {metric} | {mu:.4f} | {fmt_ci(lo, hi)} |')
    P('')

    # ── 负对照 ──
    P('## 5. 语义错配负对照（semantic-mismatch negative controls，仅附录）\n')
    P('这些源类是场景/设施区域（tennis court / train station），目标类是物体'
      '（sports ball / train），类别语义未对齐，故不计入主表均值、不参与'
      '"正增益"结论，无论正负都完整报告。它们只回答：当源-目标仅存在场景关联、'
      '而非类别对应时，原型校准能否工作。\n')
    P('| 源→目标 | baseline | zero-shot(Δ) | few-shot K10 α.5 (Δ) |')
    P('|---|---|---|---|')
    for name, r in N.items():
        base = r['baseline']
        zs = r['zero_shot']
        fs, fs_std = split_stats(r, f'K{K_FS}_a{ALPHA}')
        P(f'| {name} ({r["src_concept"]}→{name.replace("NEG-","")}) | {base:.4f} | '
          f'{zs:.4f} ({zs-base:+.4f}) | {fs:.4f} ({fs-base:+.4f}) |')
    P('')

    # ── 一致性检查 ──
    P('## 6. 内部一致性检查\n')
    P('K>0 时 α=1（源原型、排除校准图）与 zero_shot（全池）差异应小；'
      'K=1 时测试集只少 1 张。\n')
    P('| 类 | zero_shot(全池) | K=1 α=1(排1图) | 差 |')
    P('|---|---|---|---|')
    for name, r in M.items():
        zs = r['zero_shot']
        a1 = split_stats(r, 'K1_alpha1')[0]
        P(f'| {name} | {zs:.4f} | {a1:.4f} | {a1-zs:+.4f} |')
    P('')

    md = '\n'.join(L)
    OUT_MD.write_text(md, encoding='utf-8')

    summary = {
        'per_class': per_class, 'group_mean': group_mean,
        'src_concept_mean': {k: v for k, v in src_means.items()},
        'three_src_concept_mean': {'baseline': three_src, 'fewshot': three_src_fs},
        'shot_curve': curve, 'alpha_sensitivity': alpha_sens,
        'concept_stratified_bootstrap': {
            'fewshot_ap': concept_stratified_bootstrap(
                {n: split_stats(M[n], f'K{K_FS}_a{ALPHA}')[0] for n in M}),
            'delta_vs_baseline': concept_stratified_bootstrap(
                {n: split_stats(M[n], f'K{K_FS}_a{ALPHA}')[0] - M[n]['baseline'] for n in M}),
        },
        'neg_controls': {n: {'baseline': r['baseline'], 'zero': r['zero_shot'],
                             'fewshot': split_stats(r, f'K{K_FS}_a{ALPHA}')[0]}
                         for n, r in N.items()},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(md)
    print(f'\n✅ 报告 → {OUT_MD}')
    print(f'✅ 汇总 → {OUT_JSON}')


if __name__ == '__main__':
    main()
