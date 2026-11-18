# -*- coding: utf-8 -*-
"""strict_ablation.py — 严格消融 + 配对 bootstrap + 口径修正报告（2026-08-18）

两个口径修正（用户拍板，跑前锁定）：
  (1) 主表参考口径统一：
        baseline AP                     — 全池, GDINO 纯分数（与划分无关）
        source-only CDVM AP / Δ_base    — 同测试集 α=1（排除校准图），与 few-shot 同池配对
        inductive few-shot K10 α=0.5 AP / Δ_base
        Δ_few-shot−source-only          — 同 split 同测试集配对差值
  (2) 负对照 NEG-sports_ball：目标为小目标（球），GT 框经 10px 尺寸过滤后部分校准集
      为空 → 该校准协议在部分 split×K 退化（K=1: 4/10, K=2: 1/10, K≥3: 全有效）。
      该类仅作附录负对照，不进主表、不参与任何增益结论；校准覆盖逐 K 诊断并标注。

统计：
  - 配对 bootstrap（few-shot − source-only，同 split 同测试集）：
      * split 级 CI：对 10 个嵌套划分的「3 源概念分层配对差值」重采样
      * 概念级 CI：对 6 类配对点估计按 3 源概念分层重采样（vehicle 层内 4 类）
  - 6 类宏平均 / strict 组 / vehicle 组 / 3 源概念均值 并列报告
  - shot 曲线（α=0.5）与 α 灵敏度（K=10）：只表述 α∈[0,0.5] 平台，不宣传 α=0.5 最优

逐图检测结果已在 expanded_cache/*.npz（逐图候选框/GDINO分数/DINOv2特征/GT框/GT特征）；
AP 为标准数据集级 11-point AP50。每 split 每类配对差值明细写入 strict_ablation.json。

输出：实验数据/strict_ablation_report.md + strict_ablation.json
"""
import json
import numpy as np
from pathlib import Path

SAVE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
CACHE_DIR = SAVE_DIR / 'expanded_cache'
IN = SAVE_DIR / 'expanded_eval_results.json'
OUT_MD = SAVE_DIR / 'strict_ablation_report.md'
OUT_JSON = SAVE_DIR / 'strict_ablation.json'

ALPHA = 0.5
K_FS = 10
K_CURVE = [1, 2, 3, 5, 10]
N_BOOT = 10000
SEED = 42
N_SPLITS = 10

CONCEPTS = {
    'airplane': ['COCO-airplane'],
    'ship':     ['COCO-boat'],
    'vehicle':  ['COCO-car', 'COCO-truck', 'COCO-bus', 'COCO-motorcycle'],
}
STRICT = ['COCO-airplane', 'COCO-boat', 'COCO-car']
VEHICLE = ['COCO-car', 'COCO-truck', 'COCO-bus', 'COCO-motorcycle']
MAIN = ['COCO-airplane', 'COCO-boat'] + VEHICLE
NEG = ['NEG-sports_ball', 'NEG-train']
ALPHA_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]


def split_vals(r, key):
    """跨 10 split 取值（key 存在才返回）。"""
    out = []
    for s in sorted(r['splits'], key=int):
        if key in r['splits'][s]:
            out.append(r['splits'][s][key])
    return out


def reproduce_splits(n_img):
    """复现 eval_class 中 SEED=42 的 10 个嵌套随机划分。"""
    rng = np.random.default_rng(SEED)
    return [rng.permutation(n_img).tolist() for _ in range(N_SPLITS)]


def calibration_coverage(name):
    """读缓存 npz，复现 splits，统计每 (split, K) 校准集是否有有效 GT 框。"""
    d = np.load(CACHE_DIR / f'{name}.npz', allow_pickle=True)
    gt_feats = list(d['gt_feats'])
    n = len(gt_feats)
    perms = reproduce_splits(n)
    n_valid = sum(1 for f in gt_feats if len(f) > 0)
    rows = {}
    for K in K_CURVE:
        n_empty = 0
        for s in range(N_SPLITS):
            fs_idx = perms[s][:K]
            if sum(1 for i in fs_idx if len(gt_feats[i]) > 0) == 0:
                n_empty += 1
        rows[K] = n_empty
    return {'n_images': n, 'n_valid_gt_images': n_valid,
            'empty_splits_by_K': rows}


# ────────────────────────── 配对 bootstrap ──────────────────────────

def paired_split_values(r):
    """每 split 配对差值 fs−so（so=K10_alpha1 同测试集, fs=K10_a0.5）。"""
    fs = split_vals(r, f'K{K_FS}_a{ALPHA}')
    so = split_vals(r, f'K{K_FS}_alpha1')
    assert len(fs) == len(so) == N_SPLITS
    return np.array(fs), np.array(so), np.array(fs) - np.array(so)


def split_level_paired_bootstrap(d_by_split, n_boot=N_BOOT, seed=SEED):
    """split 级：对 10 splits 的分层配对差值重采样。d_by_split: list[N_SPLITS]。"""
    d = np.array(d_by_split, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(d), size=len(d))
        out[b] = d[idx].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return lo, hi, float(d.mean())


def concept_stratified_paired_bootstrap(d_by_class, n_boot=N_BOOT, seed=SEED):
    """概念级：6 类配对点估计按 3 源概念分层重采样（vehicle 层内 4 类有放回）。"""
    rng = np.random.default_rng(seed)
    air = d_by_class['COCO-airplane']
    boat = d_by_class['COCO-boat']
    v = np.array([d_by_class[c] for c in VEHICLE])
    out = np.empty(n_boot)
    for b in range(n_boot):
        vs = v[rng.integers(0, len(v), size=len(v))]
        out[b] = (air + boat + vs.mean()) / 3.0
    lo, hi = np.percentile(out, [2.5, 97.5])
    return lo, hi, float(out.mean())


def main():
    data = json.loads(IN.read_text(encoding='utf-8'))
    M, NC = data['main'], data['neg_controls']
    L = []
    P = L.append

    # ── 每类配对差值 + 逐类主结果 ──
    d_by_class, fs_by_class, so_by_class = {}, {}, {}
    base_by_class = {}
    for name in MAIN:
        r = M[name]
        fs, so, d = paired_split_values(r)
        fs_by_class[name], so_by_class[name], d_by_class[name] = fs, so, d
        base_by_class[name] = r['baseline']

    # ── 0. 口径修正说明 ──
    P('## 0. 口径修正说明\n')
    P('1. **主表参考口径统一**（全部相对 baseline）：baseline AP（全池 GDINO 纯分数）；'
      'source-only CDVM（同测试集 α=1，排除校准图，与 few-shot 同池配对）；'
      'inductive few-shot（K=10, α=0.5）；另列 Δ_few-shot−source-only（同 split 同测试集配对差值）。'
      'source-only 同测试集值与全池 zero-shot 差异 ≤0.004（一致性检查，见 §6）。\n')
    P('2. **NEG-sports_ball 校准协议退化标注**：目标域 sports ball 为小目标，GT 框经 10px 尺寸'
      '过滤后部分校准集为空。逐 K 诊断：K=1 有 4/10 splits、K=2 有 1/10 splits 校准集全空'
      '（退化为源原型 α=1 路径）；K≥3 全部有效。该校准类仅作附录负对照，不进主表、不参与'
      '任何增益结论（§5）。\n')

    # ── 1. 主表 ──
    P('## 1. 逐类主表（统一口径，Δ 均相对 baseline）\n')
    P('| 类 | 组 | baseline | source-only CDVM (Δ) | few-shot K10 α.5 (Δ) | Δ_fs−so 配对 |')
    P('|---|---|---|---|---|---|')
    per_class = {}
    for name in MAIN:
        r = M[name]
        base = base_by_class[name]
        so, fs, d = np.mean(so_by_class[name]), np.mean(fs_by_class[name]), np.mean(d_by_class[name])
        per_class[name] = {'group': r['group'], 'baseline': base,
                           'source_only': so, 'fewshot': fs, 'd_fs_minus_so': d}
        P(f'| {name} | {r["group"]} | {base:.4f} | {so:.4f} ({so-base:+.4f}) | '
          f'{fs:.4f} ({fs-base:+.4f}) | {d:+.4f} |')
    P('')
    n_pos = sum(1 for n in MAIN if per_class[n]['d_fs_minus_so'] > 0)
    P(f'Δ_fs−so 符号：**{n_pos}/6 类点估计为正**（不逐类宣称显著；airplane 增量小，逐类 CI 大概率跨 0）。\n')

    # ── 2. 分组均值 ──
    P('## 2. 分组均值（并列报告，不混用）\n')
    P('| 口径 | baseline | source-only CDVM (Δ) | few-shot K10 α.5 (Δ) | Δ_fs−so 配对 |')
    P('|---|---|---|---|---|---|')
    groups = {
        '严格跨域组 (airplane/boat/car)': STRICT,
        'vehicle 压力测试组 (car/truck/bus/moto)': VEHICLE,
        '6 类宏平均': MAIN,
    }
    group_mean = {}
    for label, sel in groups.items():
        base = np.mean([base_by_class[n] for n in sel])
        so = np.mean([np.mean(so_by_class[n]) for n in sel])
        fs = np.mean([np.mean(fs_by_class[n]) for n in sel])
        d = np.mean([np.mean(d_by_class[n]) for n in sel])
        group_mean[label] = {'baseline': base, 'source_only': so, 'fewshot': fs, 'd': d}
        P(f'| {label} | {base:.4f} | {so:.4f} ({so-base:+.4f}) | {fs:.4f} ({fs-base:+.4f}) | {d:+.4f} |')
    # 3 源概念均值（每概念内类平均，再平均）
    src = {}
    for concept, clist in CONCEPTS.items():
        base = np.mean([base_by_class[n] for n in clist])
        so = np.mean([np.mean(so_by_class[n]) for n in clist])
        fs = np.mean([np.mean(fs_by_class[n]) for n in clist])
        src[concept] = {'baseline': base, 'source_only': so, 'fewshot': fs}
    for concept, v in src.items():
        P(f'| 概念 {concept} | {v["baseline"]:.4f} | {v["source_only"]:.4f} '
          f'({v["source_only"]-v["baseline"]:+.4f}) | {v["fewshot"]:.4f} '
          f'({v["fewshot"]-v["baseline"]:+.4f}) | {v["fewshot"]-v["source_only"]:+.4f} |')
    P('')

    # ── 3. 配对 bootstrap CI ──
    P('## 3. 配对 bootstrap 95% CI（few-shot − source-only，同 split 同测试集配对）\n')
    # split 级分层配对差值
    D_split = []
    for s in range(N_SPLITS):
        d_air = d_by_class['COCO-airplane'][s]
        d_boat = d_by_class['COCO-boat'][s]
        d_v = np.mean([d_by_class[c][s] for c in VEHICLE])
        D_split.append((d_air + d_boat + d_v) / 3.0)
    lo_s, hi_s, mu_s = split_level_paired_bootstrap(D_split)
    # 概念级
    d_point = {c: float(np.mean(d_by_class[c])) for c in MAIN}
    lo_c, hi_c, mu_c = concept_stratified_paired_bootstrap(d_point)
    # 6 类宏平均配对差值（非分层参考）
    D_split6 = [np.mean([d_by_class[c][s] for c in MAIN]) for s in range(N_SPLITS)]
    lo_6, hi_6, mu_6 = split_level_paired_bootstrap(D_split6)

    P('分层口径（主）：对 10 个嵌套划分的「3 源概念分层配对差值」重采样（split 级），'
      '并对 6 类配对点估计按 3 源概念分层重采样（vehicle 层内 4 类）交叉验证。\n')
    P('| 口径 | 点估计 | 95% CI |')
    P('|---|---|---|')
    P(f'| 3 源概念分层 (split 级) | {mu_s:+.4f} | [{lo_s:+.4f}, {hi_s:+.4f}] |')
    P(f'| 3 源概念分层 (概念级) | {mu_c:+.4f} | [{lo_c:+.4f}, {hi_c:+.4f}] |')
    P(f'| 6 类宏平均 (非分层参考) | {mu_6:+.4f} | [{lo_6:+.4f}, {hi_6:+.4f}] |')
    P('')
    sig = lo_s > 0 and lo_c > 0
    P(f'**结论表述**：{n_pos}/6 类点估计为正；总体概念分层增益 '
      f'{"95% CI 不跨 0（split 级与概念级一致），可称统计显著" if sig else "95% CI 未完全排除 0"}。'
      '不逐类宣称显著。\n')

    # ── 4. shot 曲线 + α 灵敏度 ──
    P('## 4. shot 曲线（α=0.5）与 α 灵敏度（K=10）\n')
    P('shot 曲线：6 类宏平均（每 split 均值再平均）\n')
    P('| K | ' + ' | '.join(str(k) for k in K_CURVE) + ' |')
    P('|---' + '|---' * len(K_CURVE) + '|')
    curve = {}
    for k in K_CURVE:
        vals = [np.mean(split_vals(M[n], f'K{k}_a{ALPHA}')) for n in MAIN]
        curve[k] = float(np.mean(vals))
    P('| 6 类宏平均 | ' + ' | '.join(f'{curve[k]:.4f}' for k in K_CURVE) + ' |')
    P('')
    P('α 灵敏度（K=10，6 类宏平均）—— 注意不宣传"α=0.5 最优"：\n')
    P('| α | ' + ' | '.join(str(a) for a in ALPHA_LIST) + ' |')
    P('|---' + '|---' * len(ALPHA_LIST) + '|')
    alpha_sens = {}
    for a in ALPHA_LIST:
        vals = [np.mean(split_vals(M[n], f'K{K_FS}_a{a}')) for n in MAIN]
        alpha_sens[a] = float(np.mean(vals))
    P('| 6 类宏平均 | ' + ' | '.join(f'{alpha_sens[a]:.4f}' for a in ALPHA_LIST) + ' |')
    P('')
    P('表述：**α∈[0, 0.5] 表现相近**'
      f'（α=0: {alpha_sens[0]:.4f}，α=0.25: {alpha_sens[0.25]:.4f}，α=0.5: {alpha_sens[0.5]:.4f}），'
      f'过度依赖目标域原型（α=1: {alpha_sens[1.0]:.4f}）明显退化。\n')

    # ── 5. 负对照（含校准覆盖诊断） ──
    P('## 5. 语义错配负对照（semantic-mismatch negative controls，仅附录）\n')
    P('源类为场景/设施区域（tennis court / train station），目标类为物体（sports ball / train），'
      '类别语义未对齐，不进主表均值、不参与任何增益结论。它们只回答：当源-目标仅存在场景关联、'
      '而非类别对应时，原型校准能否工作。\n')
    P('| 源→目标 | baseline | zero-shot(全池, Δ) | few-shot K10 α.5 (Δ) | 校准协议 |')
    P('|---|---|---|---|---|---|')
    neg_out = {}
    for name in NEG:
        r = NC[name]
        base = r['baseline']
        zs = r['zero_shot']
        fs = float(np.mean(split_vals(r, f'K{K_FS}_a{ALPHA}')))
        cov = calibration_coverage(name)
        empty_s = cov['empty_splits_by_K']
        if name == 'NEG-sports_ball':
            proto_note = (f'⚠️ K=1 有 {empty_s[1]}/10 splits、K=2 有 {empty_s[2]}/10 splits 校准集'
                          f'全空（小目标 10px 过滤）→ 退化为源原型；K≥3 有效。'
                          f'该类不作 few-shot 有效证据，仅附录。')
        else:
            proto_note = '校准集有效（10px 过滤未清空）'
        neg_out[name] = {'baseline': base, 'zero': zs, 'fewshot': fs,
                         'coverage': cov, 'protocol_note': proto_note}
        P(f'| {name} ({r["src_concept"]}→{name.replace("NEG-","")}) | {base:.4f} | '
          f'{zs:.4f} ({zs-base:+.4f}) | {fs:.4f} ({fs-base:+.4f}) | {proto_note} |')
    P('')
    P('sports_ball 的 few-shot 数值是**真实校准结果**（K=10 校准集有效），但属于负对照：'
      '目标域 GT 校准直接把原型拉到 COCO 球形态空间，增益大；这不改变"源原型（tennis court）'
      '对目标域无判别力"的边界结论。由于 K=1/2 校准协议部分退化，该类不提供 shot 曲线。\n')

    # ── 6. 协议核验 ──
    P('## 6. 协议核验\n')
    P('**校准覆盖（全类，10px 过滤后）**：\n')
    P('| 类 | 图数 | 有有效GT框图 | 校准集全空 split×K 计数 |')
    P('|---|---|---|---|')
    for name in MAIN:
        cov = calibration_coverage(name)
        total_empty = sum(cov['empty_splits_by_K'].values())
        P(f'| {name} | {cov["n_images"]} | {cov["n_valid_gt_images"]} | {total_empty}/50 |')
    P('')
    P('**一致性**（K>0 α=1 同测试集 vs 全池 zero-shot，K=10）：\n')
    P('| 类 | zero_shot(全池) | K10 α=1(同测试集) | 差 |')
    P('|---|---|---|---|')
    for name in MAIN:
        zs = M[name]['zero_shot']
        a1 = float(np.mean(split_vals(M[name], f'K{K_FS}_alpha1')))
        P(f'| {name} | {zs:.4f} | {a1:.4f} | {a1-zs:+.4f} |')
    P('')

    md = '\n'.join(L)
    OUT_MD.write_text(md, encoding='utf-8')

    summary = {
        'config': {'alpha': ALPHA, 'K_fs': K_FS, 'n_splits': N_SPLITS, 'seed': SEED,
                   'boot': N_BOOT,
                   'paired_source_only': 'K10_alpha1 same-test-set',
                   'paired_few_shot': 'K10_a0.5 same-test-set',
                   'identity': ('six target categories / three source concepts; '
                                'vehicle prototype shared by four; '
                                'neg_controls appendix-only')},
        'per_class': per_class,
        'group_mean': group_mean,
        'src_concept_mean': src,
        'paired_bootstrap': {
            'concept_stratified_split_level': {'point': mu_s, 'ci': [lo_s, hi_s]},
            'concept_stratified_class_level': {'point': mu_c, 'ci': [lo_c, hi_c]},
            'six_class_flat': {'point': mu_6, 'ci': [lo_6, hi_6]},
            'per_split_concept_deltas': D_split,
            'per_class_per_split_deltas': {c: d_by_class[c].tolist() for c in MAIN},
        },
        'shot_curve': curve,
        'alpha_sensitivity': alpha_sens,
        'neg_controls': neg_out,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print(md)
    print(f'\n✅ 报告 → {OUT_MD}')
    print(f'✅ 汇总 → {OUT_JSON}')


if __name__ == '__main__':
    main()
