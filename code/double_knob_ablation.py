# -*- coding: utf-8 -*-
"""double_knob_ablation.py — 双旋钮判定实验（最后一轮，纯缓存重放）

协议（结果前冻结）：
  π⁺(α) = normalize((1−α)·π_S⁺ + α·π_T⁺)   ← α = 目标正原型权重
  s_c(x) = cos(f,π⁺) − β·cos(f,π⁻)           ← β ∈ {0,1}，β=1 用当前 NEG_LAM=0.5 门控
  α ∈ {0, 0.5, 1}；β ∈ {0, 1}；K ∈ {1,3,5,10}；10 splits；同 split 同测试集配对
  测试集 = 池 − 校准（K 张图），校准 = perm[:K]（嵌套，与冻结口径一致）

非 oracle 调度（看结果前冻结）：
  K ≤ 3 → (α=0.5, β=0)  保留部分源正原型 + 关闭负原型
  K ≥ 5 → (α=1.0, β=1)  偏向目标正原型 + 启用负原型

判定口径：
  - 3 源概念分层配对 bootstrap（split 级，与主论文一致）为主
  - 关键比较：schedule vs fixed(0.5,1)；schedule vs pure_tgt(1,0)
  - oracle = 每 K 6 配置里宏平均最优（事后上界，仅参考）

一致性核对：(0,1)/(0.5,1)/(1,1) ↔ expanded_eval_results.json K*_a1.0/a0.5/a0.0；
         (0,0)/(1,0) ↔ naive_baseline.json src/tgt_pos_only；(0.5,0) 为新计算。
输出：实验数据/double_knob_ablation.json + double_knob_ablation_report.md
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
import torch
from pathlib import Path
import unified_eval as U

NEG_LAM = 0.5
LAYER = 11
K_LIST = [1, 3, 5, 10]
N_SPLITS = 10
SEED = 42
N_BOOT = 10000

MAIN = [
    ('COCO-airplane', 'airplane'),
    ('COCO-boat', 'ship'),
    ('COCO-car', 'vehicle'),
    ('COCO-truck', 'vehicle'),
    ('COCO-bus', 'vehicle'),
    ('COCO-motorcycle', 'vehicle'),
]
CONCEPTS = {'airplane': ['COCO-airplane'], 'ship': ['COCO-boat'],
            'vehicle': ['COCO-car', 'COCO-truck', 'COCO-bus', 'COCO-motorcycle']}
SHORT = {'COCO-airplane': 'airplane', 'COCO-boat': 'boat', 'COCO-car': 'car',
         'COCO-truck': 'truck', 'COCO-bus': 'bus', 'COCO-motorcycle': 'motorcycle'}

ALPHAS = [0.0, 0.5, 1.0]
BETAS = [0.0, 1.0]


def reproduce_splits(n):
    rng = np.random.default_rng(SEED)
    return [rng.permutation(n).tolist() for _ in range(N_SPLITS)]


def get_neg_proto_np(bank, dior_cls):
    pos = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][LAYER]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    return (neg / neg.norm()).cpu().numpy().astype(np.float32)


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_np, neg_np):
    all_b, all_s, all_g = [], [], []
    for i in test_idx:
        boxes = cand_boxes[i]
        if len(boxes) == 0:
            all_g.extend(gt_boxes[i].tolist())
            continue
        gd = np.clip(cand_gd[i], 1e-6, 1.0)
        F = cand_feats[i]
        if len(F):
            vis = F @ P_np
            if neg_np is not None:
                vis = vis - NEG_LAM * (F @ neg_np)
            vis = np.clip(vis, 1e-6, 1.0)
            score = np.sqrt(gd * vis)
        else:
            score = gd
        all_b.extend(boxes.tolist())
        all_s.extend(score.tolist())
        all_g.extend(gt_boxes[i].tolist())
    return U.compute_ap50(all_b, all_s, all_g)


def concept_stratified_mean(per_split_cls, s):
    """3 源概念分层宏平均（split s）。per_split_cls: {class: value}。"""
    v = np.mean([per_split_cls[c] for c in CONCEPTS['airplane']])
    b = np.mean([per_split_cls[c] for c in CONCEPTS['ship']])
    vv = np.mean([per_split_cls[c] for c in CONCEPTS['vehicle']])
    return (v + b + vv) / 3.0


def split_bootstrap(d, n_boot=N_BOOT, seed=SEED):
    d = np.array(d, dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(d), size=len(d))
        out[b] = d[idx].mean()
    return float(d.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    bank = U.get_bank()

    # val[strategy_key][K] -> per-split concept-stratified mean; also 6-class macro per split
    # 这里直接存 per split 的 6 类宏平均 + 概念分层，key=(a,b)
    cs_vals = {(a, b): {k: [] for k in K_LIST} for a in ALPHAS for b in BETAS}   # 概念分层 per split
    m6_vals = {(a, b): {k: [] for k in K_LIST} for a in ALPHAS for b in BETAS}   # 6类宏 per split
    per_class_vals = {c: {(a, b): {k: [] for k in K_LIST} for a in ALPHAS for b in BETAS}
                      for c, _ in MAIN}
    calib_boxes = {k: [] for k in K_LIST}
    consistency = {}

    for name, src_cls in MAIN:
        _npz = np.load(U.SAVE_DIR / 'expanded_cache' / f'{name}.npz', allow_pickle=True)
        cand_boxes = list(_npz['cand_boxes'])
        cand_gd = list(_npz['cand_gd'])
        cand_feats = list(_npz['cand_feats'])
        gt_boxes = list(_npz['gt_boxes'])
        gt_feats = list(_npz['gt_feats'])
        n_img = len(gt_feats)
        P_src = bank[src_cls][LAYER]['prototype'].cpu().numpy().astype(np.float32)
        neg = get_neg_proto_np(bank, src_cls)
        perms = reproduce_splits(n_img)

        for s in range(N_SPLITS):
            for K in K_LIST:
                k_idx = perms[s][:K]
                test_idx = perms[s][K:]
                calib_boxes[K].append(sum(len(gt_feats[i]) for i in k_idx if len(gt_feats[i])))
                gtf = np.concatenate([gt_feats[i] for i in k_idx if len(gt_feats[i])],
                                     axis=0) if any(len(gt_feats[i]) for i in k_idx) else np.empty((0, 768), np.float32)
                if len(gtf) == 0:
                    P_tgt = P_src.copy()
                else:
                    P_tgt = normalize(gtf.mean(axis=0)).astype(np.float32)
                for a in ALPHAS:
                    P_cal = normalize((1 - a) * P_src + a * P_tgt).astype(np.float32)
                    for b in BETAS:
                        neg_np = neg if b == 1.0 else None
                        ap = eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_cal, neg_np)
                        per_class_vals[name][(a, b)][K].append(ap)

        # per split 聚合（在类的维度外还需要跨类平均，下面统一做）

    # ── 跨类聚合 ──
    for (a, b) in per_class_vals[MAIN[0][0]]:
        for K in K_LIST:
            for s in range(N_SPLITS):
                vals_cls = {c: per_class_vals[c][(a, b)][K][s] for c, _ in MAIN}
                m6_vals[(a, b)][K].append(float(np.mean(list(vals_cls.values()))))
                cs_vals[(a, b)][K].append(concept_stratified_mean(vals_cls, s))

    # ── 一致性核对 ──
    exp = json.loads((U.SAVE_DIR / 'expanded_eval_results.json').read_text(encoding='utf-8'))['main']
    nb = json.loads((U.SAVE_DIR / 'naive_baseline.json').read_text(encoding='utf-8'))
    # (a,b) → JSON key
    key_map = {(0.0, 1.0): 'K{k}_a1.0', (0.5, 1.0): 'K{k}_a0.5', (1.0, 1.0): 'K{k}_a0.0'}
    maxdiff = 0.0
    for (a, b), f in key_map.items():
        for K in K_LIST:
            for s in range(N_SPLITS):
                js = exp[MAIN[0][0]]['splits'][str(s)][f.format(k=K)]
                # 用第一类对拍
                got = per_class_vals[MAIN[0][0]][(a, b)][K][s]
                maxdiff = max(maxdiff, abs(js - got))
    # naive: (0,0)=src_pos_only, (1,0)=tgt_pos_only
    for (a, b), nbk in [((0.0, 0.0), 'src_pos_only'), ((1.0, 0.0), 'tgt_pos_only')]:
        for K in K_LIST:
            for s in range(N_SPLITS):
                got = per_class_vals[MAIN[0][0]][(a, b)][K][s]
                # naive_baseline 里 per_split[class][variant][K][s]
                ref = nb['per_split'][MAIN[0][0]][nbk][str(K)][s]
                maxdiff = max(maxdiff, abs(ref - got))
    consistency['maxdiff_airplane_all_overlap'] = maxdiff

    # ── 关键配置 ──
    def m6(a, b, k):
        return float(np.mean(m6_vals[(a, b)][k]))

    def cs(a, b, k):
        return float(np.mean(cs_vals[(a, b)][k]))

    fixed = {(a, b): (0.5, 1.0)}
    pure_tgt = (1.0, 0.0)
    schedule_cfg = {1: (0.5, 0.0), 3: (0.5, 0.0), 5: (1.0, 1.0), 10: (1.0, 1.0)}

    # oracle per K（6 类宏平均最优）
    oracle = {}
    for K in K_LIST:
        best_a, best_b, best_v = None, None, -1
        for a in ALPHAS:
            for b in BETAS:
                v = m6(a, b, K)
                if v > best_v:
                    best_a, best_b, best_v = a, b, v
        oracle[K] = {'alpha': best_a, 'beta': best_b, 'ap': best_v}

    # 配对 bootstrap：schedule vs fixed, schedule vs pure_tgt（概念分层, split级）
    boot = {}
    for K in K_LIST:
        d_sf = [cs_vals[schedule_cfg[K]][K][s] - cs_vals[(0.5, 1.0)][K][s] for s in range(N_SPLITS)]
        d_st = [cs_vals[schedule_cfg[K]][K][s] - cs_vals[pure_tgt][K][s] for s in range(N_SPLITS)]
        boot[K] = {
            'schedule_vs_fixed': dict(zip(['point', 'lo', 'hi'], split_bootstrap(d_sf))),
            'schedule_vs_pure_tgt': dict(zip(['point', 'lo', 'hi'], split_bootstrap(d_st))),
            'schedule_cfg': schedule_cfg[K],
        }
    # 低 K 合并（K=1,3）与全 K 合并
    for label, ks in [('lowK13', [1, 3]), ('allK', K_LIST)]:
        d_sf = [cs_vals[schedule_cfg[k]][k][s] - cs_vals[(0.5, 1.0)][k][s]
                for k in ks for s in range(N_SPLITS)]
        d_st = [cs_vals[schedule_cfg[k]][k][s] - cs_vals[pure_tgt][k][s]
                for k in ks for s in range(N_SPLITS)]
        boot[label] = {
            'schedule_vs_fixed': dict(zip(['point', 'lo', 'hi'], split_bootstrap(d_sf))),
            'schedule_vs_pure_tgt': dict(zip(['point', 'lo', 'hi'], split_bootstrap(d_st))),
        }
    # oracle 相对 fixed（参考，乐观上界）
    oracle_gain = {K: oracle[K]['ap'] - m6(0.5, 1.0, K) for K in K_LIST}

    # ── 报告 ──
    L = []
    L.append('## 双旋钮消融（α×β×K）— 最后一次判定实验\n')
    L.append('协议：α=目标正原型权重，π⁺=normalize((1−α)π_S⁺+απ_T⁺)；β=负原型开关'
             '（β=1 用 NEG_LAM=0.5 门控，与冻结方法一致）。同 split 同测试集配对，'
             '10 splits，SEED=42。**非 oracle 调度在看结果前冻结**：'
             'K≤3→(α=0.5,β=0)；K≥5→(α=1,β=1)。\n')

    L.append('### 1. 全网格 6 类宏平均\n')
    L.append('| (α,β) | K=1 | K=3 | K=5 | K=10 |')
    L.append('|---|---|---|---|---|')
    for a in ALPHAS:
        for b in BETAS:
            row = [m6(a, b, k) for k in K_LIST]
            tag = ''
            if (a, b) == (0.5, 1.0):
                tag = '  ←fixed'
            if (a, b) == (1.0, 0.0):
                tag = '  ←pure_tgt'
            L.append(f'| α={a:.1f},β={b:.0f} | ' + ' | '.join(f'{v:.4f}' for v in row) + f' |{tag}')
    L.append('')
    L.append('### 2. 关键配置对比\n')
    L.append('| K | fixed(0.5,1) | pure_tgt(1,0) | **schedule** | schedule−fixed | oracle(α,β) | oracle−fixed |')
    L.append('|---|---|---|---|---|---|---|')
    for K in K_LIST:
        f = m6(0.5, 1.0, K)
        pt = m6(1.0, 0.0, K)
        sc = m6(*schedule_cfg[K], K)
        o = oracle[K]
        L.append(f'| K={K} | {f:.4f} | {pt:.4f} | **{sc:.4f}** | {sc-f:+.4f} | '
                 f'{o["ap"]:.4f}(α{o["alpha"]:.1f},β{o["beta"]:.0f}) | {oracle_gain[K]:+.4f} |')
    L.append('')
    L.append('### 3. 配对 bootstrap 95% CI（3 源概念分层，split 级）\n')
    L.append('| 比较 | K | 点估计 | 95% CI |')
    L.append('|---|---|---|---|')
    for K in K_LIST:
        for cname in ['schedule_vs_fixed', 'schedule_vs_pure_tgt']:
            b = boot[K][cname]
            L.append(f'| schedule vs {"fixed" if cname=="schedule_vs_fixed" else "pure_tgt"} | K={K} | '
                     f'{b["point"]:+.4f} | [{b["lo"]:+.4f}, {b["hi"]:+.4f}] |')
    for label, ks, desc in [('lowK13', 'K=1,3 合并', '低 K'), ('allK', 'K=1,3,5,10 合并', '全 K')]:
        for cname in ['schedule_vs_fixed', 'schedule_vs_pure_tgt']:
            b = boot[label][cname]
            L.append(f'| schedule vs {"fixed" if cname=="schedule_vs_fixed" else "pure_tgt"} | {desc} | '
                     f'{b["point"]:+.4f} | [{b["lo"]:+.4f}, {b["hi"]:+.4f}] |')
    L.append('')
    L.append('### 4. 逐类（schedule 配置，α=0.5 CDVM 参考）\n')
    for K in K_LIST:
        a, b = schedule_cfg[K]
        L.append(f'**K={K}（schedule=(α={a:.1f},β={b:.0f})）**\n')
        L.append('| 类 | schedule | fixed(0.5,1) | pure_tgt(1,0) |')
        L.append('|---|---|---|---|')
        for cname, _ in MAIN:
            sc = float(np.mean(per_class_vals[cname][(a, b)][K]))
            f = float(np.mean(per_class_vals[cname][(0.5, 1.0)][K]))
            pt = float(np.mean(per_class_vals[cname][(1.0, 0.0)][K]))
            L.append(f'| {SHORT[cname]} | {sc:.4f} | {f:.4f} | {pt:.4f} |')
        L.append('')
    L.append('### 5. 一致性核对\n')
    L.append(f'- 重叠配置对拍（airplane 全 K×split）最大差 = {consistency["maxdiff_airplane_all_overlap"]:.2e}\n')
    L.append('### 6. 判定依据（逐条列，结论见正文）\n')
    sig_low = boot['lowK13']['schedule_vs_fixed']['lo'] > 0
    L.append(f'- 非 oracle 调度在低 K（1,3 合并）vs fixed：点估计 '
             f'{boot["lowK13"]["schedule_vs_fixed"]["point"]:+.4f}，CI '
             f'[{boot["lowK13"]["schedule_vs_fixed"]["lo"]:+.4f}, '
             f'{boot["lowK13"]["schedule_vs_fixed"]["hi"]:+.4f}] → '
             f'{"不跨 0" if sig_low else "跨 0"}')
    sig_all = boot['allK']['schedule_vs_fixed']['lo'] > 0
    L.append(f'- 全 K vs fixed：点估计 {boot["allK"]["schedule_vs_fixed"]["point"]:+.4f}，CI '
             f'[{boot["allK"]["schedule_vs_fixed"]["lo"]:+.4f}, {boot["allK"]["schedule_vs_fixed"]["hi"]:+.4f}] → '
             f'{"不跨 0" if sig_all else "跨 0"}')
    L.append(f'- oracle 相对 fixed 增益（每 K）：' + '；'.join(f'K{k}={oracle_gain[k]:+.4f}' for k in K_LIST))
    L.append('')

    (U.SAVE_DIR / 'double_knob_ablation_report.md').write_text('\n'.join(L), encoding='utf-8')
    summary = {
        'config': {'alphas': ALPHAS, 'betas': BETAS, 'k_list': K_LIST, 'n_splits': N_SPLITS,
                   'seed': SEED, 'neg_lam': NEG_LAM, 'boot': N_BOOT,
                   'schedule_frozen': 'K<=3:(0.5,0); K>=5:(1,1)',
                   'note': 'alpha = target-proto weight; beta = neg gate'},
        'macro6': {(f'{a}', f'{b}'): {k: m6(a, b, k) for k in K_LIST} for a in ALPHAS for b in BETAS},
        'concept_stratified_per_split': {(f'{a}', f'{b}'): {k: cs_vals[(a, b)][k] for k in K_LIST}
                                         for a in ALPHAS for b in BETAS},
        'per_class': {c: {(f'{a}', f'{b}'): {k: float(np.mean(per_class_vals[c][(a, b)][k]))
                                             for k in K_LIST} for a in ALPHAS for b in BETAS}
                      for c, _ in MAIN},
        'oracle_per_K': {k: oracle[k] for k in K_LIST},
        'oracle_gain_vs_fixed': {k: oracle_gain[k] for k in K_LIST},
        'bootstrap': {str(k): boot[k] for k in K_LIST},
        'calib_boxes_by_K': {k: float(np.mean(calib_boxes[k])) for k in K_LIST},
        'consistency': consistency,
    }
    # JSON keys must be strings
    summary['macro6'] = {f'{a}|{b}': {k: m6(a, b, k) for k in K_LIST} for a in ALPHAS for b in BETAS}
    summary['concept_stratified_per_split'] = {f'{a}|{b}': {k: cs_vals[(a, b)][k] for k in K_LIST}
                                               for a in ALPHAS for b in BETAS}
    summary['per_class'] = {c: {f'{a}|{b}': {k: float(np.mean(per_class_vals[c][(a, b)][k]))
                                             for k in K_LIST} for a in ALPHAS for b in BETAS}
                            for c, _ in MAIN}
    (U.SAVE_DIR / 'double_knob_ablation.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    # ── 控制台 ──
    print('6 类宏平均全网格')
    print('| (α,β) | K=1 | K=3 | K=5 | K=10 |')
    for a in ALPHAS:
        for b in BETAS:
            print(f'| {a:.1f},{b:.0f} | ' + ' | '.join(f'{m6(a,b,k):.4f}' for k in K_LIST) + ' |')
    print()
    print('关键配置对比')
    print('| K | fixed | pure_tgt | schedule | sched−fixed | oracle(α,β) | oracle−fixed |')
    for K in K_LIST:
        f = m6(0.5, 1.0, K); pt = m6(1.0, 0.0, K); sc = m6(*schedule_cfg[K], K); o = oracle[K]
        print(f'| K={K} | {f:.4f} | {pt:.4f} | {sc:.4f} | {sc-f:+.4f} | '
              f'{o["ap"]:.4f}(α{o["alpha"]:.1f},β{o["beta"]:.0f}) | {oracle_gain[K]:+.4f} |')
    print()
    print('配对 CI（概念分层）schedule vs fixed / vs pure_tgt')
    for K in K_LIST:
        for cname in ['schedule_vs_fixed', 'schedule_vs_pure_tgt']:
            b = boot[K][cname]
            print(f'  K={K} {cname}: {b["point"]:+.4f} [{b["lo"]:+.4f}, {b["hi"]:+.4f}]')
    for label, desc in [('lowK13', '低K'), ('allK', '全K')]:
        b = boot[label]['schedule_vs_fixed']
        print(f'  {desc} vs fixed: {b["point"]:+.4f} [{b["lo"]:+.4f}, {b["hi"]:+.4f}]')
    print(f'\n一致性 maxdiff = {consistency["maxdiff_airplane_all_overlap"]:.2e}')
    print(f'✅ 报告 → {U.SAVE_DIR / "double_knob_ablation_report.md"}')
    print(f'✅ 汇总 → {U.SAVE_DIR / "double_knob_ablation.json"}')


if __name__ == '__main__':
    main()
