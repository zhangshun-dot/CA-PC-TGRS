# -*- coding: utf-8 -*-
"""naive_prototype_baseline.py — 朴素原型对照（补方法合法性地基，纯缓存重放）

审稿人攻击：为什么不直接用 K 个目标框求均值（target prototype only）？
本脚本补 4 个最小对照（6 主类 × 5 K × 10 splits，毫秒级，从 expanded_cache 重放）：

  src_pos_only  源域正原型，无负原型抑制       vis = F·P_src
  src_cdvm      源域正−负原型（=现有 α=1）     vis = F·P_src − 0.5·F·neg
  tgt_pos_only  目标域正原型均值，无负原型抑制   vis = F·P_tgt
  tgt_cdvm      目标域正−负原型（=现有 α=0）     vis = F·P_tgt − 0.5·F·neg

核心比较 Δ_neg = AP(tgt_cdvm) − AP(tgt_pos_only)：负原型抑制在 few-shot 阶段
是否仍有贡献。可成立的贡献形式：负原型只在 K 小、目标原型不稳定时有效；
不要求 K=10 仍然强。零样本层面附注 src_pos_only vs src_cdvm（全池）。

一致性核对：src_cdvm ≡ JSON K*_alpha1；tgt_cdvm ≡ JSON K*_a0.0（逐 split）。
预算单位：K = 校准图像数；逐 K 报告实际参与校准的有效目标框数。
输出：实验数据/naive_baseline.json + naive_baseline_report.md
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
K_LIST = [1, 2, 3, 5, 10]
N_SPLITS = 10
SEED = 42
CACHE_DIR = U.SAVE_DIR / 'expanded_cache'
OUT_JSON = U.SAVE_DIR / 'naive_baseline.json'
OUT_MD = U.SAVE_DIR / 'naive_baseline_report.md'

# (name, src_dior_cls)
MAIN = [
    ('COCO-airplane', 'airplane'),
    ('COCO-boat', 'ship'),
    ('COCO-car', 'vehicle'),
    ('COCO-truck', 'vehicle'),
    ('COCO-bus', 'vehicle'),
    ('COCO-motorcycle', 'vehicle'),
]
SHORT = {'COCO-airplane': 'airplane', 'COCO-boat': 'boat', 'COCO-car': 'car',
         'COCO-truck': 'truck', 'COCO-bus': 'bus', 'COCO-motorcycle': 'motorcycle'}


def reproduce_splits(n):
    rng = np.random.default_rng(SEED)
    return [rng.permutation(n).tolist() for _ in range(N_SPLITS)]


def get_neg_proto_np(bank, dior_cls):
    pos = bank[dior_cls][LAYER]['prototype'].to(U.DEVICE)
    others = torch.stack([bank[c][LAYER]['prototype'] for c in bank if c != dior_cls]).to(U.DEVICE)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    neg = neg / neg.norm()
    return neg.cpu().numpy().astype(np.float32)


def eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_np, neg_np):
    """P_np: (768,) float32 正原型；neg_np: (768,) 或 None。与 expanded_cross_domain.ap_on 一致。"""
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


def main():
    data = json.loads((U.SAVE_DIR / 'expanded_eval_results.json').read_text(encoding='utf-8'))
    M = data['main']
    bank = U.get_bank()

    per_split = {}   # class -> {variant: {K: [AP over splits]}}
    per_class = {}   # class -> {variant: {K: mean AP}}
    coverage = {}    # class -> {'n_img', 'calib_boxes_by_K': {K: mean effective boxes}}
    zero = {}        # class -> {'src_pos_only': AP_fullpool, 'src_cdvm': AP_fullpool}

    for name, src_cls in MAIN:
        _npz = np.load(CACHE_DIR / f'{name}.npz', allow_pickle=True)
        cand_boxes = list(_npz['cand_boxes'])
        cand_gd = list(_npz['cand_gd'])
        cand_feats = list(_npz['cand_feats'])
        gt_boxes = list(_npz['gt_boxes'])
        gt_feats = list(_npz['gt_feats'])
        n_img = len(gt_feats)
        P_src = bank[src_cls][LAYER]['prototype'].cpu().numpy().astype(np.float32)
        neg = get_neg_proto_np(bank, src_cls)
        perms = reproduce_splits(n_img)
        test_all = list(range(n_img))

        # 零样本层（全池）：源正原型 ± neg
        zero[name] = {
            'src_pos_only': eval_variant(test_all, cand_boxes, cand_gd, cand_feats, gt_boxes, P_src, None),
            'src_cdvm': eval_variant(test_all, cand_boxes, cand_gd, cand_feats, gt_boxes, P_src, neg),
        }

        ps = {v: {k: [] for k in K_LIST} for v in
              ['src_pos_only', 'src_cdvm', 'tgt_pos_only', 'tgt_cdvm']}
        cov = {k: [] for k in K_LIST}

        for s in range(N_SPLITS):
            for K in K_LIST:
                k_idx = perms[s][:K]
                test_idx = perms[s][K:]  # 与 expanded_cross_domain 一致：每个 K 自己的测试集（嵌套）
                k_gtf_list = [gt_feats[i] for i in k_idx if len(gt_feats[i])]
                k_gtf = np.concatenate(k_gtf_list, axis=0) if k_gtf_list else np.empty((0, 768), np.float32)
                cov[K].append(float(len(k_gtf)))
                if len(k_gtf) == 0:
                    P_tgt = P_src.copy()  # 退化为源原型路径（与 expanded 一致）
                else:
                    v = k_gtf.mean(axis=0)
                    v = v / (np.linalg.norm(v) + 1e-12)
                    P_tgt = v.astype(np.float32)
                ps['src_pos_only'][K].append(eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_src, None))
                ps['src_cdvm'][K].append(eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_src, neg))
                ps['tgt_pos_only'][K].append(eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_tgt, None))
                ps['tgt_cdvm'][K].append(eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_tgt, neg))

        per_split[name] = ps
        per_class[name] = {v: {k: float(np.mean(x)) for k, x in ps[v].items()} for v in ps}
        coverage[name] = {'n_img': n_img, 'calib_boxes_by_K': {k: float(np.mean(x)) for k, x in cov.items()}}

    # ── 一致性核对（与 expanded_eval_results.json）──
    checks = []
    max_src_diff, max_tgt_diff = 0.0, 0.0
    for name, _ in MAIN:
        r = M[name]
        for K in K_LIST:
            for s in range(N_SPLITS):
                js_src = r['splits'][str(s)][f'K{K}_alpha1']
                js_tgt = r['splits'][str(s)][f'K{K}_a0.0']
                max_src_diff = max(max_src_diff, abs(js_src - per_split[name]['src_cdvm'][K][s]))
                max_tgt_diff = max(max_tgt_diff, abs(js_tgt - per_split[name]['tgt_cdvm'][K][s]))
    checks = {'src_cdvm_vs_alpha1_maxdiff': max_src_diff,
              'tgt_cdvm_vs_a0_maxdiff': max_tgt_diff}

    # ── 6 类宏平均 ──
    def macro(k, variant):
        return float(np.mean([per_class[c][variant][k] for c, _ in MAIN]))

    # 配对 Δ_neg（逐 split 配对再平均，与主报告口径一致）
    d_neg = {k: float(np.mean([per_split[c]['tgt_cdvm'][k][s] - per_split[c]['tgt_pos_only'][k][s]
                                for s in range(N_SPLITS) for c, _ in MAIN])) for k in K_LIST}
    # split 级 Δ_neg 明细（6 类宏平均 per split）
    d_neg_split = {k: [float(np.mean([per_split[c]['tgt_cdvm'][k][s] - per_split[c]['tgt_pos_only'][k][s]
                                       for c, _ in MAIN])) for s in range(N_SPLITS)] for k in K_LIST}

    # 报告
    L = []
    L.append('## 朴素原型对照（naive prototype baselines，纯缓存重放）\n')
    L.append('回答：**"为什么不直接用 K 个目标框求均值？"** 四个最小对照（6 主类 × 5 K × 10 splits，'
             '同一测试集配对）：\n')
    L.append('| 变体 | 正原型 | 负原型抑制 | 定义 |')
    L.append('|---|---|---|---|')
    L.append('| src_pos_only | 源域 P_src | 无 | vis = F·P_src |')
    L.append('| src_cdvm | 源域 P_src | 0.5·F·neg | 现有 α=1 口径 |')
    L.append('| tgt_pos_only | 目标域均值 P_tgt | 无 | **朴素基线** |')
    L.append('| tgt_cdvm | 目标域均值 P_tgt | 0.5·F·neg | 现有 α=0 口径 |')
    L.append('')
    L.append('### 6 类宏平均（同测试集配对）\n')
    L.append('| K | src_pos_only | src_cdvm | tgt_pos_only | tgt_cdvm | **Δ_neg** |')
    L.append('|---|---|---|---|---|---|')
    for k in K_LIST:
        s_po = macro(k, 'src_pos_only'); s_cd = macro(k, 'src_cdvm')
        t_po = macro(k, 'tgt_pos_only'); t_cd = macro(k, 'tgt_cdvm')
        L.append(f'| K={k} | {s_po:.4f} | {s_cd:.4f} | {t_po:.4f} | {t_cd:.4f} | **{d_neg[k]:+.4f}** |')
    L.append('')
    L.append('Δ_neg = AP(tgt_cdvm) − AP(tgt_pos_only)，逐 split 配对。\n')
    L.append('### 零样本层（全池，源原型 ± 负原型）\n')
    L.append('| 类 | src_pos_only | src_cdvm | Δ_neg_src |')
    L.append('|---|---|---|---|')
    for name, _ in MAIN:
        zo = zero[name]
        L.append(f'| {SHORT[name]} | {zo["src_pos_only"]:.4f} | {zo["src_cdvm"]:.4f} '
                 f'| {zo["src_cdvm"]-zo["src_pos_only"]:+.4f} |')
    L.append('')
    L.append('### 逐类表（全 K）\n')
    for name, _ in MAIN:
        L.append(f'**{SHORT[name]}**\n')
        L.append('| K | src_pos_only | src_cdvm | tgt_pos_only | tgt_cdvm | Δ_neg |')
        L.append('|---|---|---|---|---|---|')
        for k in K_LIST:
            pc = per_class[name]
            d = per_split[name]['tgt_cdvm'][k]  # not needed here
            dn = float(np.mean([per_split[name]['tgt_cdvm'][k][s] - per_split[name]['tgt_pos_only'][k][s]
                                for s in range(N_SPLITS)]))
            L.append(f'| K={k} | {pc["src_pos_only"][k]:.4f} | {pc["src_cdvm"][k]:.4f} '
                     f'| {pc["tgt_pos_only"][k]:.4f} | {pc["tgt_cdvm"][k]:.4f} | {dn:+.4f} |')
        L.append('')
    L.append('### 预算单位（K=校准图像数）\n')
    L.append('| 类 | 图数 | 平均每 K 图有效校准 GT 框数（K=1/2/3/5/10） |')
    L.append('|---|---|---|')
    for name, _ in MAIN:
        cb = coverage[name]['calib_boxes_by_K']
        L.append(f'| {SHORT[name]} | {coverage[name]["n_img"]} | '
                 f'{cb[1]:.1f} / {cb[2]:.1f} / {cb[3]:.1f} / {cb[5]:.1f} / {cb[10]:.1f} |')
    L.append('')
    L.append('### 一致性核对\n')
    L.append(f'- src_cdvm vs JSON `K*_alpha1` 最大差 = {checks["src_cdvm_vs_alpha1_maxdiff"]:.2e}')
    L.append(f'- tgt_cdvm vs JSON `K*_a0.0` 最大差 = {checks["tgt_cdvm_vs_a0_maxdiff"]:.2e}\n')

    (U.SAVE_DIR / 'naive_baseline_report.md').write_text('\n'.join(L), encoding='utf-8')

    summary = {
        'config': {'neg_lam': NEG_LAM, 'layer': 'L12', 'k_list': K_LIST,
                   'n_splits': N_SPLITS, 'seed': SEED,
                   'protocol': 'same-test-set paired; tgt prototype = calib GT-box feat mean; '
                               'empty calib -> fallback src path'},
        'zero_shot_fullpool': zero,
        'per_class': per_class,
        'per_split': per_split,
        'd_neg_macro': d_neg,
        'd_neg_macro_per_split': d_neg_split,
        'calib_coverage': coverage,
        'consistency': checks,
    }
    (U.SAVE_DIR / 'naive_baseline.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    # ── 控制台输出 ──
    print('6 类宏平均（同测试集配对）')
    print('| K | src_pos | src_cdvm | tgt_pos | tgt_cdvm | Δ_neg |')
    for k in K_LIST:
        print(f'| K={k} | {macro(k,"src_pos_only"):.4f} | {macro(k,"src_cdvm"):.4f} | '
              f'{macro(k,"tgt_pos_only"):.4f} | {macro(k,"tgt_cdvm"):.4f} | {d_neg[k]:+.4f} |')
    print(f'\n一致性: src_cdvm vs alpha1 maxdiff={checks["src_cdvm_vs_alpha1_maxdiff"]:.2e}, '
          f'tgt_cdvm vs a0 maxdiff={checks["tgt_cdvm_vs_a0_maxdiff"]:.2e}')
    print(f'\n✅ 报告 → {OUT_MD}')
    print(f'✅ 汇总 → {OUT_JSON}')


if __name__ == '__main__':
    main()
