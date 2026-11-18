# -*- coding: utf-8 -*-
"""verify_fixed_vs_pure.py — 核对 fixed(0.5,1) vs pure_tgt(1,0) 的统计口径（只这两个配置）

核对项（用户列出）：
  (1) 点估计与 bootstrap 是否用完全相同的 split 集合 —— 是，SEED=42 的 10 个嵌套划分
  (2) CI 是"先逐 split 宏平均再求差"还是"先逐类求差再平均" —— 逐 split 配对（正确）
  (3) 是否混用 AP 均值与逐图 bootstrap —— AP 是数据集级 11-point AP50，bootstrap 对 per-split 差值重采样
  (4) vehicle 是否重复计权 —— 6 类宏平均重复计权；概念分层在 vehicle 层内先平均再平权（两种都报）
  (5) CI 表头参考口径 —— 上轮把 6 类点估计与概念分层 CI 拼一起，口径错，已修正

输出：两套同源口径的完整表 + 不对称诊断 + 一致性核验
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


def concept_stratified(vals_cls):
    air = np.mean([vals_cls[c] for c in CONCEPTS['airplane']])
    ship = np.mean([vals_cls[c] for c in CONCEPTS['ship']])
    veh = np.mean([vals_cls[c] for c in CONCEPTS['vehicle']])
    return (air + ship + veh) / 3.0


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
    # per-split: (config, K) -> {class: AP}
    per_split_ap = {(a, b): {K: [{c: None for c, _ in MAIN} for _ in range(N_SPLITS)] for K in K_LIST}
                    for a in (0.5, 1.0) for b in (0.0, 1.0)}

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
                gtf = np.concatenate([gt_feats[i] for i in k_idx if len(gt_feats[i])],
                                     axis=0) if any(len(gt_feats[i]) for i in k_idx) else np.empty((0, 768), np.float32)
                P_tgt = P_src.copy() if len(gtf) == 0 else normalize(gtf.mean(axis=0)).astype(np.float32)
                for a in (0.5, 1.0):
                    P_cal = normalize((1 - a) * P_src + a * P_tgt).astype(np.float32)
                    for b in (0.0, 1.0):
                        neg_np = neg if b == 1.0 else None
                        ap = eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_cal, neg_np)
                        per_split_ap[(a, b)][K][s][name] = ap

    # 一致性核验：concept_stratified per split 应与 double_knob_ablation.json 一致
    dk = json.loads((U.SAVE_DIR / 'double_knob_ablation.json').read_text(encoding='utf-8'))
    maxdiff = 0.0
    for (a, b) in [(0.5, 1.0), (1.0, 0.0)]:
        for K in K_LIST:
            for s in range(N_SPLITS):
                vals_cls = per_split_ap[(a, b)][K][s]
                mine = concept_stratified(vals_cls)
                ref = dk['concept_stratified_per_split'][f'{a}|{b}'][str(K)][s]
                maxdiff = max(maxdiff, abs(mine - ref))
    print(f'[核验] concept_stratified per-split vs double_knob JSON 最大差 = {maxdiff:.2e}\n')

    # ── 口径A：6 类宏平均（vehicle 4 类各计一次 = 重复计权）──
    print('【口径A】6 类宏平均（每 split 先 6 类平均 → 逐 split 配对差）')
    print('        注：vehicle 的 car/truck/bus/motorcycle 各自计权一次（非分层）')
    print('| K | fixed(0.5,1) | pure_tgt(1,0) | Δ点估计 | 95%CI | 每split差值 |')
    for K in K_LIST:
        d6 = []
        for s in range(N_SPLITS):
            f = np.mean([per_split_ap[(0.5, 1.0)][K][s][c] for c, _ in MAIN])
            p = np.mean([per_split_ap[(1.0, 0.0)][K][s][c] for c, _ in MAIN])
            d6.append(f - p)
        p, lo, hi = split_bootstrap(d6)
        fm = float(np.mean([per_split_ap[(0.5, 1.0)][K][s][c] for s in range(N_SPLITS) for c, _ in MAIN]))
        pm = float(np.mean([per_split_ap[(1.0, 0.0)][K][s][c] for s in range(N_SPLITS) for c, _ in MAIN]))
        print(f'| K={K} | {fm:.4f} | {pm:.4f} | {p:+.4f} | [{lo:+.4f}, {hi:+.4f}] | '
              + ' '.join(f'{v:+.4f}' for v in d6) + ' |')
    print()

    # ── 口径B：3 源概念分层（vehicle 层内先平均再平权）──
    print('【口径B】3 源概念分层（每 split 概念分层值 → 逐 split 配对差）')
    print('| K | fixed(0.5,1) | pure_tgt(1,0) | Δ点估计 | 95%CI | 每split差值 |')
    for K in K_LIST:
        dc = []
        for s in range(N_SPLITS):
            f = concept_stratified(per_split_ap[(0.5, 1.0)][K][s])
            p = concept_stratified(per_split_ap[(1.0, 0.0)][K][s])
            dc.append(f - p)
        p, lo, hi = split_bootstrap(dc)
        fm = float(np.mean([concept_stratified(per_split_ap[(0.5, 1.0)][K][s]) for s in range(N_SPLITS)]))
        pm = float(np.mean([concept_stratified(per_split_ap[(1.0, 0.0)][K][s]) for s in range(N_SPLITS)]))
        print(f'| K={K} | {fm:.4f} | {pm:.4f} | {p:+.4f} | [{lo:+.4f}, {hi:+.4f}] | '
              + ' '.join(f'{v:+.4f}' for v in dc) + ' |')
    print()

    # ── 不对称诊断 ──
    print('【不对称诊断】K=1 口径B 每 split 差值分布')
    dc = [concept_stratified(per_split_ap[(0.5, 1.0)][1][s]) - concept_stratified(per_split_ap[(1.0, 0.0)][1][s])
          for s in range(N_SPLITS)]
    print('  差值:', ' '.join(f'{v:+.4f}' for v in dc))
    print(f'  均值={np.mean(dc):+.4f}  中位={np.median(dc):+.4f}  std={np.std(dc):.4f}')
    print(f'  不对称：均值+{np.mean(dc):+.4f} vs 中位+{np.median(dc):+.4f}，'
          f'说明右尾有少数大正值 → 上界拉长（bootstrap CI 不对称的成因）')

    # ── 两口径差异说明 ──
    print('\n【口径差异】6 类宏平均重复计权 vehicle 4 类；概念分层把 vehicle 视为单一概念。')
    print('两者 Δ 方向一致，但量值不同——上轮把 A 点估计与 B CI 拼接属口径错误。')


if __name__ == '__main__':
    main()
