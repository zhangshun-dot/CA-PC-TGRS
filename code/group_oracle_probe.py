# -*- coding: utf-8 -*-
"""group_oracle_probe.py — Sub-B 前置 oracle 判定：视角分组原型有没有出头空间？

问题：VLM 属性条件原型（Sub-B）的最小核心 = 把 K 个目标框分成多组，每组一个原型，
测试区域软路由到最近组。这个机制的「oracle 上界」先验：
  即使分组是完美的（k-means 于目标框特征 = 数据内可分性的上限），
  分组原型能否超过朴素 target-mean？
若 oracle 分组都超不过 → VLM 分组无论多准都没用 → 不用下模型，Sub-B 直接判负。

口径：boat/bus/motorcycle，SEED=42 冻结 10 划分，同 split 同测试集配对；
K∈{3,5}（K=1 无法分组，K=3/5 才能分）；DINOv2 空间，score=sqrt(gd·vis)。
变体：
  plain   vis = F·P_mean（朴素 target-mean，同冻结口径）
  max2    vis = max(F·π1, F·π2)          （最近视角原型）
  soft2   vis = softmax(τ·F·π)·(F·π)      （软路由，τ=10）
  fixed   P_cal = 0.5P_src+0.5P_tgt, β=1（现有冻结方法参考）
输出：实验数据/group_oracle_probe_report.md + json
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import numpy as np
from pathlib import Path
import unified_eval as U

SEED = 42
N_SPLITS = 10
NEG_LAM = 0.5
K_LIST = [3, 5]
N_CLUST = 2
TAU = 10.0

MAIN = [
    ('COCO-boat', 'ship'),
    ('COCO-bus', 'vehicle'),
    ('COCO-motorcycle', 'vehicle'),
]


def reproduce_splits(n):
    rng = np.random.default_rng(SEED)
    return [rng.permutation(n).tolist() for _ in range(N_SPLITS)]


def get_neg_proto_np(bank, src_cls):
    pos = bank[src_cls][11]['prototype'].to('cpu')
    others = torch_stack(bank, src_cls)
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    neg = neg / neg.norm()
    return neg.numpy().astype(np.float32)


def torch_stack(bank, src_cls):
    import torch
    return torch.stack([bank[c][11]['prototype'] for c in bank if c != src_cls])


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def kmeans2(S, iters=20):
    """S:(m,768) L2-normed → 两簇质心（归一化）。m 很小。"""
    rng = np.random.default_rng(0)
    c = S[rng.choice(len(S), 2, replace=False)].copy()
    for _ in range(iters):
        d = S @ c.T  # (m,2)
        lab = d.argmax(axis=1)
        for j in range(2):
            m = S[lab == j]
            if len(m):
                c[j] = normalize(m.mean(axis=0))
    return c


def eval_scores(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, score_fn):
    all_b, all_s, all_g = [], [], []
    for i in test_idx:
        boxes = cand_boxes[i]
        if len(boxes) == 0:
            all_g.extend(gt_boxes[i].tolist())
            continue
        gd = np.clip(cand_gd[i], 1e-6, 1.0)
        F = cand_feats[i]
        if len(F):
            vis = score_fn(F)
            vis = np.clip(vis, 1e-6, 1.0)
            s = np.sqrt(gd * vis)
        else:
            s = gd
        all_b.extend(boxes.tolist())
        all_s.extend(s.tolist())
        all_g.extend(gt_boxes[i].tolist())
    return U.compute_ap50(all_b, all_s, all_g)


def main():
    import torch
    bank = U.get_bank()
    per = {(v, K): [[] for _ in range(N_SPLITS)] for v in ['plain', 'max2', 'soft2', 'fixed']
           for K in K_LIST}
    per_class = {c: {(v, K): [] for v in ['plain', 'max2', 'soft2', 'fixed'] for K in K_LIST}
                 for c, _ in MAIN}
    for name, src_cls in MAIN:
        _npz = np.load(U.SAVE_DIR / 'expanded_cache' / f'{name}.npz', allow_pickle=True)
        cb = list(_npz['cand_boxes']); cgd = list(_npz['cand_gd'])
        cf = list(_npz['cand_feats']); gb = list(_npz['gt_boxes']); gf = list(_npz['gt_feats'])
        n = len(gf)
        P_src = bank[src_cls][11]['prototype'].numpy().astype(np.float32)
        neg = get_neg_proto_np(bank, src_cls)
        perms = reproduce_splits(n)
        for s in range(N_SPLITS):
            for K in K_LIST:
                sup = perms[s][:K]
                test = perms[s][K:]
                sf = np.concatenate([gf[i] for i in sup if len(gf[i])], axis=0)
                sf = sf.astype(np.float32)
                P_tgt = normalize(sf.mean(axis=0)) if len(sf) else P_src.copy()
                P_fixed = normalize(0.5 * P_src + 0.5 * P_tgt)
                c2 = kmeans2(sf) if len(sf) >= N_CLUST else None

                ap_plain = eval_scores(test, cb, cgd, cf, gb, lambda F: F @ P_tgt)
                ap_fixed = eval_scores(test, cb, cgd, cf, gb,
                                       lambda F: F @ P_fixed - NEG_LAM * (F @ neg))
                if c2 is not None:
                    def mx(F):
                        d = F @ c2.T
                        return d.max(axis=1)
                    ap_max = eval_scores(test, cb, cgd, cf, gb, mx)

                    def sf2(F):
                        d = F @ c2.T
                        w = np.exp(TAU * (d - d.max(axis=1, keepdims=True)))
                        w = w / w.sum(axis=1, keepdims=True)
                        return (w * d).sum(axis=1)
                    ap_soft = eval_scores(test, cb, cgd, cf, gb, sf2)
                else:
                    ap_max = ap_plain
                    ap_soft = ap_plain
                for v, ap in [('plain', ap_plain), ('max2', ap_max),
                              ('soft2', ap_soft), ('fixed', ap_fixed)]:
                    per_class[name][(v, K)].append(ap)

    # per[(v,K)][s] = 3 类宏平均
    for (v, K) in per:
        for s in range(N_SPLITS):
            per[(v, K)][s] = float(np.mean([per_class[c][(v, K)][s] for c, _ in MAIN]))

    # 聚合 + CI
    def ci(diffs, nb=10000):
        d = np.array(diffs, dtype=float)
        rng = np.random.default_rng(SEED)
        out = np.empty(nb)
        for b in range(nb):
            out[b] = rng.choice(d, size=len(d)).mean()
        return float(d.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    L = []
    L.append('## 视角分组原型 Oracle 判定（Sub-B 前置）\n')
    L.append('协议：K∈{3,5}，k-means(k=2) 于 K 个目标框特征（DINOv2 空间）＝数据内可分性上界；'
             '同 split 同测试集配对，SEED=42。plain=朴素 target-mean；max2/soft2=最近/软路由两视角原型；'
             'fixed=现有冻结方法(0.5,1)。\n')
    L.append('### 1. 每类平均 AP\n')
    for name, _ in MAIN:
        L.append(f'**{name}**\n')
        L.append('| 变体 | K=3 | K=5 |')
        L.append('|---|---|---|')
        for v in ['plain', 'max2', 'soft2', 'fixed']:
            row = [f'{np.mean(per_class[name][(v, K)]):.4f}' for K in K_LIST]
            L.append(f'| {v} | {row[0]} | {row[1]} |')
        L.append('')
    L.append('### 2. 配对 bootstrap（3 类宏平均）\n')
    L.append('| 比较 | K=3 | K=5 |')
    L.append('|---|---|---|')
    boot = {}
    for K in K_LIST:
        d_max = [per[('max2', K)][s] - per[('plain', K)][s] for s in range(N_SPLITS)]
        d_soft = [per[('soft2', K)][s] - per[('plain', K)][s] for s in range(N_SPLITS)]
        d_fix = [per[('fixed', K)][s] - per[('plain', K)][s] for s in range(N_SPLITS)]
        bm, bs, bf = ci(d_max), ci(d_soft), ci(d_fix)
        boot[K] = {'max2': dict(zip(['point', 'lo', 'hi'], bm)),
                   'soft2': dict(zip(['point', 'lo', 'hi'], bs)),
                   'fixed': dict(zip(['point', 'lo', 'hi'], bf))}

    # 双 K 列 bootstrap 表
    for comp, key in [('max2 − plain', 'max2'), ('soft2 − plain', 'soft2'),
                      ('fixed − plain', 'fixed')]:
        b3, b5 = boot[3][key], boot[5][key]
        L.append(f'| {comp} | {b3["point"]:+.4f} [{b3["lo"]:+.4f},{b3["hi"]:+.4f}] | '
                 f'{b5["point"]:+.4f} [{b5["lo"]:+.4f},{b5["hi"]:+.4f}] |')
    L.append('')
    L.append('### 3. 判定\n')
    L.append('- max2/soft2（oracle 分组上界）vs plain：若 CI 跨 0 或为负 → 目标支持集内'
             '无可分视角结构或样本太少 → VLM 分组（Sub-B）无出头空间，不用下模型。')
    L.append('- 若 max2 显著 > plain → 说明有可分结构，再下 VLM 测真实分组。')
    (U.SAVE_DIR / 'group_oracle_probe_report.md').write_text('\n'.join(L), encoding='utf-8')
    summary = {'per_class': {n: {str((v, K)): per_class[n][(v, K)] for v in
                                 ['plain', 'max2', 'soft2', 'fixed'] for K in K_LIST} for n, _ in MAIN},
               'bootstrap': {str(K): boot[K] for K in K_LIST}}
    (U.SAVE_DIR / 'group_oracle_probe.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                        encoding='utf-8')
    print('\n'.join(L))
    print(f'\n✅ 报告 → {U.SAVE_DIR / "group_oracle_probe_report.md"}')


if __name__ == '__main__':
    main()
