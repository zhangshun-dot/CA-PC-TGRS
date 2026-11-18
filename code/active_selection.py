# -*- coding: utf-8 -*-
"""active_selection.py — 主动选样消融（5 策略 × 6 主类 × 10 splits，纯缓存重放）

问题：相同 K 张标注图像预算下，怎样选择样本获得最大 few-shot 增益？
协议（用户拍板）：
  - 预算单位：K = 校准图像数（冻结；逐 K 报告实际有效校准 GT 框数）
  - 测试集 = 池 − 选定（余集法，Random 复现冻结数作一致性核验）
  - 选样阶段只用未标注信息（候选框 DINOv2 特征 + GDINO 分数 + 源原型），
    禁止使用 GT 类别/框位置/测试 AP；选定后才"揭示"GT 框构建目标原型
  - 评估口径：主报 α=0.5 CDVM（与论文冻结数可比），并列报 α=0 CDVM

策略：
  random          — perm[:K]（复现现有冻结数）
  kcenter         — 图级候选特征均值的贪心最远点（多样性，冷启动=池质心最远）
  farthest_src    — 候选框对源原型最不相似（最大域差距，1−max_sim）
  uncertainty     — 图内 GDINO 平均分数最低（检测不确定性最高）
  combine         — 域差距 + 多样性联合（贪心 λ=0.5 加权）

输出：实验数据/active_selection.json + active_selection_report.md
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
STRATEGIES = ['random', 'kcenter', 'farthest_src', 'uncertainty', 'combine']
LAMBDA_C = 0.5  # combine 里域差距权重

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
    return (neg / neg.norm()).cpu().numpy().astype(np.float32)


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


def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ── 选样策略（只用未标注信息）──

def select_random(pool, K, perms_s, **kw):
    return perms_s[:K]


def select_kcenter(pool, K, img_mean, **kw):
    """贪心最远点采样（冷启动：池质心最远）。"""
    F = np.array([img_mean[i] for i in pool])  # (P, 768)
    P_ = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    centroid = P_.mean(axis=0)
    sel_idx = [int(np.argmax(np.sum(P_ * centroid, axis=1)))]
    sel = [pool[sel_idx[0]]]
    for _ in range(K - 1):
        d = P_ @ P_[sel_idx].T  # (P, len(sel))
        dmin = d.min(axis=1)
        j = int(np.argmax(dmin))
        if j in sel_idx:
            # 退化保护：全 0 距离时取未选中的最大距离
            order = np.argsort(-dmin)
            for o in order:
                if o not in sel_idx:
                    j = int(o)
                    break
        sel_idx.append(j)
        sel.append(pool[j])
    return sel


def select_farthest_src(pool, K, img_feats, src_proto, **kw):
    """候选框对源原型最大域差距：1 − max_j cos(f_j, P_src)。"""
    gap = {}
    for i in pool:
        F = img_feats[i]
        if len(F) == 0:
            gap[i] = 0.0
        else:
            Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
            sims = Fn @ src_proto
            gap[i] = float(1.0 - sims.max())
    return sorted(pool, key=lambda i: gap[i], reverse=True)[:K]


def select_uncertainty(pool, K, cand_gd, **kw):
    """图内 GDINO 平均分数最低（检测不确定性最高）。"""
    unc = {}
    for i in pool:
        s = cand_gd[i]
        unc[i] = float(-np.mean(s)) if len(s) else -1.0  # 无候选 → 视为高不确定性
    return sorted(pool, key=lambda i: unc[i])[:K]


def select_combine(pool, K, img_mean, src_proto, **kw):
    """域差距 + 多样性联合：贪心，score = λ·gap + (1−λ)·min_dist_to_selected。"""
    P = np.array([img_mean[i] for i in pool])
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    gap = np.array([1.0 - (Pn[j] @ src_proto) if np.any(Pn[j]) else 0.0
                    for j in range(len(pool))])
    sel_idx = [int(np.argmax(gap))]
    for _ in range(K - 1):
        d = Pn @ Pn[sel_idx].T
        dmin = d.min(axis=1)
        score = LAMBDA_C * gap + (1 - LAMBDA_C) * (1 - dmin)  # 注意 dmin∈[-1,1]，取 1−dmin 为"差异"
        for o in range(len(pool)):  # 已选者置 -inf
            if o in sel_idx:
                score[o] = -np.inf
        sel_idx.append(int(np.argmax(score)))
    return [pool[j] for j in sel_idx]


SELECTORS = {
    'random': select_random,
    'kcenter': select_kcenter,
    'farthest_src': select_farthest_src,
    'uncertainty': select_uncertainty,
    'combine': select_combine,
}


def main():
    exp = json.loads((U.SAVE_DIR / 'expanded_eval_results.json').read_text(encoding='utf-8'))
    M = exp['main']
    bank = U.get_bank()

    # results[strategy][K] -> per-split 6-class macro (list of 10)
    macro = {st: {k: [] for k in K_LIST} for st in STRATEGIES}
    macro_a0 = {st: {k: [] for k in K_LIST} for st in STRATEGIES}
    per_class = {c: {st: {k: [] for k in K_LIST} for st in STRATEGIES} for c, _ in MAIN}
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
        pool = list(range(n_img))

        # 图级表示（未标注）
        img_feats = {}
        img_mean = {}
        for i in pool:
            F = cand_feats[i]
            img_feats[i] = F if len(F) else np.empty((0, 768), np.float32)
            img_mean[i] = F.mean(axis=0) if len(F) else np.zeros(768, np.float32)

        for s in range(N_SPLITS):
            for K in K_LIST:
                for st in STRATEGIES:
                    sel = SELECTORS[st](pool, K, img_feats=img_feats, img_mean=img_mean,
                                        cand_gd=cand_gd, src_proto=P_src, perms_s=perms[s])
                    # 揭示 GT → 目标原型
                    gtf = np.concatenate([gt_feats[i] for i in sel if len(gt_feats[i])],
                                         axis=0) if any(len(gt_feats[i]) for i in sel) else np.empty((0, 768), np.float32)
                    if len(gtf) == 0:
                        P_tgt = P_src.copy()
                    else:
                        P_tgt = normalize(gtf.mean(axis=0)).astype(np.float32)
                    test_idx = [i for i in pool if i not in sel]
                    # α=0.5 融合
                    P_cal = normalize(0.5 * P_src + 0.5 * P_tgt).astype(np.float32)
                    ap = eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_cal, neg)
                    # α=0 纯目标
                    ap0 = eval_variant(test_idx, cand_boxes, cand_gd, cand_feats, gt_boxes, P_tgt, neg)
                    macro[st][K].append(ap)
                    macro_a0[st][K].append(ap0)
                    per_class[name][st][K].append(ap)
                # 校准框数（random 口径）
                sel0 = perms[s][:K]
                calib_boxes[K].append(sum(len(gt_feats[i]) for i in sel0 if len(gt_feats[i])) / 1.0)

    # 一致性核验：random α=0.5 应与冻结 6 类宏平均一致
    for K in K_LIST:
        frozen = float(np.mean([np.mean([M[c]['splits'][str(s)][f'K{K}_a0.5'] for s in range(10)])
                                for c, _ in MAIN]))
        got = float(np.mean(macro['random'][K]))
        consistency[K] = {'frozen': frozen, 'random_rep': got, 'diff': got - frozen}

    # ── 报告 ──
    def m(st, k):
        return float(np.mean(macro[st][k]))

    def m0(st, k):
        return float(np.mean(macro_a0[st][k]))

    L = []
    L.append('## 主动选样消融（5 策略 × 6 主类 × 10 splits，余集法）\n')
    L.append('协议：K=校准图像数（预算冻结）；选样只用未标注信息（候选特征/GDINO分数/源原型），'
             '选定后才揭示 GT；测试集=池−选定；主报 α=0.5 CDVM，并列 α=0 CDVM。\n')
    L.append('### 6 类宏平均（α=0.5 CDVM）\n')
    L.append('| 策略 | K=1 | K=2 | K=3 | K=5 | K=10 |')
    L.append('|---|---|---|---|---|---|')
    for st in STRATEGIES:
        L.append(f'| {st} | ' + ' | '.join(f'{m(st, k):.4f}' for k in K_LIST) + ' |')
    L.append('')
    L.append('**vs random 增量（α=0.5 CDVM）**\n')
    L.append('| 策略 | K=1 | K=2 | K=3 | K=5 | K=10 |')
    L.append('|---|---|---|---|---|---|')
    for st in STRATEGIES:
        if st == 'random':
            continue
        L.append(f'| {st} | ' + ' | '.join(f'{m(st, k)-m("random", k):+.4f}' for k in K_LIST) + ' |')
    L.append('')
    L.append('### 并列：α=0 纯目标 CDVM（6 类宏平均）\n')
    L.append('| 策略 | K=1 | K=2 | K=3 | K=5 | K=10 |')
    L.append('|---|---|---|---|---|---|')
    for st in STRATEGIES:
        L.append(f'| {st} | ' + ' | '.join(f'{m0(st, k):.4f}' for k in K_LIST) + ' |')
    L.append('')
    L.append('### 逐类（α=0.5 CDVM，K=1 与 K=10）\n')
    for K in [1, 10]:
        L.append(f'**K={K}**\n')
        L.append('| 类 | random | kcenter | farthest_src | uncertainty | combine |')
        L.append('|---|---|---|---|---|---|')
        for name, _ in MAIN:
            row = [float(np.mean(per_class[name][st][K])) for st in STRATEGIES]
            L.append(f'| {SHORT[name]} | ' + ' | '.join(f'{v:.4f}' for v in row) + ' |')
        L.append('')
    L.append('### 预算单位核验（有效校准 GT 框数 / K 张图，random 口径）\n')
    for K in K_LIST:
        L.append(f'- K={K}: 平均 {float(np.mean(calib_boxes[K])):.1f} 个有效目标框参与校准')
    L.append('')
    L.append('### 一致性核验（random 复现冻结数）\n')
    for K in K_LIST:
        c = consistency[K]
        L.append(f'- K={K}: frozen={c["frozen"]:.4f}, random_rep={c["random_rep"]:.4f}, diff={c["diff"]:+.4f}')
    L.append('')

    (U.SAVE_DIR / 'active_selection_report.md').write_text('\n'.join(L), encoding='utf-8')
    summary = {
        'config': {'budget_unit': 'K calibration images', 'test_pool': 'pool-minus-selected',
                   'selection_info': 'unlabeled only (cand feats, GDINO scores, source proto)',
                   'alpha_main': 0.5, 'alpha_alt': 0.0, 'n_splits': N_SPLITS, 'seed': SEED,
                   'combine_lambda': LAMBDA_C},
        'macro_alpha05': {st: {k: m(st, k) for k in K_LIST} for st in STRATEGIES},
        'macro_alpha0': {st: {k: m0(st, k) for k in K_LIST} for st in STRATEGIES},
        'per_class_alpha05': {name: {st: {k: float(np.mean(per_class[name][st][k])) for k in K_LIST}
                                     for st in STRATEGIES} for name, _ in MAIN},
        'calib_boxes_by_K': {k: float(np.mean(calib_boxes[k])) for k in K_LIST},
        'consistency': consistency,
    }
    (U.SAVE_DIR / 'active_selection.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    # ── 控制台 ──
    print('6 类宏平均（α=0.5 CDVM）')
    print('| 策略 | ' + ' | '.join(f'K{k}' for k in K_LIST) + ' |')
    for st in STRATEGIES:
        print(f'| {st:<12} | ' + ' | '.join(f'{m(st, k):.4f}' for k in K_LIST) + ' |')
    print()
    print('vs random（α=0.5）')
    for st in STRATEGIES[1:]:
        print(f'| {st:<12} | ' + ' | '.join(f'{m(st, k)-m("random", k):+.4f}' for k in K_LIST) + ' |')
    print()
    print('一致性核验:')
    for K in K_LIST:
        c = consistency[K]
        print(f'  K={K}: random_rep={c["random_rep"]:.4f} vs frozen={c["frozen"]:.4f} (diff {c["diff"]:+.4f})')
    print(f'\n✅ 报告 → {U.SAVE_DIR / "active_selection_report.md"}')
    print(f'✅ 汇总 → {U.SAVE_DIR / "active_selection.json"}')


if __name__ == '__main__':
    main()
