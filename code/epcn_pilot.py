# -*- coding: utf-8 -*-
"""epcn_pilot.py — EPCN 可行性 Pilot（最小可学习原型校准器，纯特征训练，无新 GPU 检测）

回答：一个以 K 个目标样本集合为输入、学习「源→目标原型残差校准」的小模块，
能否在 K=1/3/5/10 稳定超过固定 α=0.5 与朴素目标均值？

设计（忠实于 EPCN 提案的最小版本）：
  P_cal = Norm( 0.5·P_src + 0.5·P_tgt + Δ )
  Δ, β = MLP([ P_src, P_tgt, d=P_src−P_tgt, P_neg, h_mean, h_max, h_std ])
  h_*   = 支持集 set-statistics（支持 K 个目标框特征）
  β 初值 = 1（=冻结 CDVM 负原型权重），范围 [0,2]，由网络学习
  评分 = sqrt( gd · max(0, F·P_cal − β·NEG_LAM·F·P_neg) )

训练：leave-one-class-out（每 held-out 类用其余 5 类训练的模块，真实类泛化测试）
  - episodic：每 episode 随机选一个训练类，K∈{1,3,5,10}，support=K 张图 GT 框特征，
    query=其他图；负例 = pre-filter 候选框中与 GT IoU<0.3 的框特征（未标注监督）
  - 损失 = margin(pos_cos−neg_cos, m=0.3) + 0.3·(1−cos(P_cal, query正例质心))
  - early stop 用 5 训练类各自保留的 val 池（类级泛化只在 held-out 评估）

评估：held-out 类，SEED=42 冻结 10 划分，K=1/3/5/10，逐 split 与固定/朴素配对。
  口径：6 类宏平均 + 3 源概念分层；bootstrap 对 per-split 配对差值重采样。

判定门槛（用户定义）：
  ① K=1/3：EPCN 稳定超 target-mean（配对 CI 不跨 0）且不低于 fixed
  ② K=5/10：至少收回 ≥60% 逐类 oracle（double_knob 网格内 per-class oracle）
输出：实验数据/epcn_pilot.json + epcn_pilot_report.md
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import unified_eval as U

NEG_LAM = 0.5
LAYER = 11
SEED = 42
N_SPLITS = 10
K_LIST = [1, 3, 5, 10]
TRAIN_K = [1, 3, 5, 10]     # 训练 episode 采样的 K
EPOCHS = 600
EPISODES_PER_EPOCH = 96
LR = 1e-3
PATIENCE = 100
MARGIN = 0.3
LAMBDA_PROTO = 0.3
NEG_IOU = 0.3              # 负例候选框与 GT 的 IoU 上限
TRAIN_POOL = 85            # 每训练类前 85 图做训练 episode，85: 为 val 池
QUERY_IMG = 8              # 每 episode query 图数

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def normalize_np(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def reproduce_splits(n):
    rng = np.random.default_rng(SEED)
    return [rng.permutation(n).tolist() for _ in range(N_SPLITS)]


def get_neg_proto_np(bank, src_cls):
    pos = bank[src_cls][LAYER]['prototype'].to('cpu')
    others = torch.stack([bank[c][LAYER]['prototype'] for c in bank if c != src_cls])
    neg = others[(others @ pos).topk(3).indices].mean(dim=0)
    neg = neg / neg.norm()
    return neg.numpy().astype(np.float32)


# ── 模型 ──

class Calibrator(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        in_dim = dim * 7  # P_src, P_tgt, d, P_neg, h_mean, h_max, h_std
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)   # Δ 初值 0 → P_cal = fixed(0.5,1)
        self.beta_net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 1))
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)  # tanh→0 → β 初值 1（冻结）

    def forward(self, P_src, P_tgt, h_mean, h_max, h_std, P_neg):
        d = P_src - P_tgt
        x = torch.cat([P_src, P_tgt, d, P_neg, h_mean, h_max, h_std], dim=-1)
        delta = self.net(x)
        base = 0.5 * (P_src + P_tgt)
        P_cal = F.normalize(base + delta, dim=-1)
        beta = 1.0 + torch.tanh(self.beta_net(x).squeeze(-1))
        return P_cal, beta


# ── 数据准备（每类：GT 特征 / 负例候选特征 / 图数） ──

def load_class(name):
    _npz = np.load(U.SAVE_DIR / 'expanded_cache' / f'{name}.npz', allow_pickle=True)
    cb = list(_npz['cand_boxes'])
    cgd = list(_npz['cand_gd'])
    cf = list(_npz['cand_feats'])
    gb = list(_npz['gt_boxes'])
    gf = list(_npz['gt_feats'])
    n = len(gf)
    neg = []
    for i in range(n):
        feats, boxes = cf[i], cb[i]
        gts = gb[i]
        m = len(boxes)
        mask = np.zeros(m, dtype=bool)
        for j in range(m):
            bj = boxes[j]
            best = 0.0
            for g in gts:
                v = U.iou(bj, g)
                if v > best:
                    best = v
            if best < NEG_IOU:
                mask[j] = True
        neg.append(feats[mask] if m else np.empty((0, 768), np.float32))
    return {'n': n, 'gt': gf, 'neg': neg, 'n_neg': [len(x) for x in neg]}


def support_stats(sf):
    """sf: (S,768) L2-normed 支持集特征 → P_tgt(768), mean/max/std(768)。"""
    S = np.asarray(sf, dtype=np.float32)
    m = S.mean(axis=0)
    return normalize_np(m), m, S.max(axis=0), S.std(axis=0)   # P_tgt, h_mean, h_max, h_std


# ── 训练 episode ──

def episode_loss(model, name, data, P_src_np, P_neg_np, rng, train):
    n = data['n']
    pool = list(range(TRAIN_POOL if train else TRAIN_POOL, n))
    if len(pool) < 3:
        return None
    K = int(rng.choice(TRAIN_K))
    # support：选 K 张有 GT 框的图
    cand_sup = [i for i in pool if len(data['gt'][i]) > 0]
    if len(cand_sup) == 0:
        return None
    support = list(rng.choice(cand_sup, size=min(K, len(cand_sup)), replace=False))
    sf = [f for i in support for f in data['gt'][i]]
    if len(sf) == 0:
        return None
    P_tgt, h_mean, h_max, h_std = support_stats(sf)
    # query：从剩余图选，要求有正例和负例
    qpool = [i for i in pool if i not in support]
    cand_q = [i for i in qpool if len(data['gt'][i]) > 0 and data['n_neg'][i] > 0]
    if len(cand_q) == 0:
        return None
    query = list(rng.choice(cand_q, size=min(QUERY_IMG, len(cand_q)), replace=False))
    pos = np.concatenate([data['gt'][i] for i in query], axis=0)
    neg = np.concatenate([data['neg'][i] for i in query], axis=0)
    if len(pos) == 0 or len(neg) == 0:
        return None

    t = lambda v: torch.tensor(v, dtype=torch.float32, device=DEVICE)
    P_cal, beta = model(t(P_src_np).unsqueeze(0), t(P_tgt).unsqueeze(0),
                        t(h_mean).unsqueeze(0), t(h_max).unsqueeze(0),
                        t(h_std).unsqueeze(0), t(P_neg_np).unsqueeze(0))
    P_cal = P_cal.squeeze(0)
    beta = beta.squeeze(0)
    pos_cos = (t(pos) @ P_cal)
    neg_cos = (t(neg) @ P_cal)
    lm = F.relu(torch.tensor(MARGIN, device=DEVICE) - (pos_cos.mean() - neg_cos.mean()))
    lp = 1.0 - (t(normalize_np(pos.mean(axis=0))) @ P_cal)
    return lm + LAMBDA_PROTO * lp


def train_fold(train_names, data, P_srcs, P_negs, fold_seed, log):
    model = Calibrator().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(fold_seed)
    best_val, best_state, wait = float('inf'), None, 0
    for ep in range(EPOCHS):
        model.train()
        tl = []
        for _ in range(EPISODES_PER_EPOCH):
            name = str(rng.choice(train_names))
            loss = episode_loss(model, name, data[name], P_srcs[name], P_negs[name], rng, train=True)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(loss.item())
        # val（只在 5 训练类的 val 池上算，不与 held-out 接触）
        model.eval()
        vls = []
        with torch.no_grad():
            for _ in range(64):
                name = str(rng.choice(train_names))
                loss = episode_loss(model, name, data[name], P_srcs[name], P_negs[name], rng, train=False)
                if loss is not None:
                    vls.append(loss.item())
        vmean = float(np.mean(vls)) if vls else float('inf')
        if vmean < best_val:
            best_val = vmean
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if ep % 100 == 0:
            log.append(f'    epoch {ep}: train {np.mean(tl):.4f} val {vmean:.4f} (best {best_val:.4f})')
            print(f'    [fold {train_names[0]}...] epoch {ep}: val {vmean:.4f} best {best_val:.4f}', flush=True)
        if wait >= PATIENCE:
            log.append(f'    early stop @ epoch {ep}, best_val {best_val:.4f}')
            print(f'    [fold {train_names[0]}...] early stop @ {ep}, best_val {best_val:.4f}', flush=True)
            break
    model.load_state_dict(best_state)
    return model, best_val


# ── 评估（held-out 类，冻结划分） ──

def eval_ap(test_idx, cb, cgd, cf, gb, P_np, neg_np, beta):
    all_b, all_s, all_g = [], [], []
    for i in test_idx:
        boxes = cb[i]
        if len(boxes) == 0:
            all_g.extend(gb[i].tolist())
            continue
        gd = np.clip(cgd[i], 1e-6, 1.0)
        F = cf[i]
        if len(F):
            vis = F @ P_np - beta * NEG_LAM * (F @ neg_np)
            vis = np.clip(vis, 1e-6, 1.0)
            score = np.sqrt(gd * vis)
        else:
            score = gd
        all_b.extend(boxes.tolist())
        all_s.extend(score.tolist())
        all_g.extend(gb[i].tolist())
    return U.compute_ap50(all_b, all_s, all_g)


def calibrate_and_score(name, data, P_src_np, P_neg_np, model, perms, K, s, mode):
    """mode: 'fixed'(0.5,1) | 'target'(1,0) | 'target_neg'(1,1) | 'epcn' | 'epcn_b1'"""
    dc = data[name]
    sup = perms[s][:K]
    test = perms[s][K:]
    sf = [f for i in sup for f in dc['gt'][i]]
    if len(sf) == 0:
        P_cal = P_src_np
        beta = 1.0
    else:
        P_tgt, h_mean, h_max, h_std = support_stats(sf)
        if mode == 'fixed':
            P_cal = normalize_np(0.5 * P_src_np + 0.5 * P_tgt)
            beta = 1.0
        elif mode in ('target', 'target_neg'):
            P_cal = P_tgt
            beta = 0.0 if mode == 'target' else 1.0
        else:  # epcn / epcn_b1
            t = lambda v: torch.tensor(v, dtype=torch.float32, device=DEVICE)
            model.eval()
            with torch.no_grad():
                P_c, b = model(t(P_src_np).unsqueeze(0), t(P_tgt).unsqueeze(0),
                               t(h_mean).unsqueeze(0), t(h_max).unsqueeze(0),
                               t(h_std).unsqueeze(0), t(P_neg_np).unsqueeze(0))
            P_cal = P_c.squeeze(0).cpu().numpy().astype(np.float32)
            beta = float(b.squeeze(0).item())
            beta = max(0.0, min(2.0, beta))
            if mode == 'epcn_b1':
                beta = 1.0
    return eval_ap(test, dc['cb'], dc['cgd'], dc['cf'], dc['gb'], P_cal, P_neg_np, beta)


def main():
    bank = U.get_bank()
    # 预载数据
    data = {}
    P_srcs, P_negs = {}, {}
    for name, src_cls in MAIN:
        d = load_class(name)
        # 需要候选框/GDINO 分数/GT 框用于评估 —— load_class 没存，重载一次并附加
        _npz = np.load(U.SAVE_DIR / 'expanded_cache' / f'{name}.npz', allow_pickle=True)
        d['cb'] = list(_npz['cand_boxes'])
        d['cgd'] = list(_npz['cand_gd'])
        d['cf'] = list(_npz['cand_feats'])
        d['gb'] = list(_npz['gt_boxes'])
        data[name] = d
        P_srcs[name] = bank[src_cls][LAYER]['prototype'].numpy().astype(np.float32)
        P_negs[name] = get_neg_proto_np(bank, src_cls)

    all_names = [n for n, _ in MAIN]
    fold_results = {}
    per_split = {name: {k: {} for k in K_LIST} for name in all_names}
    logs = {}

    for held in all_names:
        train_names = [n for n in all_names if n != held]
        log = []
        log.append(f'[fold] held-out = {held}  train = {train_names}')
        model, best_val = train_fold(train_names, data, P_srcs, P_negs,
                                     SEED + 100 + all_names.index(held), log)
        logs[held] = {'best_val': best_val, 'detail': log}
        # held-out 评估（冻结划分，逐 split 配对）
        perms_held = reproduce_splits(data[held]['n'])
        for K in K_LIST:
            for s in range(N_SPLITS):
                epcn = calibrate_and_score(held, data, P_srcs[held], P_negs[held], model, perms_held, K, s, 'epcn')
                epcn_b1 = calibrate_and_score(held, data, P_srcs[held], P_negs[held], model, perms_held, K, s, 'epcn_b1')
                fixed = calibrate_and_score(held, data, P_srcs[held], P_negs[held], None, perms_held, K, s, 'fixed')
                tgt = calibrate_and_score(held, data, P_srcs[held], P_negs[held], None, perms_held, K, s, 'target')
                tgt_neg = calibrate_and_score(held, data, P_srcs[held], P_negs[held], None, perms_held, K, s, 'target_neg')
                per_split[held][K][s] = {
                    'epcn': epcn, 'epcn_b1': epcn_b1, 'fixed': fixed, 'tgt': tgt, 'tgt_neg': tgt_neg,
                }
        log.append(f'  [done] held-out {held}')

    # ── 聚合 ──
    def macro(k, s, key):
        return float(np.mean([per_split[n][k][s][key] for n in all_names]))

    def strat(k, s, key):
        v = []
        for concept, cl in CONCEPTS.items():
            v.append(np.mean([per_split[n][k][s][key] for n in cl]))
        return float(np.mean(v))

    def ci(diffs, seed=SEED, nb=10000):
        d = np.array(diffs, dtype=float)
        rng = np.random.default_rng(seed)
        out = np.empty(nb)
        for b in range(nb):
            out[b] = rng.choice(d, size=len(d)).mean()
        return float(d.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    macro_epcn = {k: [macro(k, s, 'epcn') for s in range(N_SPLITS)] for k in K_LIST}
    macro_epcn_b1 = {k: [macro(k, s, 'epcn_b1') for s in range(N_SPLITS)] for k in K_LIST}
    macro_fixed = {k: [macro(k, s, 'fixed') for s in range(N_SPLITS)] for k in K_LIST}
    macro_tgt = {k: [macro(k, s, 'tgt') for s in range(N_SPLITS)] for k in K_LIST}
    strat_epcn = {k: [strat(k, s, 'epcn') for s in range(N_SPLITS)] for k in K_LIST}
    strat_fixed = {k: [strat(k, s, 'fixed') for s in range(N_SPLITS)] for k in K_LIST}
    strat_tgt = {k: [strat(k, s, 'tgt') for s in range(N_SPLITS)] for k in K_LIST}

    boot = {}
    for k in K_LIST:
        boot[k] = {
            'epcn_vs_fixed_6': dict(zip(['point', 'lo', 'hi'], ci([macro_epcn[k][s] - macro_fixed[k][s] for s in range(N_SPLITS)]))),
            'epcn_vs_tgt_6': dict(zip(['point', 'lo', 'hi'], ci([macro_epcn[k][s] - macro_tgt[k][s] for s in range(N_SPLITS)]))),
            'epcn_vs_fixed_cs': dict(zip(['point', 'lo', 'hi'], ci([strat_epcn[k][s] - strat_fixed[k][s] for s in range(N_SPLITS)]))),
            'epcn_vs_tgt_cs': dict(zip(['point', 'lo', 'hi'], ci([strat_epcn[k][s] - strat_tgt[k][s] for s in range(N_SPLITS)]))),
        }

    # oracle 参考（double_knob 网格 per-class oracle）
    dk = json.loads((U.SAVE_DIR / 'double_knob_ablation.json').read_text(encoding='utf-8'))
    pc = dk['per_class']
    cfgs = list(pc[all_names[0]].keys())
    oracle_ref = {}
    for k in K_LIST:
        oracle_ref[k] = float(np.mean([max(pc[n][c][str(k)] for c in cfgs) for n in all_names]))

    # ── 报告 ──
    L = []
    L.append('## EPCN Pilot 判定报告（learned residual calibration vs fixed/target-mean）\n')
    L.append('协议：leave-one-class-out（每 held-out 类用其余 5 类训练），最小 set-statistics + 残差 MLP，'
             'P_cal=Norm(0.5P_src+0.5P_tgt+Δ)，β 学习（初值=1）。训练损失=margin+原型接近。'
             '评估=SEED=42 冻结 10 划分，同 split 配对。\n')
    L.append('### 1. 6 类宏平均（逐 split 配对）\n')
    L.append('| K | fixed(0.5,1) | target-mean(1,0) | **EPCN** | EPCN−fixed | EPCN−target |')
    L.append('|---|---|---|---|---|---|')
    for k in K_LIST:
        f = float(np.mean(macro_fixed[k])); t = float(np.mean(macro_tgt[k])); e = float(np.mean(macro_epcn[k]))
        L.append(f'| K={k} | {f:.4f} | {t:.4f} | **{e:.4f}** | {e-f:+.4f} | {e-t:+.4f} |')
    L.append('')
    L.append('### 2. 配对 bootstrap 95% CI（EPCN − 基线）\n')
    L.append('| K | vs fixed(6类) | vs target(6类) | vs fixed(概念分层) | vs target(概念分层) |')
    L.append('|---|---|---|---|---|')
    for k in K_LIST:
        b = boot[k]
        L.append(f'| K={k} | {b["epcn_vs_fixed_6"]["point"]:+.4f} '
                 f'[{b["epcn_vs_fixed_6"]["lo"]:+.4f},{b["epcn_vs_fixed_6"]["hi"]:+.4f}] | '
                 f'{b["epcn_vs_tgt_6"]["point"]:+.4f} '
                 f'[{b["epcn_vs_tgt_6"]["lo"]:+.4f},{b["epcn_vs_tgt_6"]["hi"]:+.4f}] | '
                 f'{b["epcn_vs_fixed_cs"]["point"]:+.4f} '
                 f'[{b["epcn_vs_fixed_cs"]["lo"]:+.4f},{b["epcn_vs_fixed_cs"]["hi"]:+.4f}] | '
                 f'{b["epcn_vs_tgt_cs"]["point"]:+.4f} '
                 f'[{b["epcn_vs_tgt_cs"]["lo"]:+.4f},{b["epcn_vs_tgt_cs"]["hi"]:+.4f}] |')
    L.append('')
    L.append('### 3. 逐类（EPCN / fixed / target-mean）\n')
    for k in K_LIST:
        L.append(f'**K={k}**\n')
        L.append('| 类 | fixed | target | EPCN | EPCN−fixed | EPCN−target |')
        L.append('|---|---|---|---|---|---|')
        for n in all_names:
            f = float(np.mean([per_split[n][k][s]['fixed'] for s in range(N_SPLITS)]))
            t = float(np.mean([per_split[n][k][s]['tgt'] for s in range(N_SPLITS)]))
            e = float(np.mean([per_split[n][k][s]['epcn'] for s in range(N_SPLITS)]))
            L.append(f'| {SHORT[n]} | {f:.4f} | {t:.4f} | **{e:.4f}** | {e-f:+.4f} | {e-t:+.4f} |')
        L.append('')
    L.append('### 4. oracle 参考与回收率\n')
    L.append('| K | 逐类oracle(网格) | EPCN | oracle−EPCN | EPCN 回收率 |')
    L.append('|---|---|---|---|---|')
    for k in K_LIST:
        o = oracle_ref[k]; e = float(np.mean(macro_epcn[k]))
        rec = (e - float(np.mean(macro_fixed[k]))) / max(1e-9, (o - float(np.mean(macro_fixed[k]))))
        L.append(f'| K={k} | {o:.4f} | {e:.4f} | {o-e:+.4f} | {rec*100:.0f}% |')
    L.append('')
    L.append('### 5. β 学习效果（EPCN 学习 β vs 固定 β=1）\n')
    L.append('| K | EPCN | EPCN-βfix | 差 |')
    L.append('|---|---|---|---|')
    for k in K_LIST:
        e = float(np.mean(macro_epcn[k])); eb = float(np.mean(macro_epcn_b1[k]))
        L.append(f'| K={k} | {e:.4f} | {eb:.4f} | {e-eb:+.4f} |')
    L.append('')
    L.append('### 6. 训练日志（每 fold best_val）\n')
    for held in all_names:
        L.append(f'- {held}: best_val={logs[held]["best_val"]:.4f}')
        for ln in logs[held]['detail']:
            L.append(f'  {ln}')
    L.append('')
    (U.SAVE_DIR / 'epcn_pilot_report.md').write_text('\n'.join(L), encoding='utf-8')

    summary = {
        'config': {'neg_lam': NEG_LAM, 'k_list': K_LIST, 'n_splits': N_SPLITS, 'seed': SEED,
                   'margin': MARGIN, 'lambda_proto': LAMBDA_PROTO, 'epochs': EPOCHS, 'lr': LR,
                   'train_pool': TRAIN_POOL, 'leave_one_class_out': True,
                   'calib': 'P_cal=Norm(0.5P_src+0.5P_tgt+MLP(7×768)); beta learned init 1'},
        'macro6': {'fixed': {k: float(np.mean(macro_fixed[k])) for k in K_LIST},
                   'target': {k: float(np.mean(macro_tgt[k])) for k in K_LIST},
                   'epcn': {k: float(np.mean(macro_epcn[k])) for k in K_LIST},
                   'epcn_beta1': {k: float(np.mean(macro_epcn_b1[k])) for k in K_LIST}},
        'bootstrap': boot,
        'per_split_macro6': {'fixed': {k: macro_fixed[k] for k in K_LIST},
                             'target': {k: macro_tgt[k] for k in K_LIST},
                             'epcn': {k: macro_epcn[k] for k in K_LIST},
                             'epcn_beta1': {k: macro_epcn_b1[k] for k in K_LIST}},
        'per_class': {n: {k: {key: float(np.mean([per_split[n][k][s][key] for s in range(N_SPLITS)]))
                              for key in ['fixed', 'tgt', 'epcn', 'epcn_b1']}
                          for k in K_LIST} for n in all_names},
        'oracle_ref': oracle_ref,
        'train_log': logs,
    }
    (U.SAVE_DIR / 'epcn_pilot.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    # 控制台
    print('6 类宏平均')
    print('| K | fixed | target | EPCN | EPCN−fixed | EPCN−target |')
    for k in K_LIST:
        f = float(np.mean(macro_fixed[k])); t = float(np.mean(macro_tgt[k])); e = float(np.mean(macro_epcn[k]))
        print(f'| K={k} | {f:.4f} | {t:.4f} | {e:.4f} | {e-f:+.4f} | {e-t:+.4f} |')
    print('\n配对 CI（概念分层）')
    for k in K_LIST:
        b = boot[k]
        print(f'  K={k}: EPCN−fixed {b["epcn_vs_fixed_cs"]["point"]:+.4f} '
              f'[{b["epcn_vs_fixed_cs"]["lo"]:+.4f},{b["epcn_vs_fixed_cs"]["hi"]:+.4f}] | '
              f'EPCN−target {b["epcn_vs_tgt_cs"]["point"]:+.4f} '
              f'[{b["epcn_vs_tgt_cs"]["lo"]:+.4f},{b["epcn_vs_tgt_cs"]["hi"]:+.4f}]')
    print('\noracle 回收率')
    for k in K_LIST:
        o = oracle_ref[k]; e = float(np.mean(macro_epcn[k]))
        rec = (e - float(np.mean(macro_fixed[k]))) / max(1e-9, (o - float(np.mean(macro_fixed[k]))))
        print(f'  K={k}: oracle {o:.4f} EPCN {e:.4f} 回收 {rec*100:.0f}%')
    print(f'\n✅ 报告 → {U.SAVE_DIR / "epcn_pilot_report.md"}')
    print(f'✅ 汇总 → {U.SAVE_DIR / "epcn_pilot.json"}')


if __name__ == '__main__':
    main()
