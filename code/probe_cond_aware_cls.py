# -*- coding: utf-8 -*-
"""probe_cond_aware_cls.py — 条件感知训练可行性探针 (放开零样本)

目的: 验"第一层条件在训练时可被消费"——把 z_k 作可训练条件注入轻量分类 head,
      在退化域(COCO res k=2,3,4)比"不看条件的同结构训练"是否更鲁棒。

数据: COCO 6类全图(load_coco 返回全图), 每图 clean + res k∈{2,3,4} 合成退化。
特征: V4Net.encode_full → (z_c 64, z_k 标量), 全图 intended 分布(z_k 非 OOD)。
三档 head (MLP, 仅输入差):
  baseline   : 只用 z_c
  +zk        : z_c 拼接 标准化 z_k (统一空间估计的条件)
  +oracle_c  : z_c 拼接 真实退化等级 onehot (上界, 条件真值)
训练: 各档同结构 MLP, 在同训练集(clean+退化混合)训; 测试看退化域 acc。
指标: clean/test acc, degrad/test acc, 全 test acc (多 seed 均值)。
判定:
  oracle > baseline(退化域) 且 zk≈baseline → 条件感知框架有效, 但 CondEncoder(z_k)太弱需加强 → 方向成立可改进;
  oracle≈baseline → 条件感知训练本身无用 → 需端到端联合训练(RainFog路线) → 诚实收口;
  zk > baseline(退化域) → 统一空间 z_k 直接可用 → 强正, 上全量。
"""
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import phaseB_aux_probe as P
import unified_eval as U
import unified_space_v2 as US2

DEVICE = 'cpu'
COCO_MAP = P.COCO_MAP
N_PER = 20
K_DEG = [2, 3, 4]
SAVE = U.SAVE_DIR / 'probe_cond_aware_cls.json'
SEEDS = [0, 1, 2, 3, 4]


def build_dataset():
    pils, labels, cone, isdeg, orig = [], [], [], [], []
    oid = 0
    for ci, (cls, _) in enumerate(COCO_MAP):
        ims, _ = U.load_coco(cls, n=N_PER)
        for im in ims:
            pils.append(im); labels.append(ci); cone.append([1, 0, 0, 0]); isdeg.append(0); orig.append(oid)
            for ki, k in enumerate(K_DEG):
                d = US2.degrade_res(im, k)
                pils.append(d); labels.append(ci)
                oh = [0, 0, 0, 0]; oh[ki + 1] = 1; cone.append(oh); isdeg.append(1); orig.append(oid)
            oid += 1
    return pils, np.array(labels), np.array(cone), np.array(isdeg), np.array(orig)


class Head(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(din, 32), nn.ReLU(), nn.Linear(32, len(COCO_MAP)))

    def forward(self, x):
        return self.fc(x)


def acc_of(m, X, y):
    if len(X) == 0:
        return float('nan')
    with torch.no_grad():
        p = m(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return float((p == y).mean())


def train_head(Xtr, ytr, seed, n_iter=400):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Head(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.long)
    for _ in range(n_iter):
        opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    return m


def main():
    print('=== 条件感知训练探针 ===', flush=True)
    pils, labels, cone, isdeg, orig = build_dataset()
    print(f'  样本数={len(pils)} (全图, {len(COCO_MAP)}类×{N_PER}×(1+{len(K_DEG)}退化))', flush=True)
    net = P.V4Net().to(DEVICE)
    net.load_state_dict(torch.load(P.V4_PATH, map_location=DEVICE))
    net.eval()
    Zc, Zk = P.encode_full(net, pils, bs=16)
    print(f'  编码完成 Zc{Zc.shape} Zk{Zk.shape}', flush=True)

    # 按原始图分层 split (同图样本同集合, 防泄漏)
    rng = np.random.default_rng(0)
    uniq = np.unique(orig)
    rng.shuffle(uniq)
    ntr = int(0.7 * len(uniq))
    tr_orig = set(uniq[:ntr]); te_orig = set(uniq[ntr:])
    tr = np.array([o in tr_orig for o in orig])
    te = np.array([o in te_orig for o in orig])

    # 特征构造 (z_k 用训练集统计标准化)
    zk_col = Zk.reshape(-1, 1)
    mu, sd = zk_col[tr].mean(), zk_col[tr].std() + 1e-8
    zk_n = ((zk_col - mu) / sd).astype(np.float32)
    X_base = Zc.astype(np.float32)
    X_zk = np.hstack([Zc, zk_n]).astype(np.float32)
    X_orc = np.hstack([Zc, cone]).astype(np.float32)
    y = labels.astype(np.int64)

    results = {mode: {'clean': [], 'deg': [], 'all': []} for mode in ['baseline', '+zk', '+oracle']}
    for seed in SEEDS:
        for mode, X in [('baseline', X_base), ('+zk', X_zk), ('+oracle', X_orc)]:
            m = train_head(X[tr], y[tr], seed)
            a_all = acc_of(m, X[te], y[te])
            m_clean = te & (isdeg == 0); m_deg = te & (isdeg == 1)
            a_clean = acc_of(m, X[m_clean], y[m_clean])
            a_deg = acc_of(m, X[m_deg], y[m_deg])
            results[mode]['all'].append(a_all)
            results[mode]['clean'].append(a_clean)
            results[mode]['deg'].append(a_deg)
            print(f'  seed{seed} {mode}: all={a_all:.3f} clean={a_clean:.3f} deg={a_deg:.3f}', flush=True)

    out = {mode: {k: round(float(np.nanmean(v)), 3) for k, v in results[mode].items()} for mode in results}
    out['n_total'] = len(pils); out['n_train_orig'] = len(tr_orig); out['n_test_orig'] = len(te_orig)
    out['delta_deg_zk_minus_base'] = round(out['+zk']['deg'] - out['baseline']['deg'], 3)
    out['delta_deg_oracle_minus_base'] = round(out['+oracle']['deg'] - out['baseline']['deg'], 3)
    out['delta_deg_oracle_minus_zk'] = round(out['+oracle']['deg'] - out['+zk']['deg'], 3)
    out['note'] = ('退化域鲁棒性增量 = ["+zk"|"+oracle"]["deg"] - ["baseline"]["deg"]; '
                   'oracle 为上界(真实退化等级)。zk≈base 且 oracle>base ⇒ 框架有效但 CondEncoder 弱; '
                   'oracle≈base ⇒ 条件感知训练无用; zk>base ⇒ z_k 直接可用。')
    with open(SAVE, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n结果 -> {SAVE}')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
