# -*- coding: utf-8 -*-
"""autoproc_proj_b_probe.py — 形态 B/空间迁移(零标注直推式学习投影器)

用户定调(2026-10-05): 空间迁移 = 有训练的跨域投影器, 落 C'。本探针验证
"零目标标注"下能否学一个投影器把目标域图像特征映射到源对齐空间, 破 +1.25% 天花板。

设计(B, 零标注, transductive):
  - 投影器 P: MLP(512→256→512), 近恒等初始化(x + 0.1·net(x)), 不引入目标标签。
  - 训练信号: 目标域(无标签) crop 特征经 P 后, 其 2 阶统计量(均值+协方差)对齐到
    DIOR 源域特征分布 (CORAL / 分布对齐损失)。这是线性 USC 位移的非线性扩展,
    且天然不塌缩(对齐分布而非压到单点)——区别于文本侧熵最小化(−7.22pp 塌缩)。
  - 评估: evaluate() 同候选框, 只换图像特征 + 把原型也过 P 到同一映射空间。
  - 参照: 线性 CORAL 闭式(确认没比已知线性 USC 更差)。
  - 基线: 全 crop 原始特征(当前方法)。

协议: GDINO th0 → pre-filter0.03 → evaluate()。COCO 6类 N_VAL=80。
输出: 实验数据/autoproc_proj_b_results.json + 日志
"""
import sys
sys.path.insert(0, r'C:\Users/张三\Desktop\实验代码')
import json
import numpy as np
import torch
import torch.nn as nn
import usc_experiment_v12 as E
import unified_eval as U
import tge_probe as T

DATA = U.SAVE_DIR
OUT = DATA / 'autoproc_proj_b_results.json'
N_VAL = 80
PRE_FILTER = 0.03
DEVICE = U.DEVICE
SRC_CLASSES = ['airplane', 'ship', 'vehicle']   # DIOR 源域类(覆盖 6 COCO 类)
CLASSES = [
    ('airplane', 'airplane', 'airplane'),
    ('boat',     'boat',     'ship'),
    ('car',      'car',      'vehicle'),
    ('truck',    'truck',    'vehicle'),
    ('bus',     'bus',      'vehicle'),
    ('motorcycle','motorcycle','vehicle'),
]


def pool_source_features():
    feats = []
    for sc in SRC_CLASSES:
        d = E.load_dior_src(sc, n=50)
        if d is not None and d['features'] is not None:
            feats.append(d['features'])
    S = np.vstack(feats).astype(np.float64)
    return torch.tensor(S, device=DEVICE, dtype=torch.float64)


def coral_affine(T_t, S):
    """线性 CORAL: 返回把 T_t 分布对齐到 S 的仿射 (W, mu_t, mu_s)。"""
    mu_t = T_t.mean(0); mu_s = S.mean(0)
    Xt = T_t - mu_t
    cov_t = (Xt.T @ Xt) / (Xt.shape[0] - 1) + 1e-4 * torch.eye(Xt.shape[1], device=DEVICE, dtype=torch.float64)
    cov_s = (S - mu_s).T @ (S - mu_s) / (S.shape[0] - 1) + 1e-4 * torch.eye(S.shape[1], device=DEVICE, dtype=torch.float64)
    es, Vs = torch.linalg.eigh(cov_s)
    et, Vt = torch.linalg.eigh(cov_t)
    es = es.clamp(min=1e-4); et = et.clamp(min=1e-4)
    A = (Vs * es.sqrt().unsqueeze(0)) @ Vs.T
    B = (Vt * (1.0 / et.sqrt()).unsqueeze(0)) @ Vt.T
    W = A @ B
    return W, mu_t, mu_s


def apply_affine(X, W, mu_t, mu_s):
    return (X - mu_t) @ W.T + mu_s


class Proj(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), nn.Tanh(),
            nn.Linear(256, 512))
        # 近恒等: 输出 = x + 0.1·net(x)
    def forward(self, x):
        return x + 0.1 * self.net(x)


def coral_loss(Pt, S):
    mu_t = Pt.mean(0); mu_s = S.mean(0)
    dmu = (mu_t - mu_s).pow(2).sum()
    Xt = Pt - mu_t
    cov_t = (Xt.T @ Xt) / (Xt.shape[0] - 1)
    Xs = S - mu_s
    cov_s = (Xs.T @ Xs) / (Xs.shape[0] - 1)
    dcov = (cov_t - cov_s).pow(2).sum()
    return dmu + dcov


def train_proj(T_t0, S, epochs=400, lr=1e-3):
    P = Proj().double().to(DEVICE)
    opt = torch.optim.Adam(P.parameters(), lr=lr)
    for _ in range(epochs):
        Pt = P(T_t0)
        loss = coral_loss(Pt, S)
        opt.zero_grad(); loss.backward(); opt.step()
    return P


def main():
    print('=' * 70)
    print('形态B/空间迁移探针: 零标注直推式学习投影器')
    print(f'N_VAL={N_VAL}, 设备={DEVICE}, 协议=GDINO th0→pre-filter{PRE_FILTER}→evaluate()')
    print('=' * 70)
    S = pool_source_features()
    print(f'  源域特征池: {S.shape[0]} crops × {S.shape[1]}d (DIOR {SRC_CLASSES})')

    rows = {}
    for eval_name, coco_cls, src_cls in CLASSES:
        print(f'\n[{eval_name}] 加载目标域特征 ...')
        td = E.load_target_coco(coco_cls, n=N_VAL)
        if td is None or td['features'].shape[0] == 0:
            print('  无候选框, 跳过'); continue
        T0 = torch.tensor(td['features'].astype(np.float64), device=DEVICE, dtype=torch.float64)

        # 原型(原始空间 + 过投影器)
        src = E.load_dior_src(src_cls, n=50)
        proto_raw = src['prototype']
        neg_raw = E.build_neg_proto({src_cls: proto_raw}, src_cls)
        proto_tge = E.build_proto(proto_raw, src_cls, coco_cls, E.SRC_TMPL_OVERHEAD, use_tge=True)

        def ap_with_np(proto, neg, feat_np):
            tgt = {'per_img': td['per_img'], 'features': feat_np,
                   'n_crops': feat_np.shape[0]}
            return float(E.evaluate(tgt, proto, neg))

        def proj_proto_np(proto_np, map_fn):
            v = torch.tensor(proto_np, device=DEVICE, dtype=torch.float64)
            return map_fn(v).detach().cpu().numpy().reshape(-1)

        # 基线(原始空间)
        base_raw = ap_with_np(proto_raw, neg_raw, T0.cpu().numpy())
        base_tge = ap_with_np(proto_tge, neg_raw, T0.cpu().numpy())

        # 线性 CORAL (特征 + 原型都过同一仿射)
        W, mu_t, mu_s = coral_affine(T0, S)
        T_lin = apply_affine(T0, W, mu_t, mu_s)
        lin_proto_raw = proj_proto_np(proto_raw, lambda v: apply_affine(v, W, mu_t, mu_s))
        lin_proto_tge = proj_proto_np(proto_tge, lambda v: apply_affine(v, W, mu_t, mu_s))
        lin_raw = ap_with_np(lin_proto_raw, neg_raw, T_lin.detach().cpu().numpy())
        lin_tge = ap_with_np(lin_proto_tge, neg_raw, T_lin.detach().cpu().numpy())

        # 学习投影器(非线性 MLP, CORAL 损失) — 特征 + 原型都过 P
        P = train_proj(T0, S)
        T_mlp = P(T0).detach()
        mlp_proto_raw = proj_proto_np(proto_raw, lambda v: P(v))
        mlp_proto_tge = proj_proto_np(proto_tge, lambda v: P(v))
        mlp_raw = ap_with_np(mlp_proto_raw, neg_raw, T_mlp.cpu().numpy())
        mlp_tge = ap_with_np(mlp_proto_tge, neg_raw, T_mlp.cpu().numpy())

        rows[eval_name] = {
            'n_crops': int(T0.shape[0]),
            'baseline_raw': round(base_raw, 4),
            'baseline_tge': round(base_tge, 4),
            'linear_coral_raw': round(lin_raw, 4),
            'linear_coral_tge': round(lin_tge, 4),
            'mlp_proj_raw': round(mlp_raw, 4),
            'mlp_proj_tge': round(mlp_tge, 4),
            'delta_lin_vs_base_raw': round(lin_raw - base_raw, 4),
            'delta_lin_vs_base_tge': round(lin_tge - base_tge, 4),
            'delta_mlp_vs_base_raw': round(mlp_raw - base_raw, 4),
            'delta_mlp_vs_base_tge': round(mlp_tge - base_tge, 4),
        }
        print(f"  候选框={rows[eval_name]['n_crops']} 基线 raw={base_raw:.4f} tge={base_tge:.4f}")
        print(f"  线性CORAL raw={lin_raw:.4f}(Δ{lin_raw-base_raw:+.4f}) tge={lin_tge:.4f}(Δ{lin_tge-base_tge:+.4f})")
        print(f"  MLP投影  raw={mlp_raw:.4f}(Δ{mlp_raw-base_raw:+.4f}) tge={mlp_tge:.4f}(Δ{mlp_tge-base_tge:+.4f})")

    def mean(k): return float(np.mean([rows[c][k] for c in rows]))
    summary = {k: round(mean(k), 4) for k in next(iter(rows.values()))}
    print('\n' + '=' * 70)
    print('汇总 (6类均值)')
    print('=' * 70)
    for k, v in summary.items():
        print(f"  {k:<24s} {v:+.4f}" if 'delta' in k else f"  {k:<24s} {v:.4f}")
    print('\n  → 关键: 学习投影器 vs 基线 (破 +1.25% 天花板?)')
    print(f"    线性CORAL: Δraw={summary['delta_lin_vs_base_raw']:+.4f} Δtge={summary['delta_lin_vs_base_tge']:+.4f}")
    print(f"    MLP投影:   Δraw={summary['delta_mlp_vs_base_raw']:+.4f} Δtge={summary['delta_mlp_vs_base_tge']:+.4f}")

    json.dump({'config': {'n_val': N_VAL, 'pre_filter': PRE_FILTER,
                          'regime': 'B-zero-annotation transductive learned projector (CORAL/MMD dist-align)',
                          'src_classes': SRC_CLASSES,
                          'projector': 'MLP 512-256-512 near-identity, coral loss',
                          'hypothesis': '零标注学习投影器能否破 +1.25% 天花板'},
               'per_class': rows, 'summary': summary},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n✅ -> {OUT}')


if __name__ == '__main__':
    main()
