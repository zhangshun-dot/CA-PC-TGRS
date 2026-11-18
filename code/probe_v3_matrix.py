# -*- coding: utf-8 -*-
"""probe_v3_matrix.py — v3 严格验收表 (评审第4/5节)

目标: 在接 A-CSPC 之前, 先验收"表示学习本身"是否合格。四项硬门槛:

  A. z_c 内容保真 (必须: 非塌缩 + 内容可线性分)
       - z_c 线性探针 acc (zc->y) 显著 > 多数类基线(0.05)
       - z_c 有效秩 eff_rank(zc) >> 2.34 (v2c 塌缩值)
       - 成对余弦 pairwise_cos(zc) 不 ~0.99 (v2c 塌缩值)
  B. z_k 内容泄漏 (必须: 弱)
       - z_k 线性探针 acc (zk->y) 应 << z_c 探针, 且接近随机
  C. 条件依赖 (独立分辨率预测器 q, 非整体 L1)
       - 固定 zc(清晰), 变 zk 跨 4 档降级 -> 解码图 -> 独立 q 预测 k
       - q 4-class acc 高 => zk 真正驱动输出条件
       - 无参数代理: 解码图 Laplacian 高频能量应随 k 单调降 (模糊=高频少)
  D. 配对 Δzk 单调性
       - 每图 zk(k=1..4) 对 k 的逐图相关, 均值应显著 >0 (zk 随降级单调)

另含诊断(非验收证据): 交叉重建 L1 / 跨条件 zc 一致。

输出: 实验数据/probe_v3_matrix.json + 控制台验收表。
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import sys
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms

import unified_eval as U
import unified_space_v2 as US2
import unified_space_v3 as V3

DEVICE = U.DEVICE
NET_PATH = U.SAVE_DIR / 'unified_space_v3.pt'
SAVE_JSON = U.SAVE_DIR / 'probe_v3_matrix.json'
K_CHOICES = [1, 2, 3, 4]
N_CLASS = 20
SEED = 1


# ── 工具 ──
def eff_rank(Z):
    """Z: (N, D) -> 有效秩 exp(熵(归一化奇异值))。"""
    if Z.shape[0] < 2:
        return float('nan')
    Zc = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    s = s / (s.sum() + 1e-12)
    s = s[s > 1e-12]
    return float(np.exp(-(s * np.log(s)).sum()))


def pooled_zc_raw(net, paths, bs=32):
    """返回 (N, content_ch) 未归一化平均池 zc。"""
    net.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = paths[i:i + bs]
            x = torch.stack([transforms.ToTensor()(Image.open(p).convert('RGB').resize((384, 384)))
                             for p in batch]).to(DEVICE)
            zc, _ = net.encode(x)
            v = F.adaptive_avg_pool2d(zc, 1).flatten(1).cpu()
            feats.append(v)
            del x, zc
    torch.cuda.empty_cache()
    return torch.cat(feats, 0)


def zk_of(net, paths, k, bs=32):
    """返回 (N,1) zk, 对每图用降级 k 编码。"""
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            batch = paths[i:i + bs]
            x = torch.stack([transforms.ToTensor()(
                US2.degrade_res(Image.open(p).convert('RGB').resize((384, 384)), k)).to(DEVICE)
                for p in batch])
            _, zk = net.encode(x)
            out.append(zk.cpu())
            del x, zk
    torch.cuda.empty_cache()
    return torch.cat(out, 0)


def linear_probe(X, y, n_class, epochs=400, lr=1e-2, wd=1e-3):
    """最小自包含线性探针 (无 sklearn)。X:(N,D) y:(N,) 返回 (test_acc,(W,b),train_acc)。"""
    X = X.float(); y = y.long()
    n = len(y)
    idx = torch.randperm(n)
    n_tr = int(0.8 * n)
    tr, te = idx[:n_tr], idx[n_tr:]
    if len(tr) == 0 or len(te) == 0:
        return float('nan'), (None, None), float('nan')
    W = torch.zeros(X.shape[1], n_class, requires_grad=True)
    b = torch.zeros(n_class, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    xt, yt = X[tr], y[tr]
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(xt @ W + b, yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        logits = X[te] @ W + b
        ta = (logits.argmax(1) == y[te]).float().mean().item()
        logits_tr = xt @ W + b
        tra = (logits_tr.argmax(1) == yt).float().mean().item()
    return ta, (W.detach(), b.detach()), tra


class ResPredictorQ(nn.Module):
    """独立分辨率预测器: 输入解码图(3x384x384) -> 预测降级档 k∈{0..3}。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 32), nn.ReLU(),
            nn.Linear(32, 4))
    def forward(self, x):
        return self.net(x)


_LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)


def laplacian_var(imgs):
    """imgs: (N,3,H,W) -> (N,) 高频能量(灰度 Laplacian 方差)。"""
    g = imgs.mean(1, keepdim=True)
    lap = F.conv2d(g, _LAP.to(imgs.device), padding=1)
    return (lap.var(dim=(1, 2, 3)) + 1e-8).log().cpu().numpy()


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f'加载 v3 -> {NET_PATH}', flush=True)
    net = V3.CondDisentanglerV3().to(DEVICE)
    net.load_state_dict(torch.load(NET_PATH, map_location=DEVICE))
    net.eval()

    pairs = V3.build_labeled()
    paths = [p for p, _ in pairs]
    y_all = torch.tensor([c for _, c in pairs], dtype=torch.long)
    print(f'带标签 {len(pairs)} 张, {N_CLASS} 类', flush=True)

    res = {}

    # ── A. z_c 内容保真 ──
    print('\n[A] z_c 内容保真...', flush=True)
    zc = pooled_zc_raw(net, paths)                      # (N, 64)
    zc_n = zc / zc.norm(dim=1, keepdim=True)
    A_probe_acc, _, A_probe_tr = linear_probe(zc, y_all, N_CLASS)
    with torch.no_grad():
        gc_log = net.gc(zc.to(DEVICE)).cpu()
    gc_acc = (gc_log.argmax(1) == y_all).float().mean().item()
    A_eff = eff_rank(zc.numpy())
    pc = (zc_n @ zc_n.T).clamp(-1, 1).numpy()
    A_pair = float(np.mean(pc[np.triu_indices(len(pc), 1)]))
    majority = 1.0 / N_CLASS
    res['A_zc_content'] = {
        'zc_probe_acc': round(A_probe_acc, 4),
        'zc_probe_train_acc': round(A_probe_tr, 4),
        'gc_head_acc': round(gc_acc, 4),
        'eff_rank_zc': round(A_eff, 3),
        'pairwise_cos_zc': round(A_pair, 4),
        'majority_baseline': round(majority, 4),
    }
    A_pass = (A_eff > 10.0) and (A_pair < 0.95) and (A_probe_acc > 0.25)
    res['A_pass'] = bool(A_pass)
    print(f'  探针acc={A_probe_acc:.4f} gc头acc={gc_acc:.4f} 有效秩={A_eff:.3f} '
          f'成对cos={A_pair:.4f} 基线={majority:.4f} => A_PASS={A_pass}', flush=True)
    del zc, zc_n, gc_log; torch.cuda.empty_cache()

    # ── B. z_k 内容泄漏 ──
    print('\n[B] z_k 内容泄漏...', flush=True)
    zk_a = zk_of(net, paths, k=3)
    B_probe_acc, _, B_probe_tr = linear_probe(zk_a, y_all, N_CLASS)
    res['B_zk_leak'] = {
        'zk_probe_acc': round(B_probe_acc, 4),
        'zk_probe_train_acc': round(B_probe_tr, 4),
    }
    B_pass = (B_probe_acc < max(A_probe_acc * 0.5, 0.15)) and (B_probe_acc < 0.30)
    res['B_pass'] = bool(B_pass)
    print(f'  zk探针acc={B_probe_acc:.4f} (zc={A_probe_acc:.4f}) => B_PASS={B_pass}', flush=True)
    del zk_a; torch.cuda.empty_cache()

    # ── C. 条件依赖: 独立 q 预测解码图降级档 + Laplacian 高频代理 ──
    print('\n[C] 条件依赖 (独立分辨率预测器 q + Laplacian 代理)...', flush=True)
    sample_clean = paths[:80]
    with torch.no_grad():
        xc = torch.stack([transforms.ToTensor()(Image.open(p).convert('RGB').resize((384, 384)))
                          for p in sample_clean]).to(DEVICE)
        zc_spatial, _ = net.encode(xc)
    dec_imgs, dec_k = [], []
    with torch.no_grad():
        for kk in K_CHOICES:
            xa = torch.stack([transforms.ToTensor()(
                US2.degrade_res(Image.open(p).convert('RGB').resize((384, 384)), kk)).to(DEVICE)
                for p in sample_clean])
            _, zk = net.encode(xa)
            dec = net.decode(zc_spatial, zk).cpu()
            dec_imgs.append(dec)
            dec_k.append(torch.full((len(sample_clean),), kk - 1, dtype=torch.long))
    dec_imgs = torch.cat(dec_imgs, 0)
    dec_k = torch.cat(dec_k, 0)
    del xc, zc_spatial; torch.cuda.empty_cache()
    # q 训练
    n = len(dec_k); idx = torch.randperm(n); ntr = int(0.8 * n)
    q = ResPredictorQ().to(DEVICE)
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    for ep in range(60):
        q.train()
        for i in range(0, ntr, 32):
            bi = idx[i:i + 32]
            opt.zero_grad()
            loss = F.cross_entropy(q(dec_imgs[bi].to(DEVICE)), dec_k[bi].to(DEVICE))
            loss.backward(); opt.step()
    q.eval()
    with torch.no_grad():
        te = idx[ntr:]
        qpred = q(dec_imgs[te].to(DEVICE)).argmax(1).cpu()
        C_acc = (qpred == dec_k[te]).float().mean().item()
    # Laplacian 高频能量代理 (模型无关): 模糊->高频少->log-var 应随 k 降
    lapv = laplacian_var(dec_imgs)                      # (N,)
    lap_per_k = [float(lapv[dec_k == (kk - 1)].mean()) for kk in K_CHOICES]
    if np.std(lap_per_k) < 1e-6:
        lap_corr = 0.0
    else:
        lap_corr = float(np.corrcoef(lap_per_k, np.array(K_CHOICES, dtype=float))[0, 1])
    res['C_condition_dependency'] = {
        'q_resolution_acc_4class': round(C_acc, 4),
        'laplacian_logvar_per_k': [round(v, 4) for v in lap_per_k],
        'laplacian_corr_with_k': round(lap_corr, 4),
        'n_decode_samples': int(n),
    }
    C_pass = (C_acc > 0.70) and (lap_corr < -0.5)   # 高频应随 k 单调降
    res['C_pass'] = bool(C_pass)
    print(f'  q 4-class acc={C_acc:.4f} Laplacian随k相关={lap_corr:.4f} '
          f'(per_k={[round(v,3) for v in lap_per_k]}) => C_PASS={C_pass}', flush=True)
    del dec_imgs, dec_k, q, qpred; torch.cuda.empty_cache()

    # ── D. 配对 Δzk 单调性 ──
    print('\n[D] 配对 Δzk 单调性...', flush=True)
    zk_per_k = {kk: zk_of(net, paths, kk) for kk in K_CHOICES}
    ks_arr = np.array(K_CHOICES, dtype=float)
    corrs = []
    for i in range(len(paths)):
        vals = np.array([zk_per_k[kk][i, 0].item() for kk in K_CHOICES])
        if np.std(vals) < 1e-6:
            corrs.append(0.0); continue
        corrs.append(float(np.corrcoef(vals, ks_arr)[0, 1]))
    corrs = np.array(corrs)
    D_mean_corr = float(corrs.mean())
    D_frac_mono = float((corrs > 0.3).mean())
    dk14 = (zk_per_k[4] - zk_per_k[1]).abs().mean().item()
    res['D_paired_delta_zk'] = {
        'mean_perimage_corr': round(D_mean_corr, 4),
        'frac_monotonic(corr>0.3)': round(D_frac_mono, 4),
        'mean_abs_zk_delta_k1to4': round(dk14, 4),
    }
    D_pass = (D_mean_corr > 0.5) and (D_frac_mono > 0.6)
    res['D_pass'] = bool(D_pass)
    print(f'  均值相关={D_mean_corr:.4f} 单调占比={D_frac_mono:.4f} |Δzk|1->4={dk14:.4f} => D_PASS={D_pass}', flush=True)
    del zk_per_k; torch.cuda.empty_cache()

    # ── 先写 JSON (避免后续 diag OOM 丢结果) ──
    overall = A_pass and B_pass and C_pass and D_pass
    res['overall_accept'] = bool(overall)
    res['gate'] = {'A_zc_content': A_pass, 'B_zk_leak': B_pass,
                   'C_condition_dependency': C_pass, 'D_paired_delta': D_pass}
    with open(SAVE_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f'\n=== 阶段验收 overall_accept = {overall} (JSON 已落盘) ===', flush=True)

    # ── 诊断(非验收证据, 防 OOM 包裹) ──
    print('\n[diag] 交叉重建 / 跨条件 zc 一致...', flush=True)
    try:
        with torch.no_grad():
            xc = torch.stack([transforms.ToTensor()(Image.open(p).convert('RGB').resize((384, 384)))
                              for p in paths[:100]]).to(DEVICE)
            xa = torch.stack([transforms.ToTensor()(
                US2.degrade_res(Image.open(p).convert('RGB').resize((384, 384)), 3)).to(DEVICE)
                for p in paths[:100]])
            zc_c, _ = net.encode(xc)
            recon_ca, zc_a, zk_a2 = net(xa)
            cross = net.decode(zc_c, zk_a2)
            L_rec = F.l1_loss(recon_ca, xa).item()
            L_cross = F.l1_loss(cross, xa).item()
            zc_c_n = zc_c.flatten(1); zc_a_n = zc_a.flatten(1)
            zc_c_n = zc_c_n / zc_c_n.norm(dim=1, keepdim=True)
            zc_a_n = zc_a_n / zc_a_n.norm(dim=1, keepdim=True)
            cross_consist = float((zc_c_n * zc_a_n).sum(1).mean().item())
        res['diag'] = {
            'recon_L1_xa': round(L_rec, 4),
            'cross_recon_L1': round(L_cross, 4),
            'zc_cross_cond_cos': round(cross_consist, 4),
        }
        with open(SAVE_JSON, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f'  recon L1={L_rec:.4f} cross L1={L_cross:.4f} zc跨条件cos={cross_consist:.4f}', flush=True)
    except Exception as e:
        print(f'  [diag 跳过] {type(e).__name__}: {e}', flush=True)

    print(f'\n→ {SAVE_JSON}')


if __name__ == '__main__':
    main()
