# -*- coding: utf-8 -*-
"""audit_same_domain.py — 第④步收尾: 同域同条件最小审计 (只回答一个可证伪问题)

==================================================================
目的 (元 2026-09-23 指示): 彻底关闭 "合成条件 + DIOR 校准 + COCO 跨域重排" 这条线
之前, 做**一次**同域可控 sanity check, 不再跨 DIOR→COCO。只回答:
  Q: 在同域、同条件分布、可控设定下, g(z_c,z_k) 是否真的使用 z_k?
     即  g(z_c,z_k) 是否 ≠ g(z_c, z_k^shuffle) ≈ g(z_c, 0) ?
若同域也无差异 → 正式结束该路径 (条件估计量不能直接当重排信号)。
若同域有效、跨域无效 → 结论 = 结构化表示有局部条件信息但无法直接跨域迁移为检测重排收益。
==================================================================

与跨域实验的三个关键隔离 (元指出跨域失败的错配):
  1. 训练条件 = 测试条件 = 合成分辨率 k∈{1,2,3,4} (同分布);
  2. 校准域 = 测试域 = DIOR (同域, 无 DIOR→COCO 迁移);
  3. 候选框在**合成分辨率条件下生成** (对每个 k 降级图分别检测, 池内条件混合)。

评测三检查 (元指定):
  ① g 的输出是否真的随 z_k 改变      -> mean|g(zc,zk) - g(zc,zk^shuffle)|
  ② 条件变化是否改变排序             -> 逐图 AP50 / 全局 rank-AUC 在 zk vs shuffle 下的差
  ③ 条件敏感代理任务是否有效         -> (a) 独立小头 z_k->k 应可预测(且 z_c->k 差);
                                        (b) 混合条件池上 g(zc,zk) 排序质量 > g(zc,shuffle)≈g(zc,0)
"""
import os, sys, json, pickle
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import unified_eval as U
import unified_space_v2 as US2
from acspc_unified_rerank import (V4Net, encode_clip, encode_repr, eval_scores,
                                  Reranker, g_pred, SAVE_DIR, DEVICE, V4_PATH, iou)

K_LIST = [1, 2, 3, 4]
SRC_BANK = ['airplane', 'ship', 'vehicle']
N_IMG_PER_CLS = 20


# ═══════════════════ 数据: 合成分辨率条件下生成候选框 (同域 DIOR) ═══════════════════

def build_data(net):
    cache = SAVE_DIR / 'audit_same_domain_data.pkl'
    if cache.exists():
        d = pickle.load(open(cache, 'rb'))
        print(f'  [缓存] {len(d)} 图', flush=True)
        return d
    per_img = []
    for src in SRC_BANK:
        imgs, gts = U.load_dior(src, N_IMG_PER_CLS)
        prompt = f'an overhead image of {src}'
        for im, g in zip(imgs, gts):
            if not g:
                continue
            boxes_all, ks_all, crops, gd_all, y_all = [], [], [], [], []
            for k in K_LIST:
                deg = US2.degrade_res(im, k)
                b, s = U.gdino_detect([deg], prompt, 0.0)[0]
                keep = s > U.PRE_FILTER
                for box, sc in zip(b[keep], s[keep]):
                    x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                    x2 = min(im.width, x2); y2 = min(im.height, y2)
                    if x2 <= x1 + 4 or y2 <= y1 + 4:
                        continue
                    crops.append(deg.crop((x1, y1, x2, y2)))
                    boxes_all.append([x1, y1, x2, y2]); ks_all.append(k)
                    gd_all.append(float(sc))
                    y_all.append(1 if any(iou([x1, y1, x2, y2], gg) >= 0.5 for gg in g) else 0)
            if not crops:
                continue
            F = encode_clip(crops)
            zc, zk = encode_repr(net, crops)
            per_img.append({'src': src, 'boxes': np.array(boxes_all, float),
                            'gts': np.array(g, float), 'ks': np.array(ks_all),
                            'F': F, 'zc': zc, 'zk': zk,
                            'gd': np.array(gd_all), 'y': np.array(y_all)})
        print(f'  {src}: 累计 {len(per_img)} 图', flush=True)
    pickle.dump(per_img, open(cache, 'wb'))
    return per_img


# ═══════════════════ 独立小头 ═══════════════════

class Head(nn.Module):
    def __init__(self, dim, n_class, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_class))

    def forward(self, x):
        return self.net(x)


def train_heads(tr):
    Zc = torch.tensor(np.vstack([p['zc'] for p in tr]), dtype=torch.float32)
    Zk = torch.tensor(np.concatenate([p['zk'] for p in tr]), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(np.concatenate([p['y'] for p in tr]), dtype=torch.float32)
    kk = torch.tensor(np.concatenate([p['ks'] for p in tr]) - 1, dtype=torch.long)

    # g: (zc,zk)->TP  (BCE)
    g = Reranker(65).to(DEVICE)
    inp_g = torch.cat([Zc, Zk], 1).to(DEVICE); yv = y.to(DEVICE)
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    for _ in range(120):
        perm = torch.randperm(len(yv))
        for i in range(0, len(yv), 128):
            bi = perm[i:i + 128]
            loss = nn.BCELoss()(g(inp_g[bi]).flatten(), yv[bi])
            opt.zero_grad(); loss.backward(); opt.step()

    # 独立小头: z_k -> k  (应可预测) ;  z_c -> k (对照, 应差)
    hk = Head(1, 4).to(DEVICE)
    hc = Head(64, 4).to(DEVICE)
    kv = kk.to(DEVICE)
    for h, x in [(hk, Zk.to(DEVICE)), (hc, Zc.to(DEVICE))]:
        o = torch.optim.Adam(h.parameters(), lr=1e-2)
        for _ in range(300):
            loss = nn.CrossEntropyLoss()(h(x), kv)
            o.zero_grad(); loss.backward(); o.step()
    return g, hk, hc


# ═══════════════════ 评测 ═══════════════════

def rank_auc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(scores); rank = np.empty_like(order); rank[order] = np.arange(len(scores))
    return float((rank[labels == 1].mean() - (len(pos) - 1) / 2) / len(neg))


def g_scores(g, te, mode='real', zk_shuffle=None):
    out = []
    for i, p in enumerate(te):
        zc, zk = p['zc'], p['zk'].copy()
        if mode == 'shuffle':
            zk = zk_shuffle[i]
        elif mode == 'zero':
            zk = np.zeros_like(zk)
        X = np.concatenate([zc, zk.reshape(-1, 1)], 1)
        out.append(g_pred(g, X))
    return out


def main():
    print('=' * 78)
    print('同域同条件最小审计 (不跨 DIOR→COCO)')
    print('=' * 78)
    net = V4Net().to(DEVICE)
    net.load_state_dict(torch.load(V4_PATH, map_location=DEVICE)); net.eval()

    print('[1] 合成分辨率条件下生成候选框 (DIOR 同域) ...', flush=True)
    data = build_data(net)
    print(f'    共 {len(data)} 图, 候选框 {sum(len(p["boxes"]) for p in data)}')

    rng = np.random.default_rng(0)
    idx = np.arange(len(data)); rng.shuffle(idx)
    n_tr = len(idx) // 2
    tr = [data[i] for i in idx[:n_tr]]
    te = [data[i] for i in idx[n_tr:]]
    print(f'    训练 {len(tr)} 图 / 测试 {len(te)} 图 (按图划分)')

    print('[2] 训练 g(zc,zk)->TP + 独立小头 (同域) ...', flush=True)
    g, hk, hc = train_heads(tr)

    # ── 检查③a: 独立小头能否预测条件 k ──
    Zk_te = torch.tensor(np.concatenate([p['zk'] for p in te]), dtype=torch.float32).unsqueeze(1).to(DEVICE)
    Zc_te = torch.tensor(np.vstack([p['zc'] for p in te]), dtype=torch.float32).to(DEVICE)
    k_te = torch.tensor(np.concatenate([p['ks'] for p in te]) - 1, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        acc_zk = hk(Zk_te).argmax(1).eq(k_te).float().mean().item()
        acc_zc = hc(Zc_te).argmax(1).eq(k_te).float().mean().item()
    print(f'    ③a 小头 k 预测: z_k->k acc={acc_zk:.4f}  (z_c->k acc={acc_zc:.4f}, 随机0.25)')

    # ── 打乱 z_k (全局打乱, 破坏 zc-zk 对齐) ──
    all_zk = np.concatenate([p['zk'] for p in te])
    perm = np.random.default_rng(7).permutation(len(all_zk))
    zk_sh = all_zk[perm]
    zk_shuffle = []  # 按图切回
    pos = 0
    for p in te:
        zk_shuffle.append(zk_sh[pos:pos + len(p['zk'])]); pos += len(p['zk'])

    # ── 检查①: g 输出对 z_k 的敏感度 ──
    s_real = g_scores(g, te, 'real')
    s_shuf = g_scores(g, te, 'shuffle', zk_shuffle)
    s_zero = g_scores(g, te, 'zero')
    d_rs = float(np.mean(np.abs(np.concatenate(s_real) - np.concatenate(s_shuf))))
    d_rz = float(np.mean(np.abs(np.concatenate(s_real) - np.concatenate(s_zero))))
    print(f'    ① g 对 z_k 敏感度: mean|g(zk)-g(shuffle)|={d_rs:.5f}  mean|g(zk)-g(0)|={d_rz:.5f}')

    # ── 检查②③b: 排序质量 (逐图 AP50 + 全局 rank-AUC) ──
    pib = [p['boxes'] for p in te]; pig = [p['gts'] for p in te]
    y_all = np.concatenate([p['y'] for p in te])
    methods = {
        'GDINO(仅检测分)': [p['gd'] for p in te],
        'g(zc,zk)': s_real,
        'g(zc,zk_shuffle)': s_shuf,
        'g(zc,0)': s_zero,
        'gd+g(zc,zk)': [p['gd'] + s for p, s in zip(te, s_real)],
    }
    res = {}
    print('    ②③ 排序质量 (逐图 AP50 / 全局 rank-AUC):')
    for name, sp in methods.items():
        ap = eval_scores(pib, pig, sp)[0]
        auc = rank_auc(np.concatenate(sp), y_all)
        res[name] = {'ap50': round(ap, 4), 'rank_auc': round(auc, 4)}
        print(f'      {name:22s} AP50={ap:.4f}  AUC={auc:.4f}')

    # ── 判据 ──
    ap_real = res['g(zc,zk)']['ap50']; ap_shuf = res['g(zc,zk_shuffle)']['ap50']; ap_zero = res['g(zc,0)']['ap50']
    auc_real = res['g(zc,zk)']['rank_auc']; auc_shuf = res['g(zc,zk_shuffle)']['rank_auc']
    uses_zk = (d_rs > 1e-4) and (abs(ap_real - ap_shuf) > 1e-3 or abs(auc_real - auc_shuf) > 1e-3)
    verdict = ('同域亦无差异 -> 正式结束该路径: 条件估计量不能直接当重排信号'
               if not uses_zk else
               '同域 z_k 有效 -> 结论: 结构化表示有局部条件信息, 但无法直接跨域迁移为重排收益')
    print('\n' + '=' * 78)
    print(f'判据: g(zk) 是否 ≠ g(zk^shuffle) ≈ g(0) ?')
    print(f'  Δg(zk vs shuffle)={d_rs:.5f}  ΔAP50={ap_real - ap_shuf:+.4f}  ΔAUC={auc_real - auc_shuf:+.4f}')
    print(f'  条件枚举 k 可预测: z_k->k acc={acc_zk:.4f} (z_c->k acc={acc_zc:.4f})')
    print(f'  → {verdict}')

    out = {'meta': {'nature': 'same-domain same-condition minimal audit (non-cross-domain)',
                    'n_img_train': len(tr), 'n_img_test': len(te), 'k_list': K_LIST},
           'check3a_head_k': {'zk_to_k_acc': round(acc_zk, 4), 'zc_to_k_acc': round(acc_zc, 4)},
           'check1_g_sensitivity': {'d_zk_shuffle': round(d_rs, 5), 'd_zk_zero': round(d_rz, 5)},
           'check23_ranking': res,
           'verdict': verdict}
    json.dump(out, open(SAVE_DIR / 'audit_same_domain.json', 'w'), ensure_ascii=False, indent=2)
    print(f'\n结果 -> {SAVE_DIR / "audit_same_domain.json"}')


if __name__ == '__main__':
    main()
