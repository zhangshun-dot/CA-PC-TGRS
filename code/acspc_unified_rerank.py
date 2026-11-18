# -*- coding: utf-8 -*-
"""acspc_unified_rerank.py — 第④步: 玩具条件 A-CSPC 联合重排机制验证

==================================================================
⚠️ 性质声明 (必须放在最前):
本实验是【机制验证 / 合成条件健全性检查】(mechanism validation /
synthetic-condition sanity check), 不是真实遥感(RS)跨域泛化证据。
任何结论仅说明"条件感知重排是否在受控合成退化下成立", 不得外推到
真实 GSD/传感器/云/视角等 RS 条件。真实 RS 条件实验需 paired 同区域
异条件数据, 当前 DIOR↔COCO 非配对, 暂阻塞。
==================================================================

数据泄漏纪律 (元 #4):
  - v4 编码器仅在 DIOR 训练 (unified_space_v4.pt), 本脚本不训练编码器。
  - reranker 在 DIOR 独立校准子集训练 (与 DIOR 原型池图像不相交)。
  - COCO 仅用于最终测试: 不参与任何参数 / 阈值 / 条件选择。

6 对照组:
  GDINO / A-CSPC / A-CSPC+z_c / A-CSPC+z_k / A-CSPC+(z_c,z_k) / A-CSPC+shuffled z_k
最后一组(打乱 z_k)最关键: 若增益消失, 证明增益来自条件信息而非额外参数/随机性。

玩具条件公平性 (元 #4):
  同一候选框 crop 的 4 版本 x^(1..4) 对应分辨率降级 k∈{1,2,3,4}。
  Δ_degradation = perf(k=1) − perf(k=4); 若联合表示有效, 应使 Δ_joint < Δ_A-CSPC。
"""
import os, sys, json, pickle
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import unified_eval as U
import unified_space_v2 as US2
import tge_probe as T

DEVICE = U.DEVICE
SAVE_DIR = U.SAVE_DIR
LAM = 0.5
N_COCO = 200
K_LIST = [1, 2, 3, 4]
V4_PATH = SAVE_DIR / 'unified_space_v4.pt'
SRC_BANK = ['airplane', 'ship', 'vehicle']
COCO_MAP = [('airplane', 'airplane'), ('boat', 'ship'), ('car', 'vehicle'),
            ('truck', 'vehicle'), ('bus', 'vehicle'), ('motorcycle', 'vehicle')]


# ═════════════════════ V4Net (与 unified_space_v4 同构, 内联以隔离重依赖) ═══════════════════

class ContentEncoder(nn.Module):
    def __init__(self, content_ch=64, freeze=True):
        super().__init__()
        self.dino, self.pre = U.get_dinov2()
        if freeze:
            for p in self.dino.parameters():
                p.requires_grad = False
        self.dino.eval()
        self.proj = nn.Conv2d(768, content_ch, 1)

    def forward(self, x_pre):
        with torch.no_grad():
            h = self.dino(x_pre).hidden_states[11][:, 1:, :]
        B = h.shape[0]
        h = h.reshape(B, 16, 16, 768).permute(0, 3, 1, 2)
        return self.proj(h)


class CondEncoder(nn.Module):
    def __init__(self, cond_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, cond_dim))

    def forward(self, x):
        return self.net(x)


class V4Net(nn.Module):
    def __init__(self, content_ch=64, n_class=20):
        super().__init__()
        self.content = ContentEncoder(content_ch)
        self.cond = CondEncoder(1)
        self.gc = nn.Linear(content_ch, n_class)

    def zc_pool(self, zc):
        return F.adaptive_avg_pool2d(zc, 1).flatten(1)

    def content_logits(self, zc):
        return self.gc(self.zc_pool(zc))

    def forward(self, xc_pre, xa):
        zc = self.content(xc_pre)
        zk = self.cond(xa)
        return zc, zk


# ═══════════════════ 基础工具 ═════════════════

def cached_gdino(images, prompt, key):
    p = SAVE_DIR / f'{key}.pkl'
    if p.exists():
        return pickle.load(open(p, 'rb'))
    d = U.gdino_detect(images, prompt, 0.0)
    pickle.dump(d, open(p, 'wb'))
    return d


def encode_clip(crops):
    return T.encode_clip_batch(crops).cpu().numpy()  # (N,512) L2-norm


def encode_repr(net, crops, bs=64):
    """分批编码, 防 COCO 大量候选框一次 stack 爆显存 (8GB)。"""
    net.eval()
    Zc_list, Zk_list = [], []
    for i in range(0, len(crops), bs):
        batch = crops[i:i + bs]
        xc = torch.stack([net.content.pre(c.resize((224, 224))) for c in batch]).to(DEVICE)
        xa = torch.stack([transforms.ToTensor()(c.resize((384, 384))) for c in batch]).to(DEVICE)
        with torch.no_grad():
            zc, zk = net(xc, xa)
        Zc_list.append(net.zc_pool(zc).cpu().numpy())
        Zk_list.append(zk.cpu().numpy().flatten())
        del xc, xa, zc, zk
        torch.cuda.empty_cache()
    return np.vstack(Zc_list), np.concatenate(Zk_list)


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-8)


def acspc_scores_mat(gd, Feats, proto_mat, neg_mat, lam=LAM):
    pos = (Feats * proto_mat).sum(1)
    negs = (Feats * neg_mat).sum(1)
    vis = np.maximum(0.0, pos - lam * negs)
    gd = np.clip(gd, 1e-6, 1.0)
    return np.sqrt(gd * np.clip(vis, 1e-6, 1.0))


def eval_scores(per_img_boxes, per_img_gts, per_img_scores):
    """逐图匹配 11-point AP50 + Recall/Precision/F1 (全局排序, 论文口径)。"""
    n_gt = sum(len(g) for g in per_img_gts)
    if n_gt == 0:
        return 0.0, 0.0, 0.0, 0.0
    evals = []
    for boxes, gts, scores in zip(per_img_boxes, per_img_gts, per_img_scores):
        if len(boxes) == 0:
            continue
        matched = [False] * len(gts)
        idx = np.argsort(-np.asarray(scores))
        for i in idx:
            bi = boxes[i]
            best, bj = 0.0, -1
            for j, gb in enumerate(gts):
                if not matched[j]:
                    v = iou(bi, gb)
                    if v > best:
                        best, bj = v, j
            if best >= 0.5 and bj >= 0:
                evals.append((float(scores[i]), 1)); matched[bj] = True
            else:
                evals.append((float(scores[i]), 0))
    if not evals:
        return 0.0, 0.0, 0.0, 0.0
    evals.sort(key=lambda x: -x[0])
    tp = np.array([e[1] for e in evals])
    fc = np.array([1 - e[1] for e in evals])
    tc = np.cumsum(tp); fc = np.cumsum(fc)
    rec = tc / n_gt
    prec = tc / (tc + fc + 1e-8)
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        m = rec >= t
        if m.any():
            ap += float(prec[m].max())
    ap /= 11.0
    TP = int(tc[-1]); FP = int(fc[-1])
    recall = TP / n_gt
    precision = TP / (TP + FP + 1e-8)
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    return ap, recall, precision, f1


# ═══════════════════ DIOR 源原型 (原型池, 与校准池不相交) ═══════════════════

def build_dior_source(n_proto=10, n_calib=10):
    # 注: 每类 DIOR 仅 ~20 张含GT图, 故默认 10/10 对半切
    proto = {}
    for src in SRC_BANK:
        imgs, gts = U.load_dior(src, n_proto + n_calib)
        crops = []
        for im, g in zip(imgs[:n_proto], gts[:n_proto]):
            for box in g:
                x1, y1, x2, y2 = [max(0, int(v)) for v in box]
                x2 = min(im.width, x2); y2 = min(im.height, y2)
                if x2 > x1 + 4 and y2 > y1 + 4:
                    crops.append(im.crop((x1, y1, x2, y2)))
        F = encode_clip(crops)
        p = F.mean(0)
        proto[src] = p / np.linalg.norm(p)
    neg = {}
    for src in SRC_BANK:
        others = {c: proto[c] for c in SRC_BANK if c != src}
        sims = {c: float(np.dot(proto[src], v)) for c, v in others.items()}
        top3 = sorted(sims.items(), key=lambda x: -x[1])[:3]
        nv = np.mean([others[c] for c, _ in top3], axis=0)
        neg[src] = nv / np.linalg.norm(nv)
    return proto, neg


def build_calib(net, src, n_proto=10, n_calib=10):
    imgs, gts = U.load_dior(src, n_proto + n_calib)
    cimgs = imgs[n_proto:n_proto + n_calib]
    cgts = gts[n_proto:n_proto + n_calib]
    prompt = f'an overhead image of {src}'
    dets = cached_gdino(cimgs, prompt, f'acspc_rerank_calib_{src}')
    crops_all, gd_all, labels = [], [], []
    per_img_boxes, per_img_gts = [], []
    for (b, s), im, g in zip(dets, cimgs, cgts):
        keep = s > U.PRE_FILTER
        bb = b[keep]; ss = s[keep]
        per_img_boxes.append(np.array(bb, dtype=np.float64))
        per_img_gts.append(np.array(g, dtype=np.float64))
        for box, sc in zip(bb, ss):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2 = min(im.width, x2); y2 = min(im.height, y2)
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                crops_all.append(im.crop((x1, y1, max(x1 + 5, x2), max(y1 + 5, y2))))
            else:
                crops_all.append(im.crop((x1, y1, x2, y2)))
            gd_all.append(float(sc))
            lab = 1 if any(iou(list(box), list(gg)) >= 0.5 for gg in g) else 0
            labels.append(lab)
    if len(crops_all) == 0:
        return None
    Feats = encode_clip(crops_all)
    Zc, Zk = encode_repr(net, crops_all)
    return {'src': src, 'Feats': Feats, 'Zc': Zc, 'Zk': Zk,
            'gd': np.array(gd_all), 'labels': np.array(labels),
            'per_img_boxes': per_img_boxes, 'per_img_gts': per_img_gts,
            'n_crops': len(crops_all)}


# ═══════════════════ reranker ═══════════════════

class Reranker(nn.Module):
    def __init__(self, dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def train_rerankers(Zc, Zk, labels, tr, epochs=80, bs=128):
    y = torch.tensor(labels[tr], dtype=torch.float32).to(DEVICE)
    Zc_t = torch.tensor(Zc[tr], dtype=torch.float32)
    Zk_t = torch.tensor(Zk[tr], dtype=torch.float32)
    inp = {'g_full': torch.cat([Zc_t, Zk_t.unsqueeze(1)], 1).to(DEVICE),
           'g_c': Zc_t.to(DEVICE),
           'g_k': Zk_t.unsqueeze(1).to(DEVICE)}
    models = {}
    for name, dim in [('g_full', 65), ('g_c', 64), ('g_k', 1)]:
        m = Reranker(dim).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for ep in range(epochs):
            perm = torch.randperm(len(y))
            for i in range(0, len(y), bs):
                bi = perm[i:i + bs]
                pred = m(inp[name][bi]).flatten()
                loss = nn.BCELoss()(pred, y[bi])
                opt.zero_grad(); loss.backward(); opt.step()
        models[name] = m
    return models


def g_pred(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()


def tune_alpha(models, gd, Feats, proto_mat, neg_mat, Zc, Zk, per_img_boxes,
               per_img_gts, crop_img, va_img_unique, va):
    a_val = acspc_scores_mat(gd[va], Feats[va], proto_mat[va], neg_mat[va])
    # val 图像 -> crop 位置映射
    va_pos_by_img = {i: np.where(crop_img[va] == i)[0] for i in va_img_unique}
    va_per_img_boxes = [per_img_boxes[i] for i in va_img_unique]
    va_per_img_gts = [per_img_gts[i] for i in va_img_unique]

    def val_scores(s):
        return [s[va][pos] for pos in va_pos_by_img.values()]

    best = {}
    grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    for name in ['g_c', 'g_k', 'full']:
        best_a, best_ap = 0.0, -1.0
        for alpha in grid:
            if name == 'g_c':
                p = g_pred(models['g_c'], Zc[va])
                s = a_val + alpha * p
            elif name == 'g_k':
                p = g_pred(models['g_k'], Zk[va].reshape(-1, 1))
                s = a_val + alpha * p
            else:
                p = g_pred(models['g_full'], np.concatenate([Zc[va], Zk[va].reshape(-1, 1)], 1))
                s = a_val + alpha * p
            ap, _, _, _ = eval_scores(va_per_img_boxes, va_per_img_gts, val_scores(s))
            if ap > best_ap:
                best_ap, best_a = ap, alpha
        best[name] = best_a
    best['gdino'] = 0.0
    best['acspc'] = 0.0
    best['shuffled'] = best['full']
    return best


# ═══════════════════ COCO 测试集编码 (4 版本退化) ═══════════════════

def encode_coco_class(net, cls, src, cache_path):
    if cache_path.exists():
        d = pickle.load(open(cache_path, 'rb'))
        print(f'  [缓存] {cls}: {d["n_crops"]} 候选框, {d["n_img"]} 图', flush=True)
        return d
    images, gts = U.load_coco(cls, N_COCO)
    prompt = f'a photo of {cls}'
    dets = cached_gdino(images, prompt, f'acspc_rerank_coco_{cls}')
    per_img_boxes, per_img_gts, gd_list, crops_by_img = [], [], [], []
    for (b, s), im, g in zip(dets, images, gts):
        keep = s > U.PRE_FILTER
        bb = b[keep]; ss = s[keep]
        per_img_boxes.append(np.array(bb, dtype=np.float64))
        per_img_gts.append(np.array(g, dtype=np.float64))
        cr = []
        for box, sc in zip(bb, ss):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2 = min(im.width, x2); y2 = min(im.height, y2)
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                cr.append(im.crop((x1, y1, max(x1 + 5, x2), max(y1 + 5, y2))))
            else:
                cr.append(im.crop((x1, y1, x2, y2)))
            gd_list.append(float(sc))
        crops_by_img.append(cr)
    ranges = []; tot = 0
    for cr in crops_by_img:
        ranges.append((tot, tot + len(cr))); tot += len(cr)
    gd = np.array(gd_list)
    all_crops = [c for cr in crops_by_img for c in cr]
    Feats, Zc, Zk = {}, {}, {}
    for k in K_LIST:
        degraded = [US2.degrade_res(c, k) for c in all_crops]
        Feats[k] = encode_clip(degraded)
        zc, zk = encode_repr(net, degraded)
        Zc[k] = zc; Zk[k] = zk
        print(f'    k={k} 编码 {tot} 框', flush=True)
    d = {'per_img_boxes': per_img_boxes, 'per_img_gts': per_img_gts, 'ranges': ranges,
         'gd': gd, 'Feats': Feats, 'Zc': Zc, 'Zk': Zk, 'src': src,
         'n_img': len(images), 'n_crops': tot}
    pickle.dump(d, open(cache_path, 'wb'))
    print(f'  [编码] {cls}: {tot} 候选框, {len(images)} 图', flush=True)
    return d


def eval_class(d, proto, neg, alpha, g_models):
    src = d['src']; ranges = d['ranges']
    gd = d['gd']
    proto_vec = proto[src]; neg_vec = neg[src]
    proto_mat = np.tile(proto_vec, (len(gd), 1))
    neg_mat = np.tile(neg_vec, (len(gd), 1))
    per_img_boxes = d['per_img_boxes']; per_img_gts = d['per_img_gts']
    out = {}
    for k in K_LIST:
        Fk = d['Feats'][k]; Zck = d['Zc'][k]; Zkk = d['Zk'][k]
        a = acspc_scores_mat(gd, Fk, proto_mat, neg_mat)
        # 打乱 z_k: 每图内置换 (破坏 z_c,z_k 对齐)
        rng = np.random.default_rng(0)
        zk_shuff = Zkk.copy()
        for (a0, a1) in ranges:
            perm = rng.permutation(a1 - a0)
            zk_shuff[a0:a1] = Zkk[a0:a1][perm]
        variants = {
            'gdino': gd.copy(),
            'acspc': a.copy(),
            'z_c': a + alpha['g_c'] * g_pred(g_models['g_c'], Zck),
            'z_k': a + alpha['g_k'] * g_pred(g_models['g_k'], Zkk.reshape(-1, 1)),
            'full': a + alpha['full'] * g_pred(g_models['g_full'],
                                               np.concatenate([Zck, Zkk.reshape(-1, 1)], 1)),
            'shuffled': a + alpha['full'] * g_pred(g_models['g_full'],
                                                   np.concatenate([Zck, zk_shuff.reshape(-1, 1)], 1)),
        }
        res = {}
        for name, s in variants.items():
            per_scores = [s[a0:a1] for (a0, a1) in ranges]
            ap, r, p, f1 = eval_scores(per_img_boxes, per_img_gts, per_scores)
            res[name] = {'ap50': round(ap, 4), 'recall': round(r, 4),
                         'precision': round(p, 4), 'f1': round(f1, 4)}
        out[k] = res
    return out


# ═══════════════════ 主流程 ═══════════════════

def main():
    print('=' * 78)
    print('第④步 机制验证: 玩具条件 A-CSPC 联合重排 (非真实 RS 泛化证据)')
    print('=' * 78)
    net = V4Net().to(DEVICE)
    net.load_state_dict(torch.load(V4_PATH, map_location=DEVICE))
    net.eval()

    print('[1] DIOR 源原型 (原型池) ...', flush=True)
    proto, neg = build_dior_source()

    print('[2] DIOR 校准池 (reranker 训练/校准, 与原型池不相交) ...', flush=True)
    calibs = [build_calib(net, src) for src in SRC_BANK]
    calibs = [c for c in calibs if c]
    if not calibs:
        raise RuntimeError('校准池为空: 所有源类 DIOR 检测不到候选框')
    Nc = sum(c['n_crops'] for c in calibs)
    Feats = np.vstack([c['Feats'] for c in calibs])
    Zc = np.vstack([c['Zc'] for c in calibs])
    Zk = np.concatenate([c['Zk'] for c in calibs])
    labels = np.concatenate([c['labels'] for c in calibs])
    gd = np.concatenate([c['gd'] for c in calibs])
    per_img_boxes = [p for c in calibs for p in c['per_img_boxes']]
    per_img_gts = [p for c in calibs for p in c['per_img_gts']]
    print(f'    校准池 {Nc} 候选框, 正例 {int(labels.sum())}, '
          f'{len(per_img_boxes)} 图', flush=True)

    # crop -> 全局图像 id
    crop_img = []
    gid = 0
    for c in calibs:
        for ji in range(len(c['per_img_boxes'])):
            crop_img.extend([gid] * len(c['per_img_boxes'][ji]))
            gid += 1
    crop_img = np.array(crop_img, dtype=int)
    # crop 的 src 类索引 (按过滤后 calibs 顺序)
    src_of_crop = []
    for ci, c in enumerate(calibs):
        src_of_crop.extend([SRC_BANK.index(c['src'])] * c['n_crops'])
    src_of_crop = np.array(src_of_crop)
    proto_mat_all = np.stack([proto[SRC_BANK[i]] for i in src_of_crop])
    neg_mat_all = np.stack([neg[SRC_BANK[i]] for i in src_of_crop])

    print('[3] 训练 reranker (g_full/g_c/g_k) on DIOR 校准池 ...', flush=True)
    g_models = train_rerankers(Zc, Zk, labels, np.arange(Nc))
    print('[4] 调 α (组合权重, 在 DIOR 校准池; 与原型池/COCO 不相交) ...', flush=True)
    alpha = tune_alpha(g_models, gd, Feats, proto_mat_all, neg_mat_all, Zc, Zk,
                       per_img_boxes, per_img_gts, crop_img,
                       sorted(set(crop_img.tolist())), np.arange(Nc))
    print(f'    α = { {k: round(v, 3) for k, v in alpha.items()} }', flush=True)

    print('[5] COCO 测试集 (仅测试): 6 对照组 × 4 退化条件 ...', flush=True)
    results = {'meta': {
        'nature': 'mechanism_validation / synthetic-condition sanity check (NOT real-RS generalization)',
        'n_coco_per_class': N_COCO,
        'leakage': 'v4 encoder trained on DIOR only; reranker calibrated on disjoint DIOR calib subset; COCO test-only',
        'controls': ['GDINO', 'A-CSPC', 'A-CSPC+z_c', 'A-CSPC+z_k',
                     'A-CSPC+(z_c,z_k)', 'A-CSPC+shuffled_z_k'],
        'k_list': K_LIST, 'lam': LAM,
        'alpha': {k: round(float(v), 4) for k, v in alpha.items()}},
        'per_class': {}}
    for cls, src in COCO_MAP:
        print(f'  --- {cls} (src={src}) ---', flush=True)
        d = encode_coco_class(net, cls, src, SAVE_DIR / f'acspc_rerank_cache_{cls}.pkl')
        res = eval_class(d, proto, neg, alpha, g_models)
        d_ap = {m: round(res[1][m]['ap50'] - res[4][m]['ap50'], 4) for m in res[1]}
        d_f1 = {m: round(res[1][m]['f1'] - res[4][m]['f1'], 4) for m in res[1]}
        results['per_class'][cls] = {
            'src': src,
            'k': {str(k): res[k] for k in K_LIST},
            'delta_deg_ap50': d_ap,
            'delta_deg_f1': d_f1}

    # 跨类均值
    meths = ['gdino', 'acspc', 'z_c', 'z_k', 'full', 'shuffled']
    mean_k1 = {m: round(float(np.mean([results['per_class'][c]['k']['1'][m]['ap50']
                                       for c in results['per_class']])), 4) for m in meths}
    mean_dap = {m: round(float(np.mean([results['per_class'][c]['delta_deg_ap50'][m]
                                        for c in results['per_class']])), 4) for m in meths}
    results['mean'] = {'ap50_k1': mean_k1, 'delta_deg_ap50': mean_dap}

    out = SAVE_DIR / 'acspc_unified_rerank.json'
    json.dump(results, open(out, 'w'), ensure_ascii=False, indent=2)

    # 控制台汇总
    print('\n' + '=' * 78)
    print('跨类均值 AP50 @ k=1 (真实条件):')
    for m in meths:
        print(f'  {m:14s} AP50={mean_k1[m]:.4f}')
    print('\n跨类均值 Δ_degradation (AP50 k=1 − k=4, 越小越抗退化):')
    for m in meths:
        print(f'  {m:14s} Δ={mean_dap[m]:+.4f}')
    print('\n关键判据: Δ_A-CSPC(full) < Δ_A-CSPC ? '
          f'{ "✅ 成立" if mean_dap["full"] < mean_dap["acspc"] else "❌ 不成立" }')
    print('关键判据: A-CSPC(full) 增益 vs A-CSPC(shuffled) 增益:')
    g_full = mean_k1['full'] - mean_k1['acspc']
    g_shuf = mean_k1['shuffled'] - mean_k1['acspc']
    print(f'  full  增益 = {g_full:+.4f}')
    print(f'  shuffled 增益 = {g_shuf:+.4f}')
    print(f'  → {"✅ z_k 提供条件信息(打乱后增益消失)" if abs(g_shuf) < abs(g_full) * 0.5 else "⚠️ 增益不依赖 z_k 对齐"}')
    print(f'\n结果 → {out}')


if __name__ == '__main__':
    main()
