# -*- coding: utf-8 -*-
"""probe_offset_mechanism.py — 外推偏移的机制解剖 (复用已有 COCO 候选缓存, 不跑检测)

动机 (来自 probe_delta_structure):
  δ_c = E('a photo of c') - E('an overhead image of c') 的类间两两余弦 = +0.863,
  全局成分占比 0.932, 随机方向对照 ≈ 0.00
  → δ 基本是一个【类别无关的全局偏移】, 而不是"把原型搬向目标域"的方向。

本脚本检验这一"全局偏移"假说的全部可证伪后果:
  (T1) 用全局 δ̄ 替代逐类 δ_c, 增益应几乎不变;
  (T4) 只保留类别残差 (δ_c - δ̄), 增益应消失;
  (T2) 若同时平移负原型(对称施加), 增益应消失 → 证明"非对称"是承重构件;
  (T3) 换成同范数随机方向, 增益应消失;
  (T5) 反向 (-δ) 应显著更差 (已在反向探针中观察到, 此处复现)。

并做 crop 级诊断: F·δ 是否本身就是一个"前景/目标性"先验(AUC 预测 TP)。

复用: 实验数据/acspc_rerank_cache_{coco_cls}.pkl (GDINO 候选框 + CLIP 特征 + GT)
"""
import os, sys, json, pickle
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_COCO_ANN', r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
os.environ.setdefault('AC_SPC_COCO_IMG', r'C:\Users\张三\modeling\coco_data\val2017')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')

sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

from pathlib import Path
import argparse
import numpy as np
import torch
from PIL import Image

from acspc.config import Config, DIOR_ID2NAME
from acspc import data as D
from acspc import metrics as Met

CACHE_DIR = Path(r'C:\Users\张三\Desktop\实验数据')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_offset_mechanism.json')

BETA = 2.0
LAM = 0.5
NEG_TOPK = 3
N_PROTO = 20
N_RAND = 5

# COCO 评估类 -> DIOR 源类
COCO2DIOR = {
    'airplane': 'airplane',
    'boat': 'ship',
    'car': 'vehicle',
    'truck': 'vehicle',
    'bus': 'vehicle',
    'motorcycle': 'vehicle',
}
DIOR_POOL = sorted(DIOR_ID2NAME.values())


def log(*a):
    print(*a, flush=True)


# ───────── CLIP ─────────
_CLIP = {}
CLIP_ID = 'openai/clip-vit-base-patch32'


def get_clip():
    if 'model' not in _CLIP:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP['model'] = CLIPModel.from_pretrained(CLIP_ID).eval()
        _CLIP['proc'] = CLIPProcessor.from_pretrained(CLIP_ID)
    return _CLIP['model'], _CLIP['proc']


@torch.no_grad()
def clip_encode_imgs(crops, bs=64):
    model, proc = get_clip()
    out = []
    for i in range(0, len(crops), bs):
        inputs = proc(images=crops[i:i + bs], return_tensors='pt', padding=True)
        f = model.get_image_features(**inputs).float()
        out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out).cpu().numpy() if out else np.zeros((0, 512))


@torch.no_grad()
def text_emb(t):
    model, proc = get_clip()
    inputs = proc(text=[t], return_tensors='pt', padding=True, truncation=True)
    f = model.get_text_features(**inputs).float()
    return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()[0]


@torch.no_grad()
def text_emb_batch(ts, bs=64):
    model, proc = get_clip()
    out = []
    for i in range(0, len(ts), bs):
        inputs = proc(text=ts[i:i + bs], return_tensors='pt', padding=True, truncation=True)
        f = model.get_text_features(**inputs).float()
        out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out).cpu().numpy()


def unit(x):
    return x / (np.linalg.norm(x) + 1e-12)


# ───────── 度量 ─────────
def roc_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    order = np.argsort(s, kind='mergesort')
    ranks = np.arange(1, len(s) + 1, dtype=float)
    pos = ranks[y[order] == 1]
    n_pos, n_neg = len(pos), len(y) - len(pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((pos.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def iou_mat(b, g):
    if len(b) == 0 or len(g) == 0:
        return np.zeros((len(b), len(g)))
    ix1 = np.maximum(b[:, None, 0], g[None, :, 0])
    iy1 = np.maximum(b[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(b[:, None, 2], g[None, :, 2])
    iy2 = np.minimum(b[:, None, 3], g[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1); ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / np.maximum(ab[:, None] + ag[None, :] - inter, 1e-12)


# ───────── 目标侧: 复用缓存 ─────────
def load_target(coco_cls):
    p = CACHE_DIR / f'acspc_rerank_cache_{coco_cls}.pkl'
    c = pickle.load(open(p, 'rb'))
    return c


def build_per_img(c):
    """把缓存整理成统一 per_img 结构 + 全局 F"""
    F = c['Feats'][1]                      # k=1 = 未退化的 CLIP 特征 (N,512)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
    boxes, gts, ranges, gd = c['per_img_boxes'], c['per_img_gts'], c['ranges'], c['gd']
    per_img = []
    for (a, b), g, (ra, rb) in zip(ranges, gts, ranges):
        pass
    for i, (bx, gt) in enumerate(zip(boxes, gts)):
        a, b = ranges[i]
        per_img.append({
            'box': np.asarray(bx, dtype=np.float64),
            'gd': np.clip(np.asarray(gd[a:b], dtype=np.float64), 1e-6, 1.0),
            'gt': np.asarray(gt, dtype=np.float64).reshape(-1, 4),
            'idx': np.arange(a, b),
        })
    return per_img, F


# ───────── 评估 ─────────
def eval_variant(per_img, F, W, key):
    for p in per_img:
        if len(p['gd']):
            vis = np.maximum(0.0, F[p['idx']] @ W)
            p[key] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p[key] = np.zeros(0)
    preds = [(p['box'], p[key]) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return Met.compute_ap50_img(preds, gts)


def eval_baseline(per_img, F):
    for p in per_img:
        p['base'] = p['gd'].copy()
    preds = [(p['box'], p['base']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return Met.compute_ap50_img(preds, gts)


def neg_proto(protos, pos_cls, topk=NEG_TOPK):
    others = {k: v for k, v in protos.items() if k != pos_cls}
    if not others:
        return np.zeros(512)
    ks = list(others)
    P = np.stack([others[k] for k in ks])
    sim = P @ protos[pos_cls]
    order = np.argsort(-sim)[:topk]
    m = P[order].mean(0)
    return unit(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', default='airplane,boat,car')
    args = ap.parse_args()
    classes = args.classes.split(',')

    cfg = Config()
    cfg.model.device = 'cpu'

    log('=' * 78)
    log('外推偏移机制解剖 (复用 COCO 候选缓存)')
    log(f'  BETA={BETA} LAM={LAM} neg_topk={NEG_TOPK}  类={classes}')
    log('=' * 78)

    # ── 1. DIOR 源原型 ──
    log('\n[1/5] 构建 DIOR 源原型 ...')
    proto_cache = CACHE_DIR / 'probe_offset_dior_protos.pkl'
    if proto_cache.exists():
        protos = pickle.load(open(proto_cache, 'rb'))
        log(f'  [cache] {len(protos)} 个源原型')
    else:
        protos = {}
        need = sorted(set(DIOR_POOL))
        for cls in need:
            try:
                imgs, gts = D.load_dior(cfg, cls, n=N_PROTO)
            except Exception as e:
                log(f'  !! {cls}: {e}')
                continue
            crops = D.gt_crops(imgs, gts)
            if not crops:
                continue
            F = clip_encode_imgs(crops)
            protos[cls] = unit(F.mean(0))
        pickle.dump(protos, open(proto_cache, 'wb'))
    log(f'  源原型: {sorted(protos)}')

    # ── 2. δ 的构造 ──
    log('\n[2/5] 构造 δ ...')
    delta_c = {}
    for c in classes:
        delta_c[c] = text_emb(f'a photo of {c}') - text_emb(f'an overhead image of {c}')
    # 全局 δ̄: 在 DIOR 全 20 类上取平均 (不需要目标域知识)
    pool_names = [c for c in DIOR_POOL if c in protos] or DIOR_POOL
    Dm = np.stack([text_emb(f'a photo of {c}') - text_emb(f'an overhead image of {c}')
                   for c in pool_names])
    delta_bar = Dm.mean(0)
    nrm = np.linalg.norm(np.stack([delta_c[c] for c in classes]), axis=1).mean()
    log(f'  ||δ_c|| 均值={nrm:.4f}   ||δ̄||={np.linalg.norm(delta_bar):.4f}')

    rng = np.random.default_rng(0)
    rand_dirs = [unit(rng.normal(size=512)) * nrm for _ in range(N_RAND)]

    VARIANTS = {
        'baseline':        lambda c: None,
        'acspc_perclass':  lambda c: ('perclass', delta_c[c]),
        'acspc_global':    lambda c: ('global', delta_bar),
        'residual_only':   lambda c: ('resid', delta_c[c] - delta_bar),
        'anti_delta':      lambda c: ('anti', -delta_bar),
        'beta0':           lambda c: ('zero', np.zeros(512)),
    }

    results = {}
    diag = {}

    for coco_cls in classes:
        log(f'\n[3/5] {coco_cls}: 加载缓存 ...')
        c = load_target(coco_cls)
        per_img, F = build_per_img(c)
        n_crops = F.shape[0]
        src_cls = COCO2DIOR[coco_cls]
        if src_cls not in protos:
            log(f'  !! 缺源原型 {src_cls}, 跳过')
            continue
        P_src = protos[src_cls]
        P_neg = neg_proto(protos, src_cls)

        base = eval_baseline(per_img, F)
        r = {'baseline': base, 'n_img': len(per_img), 'n_crops': int(n_crops)}

        for name, fn in VARIANTS.items():
            if name == 'baseline':
                continue
            kind, dl = fn(coco_cls)
            P_hat = unit(P_src + BETA * dl)
            if name == 'residual_only':
                # 归一化到与 δ 同范数, 保证"只有类别残差"而非"幅度更小"
                dl = unit(dl) * nrm
                P_hat = unit(P_src + BETA * dl)
            W = P_hat - LAM * P_neg
            r[name] = eval_variant(per_img, F, W, 'v_' + name)

        # 对称施加: 负原型也平移
        P_hat = unit(P_src + BETA * delta_bar)
        P_neg_s = unit(P_neg + BETA * delta_bar)
        r['symmetric_global'] = eval_variant(per_img, F, P_hat - LAM * P_neg_s, 'v_sym')
        P_hat_c = unit(P_src + BETA * delta_c[coco_cls])
        P_neg_sc = unit(P_neg + BETA * delta_c[coco_cls])
        r['symmetric_perclass'] = eval_variant(per_img, F, P_hat_c - LAM * P_neg_sc, 'v_symc')

        # 只平移负原型
        r['neg_only_shift'] = eval_variant(per_img, F, P_src - LAM * unit(P_neg + BETA * delta_bar), 'v_negonly')

        # 随机方向
        rands = []
        for i, d in enumerate(rand_dirs):
            rands.append(eval_variant(per_img, F, unit(P_src + BETA * d) - LAM * P_neg, f'v_rand{i}'))
        r['random_dirs'] = rands
        r['random_mean'] = float(np.mean(rands))
        r['random_std'] = float(np.std(rands))

        # 无负原型 (λ=0)
        r['pos_only_global'] = eval_variant(per_img, F, unit(P_src + BETA * delta_bar), 'v_posonly')
        r['pos_only_nodelta'] = eval_variant(per_img, F, P_src, 'v_posnoD')

        results[coco_cls] = r

        # ── crop 级诊断: F·δ 是不是前景先验? ──
        tp, fdot, gdv = [], [], []
        for p in per_img:
            if len(p['gd']) == 0:
                continue
            m = iou_mat(p['box'], p['gt'])
            best = m.max(1) if m.size else np.zeros(len(p['gd']))
            tp.append((best >= 0.5).astype(int))
            fdot.append(F[p['idx']] @ delta_bar)
            gdv.append(p['gd'])
        tp = np.concatenate(tp); fdot = np.concatenate(fdot); gdv = np.concatenate(gdv)
        diag[coco_cls] = {
            'n': int(len(tp)), 'n_tp': int(tp.sum()),
            'auc_FdotDelta_for_TP': roc_auc(tp, fdot),
            'auc_gd_for_TP': roc_auc(tp, gdv),
            'auc_FdotProto_for_TP': roc_auc(tp, F @ P_src),
            'mean_FdotDelta_tp': float(fdot[tp == 1].mean()) if tp.sum() else None,
            'mean_FdotDelta_fp': float(fdot[tp == 0].mean()) if (1 - tp).sum() else None,
            'corr_FdotDelta_gd': float(np.corrcoef(fdot, gdv)[0, 1]),
        }
        log(f'  baseline={base:.4f}  perclass={r["acspc_perclass"]:+.4f}  '
            f'global={r["acspc_global"]:+.4f}  resid={r["residual_only"]:+.4f}  '
            f'sym={r["symmetric_global"]:+.4f}  anti={r["anti_delta"]:+.4f}  '
            f'rand={r["random_mean"]:+.4f}±{r["random_std"]:.4f}')

    out = {'results': results, 'crop_diag': diag,
           'beta': BETA, 'lam': LAM, 'neg_topk': NEG_TOPK,
           'delta_bar_norm': float(np.linalg.norm(delta_bar)),
           'delta_c_norm_mean': float(nrm)}
    json.dump(out, open(OUT, 'w'), indent=2, ensure_ascii=False)

    log('\n' + '=' * 78)
    log('汇总 (相对 baseline 的 AP50 增益)')
    log('=' * 78)
    keys = ['acspc_perclass', 'acspc_global', 'residual_only', 'symmetric_global',
            'symmetric_perclass', 'neg_only_shift', 'anti_delta', 'beta0',
            'random_mean', 'pos_only_global', 'pos_only_nodelta']
    base_m = np.mean([results[c]['baseline'] for c in results])
    log(f'  {"变体":<22s} ' + '  '.join(f'{c[:8]:>9s}' for c in results) + f' {"均值":>9s}')
    for k in keys:
        vals = [results[c][k] - results[c]['baseline'] for c in results]
        log(f'  {k:<22s} ' + '  '.join(f'{v:+9.4f}' for v in vals) + f' {np.mean(vals):+9.4f}')
    log(f'\n  baseline 均值 = {base_m:.4f}')
    log('\n[crop 级诊断] F·δ 是否前景先验:')
    for c, d in diag.items():
        log(f'  {c:<12} n={d["n"]:<6} TP={d["n_tp"]:<5} '
            f'AUC(F·δ)={d["auc_FdotDelta_for_TP"]:.4f}  AUC(gd)={d["auc_gd_for_TP"]:.4f}  '
            f'AUC(F·P_src)={d["auc_FdotProto_for_TP"]:.4f}  corr(F·δ,gd)={d["corr_FdotDelta_gd"]:+.4f}')
    log(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
