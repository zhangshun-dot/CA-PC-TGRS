# -*- coding: utf-8 -*-
"""probe_clip_diag.py — 截断(max(0,·))退化假说 与 δ 的候选级作用诊断

待解释的反常事实(汇报稿消融, N=100):
    去除正原型外推  (P_src − λP_neg)  → −0.0005  (≈ 基线, 等于没用)
    纯外推          (P̂,      λ=0)    → +0.0239
    完整            (P̂ − λP_neg)     → +0.0652
  三者**超可加**: 0.0239 + (−0.0005) ≈ 0.023, 但组合 = 0.0652。

假说: 融合分数 s = sqrt(gd · max(0, F·W))。
  - 单独减 λF·P_neg 会把大量候选压到 0 以下 → 全被截断成 0 → 排序退化回 gd → 增益≈0;
  - 单独加 βδ 只是整体平移, 无负原型时本来就没被压到 0, 平移收益有限;
  - 两者组合: 偏移把被负原型压低的候选重新抬回截断门之上, 使"减负原型"这个
    有判别力的边际项重新生效 → 超可加。
  若成立, 则 δ 的真实角色是【恢复边际项的偏置项】, 而非"域搬运"。

诊断量:
  clip_frac(F·W ≤ 0): 被截断候选占比
  mean/std(F·W)
  AUC(F·δ → TP): δ 本身是不是前景判别信号
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

from acspc.config import Config
from acspc import data as D
from acspc import metrics as Met

CACHE_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_dirmech_cache')
PROTO_CACHE = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_clip_diag.json')

BETA, LAM, NEG_TOPK, N_PROTO, N_EVAL = 2.0, 0.5, 3, 20, 30
COPOOL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
DIPOOL = ['airplane', 'ship', 'vehicle', 'harbor', 'storage tank', 'bridge']

TASKS = {
    'same_dior2dior': ('dior', [('airplane', 'airplane'), ('ship', 'ship'), ('vehicle', 'vehicle')]),
    'same_coco2coco': ('coco', [('airplane', 'airplane'), ('boat', 'boat'), ('car', 'car')]),
    'rev_coco2dior': ('coco', [('airplane', 'airplane'), ('boat', 'ship'), ('car', 'vehicle')]),
    'fwd_dior2coco': ('dior', [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]),
}

_CLIP = {}
CLIP_ID = 'openai/clip-vit-base-patch32'


def log(*a):
    print(*a, flush=True)


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


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def iou_mat(b, g):
    if len(b) == 0 or len(g) == 0:
        return np.zeros((len(b), len(g)))
    ix1 = np.maximum(b[:, None, 0], g[None, :, 0]); iy1 = np.maximum(b[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(b[:, None, 2], g[None, :, 2]); iy2 = np.minimum(b[:, None, 3], g[None, :, 3])
    iw = np.maximum(0.0, ix2 - ix1); ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    return inter / np.maximum(ab[:, None] + ag[None, :] - inter, 1e-12)


def roc_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    o = np.argsort(s, kind='mergesort'); r = np.arange(1, len(s) + 1, dtype=float)
    pos = r[y[o] == 1]; npos, nneg = len(pos), len(y) - len(pos)
    if npos == 0 or nneg == 0:
        return 0.5
    return float((pos.sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def ap_of(per_img, F, W, collect=None):
    allw = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            if collect is not None:
                collect.append(w)
            p['_s'] = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            p['_s'] = np.zeros(0)
    preds = [(p['box'], p['_s']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return Met.compute_ap50_img(preds, gts)


def ap_base(per_img, F):
    for p in per_img:
        p['_s'] = p['gd'].copy()
    return Met.compute_ap50_img([(p['box'], p['_s']) for p in per_img], [p['gt'] for p in per_img])


def load_blob(task, cls):
    p = CACHE_DIR / f'{task}_{cls}_off{N_PROTO}_n{N_EVAL}.pkl'
    return pickle.load(open(p, 'rb')) if p.exists() else None


def build_protos(cfg, domain, pool):
    allp = pickle.load(open(PROTO_CACHE, 'rb')) if PROTO_CACHE.exists() else {}
    if domain in allp and len(allp[domain]) >= len(pool):
        return allp[domain]
    protos = {}
    for cls in pool:
        ims, gts = (D.load_coco(cfg, cls, n=N_PROTO) if domain == 'coco'
                    else D.load_dior(cfg, cls, n=N_PROTO))
        crops = D.gt_crops(ims, gts)
        if not crops:
            continue
        protos[cls] = unit(clip_encode_imgs(crops).mean(0))
    allp[domain] = protos
    pickle.dump(allp, open(PROTO_CACHE, 'wb'))
    return protos


def neg_proto(protos, pos_cls, topk=NEG_TOPK):
    others = {k: v for k, v in protos.items() if k != pos_cls}
    if not others:
        return np.zeros(512)
    ks = list(others); P = np.stack([others[k] for k in ks])
    order = np.argsort(-(P @ protos[pos_cls]))[:topk]
    return unit(P[order].mean(0))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--tasks', nargs='+', default=list(TASKS.keys()))
    args = ap_.parse_args()
    cfg = Config(); cfg.model.device = 'cpu'

    delta_bar = np.mean([text_emb(f'a photo of {c}') - text_emb(f'an overhead image of {c}')
                         for c in DIPOOL + COPOOL], axis=0)
    nrm = float(np.linalg.norm(delta_bar))
    rng = np.random.default_rng(0)
    rands = [unit(rng.normal(size=512)) * nrm for _ in range(3)]

    out = {}
    for tname in args.tasks:
        src_dom, pairs = TASKS[tname]
        pool = DIPOOL if src_dom == 'dior' else COPOOL
        protos = build_protos(cfg, src_dom, pool)
        log(f'\n### {tname} (源={src_dom})')
        out[tname] = {}
        for (scls, ecls) in pairs:
            b = load_blob(tname, ecls)
            if b is None:
                log(f'  !! 无缓存 {tname}_{ecls}')
                continue
            per_img, F = b['per_img'], b['F']
            P_src = protos.get(scls)
            if P_src is None:
                continue
            P_neg = neg_proto(protos, scls)
            d_c = text_emb(f'a photo of {ecls}') - text_emb(f'an overhead image of {ecls}')

            base = ap_base(per_img, F)
            variants = {
                'no_delta': P_src - LAM * P_neg,
                'full_plus': unit(P_src + BETA * d_c) - LAM * P_neg,
                'full_minus': unit(P_src - BETA * d_c) - LAM * P_neg,
                'pos_only': unit(P_src + BETA * d_c),
                'additive': P_src - LAM * P_neg + BETA * d_c,
                'add_global': P_src - LAM * P_neg + BETA * delta_bar,
                'add_rand0': P_src - LAM * P_neg + BETA * rands[0],
                'add_rand1': P_src - LAM * P_neg + BETA * rands[1],
                'add_rand2': P_src - LAM * P_neg + BETA * rands[2],
            }
            r = {'baseline': base}
            for k, W in variants.items():
                buf = []
                r[k] = ap_of(per_img, F, W, collect=buf)
                w = np.concatenate(buf)
                r[k + '__clipfrac'] = float((w <= 0).mean())
                r[k + '__mean'] = float(w.mean())
                r[k + '__std'] = float(w.std())

            tp, idx = [], []
            for p in per_img:
                if len(p['gd']) == 0:
                    continue
                m = iou_mat(p['box'], p['gt'])
                tp.append(((m.max(1) if m.size else np.zeros(len(p['gd']))) >= 0.5).astype(int))
                idx.append(p['crop_idx'])
            tp = np.concatenate(tp); idx = np.concatenate(idx)
            r['n_crop'] = int(len(tp)); r['n_tp'] = int(tp.sum())
            r['auc_Fdelta_TP'] = roc_auc(tp, F[idx] @ delta_bar)
            r['auc_gd_TP'] = roc_auc(tp, np.concatenate([p['gd'] for p in per_img if len(p['gd'])]))
            r['auc_FPsrc_TP'] = roc_auc(tp, F[idx] @ P_src)

            out[tname][f'{scls}->{ecls}'] = r
            log(f'  {scls}->{ecls} base={base:.4f} | '
                + '  '.join(f'{k}={r[k]-base:+.4f}(clip {r[k+"__clipfrac"]:.2f})'
                            for k in ['no_delta', 'full_plus', 'full_minus', 'pos_only',
                                      'additive', 'add_global', 'add_rand0'])
                + f' | AUC(F·δ→TP)={r["auc_Fdelta_TP"]:.3f}')

    json.dump(out, open(OUT, 'w'), indent=2, ensure_ascii=False)

    log('\n' + '=' * 84)
    log('汇总 (增益 / 截断占比)')
    log('=' * 84)
    keys = ['no_delta', 'full_plus', 'full_minus', 'pos_only', 'additive', 'add_global',
            'add_rand0', 'add_rand1', 'add_rand2']
    for tname in out:
        rows = {k: v for k, v in out[tname].items() if '->' in k}
        if not rows:
            continue
        log(f'\n[{tname}]')
        log(f'  {"变体":<14s} ' + '  '.join(f'{rk:>22s}' for rk in rows) + f' {"均值":>9s}')
        for k in keys:
            g = [rows[rk][k] - rows[rk]['baseline'] for rk in rows]
            cf = [rows[rk][k + '__clipfrac'] for rk in rows]
            log(f'  {k:<14s} ' + '  '.join(f'{a:+.4f}/c{b:.2f}' for a, b in zip(g, cf))
                + f' {np.mean(g):+9.4f}')
        log(f'  {"baseline":<14s} ' + '  '.join(f'{rows[rk]["baseline"]:>22.4f}' for rk in rows))
        log(f'  AUC(F·δ→TP)    ' + '  '.join(f'{rows[rk]["auc_Fdelta_TP"]:>22.3f}' for rk in rows))
        log(f'  AUC(gd→TP)     ' + '  '.join(f'{rows[rk]["auc_gd_TP"]:>22.3f}' for rk in rows))
    log(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
