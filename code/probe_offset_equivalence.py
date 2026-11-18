# -*- coding: utf-8 -*-
"""probe_offset_equivalence.py — 外推 ≡ 加性偏移? 以及能否在【源域】估出这个偏移

背景(来自 probe_delta_structure):
  δ_c = E('a photo of c') - E('an overhead image of c') 类间两两余弦 +0.863,
  全局成分占比 0.932 (随机对照 ≈ 0.00) → δ 基本是类别无关的全局偏移。

由此推出一个可证伪的强命题:
  【等价性】A-CSPC 的"正原型外推" P̂ = unit(P_src + βδ) 在排序上等价于
            往融合权重里【直接加一个偏移】: W = P_src - λP_neg + βδ。
            若成立, 则"原型外推"这一叙述是可剥离的装饰, 真正起作用的
            就是那一个向量 βδ —— 而它只有 1 个自由度, 可以在源域估出来。

  【可估性】既然只有一个全局向量, 就可以不用手工模板, 而在【源域】拟合:
            u = unit(mean F[TP] - mean F[FP])  (源域候选框上的"目标性方向")
            完全不接触目标域 → 直接消解汇报稿边界"主配置依赖目标域GT调参"。

  【对称性】若偏移与"源→目标"无关, 则同一偏移应在 DIOR→COCO 与 COCO→DIOR
            双向都有效 → 消解边界"photo→overhead 方向外推可能失败"。

复用 probe_direction_mechanism.py 落盘的候选框缓存 (检测+CLIP 编码只做一次)。
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
from acspc import prototype as Proto

CACHE_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_dirmech_cache')
PROTO_CACHE = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_offset_equivalence.json')

BETA = 2.0
LAM = 0.5
NEG_TOPK = 3
N_PROTO = 20
N_EVAL = 30
N_RAND = 5

COPOOL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
DIPOOL = ['airplane', 'ship', 'vehicle', 'harbor', 'storage tank', 'bridge']

# 任务: 源域, 目标域, [(源类, 评测类)], 用于拟合偏移的"同域"任务
TASKS = {
    'fwd_dior2coco': ('dior', 'coco', [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')],
                      'same_dior2dior'),
    'rev_coco2dior': ('coco', 'dior', [('airplane', 'airplane'), ('boat', 'ship'), ('car', 'vehicle')],
                      'same_coco2coco'),
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


def crop_labels(per_img):
    """每个候选框是否 TP(IoU>=0.5)"""
    tp, idx = [], []
    for p in per_img:
        if len(p['gd']) == 0:
            continue
        m = iou_mat(p['box'], p['gt'])
        best = m.max(1) if m.size else np.zeros(len(p['gd']))
        tp.append((best >= 0.5).astype(int))
        idx.append(p['crop_idx'])
    return np.concatenate(tp), np.concatenate(idx)


def ap_of(per_img, F, W):
    for p in per_img:
        if len(p['gd']):
            vis = np.maximum(0.0, F[p['crop_idx']] @ W)
            p['_s'] = np.sqrt(p['gd'] * np.clip(vis, 1e-6, 1.0))
        else:
            p['_s'] = np.zeros(0)
    preds = [(p['box'], p['_s']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return Met.compute_ap50_img(preds, gts)


def ap_base(per_img, F):
    for p in per_img:
        p['_s'] = p['gd'].copy()
    preds = [(p['box'], p['_s']) for p in per_img]
    gts = [p['gt'] for p in per_img]
    return Met.compute_ap50_img(preds, gts)


def load_blob(task, cls):
    p = CACHE_DIR / f'{task}_{cls}_off{N_PROTO}_n{N_EVAL}.pkl'
    if not p.exists():
        return None
    return pickle.load(open(p, 'rb'))


def build_protos(cfg, domain, pool):
    if PROTO_CACHE.exists():
        allp = pickle.load(open(PROTO_CACHE, 'rb'))
    else:
        allp = {}
    key = f'{domain}'
    if key in allp and len(allp[key]) >= len(pool):
        return allp[key]
    protos = {}
    for cls in pool:
        ims, gts = (D.load_coco(cfg, cls, n=N_PROTO) if domain == 'coco'
                    else D.load_dior(cfg, cls, n=N_PROTO))
        crops = D.gt_crops(ims, gts)
        if not crops:
            continue
        F = clip_encode_imgs(crops)
        protos[cls] = unit(F.mean(0))
    allp[key] = protos
    pickle.dump(allp, open(PROTO_CACHE, 'wb'))
    return protos


def neg_proto(protos, pos_cls, topk=NEG_TOPK):
    others = {k: v for k, v in protos.items() if k != pos_cls}
    if not others:
        return np.zeros(512)
    ks = list(others)
    P = np.stack([others[k] for k in ks])
    sim = P @ protos[pos_cls]
    order = np.argsort(-sim)[:topk]
    return unit(P[order].mean(0))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--tasks', nargs='+', default=list(TASKS.keys()))
    args = ap_.parse_args()

    cfg = Config()
    cfg.model.device = 'cpu'

    log('=' * 80)
    log('外推 ≡ 加性偏移 ?  以及源域可估性')
    log(f'  BETA={BETA} LAM={LAM} neg_topk={NEG_TOPK} proto={N_PROTO} eval={N_EVAL}')
    log('=' * 80)

    delta_bar = np.mean([text_emb(f'a photo of {c}') - text_emb(f'an overhead image of {c}')
                         for c in DIPOOL + COPOOL], axis=0)
    nrm = np.mean([np.linalg.norm(text_emb(f'a photo of {c}') - text_emb(f'an overhead image of {c}'))
                   for c in ['airplane', 'boat', 'car']])
    log(f'  ||δ̄||={np.linalg.norm(delta_bar):.4f}   ||δ_c||均值={nrm:.4f}')
    rng = np.random.default_rng(0)
    rands = [unit(rng.normal(size=512)) * nrm for _ in range(N_RAND)]

    results = {}

    for tname in args.tasks:
        src_dom, tgt_dom, pairs, fit_task = TASKS[tname]
        pool = DIPOOL if src_dom == 'dior' else COPOOL
        protos = build_protos(cfg, src_dom, pool)
        log(f'\n### {tname}  源={src_dom} 目标={tgt_dom}  原型={sorted(protos)}')

        # ── 在【源域同域】任务上拟合偏移 u (完全不接触目标域) ──
        fit_F, fit_tp = [], []
        for (scls, ecls) in pairs:
            b = load_blob(fit_task, scls)
            if b is None:
                continue
            tp, idx = crop_labels(b['per_img'])
            fit_F.append(b['F'][idx]); fit_tp.append(tp)
        u_src = None
        if fit_F:
            Ff = np.concatenate(fit_F); tf = np.concatenate(fit_tp)
            if tf.sum() > 5 and (1 - tf).sum() > 5:
                u_src = unit(Ff[tf == 1].mean(0) - Ff[tf == 0].mean(0))
                log(f'  源域拟合: n={len(tf)} TP={int(tf.sum())}  ||u||={np.linalg.norm(Ff[tf==1].mean(0)-Ff[tf==0].mean(0)):.4f}'
                    f'  cos(u, δ̄)={float(u_src @ unit(delta_bar)):+.4f}')
            else:
                log(f'  !! 源域 TP/FP 不足, 跳过 u 拟合')
        results[tname] = {'cos_u_deltabar': float(u_src @ unit(delta_bar)) if u_src is not None else None}

        for (scls, ecls) in pairs:
            b = load_blob(tname, ecls)
            if b is None:
                log(f'  !! 缺缓存 {tname}_{ecls}, 跳过')
                continue
            per_img, F = b['per_img'], b['F']
            if scls not in protos:
                continue
            P_src = protos[scls]
            P_neg = neg_proto(protos, scls)
            d_c = text_emb(f'a photo of {ecls}') - text_emb(f'an overhead image of {ecls}')

            base = ap_base(per_img, F)
            r = {'baseline': base}

            # V1 A-CSPC 原版(归一化外推)
            P_hat = unit(P_src + BETA * d_c)
            r['acspc_extrap'] = ap_of(per_img, F, P_hat - LAM * P_neg)
            # V2 加性偏移(不归一化)
            r['additive_delta'] = ap_of(per_img, F, P_src - LAM * P_neg + BETA * d_c)
            # V3 无偏移
            r['no_delta'] = ap_of(per_img, F, P_src - LAM * P_neg)
            # V4 加性全局偏移
            r['additive_global'] = ap_of(per_img, F, P_src - LAM * P_neg + BETA * delta_bar)
            # V5 加性随机方向
            rr = [ap_of(per_img, F, P_src - LAM * P_neg + BETA * u) for u in rands]
            r['additive_random'] = rr
            r['additive_random_mean'] = float(np.mean(rr))
            # V6 只要偏移, 完全不要原型
            r['offset_only'] = ap_of(per_img, F, BETA * delta_bar)
            # V7 源域估计偏移
            if u_src is not None:
                r['src_offset'] = ap_of(per_img, F, P_src - LAM * P_neg + BETA * nrm * u_src)
                r['src_offset_only'] = ap_of(per_img, F, BETA * nrm * u_src)
            # V8 反向偏移
            r['additive_anti'] = ap_of(per_img, F, P_src - LAM * P_neg - BETA * delta_bar)

            results[tname][f'{scls}->{ecls}'] = r
            log(f'  {scls}->{ecls}: base={base:.4f} '
                f'A-CSPC={r["acspc_extrap"]-base:+.4f} 加性={r["additive_delta"]-base:+.4f} '
                f'无偏移={r["no_delta"]-base:+.4f} 全局={r["additive_global"]-base:+.4f} '
                f'随机={r["additive_random_mean"]-base:+.4f} '
                + (f'源域u={r["src_offset"]-base:+.4f} ' if u_src is not None else '')
                + f'反向={r["additive_anti"]-base:+.4f}')

    json.dump(results, open(OUT, 'w'), indent=2, ensure_ascii=False)

    log('\n' + '=' * 80)
    log('汇总: 相对 baseline 的 AP50 增益')
    log('=' * 80)
    keys = ['acspc_extrap', 'additive_delta', 'no_delta', 'additive_global',
            'additive_random_mean', 'offset_only', 'src_offset', 'src_offset_only', 'additive_anti']
    for tname in args.tasks:
        rows = {k: v for k, v in results[tname].items() if '->' in k}
        if not rows:
            continue
        log(f'\n[{tname}]')
        log(f'  {"变体":<22s} ' + '  '.join(f'{k:>16s}' for k in rows) + f' {"均值":>9s}')
        for k in keys:
            vals = [rows[rk][k] - rows[rk]['baseline'] for rk in rows if k in rows[rk]]
            if not vals:
                continue
            log(f'  {k:<22s} ' + '  '.join(f'{v:+16.4f}' for v in vals) + f' {np.mean(vals):+9.4f}')
    log(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
