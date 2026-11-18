# -*- coding: utf-8 -*-
"""run_bidirectional.py — 双向大样本主实验 (GPU)

目的: 用足够大的评测样本判别"A-CSPC 外推方向"的真实机制, 并给出可写进论文的主表。

设计要点
--------
1) 双向对称核心: 同一组 3 个类别对, 前向 DIOR→COCO 与反向 COCO→DIOR 各跑一遍。
   另加前向 6 类(与汇报稿主结果口径对齐, 便于对照 +0.0646)。
2) 方向对照:
     +δ = E('a photo of c') − E('an overhead image of c')   朝规范/自然参考系
     −δ = 反向                                              朝 RS/overhead
     β=0  无外推
     随机方向 ×3 (与 δ 同模长)
3) 结构对照(检验"全局偏移"假说):
     per-class δ_c  /  global δ̄  /  加性(不归一化)  /  只加偏移不要原型
4) 源域估计偏移 u: 在【源域】候选框上拟合 u = unit(mean F[TP] − mean F[FP]),
   完全不接触目标域 → 检验能否取代依赖目标域 GT 调参的 β。
5) crop 级诊断: 截断占比、AUC(F·δ→TP)、增益 vs 基线 AP 的关系。

用法 (GPU):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cuda \
  "C:/Users/张三/vg/Scripts/python.exe" -u run_bidirectional.py --n-eval 100
"""
import os, sys, json, pickle, time
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
from acspc import models as M
from acspc import metrics as Met
from acspc import prototype as Proto

CACHE_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_bidirectional.json')

BETA, LAM, NEG_TOPK, PRE_FILTER = 2.0, 0.5, 3, 0.03
N_PROTO = 20
N_RAND = 3

COCO_POOL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
DIOR_POOL = ['airplane', 'ship', 'vehicle', 'harbor', 'storage tank', 'bridge']

# (DIOR 类, COCO 类)
CORE = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]
EXTRA = [('vehicle', 'truck'), ('vehicle', 'bus'), ('vehicle', 'motorcycle')]

TASKS = {
    'fwd_dior2coco': ('dior', 'coco', CORE + EXTRA),
    'rev_coco2dior': ('coco', 'dior', [(c, d) for d, c in CORE]),
}

_CLIP = {}
CLIP_ID = 'openai/clip-vit-base-patch32'
DEVICE = os.environ.get('AC_SPC_DET_DEVICE', 'cuda')


def log(*a):
    print(*a, flush=True)


def get_clip():
    if 'model' not in _CLIP:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP['model'] = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE).eval()
        _CLIP['proc'] = CLIPProcessor.from_pretrained(CLIP_ID)
    return _CLIP['model'], _CLIP['proc']


def _as_vec(x):
    """transformers 4.x 返回 tensor, 5.x 可能返回 BaseModelOutputWithPooling。"""
    if hasattr(x, 'pooler_output') and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, 'image_embeds') and x.image_embeds is not None:
        return x.image_embeds
    if hasattr(x, 'text_embeds') and x.text_embeds is not None:
        return x.text_embeds
    if hasattr(x, 'last_hidden_state'):
        return x.last_hidden_state
    return x


@torch.no_grad()
def clip_encode_imgs(crops, bs=128):
    model, proc = get_clip()
    out = []
    for i in range(0, len(crops), bs):
        inputs = proc(images=crops[i:i + bs], return_tensors='pt', padding=True).to(DEVICE)
        f = _as_vec(model.get_image_features(**inputs)).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 512))


@torch.no_grad()
def text_embs(ts):
    model, proc = get_clip()
    out = []
    for i in range(0, len(ts), 64):
        inputs = proc(text=ts[i:i + 64], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
        f = _as_vec(model.get_text_features(**inputs)).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out).numpy()


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def iou_mat(b, g):
    if len(b) == 0 or len(g) == 0:
        return np.zeros((len(b), len(g)))
    ix1 = np.maximum(b[:, None, 0], g[None, :, 0]); iy1 = np.maximum(b[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(b[:, None, 2], g[None, :, 2]); iy2 = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
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


def load_split(cfg, domain, cls, offset, n):
    if domain == 'coco':
        ims, gts = D.load_coco(cfg, cls, n=offset + n)
    else:
        ims, gts = D.load_dior(cfg, cls, n=offset + n)
    ims, gts = ims[offset:offset + n], gts[offset:offset + n]
    keep = [(i, g) for i, g in zip(ims, gts) if len(g) > 0]
    if not keep:
        return [], []
    return [k[0] for k in keep], [k[1] for k in keep]


def detect_encode(cfg, images, gts, prompt, key):
    cp = CACHE_DIR / f'{key}.pkl'
    if cp.exists():
        log(f'    [cache] {cp.name}')
        return pickle.load(open(cp, 'rb'))
    t0 = time.time()
    log(f'    GDINO {len(images)} imgs prompt="{prompt}" ...')
    dets = M.gdino_detect(cfg, images, prompt, cfg.model.detection_threshold)
    per_img, crops_all = [], []
    for (boxes, scores), img, g in zip(dets, images, gts):
        bl, sl, ci = [], [], []
        for box, sc in zip(boxes, scores):
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 + 10 or y2 <= y1 + 10:
                continue
            if float(sc) < PRE_FILTER:
                continue
            bl.append([x1, y1, x2, y2]); sl.append(float(sc)); ci.append(len(crops_all))
            crops_all.append(img.crop((x1, y1, x2, y2)))
        per_img.append({
            'box': np.array(bl, dtype=np.float64) if bl else np.zeros((0, 4)),
            'gd': np.clip(np.array(sl, dtype=np.float64), 1e-6, 1.0) if sl else np.zeros(0),
            'gt': np.array(g, dtype=np.float64) if g else np.zeros((0, 4)),
            'crop_idx': np.array(ci, dtype=np.int64) if ci else np.zeros(0, dtype=np.int64),
        })
    F = clip_encode_imgs(crops_all)
    log(f'    boxes={sum(len(p["gd"]) for p in per_img)}  ({time.time()-t0:.0f}s)')
    blob = {'per_img': per_img, 'F': F}
    pickle.dump(blob, open(cp, 'wb'))
    return blob


def ap_of(per_img, F, W, collect=None):
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            if collect is not None:
                collect.append(w)
            p['_s'] = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            p['_s'] = np.zeros(0)
    return Met.compute_ap50_img([(p['box'], p['_s']) for p in per_img], [p['gt'] for p in per_img])


def ap_base(per_img, F):
    for p in per_img:
        p['_s'] = p['gd'].copy()
    return Met.compute_ap50_img([(p['box'], p['_s']) for p in per_img], [p['gt'] for p in per_img])


def crop_tp(per_img):
    tp, idx = [], []
    for p in per_img:
        if len(p['gd']) == 0:
            continue
        m = iou_mat(p['box'], p['gt'])
        tp.append(((m.max(1) if m.size else np.zeros(len(p['gd']))) >= 0.5).astype(int))
        idx.append(p['crop_idx'])
    return np.concatenate(tp), np.concatenate(idx)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--n-eval', type=int, default=100)
    ap_.add_argument('--tasks', nargs='+', default=list(TASKS.keys()))
    args = ap_.parse_args()

    cfg = Config()
    cfg.model.device = DEVICE
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    log('=' * 84)
    log(f'双向大样本主实验   device={DEVICE}  N_PROTO={N_PROTO}  N_EVAL={args.n_eval}')
    log(f'  BETA={BETA} LAM={LAM} neg_topk={NEG_TOPK} pre_filter={PRE_FILTER}')
    log('=' * 84)

    # ── 源原型池 ──
    bank = {}
    for domain, pool in (('coco', COCO_POOL), ('dior', DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = load_split(cfg, domain, cls, 0, N_PROTO)
            crops = D.gt_crops(ims, gts)
            if not crops:
                log(f'  !! {domain}:{cls} 无 crops')
                continue
            Ft = clip_encode_imgs(crops)
            pt = torch.from_numpy(Ft.mean(0))
            b[cls] = pt / pt.norm()          # 存 torch, 供 Proto.get_negative_prototype 使用
        bank[domain] = b
        log(f'[protos] {domain}: {len(b)} 类')

    # ── δ ──
    all_names = sorted(set(COCO_POOL + DIOR_POOL))
    E_photo = text_embs([f'a photo of {c}' for c in all_names])
    E_over = text_embs([f'an overhead image of {c}' for c in all_names])
    E = {c: (E_photo[i], E_over[i]) for i, c in enumerate(all_names)}
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}
    delta_c = {c: E[c][0] - E[c][1] for c in all_names}
    delta_bar = np.mean([delta_c[c] for c in all_names], axis=0)
    nrm = float(np.mean([np.linalg.norm(delta_c[c]) for c in all_names]))
    log(f'[delta] ||δ_c||均值={nrm:.4f}  ||δ̄||={np.linalg.norm(delta_bar):.4f}')
    rng = np.random.default_rng(0)
    rands = [unit(rng.normal(size=512)) * nrm for _ in range(N_RAND)]

    # ── 第一阶段: 建所有候选框缓存 ──
    blobs = {}
    for tname in args.tasks:
        src_dom, tgt_dom, pairs = TASKS[tname]
        for src_cls, eval_cls in pairs:
            images, gts = load_split(cfg, tgt_dom, eval_cls, N_PROTO, args.n_eval)
            if len(images) < 8:
                log(f'  !! {tname}/{eval_cls} 图像不足 {len(images)}')
                continue
            prompt = (f'a photo of {eval_cls}' if tgt_dom == 'coco'
                      else f'an overhead image of {eval_cls}')
            key = f'{tname}_{eval_cls}_off{N_PROTO}_n{len(images)}'
            log(f'  [{tname}] {eval_cls} ← {src_dom}:{src_cls}  N={len(images)}')
            blobs[(tname, eval_cls)] = (src_cls, detect_encode(cfg, images, gts, prompt, key))

    # ── 第二阶段: 源域拟合偏移 u ──
    # fwd 源=DIOR → 用 rev 的 DIOR 评测候选;  rev 源=COCO → 用 fwd 的 COCO 评测候选。
    # 两者对自己而言都在源域侧, 不涉及本任务的目标域。
    fit_src = {'fwd_dior2coco': ('rev_coco2dior', 'dior'), 'rev_coco2dior': ('fwd_dior2coco', 'coco')}
    u_bank = {}
    for tname, (other, dom) in fit_src.items():
        if tname not in args.tasks:
            continue
        Fs, ts = [], []
        for (tn, ec), (sc, blob) in blobs.items():
            if tn != other:
                continue
            tp, idx = crop_tp(blob['per_img'])
            Fs.append(blob['F'][idx]); ts.append(tp)
        if not Fs:
            continue
        Ff, tf = np.concatenate(Fs), np.concatenate(ts)
        if tf.sum() > 10 and (1 - tf).sum() > 10:
            u = unit(Ff[tf == 1].mean(0) - Ff[tf == 0].mean(0))
            u_bank[tname] = u
            log(f'[u] {tname}: 源域={dom} n={len(tf)} TP={int(tf.sum())} '
                f'cos(u,δ̄)={float(u @ unit(delta_bar)):+.4f}')

    # ── 第三阶段: 评估 ──
    results = {}
    for tname in args.tasks:
        src_dom, tgt_dom, pairs = TASKS[tname]
        log(f'\n{"=" * 84}\nTASK {tname}  ({src_dom} → {tgt_dom})')
        tres = {}
        for src_cls, eval_cls in pairs:
            if (tname, eval_cls) not in blobs:
                continue
            sc, blob = blobs[(tname, eval_cls)]
            per_img, F = blob['per_img'], blob['F']
            if sc not in bank[src_dom]:
                continue
            P_src_t = bank[src_dom][sc]
            P_neg_t = Proto.get_negative_prototype(bank[src_dom], sc, NEG_TOPK)
            P_src = unit(P_src_t.numpy())
            P_neg = unit(P_neg_t.numpy())
            # CA-PC 规范锚定 (A): photo(目标类) - overhead(源类)，模板角色固定、不随方向翻转
            dc_anchor = E_photo_d[eval_cls] - E_over_d[sc]
            # 角色交换 (B): photo(源类) - overhead(目标类)，现有方法按源/目标角色分配模板
            dc_roleswap = E_photo_d[sc] - E_over_d[eval_cls]
            base = ap_base(per_img, F)
            r = {'baseline': round(base, 4), 'n_img': len(per_img),
                 'n_box': int(sum(len(p['gd']) for p in per_img))}

            Vs = {
                'plus_delta': unit(P_src + BETA * dc_anchor) - LAM * P_neg,
                'minus_delta': unit(P_src + BETA * dc_roleswap) - LAM * P_neg,
                'beta0': P_src - LAM * P_neg,
                'global_delta': unit(P_src + BETA * delta_bar) - LAM * P_neg,
                'additive': P_src - LAM * P_neg + BETA * dc_anchor,
                'offset_only': BETA * dc_anchor,
                'pos_only': unit(P_src + BETA * dc_anchor),
            }
            for i, u in enumerate(rands):
                Vs[f'rand{i}'] = unit(P_src + BETA * u) - LAM * P_neg
            if tname in u_bank:
                u = u_bank[tname]
                Vs['src_offset'] = P_src - LAM * P_neg + BETA * nrm * u
                Vs['src_offset_only'] = BETA * nrm * u

            for k, W in Vs.items():
                buf = []
                r[k] = round(ap_of(per_img, F, W, collect=buf), 4)
                w = np.concatenate(buf)
                r[k + '_clip'] = round(float((w <= 0).mean()), 3)

            tp, idx = crop_tp(per_img)
            gd_all = np.concatenate([p['gd'] for p in per_img if len(p['gd'])])
            r['n_tp'] = int(tp.sum())
            r['auc_Fdelta_TP'] = round(roc_auc(tp, F[idx] @ delta_bar), 4)
            r['auc_gd_TP'] = round(roc_auc(tp, gd_all), 4)
            tres[f'{sc}->{eval_cls}'] = r
            log(f'  {sc}->{eval_cls}: base={base:.4f} | '
                + ' '.join(f'{k}={r[k]-base:+.4f}' for k in
                           ['plus_delta', 'minus_delta', 'beta0', 'global_delta', 'additive'])
                + (f' srcU={r["src_offset"]-base:+.4f}' if 'src_offset' in r else '')
                + f' | rand={np.mean([r[f"rand{i}"]-base for i in range(N_RAND)]):+.4f}'
                + f' | AUC(F·δ→TP)={r["auc_Fdelta_TP"]:.3f}')
        results[tname] = tres

    json.dump({'config': {'device': DEVICE, 'beta': BETA, 'lam': LAM, 'n_proto': N_PROTO,
                          'n_eval': args.n_eval, 'pre_filter': PRE_FILTER,
                          'detector': 'gdino-tiny', 'clip': CLIP_ID},
               'results': results},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    log('\n' + '=' * 84)
    log('主表: 相对 baseline 的 AP50 增益')
    log('=' * 84)
    keys = ['plus_delta', 'global_delta', 'additive', 'minus_delta', 'beta0',
            'offset_only', 'pos_only', 'rand0', 'rand1', 'rand2', 'src_offset', 'src_offset_only']
    for tname, tres in results.items():
        if not tres:
            continue
        log(f'\n[{tname}]  n_class={len(tres)}')
        log(f'  {"变体":<16s}' + ''.join(f'{k.split("->")[-1]:>12s}' for k in tres) + f'{"均值":>10s}')
        for k in keys:
            if k not in next(iter(tres.values())):
                continue
            g = [tres[rk][k] - tres[rk]['baseline'] for rk in tres]
            log(f'  {k:<16s}' + ''.join(f'{v:+12.4f}' for v in g) + f'{np.mean(g):+10.4f}')
        log(f'  {"baseline":<16s}' + ''.join(f'{tres[rk]["baseline"]:>12.4f}' for rk in tres)
            + f'{np.mean([tres[rk]["baseline"] for rk in tres]):>10.4f}')
    log(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
