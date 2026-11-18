# -*- coding: utf-8 -*-
"""probe_lambda_bidir.py — P2 λ 跨方向迁移: λ 也做双向扫描, 验证方向不变性

固定规范方向 (β=2.0, 正端 'a photo of {tgt}' − 负端 'an overhead image of {src}'),
扫描 λ ∈ [0, 0.25, 0.5, 0.75, 1.0, 1.5], 在 fwd(overhead→photo) 与 rev(photo→overhead) 两方向
分别求 AP 增益曲线, 找各自最优 λ*, 检验 λ*_fwd ≈ λ*_rev (方向不变), 并看固定 λ=0.5 是否近优。
"""
import os, sys, json, pickle
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from acspc import metrics as Met

BIDIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache')
PROTOS = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_lambda_bidir.json')
N_PROTO, BETA = 20, 2.0
CLIP_ID = 'openai/clip-vit-base-patch32'
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

CORE = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]
EXTRA = [('vehicle', 'truck'), ('vehicle', 'bus'), ('vehicle', 'motorcycle')]
JOBS = [('fwd_dior2coco', CORE + EXTRA, 'dior'), ('rev_coco2dior', [(c, d) for d, c in CORE], 'coco')]


def _as_vec(x):
    for a in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
        if hasattr(x, a) and getattr(x, a) is not None:
            return getattr(x, a)
    return x


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def neg_of(bank, cls):
    sub = {k: v for k, v in bank.items() if k != cls}
    if not sub:
        return np.zeros(512)
    ks = list(sub)
    P = np.stack([sub[k] for k in ks])
    o = np.argsort(-(P @ bank[cls]))[:3]
    return unit(P[o].mean(0))


def main():
    m = CLIPModel.from_pretrained(CLIP_ID).eval()
    pr = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def T(ts):
        out = []
        for i in range(0, len(ts), 64):
            f = _as_vec(m.get_text_features(
                **pr(text=ts[i:i + 64], return_tensors='pt', padding=True, truncation=True))).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(out).numpy()

    bank_all = pickle.load(open(PROTOS, 'rb'))

    def ap_of(per_img, F, W):
        for pi in per_img:
            if len(pi['gd']):
                pi['_s'] = np.sqrt(pi['gd'] * np.clip(np.maximum(0.0, F[pi['crop_idx']] @ W), 1e-6, 1.0))
            else:
                pi['_s'] = np.zeros(0)
        return Met.compute_ap50_img([(x['box'], x['_s']) for x in per_img], [x['gt'] for x in per_img])

    def ap_base(per_img, F):
        for pi in per_img:
            pi['_s'] = pi['gd'].copy()
        return Met.compute_ap50_img([(x['box'], x['_s']) for x in per_img], [x['gt'] for x in per_img])

    def open_cache(base, tname, cls, ns):
        for n in ns:
            f = base / f'{tname}_{cls}_off{N_PROTO}_n{n}.pkl'
            if f.exists():
                return pickle.load(open(f, 'rb'))
        return None

    res = {'per_unit': [], 'curves': {}}
    SHORT = {'fwd_dior2coco': 'fwd', 'rev_coco2dior': 'rev'}
    curve = {'fwd': {l: [] for l in LAMBDAS}, 'rev': {l: [] for l in LAMBDAS}}
    for tname, pairs, dom in JOBS:
        sdir = SHORT[tname]
        bank = bank_all.get(dom, {})
        for src_cls, tgt_cls in pairs:
            b = open_cache(BIDIR, tname, tgt_cls, [100, 77])
            if b is None or src_cls not in bank:
                continue
            per_img, F = b['per_img'], b['F']
            P_src, P_neg = bank[src_cls], neg_of(bank, src_cls)
            base = ap_base(per_img, F)
            d = T([f'a photo of {tgt_cls}'])[0] - T([f'an overhead image of {src_cls}'])[0]
            row = {'dir': tname, 'src': src_cls, 'tgt': tgt_cls, 'base': round(base, 4)}
            for lam in LAMBDAS:
                W = unit(P_src + BETA * d) - lam * P_neg
                g = ap_of(per_img, F, W) - base
                row[f'lam{lam}'] = round(float(g), 4)
                curve[sdir][lam].append(float(g))
            res['per_unit'].append(row)

    curves_mean = {}
    for tname in ('fwd', 'rev'):
        means = {l: float(np.mean(curve[tname][l])) for l in LAMBDAS}
        best = max(LAMBDAS, key=lambda l: means[l])
        curves_mean[tname] = {'mean_gain': {str(l): round(means[l], 4) for l in LAMBDAS},
                              'best_lambda': best, 'best_gain': round(means[best], 4),
                              'gain_at_fixed05': round(means[0.5], 4)}
    res['curves'] = curves_mean
    json.dump(res, open(OUT, 'w'), indent=2, ensure_ascii=False)
    print('[lambda bidir]')
    for tname in ('fwd', 'rev'):
        cm = curves_mean[tname]
        print(f'  {tname}: best λ={cm["best_lambda"]} (gain {cm["best_gain"]:+.4f}), λ=0.5 → {cm["gain_at_fixed05"]:+.4f}')
        print('    ' + ' '.join(f'λ={l}:{cm["mean_gain"][str(l)]:+.4f}' for l in LAMBDAS))
    print(f'  saved {OUT}')


if __name__ == '__main__':
    main()
