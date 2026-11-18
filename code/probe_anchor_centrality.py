# -*- coding: utf-8 -*-
"""probe_anchor_centrality.py — P2 锚点模板扩展: 验证"中心性越高的模板越好"是否单调

中心性定义: 候选目标类提示词文本嵌入 与 规范锚点 'a photo of {cls}' 文本嵌入的余弦相似度。
越高 = 越接近规范/原型提示(越"中心")。
对 fwd(overhead→photo) 与 rev(photo→overhead) 两方向, 在每个单元上用候选提示作 CA-PC 正端
(β=2.0 固定, 负端固定 'an overhead image of {src}'), 算 AP 增益; 然后检验
增益 是否随 中心性 单调递增 (Spearman / Kendall τ / 逆序数)。
"""
import os, sys, json, pickle
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, kendalltau
import torch
from transformers import CLIPModel, CLIPProcessor
from acspc import metrics as Met

BIDIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache')
PROTOS = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_anchor_centrality.json')
LAM, N_PROTO, BETA = 0.5, 20, 2.0
CLIP_ID = 'openai/clip-vit-base-patch32'

CORE = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]
EXTRA = [('vehicle', 'truck'), ('vehicle', 'bus'), ('vehicle', 'motorcycle')]
JOBS = [('fwd_dior2coco', CORE + EXTRA, 'dior'), ('rev_coco2dior', [(c, d) for d, c in CORE], 'coco')]
HYP = {'airplane': 'aircraft', 'ship': 'watercraft', 'vehicle': 'vehicle', 'boat': 'watercraft',
       'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'motorcycle': 'vehicle'}

CAND = [
    ('canonical', 'a photo of {}'),
    ('natural', 'a natural image of {}'),
    ('picture', 'a picture of {}'),
    ('image', 'an image of {}'),
    ('plain', '{}'),
    ('object', 'a {} object'),
    ('hyper', 'a {}'),          # filled with hypernym
    ('painting', 'a painting of {}'),
    ('sketch', 'a sketch of {}'),
    ('satellite', 'a satellite image of {}'),
]


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

    res = {'per_unit': [], 'pooled': {}}
    all_c, all_g = [], []
    for tname, pairs, dom in JOBS:
        bank = bank_all.get(dom, {})
        for src_cls, tgt_cls in pairs:
            b = open_cache(BIDIR, tname, tgt_cls, [100, 77])
            if b is None or src_cls not in bank:
                continue
            per_img, F = b['per_img'], b['F']
            P_src, P_neg = bank[src_cls], neg_of(bank, src_cls)
            base = ap_base(per_img, F)
            over = T([f'an overhead image of {src_cls}'])[0]
            canon = T([f'a photo of {tgt_cls}'])[0]
            prompts = []
            for name, tpl in CAND:
                txt = tpl.format(HYP.get(tgt_cls, tgt_cls)) if name == 'hyper' else tpl.format(tgt_cls)
                prompts.append((name, txt))
            embs = {name: T([txt])[0] for name, txt in prompts}
            cen = {name: float(embs[name] @ canon) for name in embs}
            row = {'dir': tname, 'src': src_cls, 'tgt': tgt_cls, 'base': round(base, 4)}
            for name, txt in prompts:
                d = embs[name] - over
                g = ap_of(per_img, F, unit(P_src + BETA * d) - LAM * P_neg) - base
                row[name] = round(float(g), 4)
                row[name + '_cen'] = round(cen[name], 4)
                all_c.append(cen[name])
                all_g.append(float(g))
            res['per_unit'].append(row)

    all_c = np.array(all_c)
    all_g = np.array(all_g)
    rho_s, _ = spearmanr(all_c, all_g)
    tau, _ = kendalltau(all_c, all_g)
    order = np.argsort(all_c)
    gs = all_g[order]
    inv = int(sum(1 for i in range(len(gs)) for j in range(i + 1, len(gs)) if gs[i] > gs[j] + 1e-9))
    res['pooled'] = {'n': int(len(all_c)), 'spearman_rho': round(float(rho_s), 4),
                     'kendall_tau': round(float(tau), 4), 'inversions': inv,
                     'monotonic_nondec': bool(inv == 0)}
    json.dump(res, open(OUT, 'w'), indent=2, ensure_ascii=False)
    print(f'[centrality] n={len(all_c)} spearman={rho_s:.4f} kendall={tau:.4f} inversions={inv} mono={inv == 0}')
    print(f'  saved {OUT}')


if __name__ == '__main__':
    main()
