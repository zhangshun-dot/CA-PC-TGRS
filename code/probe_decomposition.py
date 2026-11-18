# -*- coding: utf-8 -*-
"""probe_decomposition.py — 把 δ 拆成【类迁移项】与【模板项】, 定位方向不对称的承担者

δ_ACSPC = E('a photo of {tgt}') − E('an overhead image of {src}')
        = [E_photo(tgt) − E_photo(src)]          ← 类迁移项 (源粗类 → 目标细类)
        + [E_photo(src) − E_overhead(src)]       ← 模板项   (朝 CLIP 规范参考系)

已确证(probe_delta_structure): 模板项 93.2% 是类别无关的全局方向。
本脚本检验:
  Q1 类迁移项与模板项各自贡献多少?
  Q2 方向不对称(+δ 有效 / −δ 失效)由哪一项承担?
  Q3 模板项能否用【全局 δ̄】替代(→ 不需要逐类文本 delta)?
  Q4 组合 "类迁移项 + 固定规范方向 δ̄" 能否在双向都等价于完整 δ?
     若成立 → 方法不再需要知道目标域的成像风格(只需目标类名), 且双向通用。

全部复用 n=100 候选缓存, 不跑检测。
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

CACHE = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache')
PROTOS = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_decomposition.json')

BETA, LAM, NEG_TOPK, N_PROTO = 2.0, 0.5, 3, 20
CLIP_ID = 'openai/clip-vit-base-patch32'

CORE = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]
EXTRA = [('vehicle', 'truck'), ('vehicle', 'bus'), ('vehicle', 'motorcycle')]
# (任务, [(源类, 目标类)], 原型域)
JOBS = [('fwd_dior2coco', CORE + EXTRA, 'dior'), ('rev_coco2dior', [(c, d) for d, c in CORE], 'coco')]

VOCAB = ['airplane', 'ship', 'vehicle', 'boat', 'car', 'truck', 'bus', 'motorcycle',
         'harbor', 'storage tank', 'bridge', 'train station', 'airport', 'stadium']


def _as_vec(x):
    for a in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
        if hasattr(x, a) and getattr(x, a) is not None:
            return getattr(x, a)
    return x


def log(*a):
    print(*a, flush=True)


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def neg_of(bank, cls):
    sub = {k: v for k, v in bank.items() if k != cls}
    if not sub:
        return np.zeros(512)
    ks = list(sub); P = np.stack([sub[k] for k in ks])
    o = np.argsort(-(P @ bank[cls]))[:NEG_TOPK]
    return unit(P[o].mean(0))


def main():
    m = CLIPModel.from_pretrained(CLIP_ID).eval()
    p = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def T(ts):
        out = []
        for i in range(0, len(ts), 64):
            f = _as_vec(m.get_text_features(
                **p(text=ts[i:i + 64], return_tensors='pt', padding=True, truncation=True))).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(out).numpy()

    photo = {c: T([f'a photo of {c}'])[0] for c in set(VOCAB)}
    over = {c: T([f'an overhead image of {c}'])[0] for c in set(VOCAB)}
    dbar = np.mean([photo[c] - over[c] for c in VOCAB], axis=0)
    nbar = float(np.linalg.norm(dbar))
    log(f'||δ̄||={nbar:.4f}')

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

    rng = np.random.default_rng(0)
    res = {}
    for tname, pairs, dom in JOBS:
        bank = bank_all.get(dom, {})
        rows = {}
        for src_cls, tgt_cls in pairs:
            fp = None
            for n in (100, 77):
                c = CACHE / f'{tname}_{tgt_cls}_off{N_PROTO}_n{n}.pkl'
                if c.exists():
                    fp = c; break
            if fp is None or src_cls not in bank:
                continue
            b = pickle.load(open(fp, 'rb'))
            per_img, F = b['per_img'], b['F']
            P_src, P_neg = bank[src_cls], neg_of(bank, src_cls)
            base = ap_base(per_img, F)

            d_cls = photo[tgt_cls] - photo[src_cls]        # 类迁移项
            d_tm_src = photo[src_cls] - over[src_cls]      # 模板项(源类)
            d_tm_tgt = photo[tgt_cls] - over[tgt_cls]      # 模板项(目标类)
            d_full = photo[tgt_cls] - over[src_cls]        # A-CSPC 原式 (规范锚定, 朝 B)
            d_roleswap = photo[src_cls] - over[tgt_cls]     # 按角色指派: 源=photo 目标=overhead

            Vs = {
                'full_ACSPC': d_full,
                'class_only': d_cls,
                'tmpl_src': d_tm_src,
                'tmpl_tgt': d_tm_tgt,
                'tmpl_bar': dbar,
                'class_plus_bar': d_cls + dbar,
                'class_minus_bar': d_cls - dbar,
                'neg_full': -d_full,
                'roleswap': d_roleswap,
            }
            rr = [unit(rng.normal(size=512)) * np.linalg.norm(d_full) for _ in range(3)]
            for i, u in enumerate(rr):
                Vs[f'rand{i}'] = u

            row = {'baseline': round(base, 4)}
            for k, d in Vs.items():
                row[k] = round(ap_of(per_img, F, unit(P_src + BETA * d) - LAM * P_neg) - base, 4)
            row['beta0'] = round(ap_of(per_img, F, P_src - LAM * P_neg) - base, 4)
            row['rand_mean'] = round(float(np.mean([row[f'rand{i}'] for i in range(3)])), 4)
            rows[f'{src_cls}->{tgt_cls}'] = row
        res[tname] = rows

        log(f'\n[{tname}]')
        keys = ['full_ACSPC', 'class_only', 'tmpl_src', 'tmpl_tgt', 'tmpl_bar',
                'class_plus_bar', 'class_minus_bar', 'neg_full', 'roleswap', 'beta0', 'rand_mean']
        log('  ' + ' '.join(f'{k.split("->")[-1]:>12s}' for k in rows))
        for k in keys:
            log(f'  {k:<16s}' + ' '.join(f'{rows[r][k]:+12.4f}' for r in rows)
                + f'  |均值 {np.mean([rows[r][k] for r in rows]):+.4f}')

    json.dump(res, open(OUT, 'w'), indent=2, ensure_ascii=False)

    log('\n' + '=' * 92)
    log('分解汇总 (均值增益)')
    log('=' * 92)
    keys = ['full_ACSPC', 'class_only', 'tmpl_src', 'tmpl_bar', 'class_plus_bar',
            'class_minus_bar', 'neg_full', 'beta0', 'rand_mean']
    log(f'  {"分量":<18s}' + ''.join(f'{t:>22s}' for t in res))
    for k in keys:
        log(f'  {k:<18s}' + ''.join(f'{np.mean([r[k] for r in res[t].values()]):+22.4f}' for t in res))
    log(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
