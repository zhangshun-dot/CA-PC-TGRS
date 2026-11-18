# -*- coding: utf-8 -*-
"""probe_fidelity.py — 为什么我的前向增益 (+0.0136) 远低于汇报稿 (+0.0646)?

排查三个可能的实现差异(均可用 n=100 缓存直接重算, 不跑检测):
  D1 δ 的类别: 我用的是 photo(src_cls) − overhead(src_cls);
               汇报稿 prototype.compute_domain_shift 是 photo(tgt_cls) − overhead(src_cls)。
               对 airplane→airplane 相同, 对 ship→boat / vehicle→car 不同。
  D2 负原型池: 我只用了 6 个 DIOR 类; 汇报稿 DIOR 有 20 类, top-3 近邻应从 20 类里选。
  D3 两者叠加。

若 D1/D2 能把前向增益抬到接近 +0.0646, 则主表应改用与汇报稿一致的实现重报。
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
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from acspc.config import Config, DIOR_ID2NAME
from acspc import data as D
from acspc import metrics as Met
from acspc import prototype as Proto

CACHE = Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache')
PROTOS = Path(r'C:\Users\张三\Desktop\实验数据\probe_fidelity_protos.pkl')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_fidelity.json')

BETA, LAM, NEG_TOPK, N_PROTO = 2.0, 0.5, 3, 20
CLIP_ID = 'openai/clip-vit-base-patch32'
DIOR20 = sorted(DIOR_ID2NAME.values())
COCO6 = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']

CORE = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]
EXTRA = [('vehicle', 'truck'), ('vehicle', 'bus'), ('vehicle', 'motorcycle')]
FWD = CORE + EXTRA
REV = [(c, d) for d, c in CORE]


def _as_vec(x):
    for a in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
        if hasattr(x, a) and getattr(x, a) is not None:
            return getattr(x, a)
    return x


def log(*a):
    print(*a, flush=True)


def main():
    cfg = Config(); cfg.model.device = 'cpu'
    m = CLIPModel.from_pretrained(CLIP_ID).eval()
    p = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def text_embs(ts):
        out = []
        for i in range(0, len(ts), 64):
            f = _as_vec(m.get_text_features(
                **p(text=ts[i:i + 64], return_tensors='pt', padding=True, truncation=True))).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def clip_encode(crops, bs=64):
        out = []
        for i in range(0, len(crops), bs):
            f = _as_vec(m.get_image_features(
                **p(images=crops[i:i + bs], return_tensors='pt', padding=True))).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(out).numpy() if out else np.zeros((0, 512))

    def unit(x):
        n = np.linalg.norm(x)
        return x / n if n > 1e-12 else x

    # ── 原型: DIOR 20 类 + COCO 6 类 ──
    if PROTOS.exists():
        bank_np = pickle.load(open(PROTOS, 'rb'))
    else:
        bank_np = {}
        for dom, pool in (('dior', DIOR20), ('coco', COCO6)):
            b = {}
            for cls in pool:
                try:
                    ims, gts = (D.load_coco(cfg, cls, n=N_PROTO) if dom == 'coco'
                                else D.load_dior(cfg, cls, n=N_PROTO))
                except Exception:
                    continue
                crops = D.gt_crops(ims, gts)
                if not crops:
                    continue
                b[cls] = unit(clip_encode(crops).mean(0))
            bank_np[dom] = b
            log(f'[protos] {dom}: {len(b)} 类')
        pickle.dump(bank_np, open(PROTOS, 'wb'))

    def neg_of(bank, cls, pool):
        sub = {k: v for k, v in bank.items() if k in pool and k != cls}
        if not sub:
            return np.zeros(512)
        ks = list(sub); P = np.stack([sub[k] for k in ks])
        o = np.argsort(-(P @ bank[cls]))[:NEG_TOPK]
        return unit(P[o].mean(0))

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

    def run(tname, pairs, src_dom, delta_mode, neg_pool):
        rows = {}
        for src_cls, eval_cls in pairs:
            fp = None
            for n in (100, 77):
                c = CACHE / f'{tname}_{eval_cls}_off{N_PROTO}_n{n}.pkl'
                if c.exists():
                    fp = c
                    break
            if fp is None:
                continue
            b = pickle.load(open(fp, 'rb'))
            per_img, F = b['per_img'], b['F']
            bank = bank_np[src_dom]
            if src_cls not in bank:
                continue
            P_src = bank[src_cls]
            P_neg = neg_of(bank, src_cls, neg_pool)
            if delta_mode == 'srcsrc':
                d = text_embs([f'a photo of {src_cls}'])[0] - text_embs([f'an overhead image of {src_cls}'])[0]
            else:  # srctgt = 汇报稿口径
                d = text_embs([f'a photo of {eval_cls}'])[0] - text_embs([f'an overhead image of {src_cls}'])[0]
            base = ap_base(per_img, F)
            dn = np.linalg.norm(d)
            rr = [unit(rng.normal(size=512)) * dn for _ in range(3)]
            rows[f'{src_cls}->{eval_cls}'] = {
                'baseline': round(base, 4),
                'plus': round(ap_of(per_img, F, unit(P_src + BETA * d) - LAM * P_neg) - base, 4),
                'minus': round(ap_of(per_img, F, unit(P_src - BETA * d) - LAM * P_neg) - base, 4),
                'beta0': round(ap_of(per_img, F, P_src - LAM * P_neg) - base, 4),
                'rand': round(float(np.mean([ap_of(per_img, F, unit(P_src + BETA * u) - LAM * P_neg) - base
                                             for u in rr])), 4),
            }
        return rows

    dior6 = ['airplane', 'ship', 'vehicle', 'harbor', 'storage tank', 'bridge']
    combos = [
        ('A 我原实现 (δ=src/src, 负池6类)', 'srcsrc', dior6),
        ('B 汇报稿 δ 口径 (δ=tgt/src, 负池6类)', 'srctgt', dior6),
        ('C 负原型池扩到 DIOR20 (δ=src/src)', 'srcsrc', DIOR20),
        ('D 两者都用汇报稿口径 (δ=tgt/src, 负池20类)', 'srctgt', DIOR20),
    ]

    out = {}
    for label, dmode, npool in combos:
        log(f'\n### {label}')
        res = {}
        for tname, pairs, sdom in (('fwd_dior2coco', FWD, 'dior'), ('rev_coco2dior', REV, 'coco')):
            rows = run(tname, pairs, sdom, dmode, npool)
            if not rows:
                continue
            res[tname] = rows
            log(f'  [{tname}] ' + '  '.join(
                f'{k.split("->")[-1]}:+{v["plus"]:.4f}' for k, v in rows.items()))
            for kk in ['plus', 'minus', 'beta0', 'rand']:
                log(f'      {kk:<6s} ' + ' '.join(f'{v[kk]:+.4f}' for v in rows.values())
                    + f'  均值={np.mean([v[kk] for v in rows.values()]):+.4f}')
        out[label] = res

    json.dump(out, open(OUT, 'w'), indent=2, ensure_ascii=False)

    log('\n' + '=' * 90)
    log('前向 6 类均值对照 (汇报稿口径 +0.0646)')
    log('=' * 90)
    log(f'  {"配置":<38s} {"+δ":>9s} {"−δ":>9s} {"β=0":>9s} {"随机":>9s}')
    for label in out:
        if 'fwd_dior2coco' not in out[label]:
            continue
        r = out[label]['fwd_dior2coco']
        log(f'  {label:<38s} ' + ' '.join(
            f'{np.mean([v[k] for v in r.values()]):+9.4f}' for k in ['plus', 'minus', 'beta0', 'rand']))
    log('')
    log(f'  {"配置":<38s} {"+δ":>9s} {"−δ":>9s} {"β=0":>9s} {"随机":>9s}   [反向 3 类]')
    for label in out:
        if 'rev_coco2dior' not in out[label]:
            continue
        r = out[label]['rev_coco2dior']
        log(f'  {label:<38s} ' + ' '.join(
            f'{np.mean([v[k] for v in r.values()]):+9.4f}' for k in ['plus', 'minus', 'beta0', 'rand']))
    log(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
