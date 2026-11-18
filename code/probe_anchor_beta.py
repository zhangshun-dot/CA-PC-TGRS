# -*- coding: utf-8 -*-
"""probe_anchor_beta.py — 锚点模板 ablation + β 源域可选性 (N=100, 复用缓存)

Q1 锚点 ablation: "朝规范方向"到底是"朝 'a photo of' 这一个模板", 还是"朝任何非 RS 模板"?
    固定负端为 'an overhead image of {src}', 正端换成不同锚点模板:
      photo / natural / plain(裸类名) / painting / sketch / satellite(同为 RS 类)
    若只有 canonical 锚点(photo/natural)有效而 painting/sketch 无效 → "规范性"假说成立;
    若都有效 → 只是"远离 RS 措辞"而已。

Q2 β 源域可选性: 在【源域同域任务】上扫 β 选最优, 套到跨域任务, 与目标域 oracle 对比。
    (同域 N=30 缓存用于选 β; 跨域 N=100 缓存用于评测)
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
SAME = Path(r'C:\Users\张三\Desktop\实验数据\probe_dirmech_cache')
PROTOS = Path(r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_anchor_beta.json')

LAM, NEG_TOPK, N_PROTO = 0.5, 3, 20
CLIP_ID = 'openai/clip-vit-base-patch32'

ANCHORS = {
    'photo': 'a photo of {}',
    'natural': 'a natural image of {}',
    'plain': '{}',
    'painting': 'a painting of {}',
    'sketch': 'a sketch of {}',
    'satellite': 'a satellite image of {}',
}
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

CORE = [('airplane', 'airplane'), ('ship', 'boat'), ('vehicle', 'car')]
EXTRA = [('vehicle', 'truck'), ('vehicle', 'bus'), ('vehicle', 'motorcycle')]
JOBS = [('fwd_dior2coco', CORE + EXTRA, 'dior'), ('rev_coco2dior', [(c, d) for d, c in CORE], 'coco')]
# 同域任务(N=30)用于选 β:  (同域任务名, [(源类, 评测类)])
SAME_JOBS = {'fwd_dior2coco': ('same_dior2dior', [('airplane', 'airplane'), ('ship', 'ship'), ('vehicle', 'vehicle')]),
             'rev_coco2dior': ('same_coco2coco', [('airplane', 'airplane'), ('boat', 'boat'), ('car', 'car')])}


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

    res = {'anchor': {}, 'beta': {}}

    # ───────── Q1 锚点 ablation ─────────
    log('=' * 90); log('Q1 锚点模板 ablation (β=2.0)'); log('=' * 90)
    for tname, pairs, dom in JOBS:
        bank = bank_all.get(dom, {})
        rows = {}
        for src_cls, tgt_cls in pairs:
            b = open_cache(BIDIR, tname, tgt_cls, [100, 77])
            if b is None or src_cls not in bank:
                continue
            per_img, F = b['per_img'], b['F']
            P_src, P_neg = bank[src_cls], neg_of(bank, src_cls)
            base = ap_base(per_img, F)
            over_src = T([f'an overhead image of {src_cls}'])[0]
            row = {'baseline': round(base, 4)}
            for an, tpl in ANCHORS.items():
                d = T([tpl.format(tgt_cls)])[0] - over_src      # 与 A-CSPC 同口径: 正端用目标类
                row[an] = round(ap_of(per_img, F, unit(P_src + 2.0 * d) - LAM * P_neg) - base, 4)
                row[an + '_flip'] = round(ap_of(per_img, F, unit(P_src - 2.0 * d) - LAM * P_neg) - base, 4)
            rows[f'{src_cls}->{tgt_cls}'] = row
        res['anchor'][tname] = rows
        log(f'\n[{tname}]')
        log('  ' + ' '.join(f'{k.split("->")[-1]:>11s}' for k in rows) + f'{"均值":>10s}')
        for an in ANCHORS:
            v = [rows[r][an] for r in rows]
            vf = [rows[r][an + '_flip'] for r in rows]
            log(f'  {an:<11s}' + ' '.join(f'{x:+11.4f}' for x in v)
                + f'{np.mean(v):+10.4f}   | 翻向 {np.mean(vf):+.4f}')

    # ───────── Q2 β 源域可选性 ─────────
    log('\n' + '=' * 90); log('Q2 β 扫描 (跨域 N=100) 与源域选 β'); log('=' * 90)
    for tname, pairs, dom in JOBS:
        bank = bank_all.get(dom, {})
        rows = {}
        for src_cls, tgt_cls in pairs:
            b = open_cache(BIDIR, tname, tgt_cls, [100, 77])
            if b is None or src_cls not in bank:
                continue
            per_img, F = b['per_img'], b['F']
            P_src, P_neg = bank[src_cls], neg_of(bank, src_cls)
            base = ap_base(per_img, F)
            d = T([f'a photo of {tgt_cls}'])[0] - T([f'an overhead image of {src_cls}'])[0]
            row = {'baseline': round(base, 4)}
            for beta in BETAS:
                if beta == 0:
                    W = P_src - LAM * P_neg
                else:
                    W = unit(P_src + beta * d) - LAM * P_neg
                row[f'plus_b{beta}'] = round(ap_of(per_img, F, W) - base, 4)
                if beta > 0:
                    row[f'minus_b{beta}'] = round(
                        ap_of(per_img, F, unit(P_src - beta * d) - LAM * P_neg) - base, 4)
            rows[f'{src_cls}->{tgt_cls}'] = row
        res['beta'][tname] = rows
        log(f'\n[{tname}]  β: +δ / −δ 均值')
        for beta in BETAS:
            gp = [rows[r][f'plus_b{beta}'] for r in rows]
            gm = ([rows[r][f'minus_b{beta}'] for r in rows] if beta > 0 else gp)
            log(f'  β={beta:<4.1f}  {np.mean(gp):+.4f}   {np.mean(gm):+.4f}')

    # 源域(同域 N=30)选 β, 套到跨域 N=100
    sel = {}
    for tname, (same_task, same_pairs) in SAME_JOBS.items():
        bank = bank_all.get('dior' if tname == 'fwd_dior2coco' else 'coco', {})
        srows = {}
        for src_cls, eval_cls in same_pairs:
            b = open_cache(SAME, same_task, eval_cls, [30])
            if b is None or src_cls not in bank:
                continue
            per_img, F = b['per_img'], b['F']
            P_src, P_neg = bank[src_cls], neg_of(bank, src_cls)
            base = ap_base(per_img, F)
            d = T([f'a photo of {eval_cls}'])[0] - T([f'an overhead image of {src_cls}'])[0]
            r = {}
            for beta in BETAS:
                if beta == 0:
                    W = P_src - LAM * P_neg
                else:
                    W = unit(P_src + beta * d) - LAM * P_neg
                r[beta] = ap_of(per_img, F, W) - base
            srows[eval_cls] = r
        if not srows:
            continue
        best_b, best_g = max(BETAS[1:], key=lambda b: np.mean([srows[k][b] for k in srows])), -9
        best_g = float(np.mean([srows[k][best_b] for k in srows]))
        cross = res['beta'].get(tname, {})
        if cross:
            g_at_src = np.mean([cross[k][f'plus_b{best_b}'] for k in cross])
            ob = max(BETAS[1:], key=lambda b: np.mean([cross[k][f'plus_b{b}'] for k in cross]))
            g_or = np.mean([cross[k][f'plus_b{ob}'] for k in cross])
            sel[tname] = {'selected_on': same_task, 'beta_src': best_b, 'src_gain': round(best_g, 4),
                          'cross_at_src_beta': round(float(g_at_src), 4),
                          'oracle_beta': ob, 'cross_at_oracle': round(float(g_or), 4),
                          'cross_at_beta2': round(float(np.mean([cross[k]['plus_b2.0'] for k in cross])), 4)}
            log(f'\n  {tname}: 源域({same_task})选 β={best_b} (源域增益 {best_g:+.4f})')
            log(f'      套到跨域: β={best_b} → {g_at_src:+.4f} | '
                f'oracle β={ob} → {g_or:+.4f} | 固定 β=2 → {sel[tname]["cross_at_beta2"]:+.4f}')
    res['beta_selection'] = sel

    json.dump(res, open(OUT, 'w'), indent=2, ensure_ascii=False)
    log(f'\n已保存: {OUT}')


if __name__ == '__main__':
    main()
