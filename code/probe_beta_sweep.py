# -*- coding: utf-8 -*-
"""probe_beta_sweep.py — β 能否在【源域】选出? (纯缓存分析, 不跑检测)

汇报稿自陈边界: 「主配置依赖目标域GT调参（非zero-shot）」——β=2.0, λ=0.5 是
在目标域 GT 上选的。目标域无关配置(β=1.0,λ=0.25, 在 DIOR→VOC 开发域选)只有 +0.0166。

本脚本检验: 若只调 β 这一个标量, 能否在【源域同域任务】上选出来, 并迁移到跨域任务?
  - 对每个任务/类别扫 β ∈ {0,0.5,1,1.5,2,3,4} × {+δ, −δ}
  - 选出"源域同域任务"上平均增益最大的 β*(和符号), 套到跨域任务上报增益
  - 对照: 目标域上 oracle 选 β (上界)

同时给出 β 的鲁棒性曲线 —— 若增益对 β 很敏感, 则"依赖调参"是实质问题;
若不敏感(宽平台), 则边界可被弱化。
"""
import os, sys, json, pickle
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')

from pathlib import Path
import argparse
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from acspc import prototype as Proto
from acspc import metrics as Met

CLIP_ID = 'openai/clip-vit-base-patch32'
DEFAULT_CACHE = Path(r'C:\Users\张三\Desktop\实验数据\probe_dirmech_cache')
LAM = 0.5
NEG_TOPK = 3

# 注意: 这里第二项是【源原型域】(跨域任务的原型来自源域), 评测域名另见 EVAL_DOM
TASKS = {
    'same_dior2dior': ('dior', ['airplane', 'ship', 'vehicle']),
    'same_coco2coco': ('coco', ['airplane', 'boat', 'car']),
    'rev_coco2dior': ('coco', ['airplane', 'ship', 'vehicle']),
    'fwd_dior2coco': ('dior', ['airplane', 'boat', 'car']),
}
EVAL_DOM = {'same_dior2dior': 'dior', 'same_coco2coco': 'coco',
            'rev_coco2dior': 'dior', 'fwd_dior2coco': 'coco'}
# 每个跨域任务的"源域同域任务"(用于选 β)
SRC_SAME = {'rev_coco2dior': 'same_coco2coco', 'fwd_dior2coco': 'same_dior2dior'}
# 类别对应: 跨域任务评测类 -> 同域任务里对应的类
PAIRMAP = {
    'rev_coco2dior': {'airplane': 'airplane', 'ship': 'boat', 'vehicle': 'car'},
    'fwd_dior2coco': {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'},
}
# 原型类(源域里的类名)
SRCCLS = {
    'rev_coco2dior': {'airplane': 'airplane', 'ship': 'boat', 'vehicle': 'car'},
    'fwd_dior2coco': {'airplane': 'airplane', 'boat': 'ship', 'car': 'vehicle'},
}

BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def _as_vec(x):
    for a in ('pooler_output', 'image_embeds', 'text_embeds', 'last_hidden_state'):
        if hasattr(x, a) and getattr(x, a) is not None:
            return getattr(x, a)
    return x


def load_text():
    m = CLIPModel.from_pretrained(CLIP_ID).eval()
    p = CLIPProcessor.from_pretrained(CLIP_ID)
    return m, p


@torch.no_grad()
def text_emb(m, p, t):
    f = _as_vec(m.get_text_features(**p(text=[t], return_tensors='pt', padding=True,
                                        truncation=True))).float()
    return (f / f.norm(dim=-1, keepdim=True)).numpy()[0]


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def ap_of(per_img, F, W):
    for pi in per_img:
        if len(pi['gd']):
            pi['_s'] = np.sqrt(pi['gd'] * np.clip(np.maximum(0.0, F[pi['crop_idx']] @ W), 1e-6, 1.0))
        else:
            pi['_s'] = np.zeros(0)
    return Met.compute_ap50_img([(p['box'], p['_s']) for p in per_img], [p['gt'] for p in per_img])


def ap_base(per_img, F):
    for pi in per_img:
        pi['_s'] = pi['gd'].copy()
    return Met.compute_ap50_img([(p['box'], p['_s']) for p in per_img], [p['gt'] for p in per_img])


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--cache-dir', default=str(DEFAULT_CACHE))
    ap_.add_argument('--n', default=30)
    ap_.add_argument('--proto-pkl', default=r'C:\Users\张三\Desktop\实验数据\probe_equiv_protos.pkl')
    args = ap_.parse_args()
    CACHE = Path(args.cache_dir)

    # 原型(由 probe_clip_diag 落盘)
    protos_np = pickle.load(open(args.proto_pkl, 'rb')) if Path(args.proto_pkl).exists() else {}

    m, p = load_text()
    names = ['airplane', 'ship', 'vehicle', 'boat', 'car', 'truck', 'bus', 'motorcycle']
    dc = {c: text_emb(m, p, f'a photo of {c}') - text_emb(m, p, f'an overhead image of {c}')
          for c in names}

    def neg_proto(bank, cls):
        others = {k: v for k, v in bank.items() if k != cls}
        if not others:
            return np.zeros(512)
        ks = list(others); P = np.stack([others[k] for k in ks])
        o = np.argsort(-(P @ bank[cls]))[:NEG_TOPK]
        return unit(P[o].mean(0))

    out = {}
    for tname, (dom, classes) in TASKS.items():
        bank = protos_np.get(dom, {})
        if not bank:
            print(f'!! 无 {dom} 原型, 跳过 {tname}')
            continue
        out[tname] = {}
        for cls in classes:
            fp = CACHE / f'{tname}_{cls}_off20_n{args.n}.pkl'
            if not fp.exists():
                continue
            b = pickle.load(open(fp, 'rb'))
            per_img, F = b['per_img'], b['F']
            src_cls = SRCCLS.get(tname, {}).get(cls, cls)
            if src_cls not in bank:
                continue
            P_src = bank[src_cls]; P_neg = neg_proto(bank, src_cls)
            d = dc[src_cls]
            base = ap_base(per_img, F)
            r = {'baseline': round(base, 4)}
            for sgn, tag in ((1, 'plus'), (-1, 'minus')):
                for beta in BETAS:
                    W = (unit(P_src + sgn * beta * d) if beta > 0 else P_src) - LAM * P_neg
                    r[f'{tag}_b{beta}'] = round(ap_of(per_img, F, W) - base, 4)
            out[tname][cls] = r

    # 源域选 β
    sel = {}
    for tname, same in SRC_SAME.items():
        if tname not in out or same not in out:
            continue
        common = [c for c in out[same] if c in PAIRMAP.get(tname, {})]
        # 在源域同域任务上按 (符号, β) 平均增益选最优
        best, bestk = -9, None
        for sgn, tag in ((1, 'plus'), (-1, 'minus')):
            for beta in BETAS:
                if beta == 0:
                    continue
                g = np.mean([out[same][c][f'{tag}_b{beta}'] for c in out[same]])
                if g > best:
                    best, bestk = g, (tag, beta)
        sel[tname] = {'selected_on': same, 'source_best': bestk, 'source_gain': round(float(best), 4)}
        # 套到跨域任务
        cross = [c for c in out[tname]]
        if cross and bestk:
            tag, beta = bestk
            g = np.mean([out[tname][c][f'{tag}_b{beta}'] for c in cross])
            sel[tname]['cross_gain_at_source_beta'] = round(float(g), 4)
            # oracle(在目标域上选)
            ob, obk = -9, None
            for t2 in (('plus', 1), ('minus', -1)):
                for b2 in BETAS:
                    if b2 == 0:
                        continue
                    gg = np.mean([out[tname][c][f'{t2[0]}_b{b2}'] for c in cross])
                    if gg > ob:
                        ob, obk = gg, (t2[0], b2)
            sel[tname]['oracle_best'] = obk
            sel[tname]['cross_gain_at_oracle'] = round(float(ob), 4)

    res = {'per_task': out, 'beta_selection': sel, 'betas': BETAS}
    op = Path(r'C:\Users\张三\Desktop\实验数据') / f'probe_beta_sweep_n{args.n}.json'
    json.dump(res, open(op, 'w'), indent=2, ensure_ascii=False)

    print('\n' + '=' * 88)
    print('β 扫描 (相对 baseline 的增益)')
    print('=' * 88)
    for tname, d in out.items():
        if not d:
            continue
        print(f'\n[{tname}]')
        print('  β      ' + ''.join(f'{c:>12s}' for c in d) + f'{"均值":>10s}  (+δ / −δ)')
        for beta in BETAS:
            gp = [d[c][f'plus_b{beta}'] for c in d]
            gm = [d[c][f'minus_b{beta}'] for c in d]
            print(f'  {beta:<6.1f} ' + ''.join(f'{v:+12.4f}' for v in gp)
                  + f'{np.mean(gp):+10.4f}  |'
                  + ''.join(f'{v:+10.4f}' for v in gm) + f'{np.mean(gm):+10.4f}')
    print('\n' + '=' * 88)
    print('β 能否在源域选出?')
    for tname, s in sel.items():
        print(f'  {tname}: 在 {s["selected_on"]} 上选到 {s["source_best"]} '
              f'(源域增益 {s["source_gain"]:+.4f})')
        print(f'      套到跨域: {s.get("cross_gain_at_source_beta")}   '
              f'oracle(目标域选 {s.get("oracle_best")}): {s.get("cross_gain_at_oracle")}')
    print(f'\n已保存: {op}')


if __name__ == '__main__':
    main()
