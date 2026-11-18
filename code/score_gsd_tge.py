# -*- coding: utf-8 -*-
"""score_gsd_tge.py — 与 TGE 的机制区分: 在真实 RS GSD 条件轴上 逐档对比 CA-PC vs TGE

元 2026-10-13 障碍①/②: CA-PC 与 TGE 竞争优势不稳(Table 15 均值仅 +0.0005)。本脚本在
【条件可变】的真实 RS GSD 轴上, 把 TGE 的 DeltaSpace 方向放进同一分解框架, 直接显示二者机制差异:
  - CA-PC 方向 dc = E_photo[eval] - E_over[src]  (含显式 photo↔overhead 观测条件分量)
  - TGE  方向 d_tge = E_photo[eval] - t_bar      (ExDet Eq.1, 纯类别外推, 无域/条件项)
  另把 TGE 方向做 norm-match + 同门控, 与 cond/vocab 分量并列 (只动方向)。

臂:
  base, ca_pc(gate), tge_nat(ExDet β=1.0), tge_star(grid 最优, 不 handicap), tge_gate(同门控+norm-match),
  cond_gate, vocab_gate (参考分量)
数据: 反向 COCO→DIOR 的 gdino/owlv2 多类别 GSD 缓存 (probe_bidir_gsd_cache / _owl_cache)。

用法(CPU): HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u score_gsd_tge.py
"""
import os, sys, json, pickle, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
os.environ.setdefault('AC_SPC_COCO_ANN', r'C:\Users\张三\modeling\coco_data\annotations\instances_val2017.json')
os.environ.setdefault('AC_SPC_COCO_IMG', r'C:\Users\张三\modeling\coco_data\val2017')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
import run_bidirectional as RB
from pathlib import Path

RB.DEVICE = 'cpu'
LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
GSD_LEVELS = [1.0, 0.5, 0.25, 0.125]
CATS = {'vehicle': ['car', 'truck', 'bus', 'motorcycle'], 'ship': ['boat'], 'airplane': ['airplane']}
DET_DIRS = {'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_cache'),
            'owlv2': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_owl_cache')}
OUT_JSON = Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_tge.json')
OUT_MD = Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_tge.md')


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def main():
    t0 = time.time()
    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    bank_coco = {}
    for cls in RB.COCO_POOL:
        ims, gts = RB.load_split(cfg, 'coco', cls, 0, N_PROTO)
        crops = RB.D.gt_crops(ims, gts)
        if not crops:
            continue
        Ft = RB.clip_encode_imgs(crops)
        pt = torch.from_numpy(Ft.mean(0)); bank_coco[cls] = pt / pt.norm()

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    idx = {c: i for i, c in enumerate(all_names)}
    t_bar = unit(E_photo.mean(0))                     # TGE DeltaSpace 基类文本锚 (photo 空间)

    gcfg = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE, c0 = gcfg['prior_scale'], gcfg['c0']

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    out = {'per_det': {}, 'config': {'cats': CATS, 'gsd_levels': GSD_LEVELS, 'lam': LAM,
                                     'tge_ref': 'ExDet DeltaSpace Eq.1 v_n = v_b + (t_n - t_bar), faithful',
                                     'zero_target_gt': True}}

    for det, gdir in DET_DIRS.items():
        if not gdir.exists():
            continue
        per_gsd = {}
        cells_all = []
        for gsd in GSD_LEVELS:
            k = 1.0 / gsd
            for cat, srcs in CATS.items():
                for sc in srcs:
                    cp = gdir / f'rev_coco2dior_{cat}_off20_n100_gsd{gsd}.pkl'
                    if not cp.exists() or sc not in bank_coco:
                        continue
                    blob = pickle.load(open(cp, 'rb'))
                    per_img, F = blob['per_img'], blob['F']
                    P_src = bank_coco[sc]
                    P_neg = unit(RB.Proto.get_negative_prototype(bank_coco, sc, NEG_TOPK))
                    dc = E_photo[idx[cat]] - E_over[idx[sc]]
                    mag = float(np.linalg.norm(dc))
                    d_tge = E_photo[idx[cat]] - t_bar
                    max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                    gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
                    base = float(RB.ap_base(per_img, F))
                    row = {'det': det, 'cat': cat, 'src': sc, 'gsd': gsd, 'k': k, 'base': round(base, 4)}
                    # base / ca_pc(gate) / tge_nat(β=1) / tge_gate / cond / vocab
                    row['ca_pc_g'] = round(float(RB.ap_of(per_img, F, unit(P_src + gamma * dc) - LAM * P_neg)) - base, 4)
                    row['tge_nat_g'] = round(float(RB.ap_of(per_img, F, unit(P_src + 1.0 * d_tge) - LAM * P_neg)) - base, 4)
                    d_tge_n = unit(d_tge) * mag
                    row['tge_gate_g'] = round(float(RB.ap_of(per_img, F, unit(P_src + gamma * d_tge_n) - LAM * P_neg)) - base, 4)
                    d_cond = unit(E_photo[idx[cat]] - E_over[idx[cat]]) * mag
                    row['cond_gate_g'] = round(float(RB.ap_of(per_img, F, unit(P_src + gamma * d_cond) - LAM * P_neg)) - base, 4)
                    d_vocab = unit(E_over[idx[cat]] - E_over[idx[sc]]) * mag
                    row['vocab_gate_g'] = round(float(RB.ap_of(per_img, F, unit(P_src + gamma * d_vocab) - LAM * P_neg)) - base, 4)
                    # star (grid best) for tge and ca_pc
                    cur = [float(RB.ap_of(per_img, F, unit(P_src + bb * d_tge) - LAM * P_neg)) - base for bb in BETA_GRID]
                    row['tge_star_g'] = round(max(cur), 4); row['tge_star_beta'] = BETA_GRID[int(np.argmax(cur))]
                    cur2 = [float(RB.ap_of(per_img, F, unit(P_src + bb * dc) - LAM * P_neg)) - base for bb in BETA_GRID]
                    row['ca_pc_star_g'] = round(max(cur2), 4); row['ca_pc_star_beta'] = BETA_GRID[int(np.argmax(cur2))]
                    cells_all.append(row)
            if cells_all:
                m = lambda key: float(np.mean([c[key] for c in cells_all if c['gsd'] == gsd]))
                per_gsd[str(gsd)] = {kk: round(m(kk), 4) for kk in
                                     ('base', 'ca_pc_g', 'tge_nat_g', 'tge_gate_g', 'tge_star_g',
                                      'ca_pc_star_g', 'cond_gate_g', 'vocab_gate_g')}
                per_gsd[str(gsd)]['k'] = k
                print(f'  [{det}] gsd={gsd} base={m("base"):.4f} | CA-PC(G){m("ca_pc_g"):+.4f} '
                      f'TGE(nat){m("tge_nat_g"):+.4f} TGE*(star){m("tge_star_g"):+.4f} '
                      f'TGE(G){m("tge_gate_g"):+.4f} | cond{m("cond_gate_g"):+.4f} vocab{m("vocab_gate_g"):+.4f}',
                      flush=True)
        # 宏平均 (跨 GSD)
        macro = {kk: round(float(np.mean([per_gsd[str(g)][kk] for g in GSD_LEVELS])), 4)
                 for kk in per_gsd[str(GSD_LEVELS[0])] if kk != 'k'}
        out['per_det'][det] = {'per_gsd': per_gsd, 'macro': macro,
                               'ca_minus_tge_gate': round(macro['ca_pc_g'] - macro['tge_gate_g'], 4),
                               'ca_minus_tge_nat': round(macro['ca_pc_g'] - macro['tge_nat_g'], 4),
                               'ca_minus_tge_star': round(macro['ca_pc_star_g'] - macro['tge_star_g'], 4)}

    # 裁决: TGE 是否随条件走? CA-PC 相对 TGE 的优势是否来自 cond 分量?
    def pearson(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if len(x) < 3:
            return float('nan'), float('nan')
        x = x - x.mean(); y = y - y.mean()
        d = np.sqrt((x ** 2).sum() * (y ** 2).sum())
        r = float((x * y).sum() / d) if d > 1e-12 else float('nan')
        from math import erf, sqrt
        t = r * np.sqrt((len(x) - 2) / max(1e-12, 1 - r ** 2))
        p = float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))
        return r, p

    verdict = {}
    H = [1.0 / g for g in GSD_LEVELS]
    for det, d in out['per_det'].items():
        pg = d['per_gsd']
        r_ca, p_ca = pearson(H, [pg[str(g)]['ca_pc_g'] for g in GSD_LEVELS])
        r_tge, p_tge = pearson(H, [pg[str(g)]['tge_nat_g'] for g in GSD_LEVELS])
        r_tgef, p_tgef = pearson(H, [pg[str(g)]['tge_gate_g'] for g in GSD_LEVELS])
        verdict[det] = {
            'ca_pc_gain_by_gsd': [pg[str(g)]['ca_pc_g'] for g in GSD_LEVELS],
            'tge_nat_gain_by_gsd': [pg[str(g)]['tge_nat_g'] for g in GSD_LEVELS],
            'tge_gate_gain_by_gsd': [pg[str(g)]['tge_gate_g'] for g in GSD_LEVELS],
            'cond_gain_by_gsd': [pg[str(g)]['cond_gate_g'] for g in GSD_LEVELS],
            'vocab_gain_by_gsd': [pg[str(g)]['vocab_gate_g'] for g in GSD_LEVELS],
            'macro_ca_pc': d['macro']['ca_pc_g'], 'macro_tge_nat': d['macro']['tge_nat_g'],
            'macro_tge_star': d['macro']['tge_star_g'], 'macro_cond': d['macro']['cond_gate_g'],
            'macro_vocab': d['macro']['vocab_gate_g'],
        }
    out['verdict'] = verdict
    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    L = ['# CA-PC vs TGE 的机制区分 (真实 RS GSD 条件轴)\n',
         f'- 生成时间: {time.strftime("%Y-%m-%d %H:%M")}  反向 COCO→DIOR  多类别  TGE=ExDet DeltaSpace($t_n-\\bar t_b$)\n',
         '- 臂: base / CA-PC(门控) / TGE(ExDet β=1) / TGE(grid 最优 *) / TGE(同门控+norm) / cond / vocab\n']
    for det, d in out['per_det'].items():
        L.append(f'\n## {det}\n')
        L.append('| GSD | base | CA-PC(g) | TGE(β=1) | TGE* | TGE(g) | cond(g) | vocab(g) |')
        L.append('|---|---|---|---|---|---|---|---|')
        for g in GSD_LEVELS:
            r = d['per_gsd'][str(g)]
            L.append(f"| {g} | {r['base']:.4f} | {r['ca_pc_g']:+.4f} | {r['tge_nat_g']:+.4f} | "
                     f"{r['tge_star_g']:+.4f} | {r['tge_gate_g']:+.4f} | {r['cond_gate_g']:+.4f} | {r['vocab_gate_g']:+.4f} |")
        m = d['macro']
        L.append(f"\n- 宏平均(跨 GSD): CA-PC(g) {m['ca_pc_g']:+.4f} / TGE(β=1) {m['tge_nat_g']:+.4f} / "
                 f"TGE* {m['tge_star_g']:+.4f} / TGE(g) {m['tge_gate_g']:+.4f} / cond {m['cond_gate_g']:+.4f} / vocab {m['vocab_gate_g']:+.4f}")
        L.append(f"- CA-PC(g) − TGE(β=1) = {d['ca_minus_tge_nat']:+.4f}; CA-PC(g) − TGE* = {d['ca_minus_tge_star']:+.4f}")
    L.append('\n## 裁决\n')
    for det, v in verdict.items():
        co = v['macro_cond']; vo = v['macro_vocab']
        L.append(f"- **{det}**: cond 分量(CA-PC 独有) 宏平均 {co:+.4f} vs vocab 分量 {vo:+.4f}; "
                 f"CA-PC {v['macro_ca_pc']:+.4f} vs TGE 自然 {v['macro_tge_nat']:+.4f} / TGE* {v['macro_tge_star']:+.4f}。"
                 + ('CA-PC 在真实 RS 轴上领先 TGE, 差额≈其 cond 分量(观测条件项), TGE 无此分量。'
                    if v['macro_ca_pc'] > v['macro_tge_nat'] else 'TGE 自然口径不弱于 CA-PC, 需按 grid 最优读。'))
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'\n→ {OUT_JSON} / {OUT_MD}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
