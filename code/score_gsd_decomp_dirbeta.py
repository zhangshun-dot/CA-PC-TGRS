# -*- coding: utf-8 -*-
"""score_gsd_decomp_dirbeta.py — 方向臂 × 强度(oracle β*) 对照 (第10版审稿 MC1 专用)

问题: "CA-PC 去掉 condition 分量(= 只保留类词方向, 即 TGE 同族的 vocab-only)后,
        即使给该方向最优强度, 能否追上 full(完整 canonical shift)?"

做法: 在真实 RS GSD 条件轴(反向 COCO->DIOR)上, 四臂 full/vocab/cond/wrong
      各自扫 β∈BETA_GRID 取 oracle β* 与最优 gain; 并保留固定 γ(decomp_multi 口径)对照。
      缓存复用 probe_bidir_gsd_cache / probe_bidir_gsd_owl_cache, CPU 重算, 不重跑 GPU。

用法 (CPU): HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu \
            "C:/Users/张三/vg/Scripts/python.exe" -u score_gsd_decomp_dirbeta.py
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
CATS = {'vehicle': ['car', 'truck', 'bus', 'motorcycle'],
        'ship': ['boat'],
        'airplane': ['airplane']}
DET_DIRS = {
    'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_cache'),
    'owlv2': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_owl_cache'),
}
OUT_JSON = Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_decomp_dirbeta.json')
OUT_MD = Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_decomp_dirbeta.md')


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

    gcfg = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE, c0 = gcfg['prior_scale'], gcfg['c0']

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    out = {'config': {'cats': CATS, 'gsd_levels': GSD_LEVELS, 'beta_grid': BETA_GRID,
                      'lam': LAM, 'neg_topk': NEG_TOPK, 'zero_target_gt': True,
                      'gate': {'prior_scale': PRIOR_SCALE, 'c0': c0},
                      'note': 'per-direction oracle beta* vs fixed-gamma; reverse COCO->DIOR; '
                              'answers whether dropping the condition component (vocab-only direction) '
                              'can reach the full canonical shift even at its own optimal strength'},
           'per_det': {}}

    for det, gdir in DET_DIRS.items():
        if not gdir.exists():
            print(f'  !! 缺目录 {gdir}', flush=True); continue
        cells_all = []
        for gsd in GSD_LEVELS:
            for cat, srcs in CATS.items():
                for sc in srcs:
                    cp = gdir / f'rev_coco2dior_{cat}_off20_n100_gsd{gsd}.pkl'
                    if not cp.exists() or sc not in bank_coco:
                        continue
                    blob = pickle.load(open(cp, 'rb'))
                    per_img, F = blob['per_img'], blob['F']
                    P_src = bank_coco[sc].numpy()
                    P_neg = unit(RB.Proto.get_negative_prototype(bank_coco, sc, NEG_TOPK))
                    dc = E_photo[idx[cat]] - E_over[idx[sc]]
                    mag = float(np.linalg.norm(dc))
                    max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                    gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
                    Vs = {'full': dc,
                          'vocab': unit(E_over[idx[cat]] - E_over[idx[sc]]) * mag,
                          'cond': unit(E_photo[idx[cat]] - E_over[idx[cat]]) * mag,
                          'wrong': unit(E_over[idx[cat]] - E_photo[idx[sc]]) * mag}
                    base = float(RB.ap_base(per_img, F))
                    row = {'det': det, 'cat': cat, 'src': sc, 'gsd': gsd, 'base': round(base, 4),
                           'gamma_fixed': round(gamma, 4), 'n_box': int(sum(len(p['gd']) for p in per_img))}
                    for kk, v in Vs.items():
                        # 固定 γ (decomp_multi 口径)
                        g_fix = float(RB.ap_of(per_img, F, unit(P_src + gamma * v) - LAM * P_neg)) - base
                        # oracle β* 扫描
                        best_ap, best_b = -1.0, 0.0
                        for b in BETA_GRID:
                            ap = float(RB.ap_of(per_img, F, unit(P_src + b * v) - LAM * P_neg))
                            if ap > best_ap:
                                best_ap, best_b = ap, b
                        row[kk + '_fix'] = round(g_fix, 4)
                        row[kk + '_or'] = round(best_ap - base, 4)
                        row[kk + '_bstar'] = best_b
                    row['d_or_full_minus_vocab'] = round(row['full_or'] - row['vocab_or'], 4)
                    row['d_fix_full_minus_vocab'] = round(row['full_fix'] - row['vocab_fix'], 4)
                    cells_all.append(row)
            # 该 gsd 的均值
            sub = [c for c in cells_all if c['gsd'] == gsd]
            if sub:
                m = lambda k: float(np.mean([c[k] for c in sub]))
                print(f'  [{det}] gsd={gsd} base={m("base"):.4f} | '
                      f'OR full{m("full_or"):+.4f} vocab{m("vocab_or"):+.4f} cond{m("cond_or"):+.4f} '
                      f'wrong{m("wrong_or"):+.4f} | FIX full{m("full_fix"):+.4f} vocab{m("vocab_fix"):+.4f} '
                      f'| d(full-vocab) OR {m("d_or_full_minus_vocab"):+.4f}', flush=True)

        if not cells_all:
            continue
        macro = lambda k: float(np.mean([c[k] for c in cells_all]))
        per_cat = {}
        for cat in CATS:
            sub = [c for c in cells_all if c['cat'] == cat]
            if sub:
                per_cat[cat] = {'n': len(sub),
                                **{k: round(float(np.mean([c[k] for c in sub])), 4)
                                   for k in ('base', 'full_or', 'vocab_or', 'cond_or', 'wrong_or',
                                             'full_fix', 'vocab_fix', 'd_or_full_minus_vocab',
                                             'd_fix_full_minus_vocab')}}
        out['per_det'][det] = {
            'n_cells': len(cells_all),
            'macro': {k: round(macro(k), 4) for k in
                      ('base', 'full_or', 'vocab_or', 'cond_or', 'wrong_or',
                       'full_fix', 'vocab_fix', 'cond_fix', 'wrong_fix',
                       'd_or_full_minus_vocab', 'd_fix_full_minus_vocab')},
            'beta_star_mean': {k: round(float(np.mean([c[k + '_bstar'] for c in cells_all])), 3)
                               for k in ('full', 'vocab', 'cond', 'wrong')},
            'per_cat': per_cat,
            'cells': cells_all,
        }

    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # ---- MD ----
    L = ['# 方向臂 × 最优强度(oracle beta*) 对照 — 回答"去掉 condition 分量能否追上 full"\n',
         f'- 生成时间: {time.strftime("%Y-%m-%d %H:%M")}  反向 COCO->DIOR(RS 目标域), 零目标 GT',
         f'- 四臂: full=dc(joint) / vocab=(E_over^cat - E_over^src)(**去掉 condition 分量**) / '
         f'cond=(E_photo^cat - E_over^cat) / wrong(反轴); 范数匹配 ‖dc‖',
         f'- OR = 每臂各自扫 beta∈{BETA_GRID} 的 oracle 最优; FIX = 同固定 gamma(decomp_multi 口径)\n']
    for det, dr in out['per_det'].items():
        m = dr['macro']
        L.append(f'## {det}  (n_cells={dr["n_cells"]})\n')
        L.append('| 臂 | oracle gain | oracle beta* (均值) | 固定γ gain |')
        L.append('|---|---|---|---|')
        for arm in ('full', 'vocab', 'cond', 'wrong'):
            L.append(f'| {arm} | {m[arm+"_or"]:+.4f} | {dr["beta_star_mean"][arm]} | '
                     f'{m.get(arm+"_fix", float("nan")):+.4f} |')
        L.append(f'\n- **full_oracle − vocab_oracle = {m["d_or_full_minus_vocab"]:+.4f}**  '
                 f'(固定γ 下为 {m["d_fix_full_minus_vocab"]:+.4f})')
        L.append(f'- 即: 即使给"去掉 condition 分量"的方向配**它自己的最优强度**, '
                 f'仍比 full 低 {m["d_or_full_minus_vocab"]:+.4f} AP50')
        L.append('\n每类别(oracle):\n')
        L.append('| 类别 | full_or | vocab_or | cond_or | full−vocab |')
        L.append('|---|---|---|---|---|')
        for cat, d in dr['per_cat'].items():
            L.append(f'| {cat} | {d["full_or"]:+.4f} | {d["vocab_or"]:+.4f} | {d["cond_or"]:+.4f} | '
                     f'{d["d_or_full_minus_vocab"]:+.4f} |')
        L.append('')
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'\n→ {OUT_JSON}\n→ {OUT_MD}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
