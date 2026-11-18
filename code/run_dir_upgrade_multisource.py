# -*- coding: utf-8 -*-
"""run_dir_upgrade_multisource.py — 4 源正向验证: 部署口径下"换注入方向"是否真提分

背景 (10-11 部署口径读数, 同 γ=‖dc‖·gate 门控, 零目标 GT):
  gdino 正向6cell : full(d_c) +0.0594 | par +0.0745 | 类均值方向 +0.0951 | vocab +0.0777
  owlv2 正向6cell : full      +0.0157 | par +0.0333 | 类均值方向 +0.0517 | vocab +0.0092
两检测器一致 -> 现行 d_c 不是部署口径下的最优方向。本脚本把它放到论文口径上验:
  **4 源 (DIOR/DOTA/NWPU/VEDAI) x 17 单元 -> COCO, 与论文主表逐字同源**

关键: 门控与步长完全不动 (γ = PRIOR_SCALE·‖dc‖·gate_conf), 只换方向向量; 所有向量范数匹配 ‖dc‖,
      因此 full 一行必须**逐位复现论文的 +0.0398 / 11-17**。若不复现就是脚手架错, 读数作废。
另附 TGE 原生尺度 (β=2.0) 作对照, 区分"换方向"与"换步长"。

用法: HF_HUB_OFFLINE=1 MD_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u run_dir_upgrade_multisource.py
"""
import os, sys, json, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, r'C:/Users/张三/Desktop/实验代码')
sys.path.insert(0, r'C:/Users/张三/Desktop/A-CSPC')
import numpy as np, torch
from pathlib import Path
import run_bidirectional as RB

RB.DEVICE = 'cpu'
MD = __import__('run_closed_loop_multidomain_v2')   # 复用 UNITS / load_src_imgs / load_coco_cache / ap_of
UNITS = MD.UNITS
COCO_EVAL = ['airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle']
PRIOR_SCALE, c0 = MD.PRIOR_SCALE, MD.c0
LAM, NEG_TOPK, N_PROTO = RB.LAM, RB.NEG_TOPK, RB.N_PROTO
OUT = Path(r'C:/Users/张三/Desktop/实验数据/run_dir_upgrade_multisource.json')
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
VK = ['full', 'par_ph', 'tge_nm', 'vocab', 'tge_nat']


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def main():
    t0 = time.time()
    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0; cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    # ── 文本锚 (与 run_sota_tge / probe_dir_decomp 同源) ──
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    idx = {c: i for i, c in enumerate(all_names)}
    _txt = {}

    def EP(n):   # 'a photo of {n}'  (任意名字, 含 DOTA/NWPU/VEDAI 类名)
        k = ('p', n)
        if k not in _txt: _txt[k] = RB.text_embs([f'a photo of {n}'])[0]
        return _txt[k]

    def EO(n):   # 'an overhead image of {n}'
        k = ('o', n)
        if k not in _txt: _txt[k] = RB.text_embs([f'an overhead image of {n}'])[0]
        return _txt[k]

    t_bar = unit(np.mean([EP(c) for c in all_names], axis=0))   # TGE DeltaSpace 锚 (仅具名类)
    Bmat = np.stack([EP(c) - t_bar for c in all_names], axis=1)
    U, S, _ = np.linalg.svd(Bmat, full_matrices=False); U = U[:, S > S[0] * 1e-8]
    P_ph = U @ U.T
    print(f'[text] t_bar + 类子空间 rank={U.shape[1]}', flush=True)

    # ── 4 源原型 bank ──
    src_banks = {d: {} for d in ('dior', 'dota', 'nwpu', 'vedai')}
    for ds in ('dior', 'dota', 'nwpu', 'vedai'):
        for sc in sorted({s for d, s, _, _ in UNITS if d == ds}):
            ims, gts = MD.load_src_imgs(ds, sc, N_PROTO)
            if not ims:
                print(f'  !! {ds}/{sc} 无图像', flush=True); continue
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                print(f'  !! {ds}/{sc} 无 crop', flush=True); continue
            Ft = RB.clip_encode_imgs(crops)
            v = torch.from_numpy(Ft.mean(0)); src_banks[ds][sc] = v / v.norm()
            print(f'  [src] {ds}/{sc}: {len(crops)} crops', flush=True)

    # ── 逐单元 ──
    results, gain = {}, {k: [] for k in VK}
    for ds, src_cls, eval_cls, pname in UNITS:
        blob = MD.load_coco_cache(eval_cls)
        if blob is None or src_cls not in src_banks[ds]:
            print(f'  !! skip {ds}/{src_cls}->{eval_cls}', flush=True); continue
        per_img, F = blob['per_img'], blob['F']
        P_src = unit(src_banks[ds][src_cls].numpy())
        P_neg = unit(RB.Proto.get_negative_prototype(src_banks[ds], src_cls, NEG_TOPK).numpy())
        dc = EP(eval_cls) - EO(src_cls)
        mag = float(np.linalg.norm(dc))
        max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
        g = float(np.clip(PRIOR_SCALE * mag * MD.gate_conf(max_conf), 0.0, 4.0))
        base = MD.ap_of(per_img, F, unit(P_src) - LAM * P_neg)
        Vs = {'full': dc,
              'par_ph': unit(P_ph @ dc) * mag,
              'tge_nm': unit(EP(eval_cls) - t_bar) * mag,
              'vocab': unit(EO(eval_cls) - EO(src_cls)) * mag}
        rec = {'unit': f'{ds}_{src_cls}->{eval_cls}', 'base': round(base, 4),
               'dc_mag': round(mag, 4), 'gamma': round(g, 4), 'max_conf': round(max_conf, 4)}
        for k in VK:
            if k == 'tge_nat':
                rec[k] = round(MD.ap_of(per_img, F, unit(P_src + 2.0 * (EP(eval_cls) - t_bar)) - LAM * P_neg), 4)
            else:
                rec[k] = round(MD.ap_of(per_img, F, unit(P_src + g * Vs[k]) - LAM * P_neg), 4)
            rec[k + '_gain'] = round(rec[k] - base, 4)
            gain[k].append(rec[k] - base)
        # oracle 峰值 (仅 full / tge_nm)
        for k in ('full', 'tge_nm'):
            v = Vs[k]; best = base
            for bb in BETA_GRID:
                a = MD.ap_of(per_img, F, unit(P_src + bb * v) - LAM * P_neg)
                if a > best: best = a
            rec[k + '_star_gain'] = round(best - base, 4)
        results[rec['unit']] = rec
        print(f'  [{rec["unit"]:<28}] base={base:.4f} | '
              + ' | '.join(f'{k}={rec[k]:.4f}({rec[k+"_gain"]:+.4f})' for k in VK), flush=True)

    n = len(results)
    summary = {k: {'mean_gain': float(np.mean(gain[k])), 'n_pos': int(sum(1 for v in gain[k] if v > 0)), 'n': n}
               for k in VK}
    summary['_oracle'] = {'full': float(np.mean([r['full_star_gain'] for r in results.values()])),
                          'tge_nm': float(np.mean([r['tge_nm_star_gain'] for r in results.values()]))}
    print('\n' + '=' * 88, flush=True)
    print(f'4 源正向 {n} 单元 —— 部署口径 (同 γ 门控, 零目标 GT):', flush=True)
    print(f'  {"方向":<10}{"均值增益":>12}{"正号单元":>12}{"oracle峰值":>12}', flush=True)
    for k in VK:
        s = summary[k]
        oc = summary['_oracle'].get(k)
        pos = '%d/%d' % (s['n_pos'], n)
        orc = ('%+.4f' % oc) if oc is not None else '-'
        print(f'  {k:<10}{s["mean_gain"]:>+12.4f}{pos:>12}{orc:>12}', flush=True)
    print('\n  论文现行口径 = full 行, 应复现 +0.0398 / 11-17', flush=True)
    json.dump({'config': {'units': n, 'sources': ['dior', 'dota', 'nwpu', 'vedai'], 'device': 'cpu',
                          'gate': {'prior_scale': PRIOR_SCALE, 'c0': c0}, 'lam': LAM,
                          'norm_matched_to': '||dc||', 'zero_target_gt': True,
                          'note': 'same gate & step as paper; only the injected DIRECTION differs; '
                                  'full must reproduce the paper headline'},
               'summary': summary, 'results': results, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
