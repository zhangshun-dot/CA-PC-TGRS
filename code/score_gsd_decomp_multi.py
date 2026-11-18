# -*- coding: utf-8 -*-
"""score_gsd_decomp_multi.py — 多 RS 类别 × 多检测器 的真实 GSD 观测条件归因

在原 score_gsd_decomp.py (单 vehicle + 单 gdino) 基础上扩展 (元 2026-10-13 评审要求):
  类别: vehicle(car/truck/bus/motorcycle→vehicle, 非零 vocab) ;
        ship(boat→ship, 非零 vocab) ; airplane(airplane→airplane, 零 vocab 对照)
  检测器: gdino (probe_bidir_gsd_cache) + owlv2 (probe_bidir_gsd_owl_cache)
  GSD: 1.0/0.5/0.25/0.125 ; 反向 COCO→DIOR(RS 目标域)
方向正交分解 vocab/cond/full/wrong, 范数全匹配 ‖dc‖, 同门控 γ (只动方向)。
另加 image-space CLIP drift ΔF(每 GSD 相对 clean), 量化条件严重度, 并回答
"文本 Anchor 恒定 → 是否定量感知分辨率严重度"。

用法 (CPU): HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u score_gsd_decomp_multi.py
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
GSD_LEVELS = [1.0, 0.5, 0.25, 0.125]
CATS = {'vehicle': ['car', 'truck', 'bus', 'motorcycle'],
        'ship': ['boat'],
        'airplane': ['airplane']}
DET_DIRS = {
    'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_cache'),
    'owlv2': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_owl_cache'),
}
OUT_JSON = Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_decomp_multi.json')
OUT_MD = Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_decomp_multi.md')


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


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

    out = {'per_det': {}, 'config': {'cats': CATS, 'gsd_levels': GSD_LEVELS, 'lam': LAM,
                                     'neg_topk': NEG_TOPK, 'zero_target_gt': True,
                                     'gate': {'prior_scale': PRIOR_SCALE, 'c0': c0},
                                     'note': 'multi-category x multi-detector real RS GSD probe; '
                                             'same mag + same gate across directions'}}

    for det, gdir in DET_DIRS.items():
        if not gdir.exists():
            print(f'  !! 缺目录 {gdir}', flush=True); continue
        det_res = {}
        # image drift ΔF per gsd (severity proxy, in CLIP feature space)
        means = {}
        for gsd in GSD_LEVELS:
            Fs = []
            for cls in 'vehicle ship airplane'.split():
                cp = gdir / f'rev_coco2dior_{cls}_off20_n100_gsd{gsd}.pkl'
                if cp.exists():
                    b = pickle.load(open(cp, 'rb'))
                    if len(b['F']):
                        Fs.append(b['F'].mean(0))
            if Fs:
                means[gsd] = unit(np.mean(Fs, 0))
        drift = {str(g): (float(np.linalg.norm(means[g] - means[1.0])) if (g in means and 1.0 in means) else None)
                 for g in GSD_LEVELS}

        per_gsd = {}
        cells_all = []
        for gsd in GSD_LEVELS:
            k = 1.0 / gsd
            cat_rows = {}
            for cat, srcs in CATS.items():
                cg = []
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
                    max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                    gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
                    Vs = {'full': dc,
                          'vocab': unit(E_over[idx[cat]] - E_over[idx[sc]]) * mag,
                          'cond': unit(E_photo[idx[cat]] - E_over[idx[cat]]) * mag,
                          'wrong': unit(E_over[idx[cat]] - E_photo[idx[sc]]) * mag}
                    base = float(RB.ap_base(per_img, F))
                    row = {'det': det, 'cat': cat, 'src': sc, 'gsd': gsd, 'k': k,
                           'base': round(base, 4), 'gamma': round(gamma, 4),
                           'n_box': int(sum(len(p['gd']) for p in per_img))}
                    for kk, v in Vs.items():
                        g = float(RB.ap_of(per_img, F, unit(P_src + gamma * v) - LAM * P_neg)) - base
                        row[kk + '_g'] = round(g, 4)
                    row['margin_fv'] = round(row['full_g'] - row['vocab_g'], 4)
                    row['margin_fc'] = round(row['full_g'] - row['cond_g'], 4)
                    cg.append(row); cells_all.append(row)
                if cg:
                    m = lambda key: float(np.mean([c[key] for c in cg]))
                    cat_rows[cat] = {'n': len(cg), 'base': round(m('base'), 4),
                                     'full_g': round(m('full_g'), 4), 'vocab_g': round(m('vocab_g'), 4),
                                     'cond_g': round(m('cond_g'), 4), 'wrong_g': round(m('wrong_g'), 4),
                                     'margin_fv': round(m('margin_fv'), 4)}
            # overall mean over all cells at this gsd
            if cells_all:
                m = lambda key: float(np.mean([c[key] for c in cells_all if c['gsd'] == gsd]))
                per_gsd[str(gsd)] = {'k': k,
                                     'base': round(m('base'), 4), 'full_g': round(m('full_g'), 4),
                                     'vocab_g': round(m('vocab_g'), 4), 'cond_g': round(m('cond_g'), 4),
                                     'wrong_g': round(m('wrong_g'), 4), 'margin_fv': round(m('margin_fv'), 4),
                                     'cats': cat_rows,
                                     'drift_dF': (round(drift[str(gsd)], 4) if drift[str(gsd)] is not None else None)}
                print(f'  [{det}] gsd={gsd} base={m("base"):.4f} full{m("full_g"):+.4f} '
                      f'vocab{m("vocab_g"):+.4f} cond{m("cond_g"):+.4f} wrong{m("wrong_g"):+.4f} '
                      f'ΔF={drift[str(gsd)]}', flush=True)

        H = [1.0 / g for g in GSD_LEVELS]
        r_base, p_base = pearson(H, [per_gsd[str(g)]['base'] for g in GSD_LEVELS])
        r_full, p_full = pearson(H, [per_gsd[str(g)]['full_g'] for g in GSD_LEVELS])
        r_vocab, p_vocab = pearson(H, [per_gsd[str(g)]['vocab_g'] for g in GSD_LEVELS])
        r_cond, p_cond = pearson(H, [per_gsd[str(g)]['cond_g'] for g in GSD_LEVELS])
        r_mfv, p_mfv = pearson(H, [per_gsd[str(g)]['margin_fv'] for g in GSD_LEVELS])
        r_wrong, p_wrong = pearson(H, [per_gsd[str(g)]['wrong_g'] for g in GSD_LEVELS])
        r_drift, p_drift = pearson(H, [per_gsd[str(g)]['drift_dF'] for g in GSD_LEVELS])
        # 单元格级 (n=4gsd * 6pairs) 相关, 更稳
        rC_mfv, pC_mfv = pearson([c['k'] for c in cells_all], [c['margin_fv'] for c in cells_all])
        rC_vocab, pC_vocab = pearson([c['k'] for c in cells_all], [c['vocab_g'] for c in cells_all])
        rC_cond, pC_cond = pearson([c['k'] for c in cells_all], [c['cond_g'] for c in cells_all])
        mean_vocab = float(np.mean([c['vocab_g'] for c in cells_all]))
        mean_cond = float(np.mean([c['cond_g'] for c in cells_all]))
        mean_full = float(np.mean([c['full_g'] for c in cells_all]))
        mean_wrong = float(np.mean([c['wrong_g'] for c in cells_all]))
        det_res = {'per_gsd': per_gsd,
                   'corr_gsd_mean': {'base': [round(r_base, 4), round(p_base, 4)],
                                     'full_g': [round(r_full, 4), round(p_full, 4)],
                                     'vocab_g': [round(r_vocab, 4), round(p_vocab, 4)],
                                     'cond_g': [round(r_cond, 4), round(p_cond, 4)],
                                     'margin_fv': [round(r_mfv, 4), round(p_mfv, 4)],
                                     'wrong_g': [round(r_wrong, 4), round(p_wrong, 4)],
                                     'drift_dF': [round(r_drift, 4), round(p_drift, 4)]},
                   'corr_cell': {'margin_fv': [round(rC_mfv, 4), round(pC_mfv, 4)],
                                 'vocab_g': [round(rC_vocab, 4), round(pC_vocab, 4)],
                                 'cond_g': [round(rC_cond, 4), round(pC_cond, 4)]},
                   'mean_all_cells': {'n_cells': len(cells_all), 'full_g': round(mean_full, 4),
                                      'vocab_g': round(mean_vocab, 4), 'cond_g': round(mean_cond, 4),
                                      'wrong_g': round(mean_wrong, 4)},
                   'airplane_control_full_g': {str(g): per_gsd[str(g)]['cats'].get('airplane', {}).get('full_g')
                                               for g in GSD_LEVELS}}
        out['per_det'][det] = det_res

    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    # ---- MD ----
    L = []
    L.append('# 多 RS 类别 × 多检测器 GSD 观测条件归因 (扩展 Table 19)\n')
    L.append(f'- 生成时间: {time.strftime("%Y-%m-%d %H:%M")}  反向 COCO→DIOR (RS 目标域)  零目标 GT\n')
    L.append('- 类别: vehicle(4 对, 非零 vocab) / ship(1 对, 非零 vocab) / airplane(1 对, 零 vocab 对照)\n')
    L.append('- 方向: full=dc, vocab-only, cond-only, wrong(反轴); 范数匹配 ‖dc‖, 同门控 γ\n')
    for det, dr in out['per_det'].items():
        L.append(f'\n## 检测器: {det}\n')
        L.append('| GSD | k | base | full_g | vocab_g | cond_g | wrong_g | Δ(full−vocab) | ΔF |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for g in GSD_LEVELS:
            d = dr['per_gsd'][str(g)]
            L.append(f"| {g} | {d['k']} | {d['base']:.4f} | {d['full_g']:+.4f} | {d['vocab_g']:+.4f} | "
                     f"{d['cond_g']:+.4f} | {d['wrong_g']:+.4f} | {d['margin_fv']:+.4f} | "
                     f"{d['drift_dF'] if d['drift_dF'] is None else round(d['drift_dF'],4)} |")
        L.append('\n每类别 (GSD 均值):\n')
        L.append('| 类别(n对) | base | full_g | vocab_g | cond_g |')
        L.append('|---|---|---|---|---|')
        for cat in CATS:
            vals = [(dr['per_gsd'][str(g)]['cats'].get(cat), g) for g in GSD_LEVELS]
            vals = [v for v in vals if v[0]]
            if not vals:
                continue
            n = vals[0][0]['n']
            mg = lambda key: float(np.mean([v[0][key] for v in vals]))
            L.append(f"| {cat}({n}) | {mg('base'):.4f} | {mg('full_g'):+.4f} | {mg('vocab_g'):+.4f} | {mg('cond_g'):+.4f} |")
        c = dr['corr_gsd_mean']; cc = dr['corr_cell']; ma = dr['mean_all_cells']
        L.append(f"\n- 全单元均值(n={ma['n_cells']}): full {ma['full_g']:+.4f} / vocab {ma['vocab_g']:+.4f} / "
                 f"cond {ma['cond_g']:+.4f} / wrong {ma['wrong_g']:+.4f}")
        L.append(f"- 相关(GSD 均值 vs H=k): base {c['base'][0]:+.3f}(p={c['base'][1]:.3f}); "
                 f"full {c['full_g'][0]:+.3f}(p={c['full_g'][1]:.3f}); vocab {c['vocab_g'][0]:+.3f}(p={c['vocab_g'][1]:.3f}); "
                 f"cond {c['cond_g'][0]:+.3f}(p={c['cond_g'][1]:.3f}); margin {c['margin_fv'][0]:+.3f}(p={c['margin_fv'][1]:.3f})")
        L.append(f"- 相关(单元格级 n={ma['n_cells']} vs H): vocab {cc['vocab_g'][0]:+.3f}(p={cc['vocab_g'][1]:.3f}); "
                 f"cond {cc['cond_g'][0]:+.3f}(p={cc['cond_g'][1]:.3f}); margin {cc['margin_fv'][0]:+.3f}(p={cc['margin_fv'][1]:.3f})")
        L.append(f"- **ΔF(图像 CLIP drift vs clean) 随 H: r={c['drift_dF'][0]:+.3f}(p={c['drift_dF'][1]:.3f})** "
                 f"→ {'图像漂移随退化增大, 严重度轴在图像空间成立' if c['drift_dF'][0]>0.5 else '图像漂移未明显随退化'}")
        L.append(f"- 零 vocab 对照 airplane 的 full_g: {dr['airplane_control_full_g']} (应≈cond, vocab≡0)")
    L.append('\n## 裁决 (多类别 × 多检测器)\n')
    for det, dr in out['per_det'].items():
        ma = dr['mean_all_cells']
        L.append(f'- **{det}**: vocab 均值 {ma["vocab_g"]:+.4f} vs cond 均值 {ma["cond_g"]:+.4f} '
                 f'→ cond {"明显强于" if ma["cond_g"] > ma["vocab_g"] + 0.01 else "≈"} vocab; '
                 f'wrong {ma["wrong_g"]:+.4f} ({"有损, 方向正确" if ma["wrong_g"] < 0 else "也有增益"})')
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'\n→ {OUT_JSON} / {OUT_MD}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
