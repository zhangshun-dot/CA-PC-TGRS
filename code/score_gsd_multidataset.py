# -*- coding: utf-8 -*-
"""score_gsd_multidataset.py — 多数据集 (COCO→DOTA / COCO→NWPU) 真实 RS GSD 观测条件归因

把 §exp-decomp-gsd 的受控轴扩到【多个 RS 数据集】(DIOR 之外再加 DOTA / NWPU)，同一正交分解
vocab/cond/full/wrong + 同门控 γ。回答元的要求: 结论能否在多个数据集上成立。

数据: probe_bidir_gsd_dota_cache / probe_bidir_gsd_nwpu_cache (gdino)
单元:
  DOTA: eval plane/ship/small-vehicle/large-vehicle ← COCO airplane/boat/car/truck
  NWPU: eval airplane/ship/vehicle ← COCO airplane/boat/car
用法(CPU): HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u score_gsd_multidataset.py
"""
import os, sys, json, pickle, glob, time
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

# (key_idx, eval_cls_name, src_coco_cls)
DATASETS = {
    'dota': {'dir': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_dota_cache'),
             'prefix': 'rev_coco2dota',
             'units': [('plane', 'plane', 'airplane'), ('ship', 'ship', 'boat'),
                       ('small-vehicle', 'small-vehicle', 'car'), ('large-vehicle', 'large-vehicle', 'truck')]},
    'nwpu': {'dir': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_gsd_nwpu_cache'),
             'prefix': 'rev_coco2nwpu',
             'units': [('1', 'airplane', 'airplane'), ('2', 'ship', 'boat'), ('10', 'vehicle', 'car')]},
}
EXTRA_NAMES = ['plane', 'small-vehicle', 'large-vehicle']
OUT_JSON = Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_multidataset.json')
OUT_MD = Path(r'C:\Users\张三\Desktop\实验数据\score_gsd_multidataset.md')


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


def find_blob(d, prefix, key, gsd):
    hits = glob.glob(str(d / f'{prefix}_{key}_off20_n*_gsd{gsd}.pkl'))
    return hits[0] if hits else None


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

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL + EXTRA_NAMES))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    idx = {c: i for i, c in enumerate(all_names)}

    gcfg = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE, c0 = gcfg['prior_scale'], gcfg['c0']

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    out = {'per_ds': {}, 'config': {'datasets': list(DATASETS), 'gsd_levels': GSD_LEVELS, 'lam': LAM,
                                    'zero_target_gt': True, 'detector': 'grounding-dino-tiny'}}

    for ds, spec in DATASETS.items():
        d = spec['dir']
        if not d.exists():
            print(f'  !! 缺目录 {d}', flush=True); continue
        per_gsd = {}
        cells_all = []
        drift = {}
        for gsd in GSD_LEVELS:
            means = []
            for key_idx, eval_cls, src in spec['units']:
                hp = find_blob(d, spec['prefix'], key_idx, gsd)
                if not hp or src not in bank_coco:
                    continue
                blob = pickle.load(open(hp, 'rb'))
                per_img, F = blob['per_img'], blob['F']
                if len(F):
                    means.append(F.mean(0))
                P_src = bank_coco[src]
                P_neg = unit(RB.Proto.get_negative_prototype(bank_coco, src, NEG_TOPK))
                dc = E_photo[idx[eval_cls]] - E_over[idx[src]]
                mag = float(np.linalg.norm(dc))
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
                Vs = {'full': dc,
                      'vocab': unit(E_over[idx[eval_cls]] - E_over[idx[src]]) * mag,
                      'cond': unit(E_photo[idx[eval_cls]] - E_over[idx[eval_cls]]) * mag,
                      'wrong': unit(E_over[idx[eval_cls]] - E_photo[idx[src]]) * mag}
                base = float(RB.ap_base(per_img, F))
                row = {'ds': ds, 'eval': eval_cls, 'src': src, 'gsd': gsd, 'k': 1.0 / gsd,
                       'base': round(base, 4), 'n_img': len(per_img)}
                for kk, v in Vs.items():
                    row[kk + '_g'] = round(float(RB.ap_of(per_img, F, unit(P_src + gamma * v) - LAM * P_neg)) - base, 4)
                row['margin_fv'] = round(row['full_g'] - row['vocab_g'], 4)
                cells_all.append(row)
            if means:
                m = unit(np.mean(means, 0))
                drift[gsd] = m
            if cells_all:
                agg = lambda key: float(np.mean([c[key] for c in cells_all if c['gsd'] == gsd]))
                per_gsd[str(gsd)] = {'k': 1.0 / gsd, 'n': sum(1 for c in cells_all if c['gsd'] == gsd),
                                     'base': round(agg('base'), 4), 'full_g': round(agg('full_g'), 4),
                                     'vocab_g': round(agg('vocab_g'), 4), 'cond_g': round(agg('cond_g'), 4),
                                     'wrong_g': round(agg('wrong_g'), 4), 'margin_fv': round(agg('margin_fv'), 4)}
        dF = {str(g): (round(float(np.linalg.norm(drift[g] - drift[1.0])), 4) if (g in drift and 1.0 in drift) else None)
              for g in GSD_LEVELS}
        for g in GSD_LEVELS:
            if str(g) in per_gsd:
                per_gsd[str(g)]['drift_dF'] = dF[str(g)]
        # 逐 eval_cls 均值
        per_cls = {}
        for key_idx, eval_cls, src in spec['units']:
            cs = [c for c in cells_all if c['eval'] == eval_cls]
            if cs:
                per_cls[eval_cls] = {'n': len(cs),
                                     'full_g': round(float(np.mean([c['full_g'] for c in cs])), 4),
                                     'vocab_g': round(float(np.mean([c['vocab_g'] for c in cs])), 4),
                                     'cond_g': round(float(np.mean([c['cond_g'] for c in cs])), 4)}
        # 相关
        H = [1.0 / g for g in GSD_LEVELS]
        keys = [str(g) for g in GSD_LEVELS]
        r_base, p_base = pearson(H, [per_gsd[k]['base'] for k in keys])
        r_full, p_full = pearson(H, [per_gsd[k]['full_g'] for k in keys])
        r_vocab, p_vocab = pearson(H, [per_gsd[k]['vocab_g'] for k in keys])
        r_cond, p_cond = pearson(H, [per_gsd[k]['cond_g'] for k in keys])
        r_dF, p_dF = pearson(H, [per_gsd[k]['drift_dF'] for k in keys])
        mean_all = {kk: round(float(np.mean([c[kk] for c in cells_all])), 4)
                    for kk in ('full_g', 'vocab_g', 'cond_g', 'wrong_g')}
        out['per_ds'][ds] = {'per_gsd': per_gsd, 'per_cls': per_cls, 'mean_all': mean_all, 'n_cells': len(cells_all),
                             'corr': {'base': [round(r_base, 4), round(p_base, 4)],
                                      'full_g': [round(r_full, 4), round(p_full, 4)],
                                      'vocab_g': [round(r_vocab, 4), round(p_vocab, 4)],
                                      'cond_g': [round(r_cond, 4), round(p_cond, 4)],
                                      'drift_dF': [round(r_dF, 4), round(p_dF, 4)]}}
        print(f'[{ds}] n={len(cells_all)} 全单元: full{mean_all["full_g"]:+.4f} vocab{mean_all["vocab_g"]:+.4f} '
              f'cond{mean_all["cond_g"]:+.4f} wrong{mean_all["wrong_g"]:+.4f}', flush=True)
    json.dump(out, open(OUT_JSON, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    L = ['# 多数据集真实 RS GSD 观测条件归因 (COCO→DOTA / COCO→NWPU)\n',
         f'- 生成时间: {time.strftime("%Y-%m-%d %H:%M")}  gdino  零目标 GT  GSD [1.0,0.5,0.25,0.125]\n']
    for ds, r in out['per_ds'].items():
        L.append(f'\n## 数据集: {ds} (n_cells={r["n_cells"]})\n')
        L.append('| GSD | k | base | full_g | vocab_g | cond_g | wrong_g | Δ(full−vocab) | ΔF |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        for g in GSD_LEVELS:
            if str(g) not in r['per_gsd']:
                continue
            x = r['per_gsd'][str(g)]
            L.append(f"| {g} | {x['k']} | {x['base']:.4f} | {x['full_g']:+.4f} | {x['vocab_g']:+.4f} | "
                     f"{x['cond_g']:+.4f} | {x['wrong_g']:+.4f} | {x['margin_fv']:+.4f} | {x['drift_dF']} |")
        L.append('\n每类别: ' + '; '.join(f"{k}(n{v['n']}): full{v['full_g']:+.4f}/vocab{v['vocab_g']:+.4f}/cond{v['cond_g']:+.4f}"
                                        for k, v in r['per_cls'].items()))
        ma = r['mean_all']; cc = r['corr']
        L.append(f"\n- 全单元均值: full {ma['full_g']:+.4f} / vocab {ma['vocab_g']:+.4f} / cond {ma['cond_g']:+.4f} / wrong {ma['wrong_g']:+.4f}")
        L.append(f"- 相关(vs H): base {cc['base'][0]:+.3f}(p={cc['base'][1]:.3f}); vocab {cc['vocab_g'][0]:+.3f}(p={cc['vocab_g'][1]:.3f}); "
                 f"cond {cc['cond_g'][0]:+.3f}(p={cc['cond_g'][1]:.3f}); ΔF {cc['drift_dF'][0]:+.3f}(p={cc['drift_dF'][1]:.3f})")
    L.append('\n## 裁决 (多数据集)\n')
    for ds, r in out['per_ds'].items():
        ma = r['mean_all']
        L.append(f"- **{ds}**: cond {ma['cond_g']:+.4f} vs vocab {ma['vocab_g']:+.4f} "
                 f"→ {'cond 强于 vocab (条件项是驱动)' if ma['cond_g'] > ma['vocab_g'] else '未显现 cond 优势'}; "
                 f"wrong {ma['wrong_g']:+.4f} ({'有损' if ma['wrong_g'] < 0 else '也正'})")
    open(OUT_MD, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'\n→ {OUT_JSON} / {OUT_MD}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
