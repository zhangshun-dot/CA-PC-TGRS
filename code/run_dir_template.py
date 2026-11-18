# -*- coding: utf-8 -*-
"""run_dir_template.py — 新路检验: 类判别子空间该用【哪个域的模板】建?

背景 (probe_dir_decomp 读数): 反向迁移 rev_coco2dior 上, 用 E_photo 建的类子空间投影 par=+0.0003 (3/3 全零),
                              TGE 方向也塌 (+0.0043)。两者都由 E_photo 搭, 而反向任务的目标域是 DIOR(俯视)。
假说:
  H1 (配对目标域): 反向应用 E_over 建类轴 -> 救回来; 正向用 E_over 应更差。
  H2 (配对源域)  : 正向用 E_over 更好; 反向用 E_over 更差。
两假说与现有数据都相容, 本 2x2 劈开它们。

向量 (全部范数匹配到 ||d_c||, 同 β 网格, 同门控):
  full        = d_c
  par_ph      = ||dc|| * unit(P_photo @ dc)         (旧 par)
  par_ov      = ||dc|| * unit(P_over  @ dc)         (新)
  tge_ph      = ||dc|| * unit(E_photo(eval) - t_bar_ph)
  tge_ov      = ||dc|| * unit(E_over(eval)  - t_bar_ov)   (新)
  perp_ph     = ||dc|| * unit(dc - P_photo @ dc)    (旧 perp)

纯 B-pipeline 零目标 GT; 产物独立: run_dir_template_{det}.json + 合并 run_dir_template.json
用法: HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u run_dir_template.py --dets gdino owlv2
"""
import os, sys, json, pickle, glob, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
sys.path.insert(0, r'C:/Users/张三/Desktop/实验代码')
sys.path.insert(0, r'C:/Users/张三/Desktop/A-CSPC')
import numpy as np, torch
from pathlib import Path
import run_bidirectional as RB

RB.DEVICE = 'cpu'
BETA = RB.BETA; LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
VEC_KEYS = ['full', 'par_ph', 'par_ov', 'tge_ph', 'tge_ov', 'perp_ph']
VEC_DESC = {'full': 'd_c (full)', 'par_ph': 'par(E_photo 类轴)', 'par_ov': 'par(E_over 类轴)',
            'tge_ph': 'TGE(E_photo)', 'tge_ov': 'TGE(E_over)', 'perp_ph': 'perp(E_photo 残差)'}
DETECTORS = {
    'gdino':  Path(r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache'),
    'owlv2':  Path(r'C:/Users/张三/Desktop/实验数据/probe_owl_cache'),
}
DET_LABEL = {'gdino': 'grounding-tiny', 'owlv2': 'OWL-ViT'}
OUT = Path(r'C:/Users/张三/Desktop/实验数据/run_dir_template.json')


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(cache_dir, tname, eval_cls):
    hits = glob.glob(str(cache_dir / f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


def score_rows(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return rows


def eval_pair(per_img, F, W):
    rows = score_rows(per_img, F, W)
    return (float(RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])),
            float(RB.Met.compute_ap_coco(rows, [p['gt'] for p in per_img])))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--dets', nargs='+', default=['gdino', 'owlv2'])
    args = ap_.parse_args()
    t0 = time.time()

    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                continue
            Ft = RB.clip_encode_imgs(crops)
            pt = torch.from_numpy(Ft.mean(0)); b[cls] = pt / pt.norm()
        bank[domain] = b
        print(f'[protos] {domain}: {len(b)} 类', flush=True)

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    idx = {c: i for i, c in enumerate(all_names)}
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])

    def build_P(Et):
        tb = unit(Et.mean(0))
        B = np.stack([Et[idx[c]] - tb for c in all_names], axis=1)
        U, S, _ = np.linalg.svd(B, full_matrices=False)
        U = U[:, S > S[0] * 1e-8]
        return U @ U.T, tb
    P_ph, tbar_ph = build_P(E_photo)
    P_ov, tbar_ov = build_P(E_over)
    print(f'[subspace] rank photo={int(np.trace(P_ph))} overhead={int(np.trace(P_ov))}', flush=True)

    gcfg = json.load(open(r'C:/Users/张三/Desktop/实验数据/run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE, c0 = gcfg['prior_scale'], gcfg['c0']
    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    out = {}
    if OUT.exists():
        try: out = json.load(open(OUT, encoding='utf-8')).get('per_det', {})
        except Exception: out = {}
    for det in args.dets:
        if det not in DETECTORS: continue
        cache_dir = DETECTORS[det]; cells = []
        for tname in RB.TASKS:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cache_dir, tname, eval_cls)
                if blob is None or src_cls not in bank[src_dom]:
                    print(f'  !! {det}/{tname}/{eval_cls} 缺缓存', flush=True); continue
                per_img = blob['per_img']; F = blob['F']
                P_src = unit(bank[src_dom][src_cls].numpy())
                P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())
                dc = E_photo[idx[eval_cls]] - E_over[idx[src_cls]]
                mag = float(np.linalg.norm(dc))
                par_ph = P_ph @ dc; par_ov = P_ov @ dc
                Vs = {
                    'full':    dc,
                    'par_ph':  unit(par_ph) * mag if np.linalg.norm(par_ph) > 1e-9 else np.zeros_like(dc),
                    'par_ov':  unit(par_ov) * mag if np.linalg.norm(par_ov) > 1e-9 else np.zeros_like(dc),
                    'tge_ph':  unit(E_photo[idx[eval_cls]] - tbar_ph) * mag,
                    'tge_ov':  unit(E_over[idx[eval_cls]] - tbar_ov) * mag,
                    'perp_ph': unit(dc - par_ph) * mag,
                }
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
                cell = {'det': det, 'tname': tname, 'src': src_cls, 'eval': eval_cls,
                        'n_img': len(per_img), 'dc_mag': round(mag, 4),
                        'gamma': round(gamma, 4), 'max_conf': round(max_conf, 4),
                        'frac_ph': round(float(np.linalg.norm(par_ph) / max(mag, 1e-12)), 4),
                        'frac_ov': round(float(np.linalg.norm(par_ov) / max(mag, 1e-12)), 4)}
                b50, bm = eval_pair(per_img, F, unit(P_src) - LAM * P_neg)
                cell['base'], cell['base_m'] = round(b50, 4), round(bm, 4)
                for k in VEC_KEYS:
                    v = Vs[k]; c50, cn = [], []
                    for bb in BETA_GRID:
                        a50, am = eval_pair(per_img, F, unit(P_src + bb * v) - LAM * P_neg)
                        c50.append(round(a50, 4)); cn.append(round(am, 4))
                    g50, gm = eval_pair(per_img, F, unit(P_src + gamma * v) - LAM * P_neg)
                    cell[k + '_curve'] = c50; cell[k + '_curvem'] = cn
                    cell[k + '_star'] = max(c50)
                    cell[k + '_star_beta'] = BETA_GRID[int(np.argmax(c50))]
                    cell[k + '_g'] = round(g50, 4); cell[k + '_g_m'] = round(gm, 4)
                cells.append(cell)
                print(f'  [{det}] {tname[:3]} {src_cls}->{eval_cls} base={b50:.4f} '
                      f'frac_ph={cell["frac_ph"]:.2f} frac_ov={cell["frac_ov"]:.2f} | '
                      f'par_ph*={cell["par_ph_star"]:.4f} par_ov*={cell["par_ov_star"]:.4f} '
                      f'tge_ph*={cell["tge_ph_star"]:.4f} tge_ov*={cell["tge_ov_star"]:.4f}', flush=True)
        out[det] = cells
        json.dump({'config': {'beta_grid': BETA_GRID, 'lam': LAM, 'zero_target_gt': True,
                              'note': 'test WHICH TEMPLATE builds the class subspace: E_photo vs E_over',
                              'H1_target_paired': 'par(E_over) should win on rev_coco2dior (eval=DIOR)',
                              'H2_source_paired': 'par(E_over) should win on fwd_dior2coco (src=DIOR)'},
                   'per_det': out, 'elapsed_s': round(time.time() - t0, 1)},
                  open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f'[save] {det} -> {OUT}', flush=True)

    print('\n' + '=' * 96, flush=True)
    print('类轴模板 2x2: 增益 (均值, vs base)', flush=True)
    print('=' * 96, flush=True)
    summ = {}
    for det, cells in out.items():
        if not cells: continue
        summ[det] = {}
        for t in ['fwd_dior2coco', 'rev_coco2dior']:
            rr = [c for c in cells if c['tname'] == t]
            if not rr: continue
            row = {'n': len(rr), 'base': float(np.mean([c['base'] for c in rr]))}
            for k in VEC_KEYS:
                row[k + '_star'] = float(np.mean([c[k + '_star'] - c['base'] for c in rr]))
                row[k + '_g'] = float(np.mean([c[k + '_g'] - c['base'] for c in rr]))
            summ[det][t] = row
            print(f'\n【{DET_LABEL[det]}】{t}  n={len(rr)}  base={row["base"]:.3f}', flush=True)
            print(f'  {"方向":<22}{"oracle峰值增益":>16}{"门控增益":>14}', flush=True)
            for k in VEC_KEYS:
                print(f'  {VEC_DESC[k]:<22}{row[k+"_star"]:>+16.4f}{row[k+"_g"]:>+14.4f}', flush=True)
        if 'fwd_dior2coco' in summ[det] and 'rev_coco2dior' in summ[det]:
            f, r = summ[det]['fwd_dior2coco'], summ[det]['rev_coco2dior']
            h1 = r['par_ov_star'] > r['par_ph_star'] and f['par_ov_star'] < f['par_ph_star']
            h2 = f['par_ov_star'] > f['par_ph_star']
            print(f'\n  ★ H1(配目标域) 读数: rev par_ov {r["par_ov_star"]:+.4f} vs par_ph {r["par_ph_star"]:+.4f}; '
                  f'fwd par_ov {f["par_ov_star"]:+.4f} vs par_ph {f["par_ph_star"]:+.4f}  -> {"成立" if h1 else "不成立"}', flush=True)
            print(f'  ★ H2(配源域)   读数: fwd par_ov {f["par_ov_star"]:+.4f} vs par_ph {f["par_ph_star"]:+.4f}  '
                  f'-> {"成立" if h2 else "不成立"}', flush=True)
            print(f'  ★ 反向 TGE 模板: tge_ov {r["tge_ov_star"]:+.4f} vs tge_ph {r["tge_ph_star"]:+.4f} '
                  f'(旧读数 +0.0043)', flush=True)

    prev = {}
    if OUT.exists():
        try: prev = json.load(open(OUT, encoding='utf-8'))
        except Exception: prev = {}
    prev.update({'summary': summ, 'per_det': out, 'elapsed_s': round(time.time() - t0, 1)})
    if 'config' not in prev:
        prev['config'] = {'beta_grid': BETA_GRID, 'lam': LAM, 'zero_target_gt': True}
    json.dump(prev, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
