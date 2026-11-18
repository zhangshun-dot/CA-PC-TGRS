# -*- coding: utf-8 -*-
"""probe_dir_decomp.py — 方向分解: 权重空间里把 d_c 拆成【类判别分量】与【正交残差(观测/模板)分量】

元的规格 (2026-10-11, 预先注册):
  1. 类判别子空间 = span{ t_photo(cls) - t_bar }, t_bar = unit(mean_cls t_photo)
     d_c 正交投影 -> d_c// (类分量) 与 d_c^perp = d_c - d_c// (观测/模板分量)
  2. 注入向量【范数全部匹配到 ||d_c||】, 同一 β 网格 + 同一门控:
       full  = d_c
       par   = ||d_c|| * unit(d_c//)
       perp  = ||d_c|| * unit(d_c^perp)
       tge   = ||d_c|| * unit(t_photo(eval) - t_bar)
     另加两条与论文分解口径对齐的交叉校验向量 (同样范数匹配):
       vocab = ||d_c|| * unit(t_over(eval) - t_over(src))    # 论文 d_vocab
       cond  = ||d_c|| * unit(t_photo(eval) - t_over(eval))  # 论文 d_cond
  3. 两个检测器 gdino + owlv2 (避免单检测器结论)
  4. 判读预先注册:
       perp 边际增益 > 0 且两检测器同号  => 观测项有独立 AP 贡献
       par ≈ full 且 perp ≈ 0            => 观测项死重, 方向效率病根坐实

红线: 纯 B-pipeline 零目标 GT; 打分/评估与 run_sota_tge.py 逐字同源。
绝不写 run_sota_tge.json (产物独立: probe_dir_decomp.json)。
用法: HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u probe_dir_decomp.py --dets gdino owlv2
"""
import os, sys, json, pickle, glob, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from pathlib import Path
import run_bidirectional as RB

RB.DEVICE = 'cpu'
BETA = RB.BETA; LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
VEC_KEYS = ['full', 'par', 'perp', 'tge', 'vocab', 'cond']
VEC_DESC = {'full': 'd_c (full)', 'par': 'd_c// (class)', 'perp': 'd_c_perp (obs/template)',
            'tge': 'TGE dir', 'vocab': 'paper d_vocab', 'cond': 'paper d_cond'}

DETECTORS = {
    'gdino':  Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'),
    'owlv2':  Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache'),
    'yolo':   Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache'),
    'rtdetr': Path(r'C:\Users\张三\Desktop\实验数据\probe_rtdetr_cache'),
}
DET_LABEL = {'gdino': 'grounding-tiny', 'owlv2': 'OWL-ViT', 'yolo': 'YOLO-style', 'rtdetr': 'RT-DETR'}
OUT = Path(r'C:\Users\张三\Desktop\实验数据\probe_dir_decomp.json')


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

    # ── 源原型池 ──
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

    # ── 文本原型 + 类判别子空间 ──
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    idx = {c: i for i, c in enumerate(all_names)}
    t_bar = unit(E_photo.mean(0))
    Bmat = np.stack([E_photo[idx[c]] - t_bar for c in all_names], axis=1)   # 512 x n_cls
    U, S, _ = np.linalg.svd(Bmat, full_matrices=False)
    U = U[:, S > S[0] * 1e-8]
    P = U @ U.T
    print(f'[subspace] 类判别子空间 rank={U.shape[1]} (of {len(all_names)} cls), 投影器 {P.shape}', flush=True)

    # ── 门控尺度 (与 tab:det / run_sota_tge.py 同源) ──
    gcfg = json.load(open(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json'))['config']['gate']
    PRIOR_SCALE, c0 = gcfg['prior_scale'], gcfg['c0']
    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    out = {}
    if OUT.exists():                      # 合并写入: 子集重跑不截断其它检测器
        try: out = json.load(open(OUT, encoding='utf-8')).get('per_det', {})
        except Exception: out = {}
    for det in args.dets:
        if det not in DETECTORS:
            continue
        cache_dir = DETECTORS[det]
        cells = []
        for tname in RB.TASKS:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cache_dir, tname, eval_cls)
                if blob is None or src_cls not in bank[src_dom]:
                    print(f'  !! {det}/{tname}/{eval_cls} 缺缓存', flush=True)
                    continue
                per_img = blob['per_img']; F = blob['F']
                P_src = unit(bank[src_dom][src_cls].numpy())
                P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())

                dc = E_photo[idx[eval_cls]] - E_over[idx[src_cls]]
                mag = float(np.linalg.norm(dc))
                par = P @ dc
                perp = dc - par
                # 全部范数匹配到 ||dc||
                Vs = {
                    'full':  dc,
                    'par':   unit(par) * mag if np.linalg.norm(par) > 1e-9 else np.zeros_like(dc),
                    'perp':  unit(perp) * mag if np.linalg.norm(perp) > 1e-9 else np.zeros_like(dc),
                    'tge':   unit(E_photo[idx[eval_cls]] - t_bar) * mag,
                    'vocab': unit(E_over[idx[eval_cls]] - E_over[idx[src_cls]]) * mag,
                    'cond':  unit(E_photo[idx[eval_cls]] - E_over[idx[eval_cls]]) * mag,
                }
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))

                cell = {'det': det, 'tname': tname, 'src': src_cls, 'eval': eval_cls,
                        'n_img': len(per_img), 'dc_mag': round(mag, 4),
                        'par_frac': round(float(np.linalg.norm(par) / max(mag, 1e-12)), 4),
                        'perp_frac': round(float(np.linalg.norm(perp) / max(mag, 1e-12)), 4),
                        'gamma': round(gamma, 4), 'max_conf': round(max_conf, 4)}
                # base
                b50, bm = eval_pair(per_img, F, unit(P_src) - LAM * P_neg)
                cell['base'], cell['base_m'] = round(b50, 4), round(bm, 4)
                for k in VEC_KEYS:
                    v = Vs[k]
                    curve50, curven = [], []
                    for bb in BETA_GRID:
                        a50, am = eval_pair(per_img, F, unit(P_src + bb * v) - LAM * P_neg)
                        curve50.append(round(a50, 4)); curven.append(round(am, 4))
                    g50, gm = eval_pair(per_img, F, unit(P_src + gamma * v) - LAM * P_neg)
                    cell[k + '_curve'] = curve50
                    cell[k + '_curvem'] = curven
                    cell[k + '_star'] = max(curve50)
                    cell[k + '_star_beta'] = BETA_GRID[int(np.argmax(curve50))]
                    cell[k + '_g'] = round(g50, 4); cell[k + '_g_m'] = round(gm, 4)
                cells.append(cell)
                print(f'  [{det}] {src_cls}->{eval_cls}  base={b50:.4f} '
                      f'par_frac={cell["par_frac"]:.2f} perp_frac={cell["perp_frac"]:.2f} '
                      f'| full*={cell["full_star"]:.4f} par*={cell["par_star"]:.4f} '
                      f'perp*={cell["perp_star"]:.4f} tge*={cell["tge_star"]:.4f}', flush=True)
        out[det] = cells
        # 增量落盘 (每检测器写完就存, 防长跑丢失)
        json.dump({'config': {'beta_grid': BETA_GRID, 'lam': LAM, 'neg_topk': NEG_TOPK,
                              'n_proto': N_PROTO, 'zero_target_gt': True,
                              'note': 'all injection vectors norm-matched to ||d_c||; same beta grid + same gate; '
                                      'only the DIRECTION differs',
                              'judgement_prereg': {'perp_independent': 'perp marginal gain > 0 and same sign on both detectors',
                                                   'perp_dead': 'par ~= full and perp ~= 0'}},
                   'per_det': out, 'elapsed_s': round(time.time() - t0, 1)},
                  open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f'[save] {det} 已写 {OUT}', flush=True)

    # ── 汇总 + 预注册判读 ──
    print('\n' + '=' * 104, flush=True)
    print('方向分解: 范数匹配到 ||d_c|| 后, 各方向在同一 β 网格/同一门控下的宏平均增益 (vs base)', flush=True)
    print('=' * 104, flush=True)
    summary = {}
    for det, cells in out.items():
        if not cells:
            continue
        m = lambda k: float(np.mean([c[k] for c in cells]))
        base = m('base')
        row = {'base': base, 'n_cells': len(cells)}
        for k in VEC_KEYS:
            row[k + '_star'] = m(k + '_star') - base      # oracle β 峰值增益
            row[k + '_g'] = m(k + '_g') - base            # 门控(零GT 部署口径)增量
            row[k + '_star_abs'] = m(k + '_star')
            row[k + '_g_abs'] = m(k + '_g')
        row['par_frac'] = m('par_frac'); row['perp_frac'] = m('perp_frac'); row['dc_mag'] = m('dc_mag')
        summary[det] = row
        print(f'\n【{DET_LABEL[det]}】 base={base:.4f}  n={len(cells)}  '
              f'类分量占比 par_frac={row["par_frac"]:.3f} / 残差占比 perp_frac={row["perp_frac"]:.3f}', flush=True)
        print(f'  {"方向":<24}{"oracle峰值增益":>16}{"门控增益(零GT)":>16}', flush=True)
        for k in VEC_KEYS:
            print(f'  {VEC_DESC[k]:<24}{row[k+"_star"]:>+16.4f}{row[k+"_g"]:>+16.4f}', flush=True)
        # 预注册判读
        pf = row['perp_frac']
        print(f'  判读: perp oracle 增益 {row["perp_star"]:+.4f} | par {row["par_star"]:+.4f} | '
              f'full {row["full_star"]:+.4f} | tge {row["tge_star"]:+.4f}', flush=True)

    if len(summary) >= 2:
        ds = list(summary.keys())
        a, b = summary[ds[0]], summary[ds[1]]
        print('\n' + '-' * 104, flush=True)
        print('★ 预注册判据读数:', flush=True)
        perp_sign_same = (np.sign(a['perp_star']) == np.sign(b['perp_star'])) and a['perp_star'] != 0
        print(f'  (i) perp 边际增益 > 0 且跨检测器同号?  {ds[0]} {a["perp_star"]:+.4f} / '
              f'{ds[1]} {b["perp_star"]:+.4f}  ->  {"是" if (a["perp_star"]>0 and b["perp_star"]>0 and perp_sign_same) else "否"}', flush=True)
        par_close = all(abs(summary[d]['par_star'] - summary[d]['full_star']) < 0.002 for d in ds)
        perp_zero = all(abs(summary[d]['perp_star']) < 0.002 for d in ds)
        print(f'  (ii) par ≈ full 且 perp ≈ 0?  par-full: '
              f'{abs(a["par_star"]-a["full_star"]):.4f}/{abs(b["par_star"]-b["full_star"]):.4f}; '
              f'|perp|: {abs(a["perp_star"]):.4f}/{abs(b["perp_star"]):.4f}  ->  {"是" if (par_close and perp_zero) else "否"}', flush=True)
        print(f'  参考: 论文口径交叉校验 vocab {a["vocab_star"]:+.4f}/{b["vocab_star"]:+.4f} vs '
              f'cond {a["cond_star"]:+.4f}/{b["cond_star"]:+.4f}', flush=True)
        # TGE 对比
        print(f'  TGE 方向: {a["tge_star"]:+.4f}/{b["tge_star"]:+.4f}  vs full {a["full_star"]:+.4f}/{b["full_star"]:+.4f}', flush=True)

    json.dump({'config': {'beta_grid': BETA_GRID, 'lam': LAM, 'neg_topk': NEG_TOPK,
                          'n_proto': N_PROTO, 'zero_target_gt': True,
                          'note': 'all injection vectors norm-matched to ||d_c||; same beta grid + same gate; '
                                  'only the DIRECTION differs'},
               'summary': summary, 'per_det': out, 'elapsed_s': round(time.time() - t0, 1)},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
