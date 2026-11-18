# -*- coding: utf-8 -*-
"""run_reanchor.py — 锚点域一致性检验: d_c 的两个文本锚是否各自用了【自己那个域】的模板?

诊断 (读管线时发现): run_bidirectional / run_closed_loop_multidomain_v2 里
    dc = E_photo[eval_cls] - E_over[src_cls]        # 硬编码
  正向 (src=遥感/over, tgt=COCO/photo) 时正确;
  反向 (src=COCO/photo, tgt=遥感/over) 时 **两个锚全部错域** -> 反向 par 归零、tge 塌。

修法 (域模板一致):  dc_fix = E_{tgt模板}[eval_cls] - E_{src模板}[src_cls]
  正向 dc_fix == dc_old (内置恒等校验); 反向才是真正的对照。

向量 (同一 β 网格 0~4, 同一门控, 零目标 GT):
  dc_old    = E_photo[eval] - E_over[src]                   (现行)
  dc_fix    = E_{tgt}[eval] - E_{src}[src]                  (域一致, 自然范数)
  dc_fix_nm = ||dc_old|| * unit(dc_fix)                     (范数匹配, 排除步长混淆)
  par_fix   = ||dc_old|| * unit(P_{tgt} @ dc_fix)           (修锚 + 修子空间)
  tge_fix   = ||dc_old|| * unit(E_{tgt}[eval] - t_bar_{tgt})  (TGE 同域模板版)

产物: run_reanchor_{det}.json + 合并 run_reanchor.json (绝不写其它产物)
用法: HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu "C:/Users/张三/vg/Scripts/python.exe" -u run_reanchor.py --dets gdino owlv2
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
LAM = RB.LAM; NEG_TOPK = RB.NEG_TOPK; N_PROTO = RB.N_PROTO
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
VEC_KEYS = ['dc_old', 'dc_fix', 'dc_fix_nm', 'par_fix', 'tge_fix']
VEC_DESC = {'dc_old': 'd_c 现行(photo-eval/over-src)', 'dc_fix': 'd_c 域一致(自然范数)',
            'dc_fix_nm': 'd_c 域一致(范数匹配)', 'par_fix': 'par 修锚+修子空间',
            'tge_fix': 'TGE 同域模板版'}
DETECTORS = {'gdino': Path(r'C:/Users/张三/Desktop/实验数据/probe_bidir_cache'),
             'owlv2': Path(r'C:/Users/张三/Desktop/实验数据/probe_owl_cache')}
DET_LABEL = {'gdino': 'grounding-tiny', 'owlv2': 'OWL-ViT'}
OUT = Path(r'C:/Users/张三/Desktop/实验数据/run_reanchor.json')


def unit(x):
    x = np.asarray(x, float); n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(cache_dir, tname, eval_cls):
    hits = glob.glob(str(cache_dir / f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


def eval_pair(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return (float(RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])),
            float(RB.Met.compute_ap_coco(rows, [p['gt'] for p in per_img])))


def main():
    ap_ = argparse.ArgumentParser(); ap_.add_argument('--dets', nargs='+', default=['gdino', 'owlv2'])
    args = ap_.parse_args(); t0 = time.time()
    cfg = RB.Config(); cfg.model.device = 'cpu'
    cfg.model.detection_threshold = 0.0; cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops: continue
            Ft = RB.clip_encode_imgs(crops)
            pt = torch.from_numpy(Ft.mean(0)); b[cls] = pt / pt.norm()
        bank[domain] = b
        print(f'[protos] {domain}: {len(b)} 类', flush=True)

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL)); idx = {c: i for i, c in enumerate(all_names)}
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])

    def build_P(Et):
        tb = unit(Et.mean(0)); B = np.stack([Et[idx[c]] - tb for c in all_names], axis=1)
        U, S, _ = np.linalg.svd(B, full_matrices=False); U = U[:, S > S[0] * 1e-8]
        return U @ U.T, tb
    P_ph, tbar_ph = build_P(E_photo); P_ov, tbar_ov = build_P(E_over)
    TMPL = {'coco': ('photo', E_photo, P_ph, tbar_ph), 'dior': ('over', E_over, P_ov, tbar_ov)}

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
            _, E_src_t, P_tgt, tbar_tgt = TMPL[src_dom][0], TMPL[src_dom][1], TMPL[tgt_dom][2], TMPL[tgt_dom][3]
            E_tgt_t = TMPL[tgt_dom][1]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cache_dir, tname, eval_cls)
                if blob is None or src_cls not in bank[src_dom]:
                    print(f'  !! {det}/{tname}/{eval_cls} 缺缓存', flush=True); continue
                per_img = blob['per_img']; F = blob['F']
                P_src = unit(bank[src_dom][src_cls].numpy())
                P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())
                dc_old = E_photo[idx[eval_cls]] - E_over[idx[src_cls]]
                dc_fix = E_tgt_t[idx[eval_cls]] - E_src_t[idx[src_cls]]
                mag = float(np.linalg.norm(dc_old)); magf = float(np.linalg.norm(dc_fix))
                pf = P_tgt @ dc_fix
                Vs = {'dc_old': dc_old, 'dc_fix': dc_fix,
                      'dc_fix_nm': unit(dc_fix) * mag,
                      'par_fix': unit(pf) * mag if np.linalg.norm(pf) > 1e-9 else np.zeros_like(dc_fix),
                      'tge_fix': unit(E_tgt_t[idx[eval_cls]] - tbar_tgt) * mag}
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                gamma = float(np.clip(PRIOR_SCALE * mag * gate_conf(max_conf), 0.0, 4.0))
                cell = {'det': det, 'tname': tname, 'src': src_cls, 'eval': eval_cls, 'n_img': len(per_img),
                        'src_tmpl': TMPL[src_dom][0], 'tgt_tmpl': TMPL[tgt_dom][0],
                        'mag_old': round(mag, 4), 'mag_fix': round(magf, 4),
                        'cos_old_fix': round(float(np.dot(unit(dc_old), unit(dc_fix))), 4),
                        'gamma': round(gamma, 4), 'max_conf': round(max_conf, 4)}
                b50, bm = eval_pair(per_img, F, unit(P_src) - LAM * P_neg)
                cell['base'], cell['base_m'] = round(b50, 4), round(bm, 4)
                for k in VEC_KEYS:
                    v = Vs[k]; c50, cn = [], []
                    for bb in BETA_GRID:
                        a50, am = eval_pair(per_img, F, unit(P_src + bb * v) - LAM * P_neg)
                        c50.append(round(a50, 4)); cn.append(round(am, 4))
                    g50, gm = eval_pair(per_img, F, unit(P_src + gamma * v) - LAM * P_neg)
                    cell[k + '_curve'] = c50; cell[k + '_curvem'] = cn
                    cell[k + '_star'] = max(c50); cell[k + '_star_beta'] = BETA_GRID[int(np.argmax(c50))]
                    cell[k + '_g'] = round(g50, 4); cell[k + '_g_m'] = round(gm, 4)
                cells.append(cell)
                print(f'  [{det}] {tname[:3]} {src_cls}->{eval_cls} base={b50:.4f} '
                      f'|dc| {mag:.3f}->{magf:.3f} cos={cell["cos_old_fix"]:+.2f} | '
                      f'old*={cell["dc_old_star"]:.4f} fix*={cell["dc_fix_star"]:.4f} '
                      f'fixnm*={cell["dc_fix_nm_star"]:.4f} par*={cell["par_fix_star"]:.4f} '
                      f'tgefix*={cell["tge_fix_star"]:.4f}', flush=True)
        out[det] = cells
        json.dump({'config': {'beta_grid': BETA_GRID, 'lam': LAM, 'zero_target_gt': True,
                              'note': 'anchor-domain consistency: each text anchor uses its OWN domain template'},
                   'per_det': out, 'elapsed_s': round(time.time() - t0, 1)},
                  open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        print(f'[save] {det} -> {OUT}', flush=True)

    print('\n' + '=' * 92, flush=True)
    summ = {}
    for det, cells in out.items():
        if not cells: continue
        summ[det] = {}
        for t in ['fwd_dior2coco', 'rev_coco2dior']:
            rr = [c for c in cells if c['tname'] == t]
            if not rr: continue
            row = {'n': len(rr), 'base': float(np.mean([c['base'] for c in rr])),
                   'cos_old_fix': float(np.mean([c['cos_old_fix'] for c in rr]))}
            for k in VEC_KEYS:
                row[k + '_star'] = float(np.mean([c[k + '_star'] - c['base'] for c in rr]))
                row[k + '_g'] = float(np.mean([c[k + '_g'] - c['base'] for c in rr]))
            summ[det][t] = row
            print(f'\n【{DET_LABEL[det]}】{t}  n={len(rr)}  base={row["base"]:.3f}  '
                  f'cos(dc_old,dc_fix)={row["cos_old_fix"]:+.3f}', flush=True)
            print(f'  {"向量":<30}{"oracle峰值增益":>16}{"门控增益":>14}', flush=True)
            for k in VEC_KEYS:
                print(f'  {VEC_DESC[k]:<30}{row[k+"_star"]:>+16.4f}{row[k+"_g"]:>+14.4f}', flush=True)
    prev = {}
    if OUT.exists():
        try: prev = json.load(open(OUT, encoding='utf-8'))
        except Exception: prev = {}
    prev.update({'summary': summ, 'per_det': out, 'elapsed_s': round(time.time() - t0, 1)})
    json.dump(prev, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
