# -*- coding: utf-8 -*-
"""run_closed_loop_cross_det.py — 跨检测器闭环验证（OWLv2）

复用 run_bidirectional 的全部共享逻辑（源原型 CLIP、文本锚点、CA-PC 应用、β 网格），
仅把检测缓存目录从 GDINO-tiny 的 probe_bidir_cache 换成 OWLv2 的 probe_owl_cache，
并把 load_blob 的命名匹配放宽（OWLv2 缓存文件无 off{N_PROTO} 段）。

闭环数学与检测器无关：源原型来自 CLIP GT crop、dc_mag 来自 CLIP 文本、CA-PC 作用于
检测器产出的 (box, gd, F) —— 同一套先验 R(c)=dc_mag 应跨检测器驱动 γ_c。
本脚本验证方向1(dc_mag 预测 OWLv2 每类增益)、方向3(零GT驱动达 OWLv2 oracle 比例)、
方向2(OWLv2 双向 CA-PC) 是否站得住。

用法 (CPU 即可，复用 OWLv2 缓存):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu ^
  "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_cross_det.py
"""
import os, sys, json, pickle, glob, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')

sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from scipy.stats import spearmanr
from pathlib import Path

import run_bidirectional as RB

RB.DEVICE = 'cpu'
DEVICE = 'cpu'
USE_FULL = True    # 全量（每单元缓存自带 n：OWLv2 fwd airplane=n77, 其余 n100）
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LAM = RB.LAM
NEG_TOPK = RB.NEG_TOPK
N_PROTO = RB.N_PROTO
CACHE_DIR = Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache')   # OWLv2 缓存
DETECTOR = 'owlv2-base-patch16-ensemble'
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_owlv2.json')


def unit(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(tname, eval_cls):
    # OWLv2 命名: {tname}_{eval_cls}_n*.pkl ；GDINO 命名: {tname}_{eval_cls}_off{N_PROTO}_n*.pkl
    hits = glob.glob(str(CACHE_DIR / f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(str(CACHE_DIR / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
    if not hits:
        return None
    return pickle.load(open(hits[0], 'rb'))


def ap_of(per_img, F, W):
    rows = []
    for p in per_img:
        if len(p['gd']):
            w = F[p['crop_idx']] @ W
            s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
        else:
            s = np.zeros(0)
        rows.append((p['box'], s))
    return RB.Met.compute_ap50_img(rows, [p['gt'] for p in per_img])


def main():
    t0 = time.time()
    cfg = RB.Config()
    cfg.model.device = DEVICE
    cfg.model.detection_threshold = 0.0
    cfg.model.pre_filter_threshold = RB.PRE_FILTER
    cfg.model.gdino_cache = 'IDEA-Research/grounding-dino-tiny'

    # ── 源原型池（CPU CLIP, GT crop；与检测器无关）──
    bank = {}
    for domain, pool in (('coco', RB.COCO_POOL), ('dior', RB.DIOR_POOL)):
        b = {}
        for cls in pool:
            ims, gts = RB.load_split(cfg, domain, cls, 0, N_PROTO)
            crops = RB.D.gt_crops(ims, gts)
            if not crops:
                continue
            Ft = RB.clip_encode_imgs(crops)
            pt = torch.from_numpy(Ft.mean(0))
            b[cls] = pt / pt.norm()
        bank[domain] = b
        print(f'[protos] {domain}: {len(b)} 类', flush=True)

    # ── 文本锚点（与检测器无关）──
    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}

    cells = []
    rows = []
    for tname in RB.TASKS:
        src_dom, tgt_dom, pairs = RB.TASKS[tname]
        for src_cls, eval_cls in pairs:
            blob = load_blob(tname, eval_cls)
            if blob is None or src_cls not in bank[src_dom]:
                print(f'  !! {tname}/{eval_cls} 无缓存/无原型', flush=True)
                continue
            per_img = blob['per_img']
            F = blob['F']
            P_src = unit(bank[src_dom][src_cls].numpy())
            P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())
            dc = E_photo_d[eval_cls] - E_over_d[src_cls]
            d_tgt = E_photo_d[eval_cls]
            base = ap_of(per_img, F, P_src - LAM * P_neg)

            best_b, best_ap = 0.0, -1
            for bb in BETA_GRID:
                W = unit(P_src + bb * dc) - LAM * P_neg
                ap = ap_of(per_img, F, W)
                if ap > best_ap:
                    best_ap, best_b = ap, bb

            dc_mag = float(np.linalg.norm(dc))
            cls_mig = float(np.linalg.norm(E_photo_d[eval_cls] - E_photo_d[src_cls]))
            dom_gap = float(np.linalg.norm(E_photo_d[eval_cls] - E_over_d[eval_cls]))
            src_align = float(P_src @ unit(dc))
            n_pred = sum(len(p['box']) for p in per_img)
            cand = {'dc_mag': dc_mag, 'cls_mig': cls_mig, 'dom_gap': dom_gap,
                    'src_align': src_align, 'n_pred_log': float(np.log(n_pred + 1))}
            gain = best_ap - base

            cells.append({'tname': tname, 'src': src_cls, 'eval': eval_cls,
                          'cand': cand, 'beta_star': best_b, 'gain': gain})
            rows.append({'tname': tname, 'src': src_cls, 'eval': eval_cls,
                         'baseline': round(base, 4), 'beta_star': best_b,
                         'ap_beta_star': round(best_ap, 4),
                         'ap_fixed2': round(ap_of(per_img, F, unit(P_src + 2.0 * dc) - LAM * P_neg), 4),
                         'dc_mag': round(dc_mag, 4), 'cls_mig': round(cls_mig, 4),
                         'dom_gap': round(dom_gap, 4), 'src_align': round(src_align, 3),
                         'n_pred': n_pred, 'gain': round(gain, 4)})
            print(f'  {tname} {src_cls}->{eval_cls}: base={base:.4f} β*={best_b} '
                  f'ap(β*)={best_ap:.4f} ap(β=2)={rows[-1]["ap_fixed2"]:.4f} '
                  f'dc_mag={dc_mag:.3f} cls_mig={cls_mig:.3f} dom_gap={dom_gap:.3f} '
                  f'align={src_align:+.3f} gain={gain:+.4f}', flush=True)

    # ── 诊断：各零GT候选量 vs β*(oracle) / gain ──
    y = np.array([c['beta_star'] for c in cells])
    gains = np.array([c['gain'] for c in cells])
    cand_names = list(cells[0]['cand'].keys())
    print('\n[诊断] 零GT候选量 vs β*(oracle) / gain 的 Spearman (@ OWLv2):')
    corr_beta, corr_gain = {}, {}
    for k in cand_names:
        v = np.array([c['cand'][k] for c in cells])
        rb, pb = spearmanr(v, y)
        rg, pg = spearmanr(v, gains)
        corr_beta[k] = (rb, pb)
        corr_gain[k] = (rg, pg)
        print(f'  {k:<12} vs β*: ρ={rb:+.3f}(p={pb:.3f})   vs gain: ρ={rg:+.3f}(p={pg:.3f})')
    top2 = sorted(cand_names, key=lambda k: abs(corr_beta[k][0]) if np.isfinite(corr_beta[k][0]) else 0, reverse=True)[:2]
    print(f'[闭环] LOO 选用先验: {top2}')
    X = np.array([[c['cand'][k] for k in top2] for c in cells])
    beta_auto = np.zeros(len(cells))
    for i in range(len(cells)):
        Xtr, ytr = np.delete(X, i, 0), np.delete(y, i)
        if len(np.unique(ytr)) > 1:
            from sklearn.linear_model import LinearRegression
            m = LinearRegression().fit(Xtr, ytr)
            beta_auto[i] = float(np.clip(m.predict(X[i:i + 1])[0], 0.5, 4.0))
        else:
            beta_auto[i] = float(ytr.mean())

    PRIOR_SCALE = 2.0 / float(np.mean([c['cand']['dc_mag'] for c in cells]))
    for k, c in enumerate(cells):
        tname = c['tname']; src_cls = c['src']; eval_cls = c['eval']
        blob = load_blob(tname, eval_cls)
        per_img = blob['per_img']; F = blob['F']
        P_src = unit(bank[{'fwd_dior2coco': 'dior', 'rev_coco2dior': 'coco'}[tname]][src_cls].numpy())
        P_neg = unit(RB.Proto.get_negative_prototype(
            bank[{'fwd_dior2coco': 'dior', 'rev_coco2dior': 'coco'}[tname]], src_cls, NEG_TOPK).numpy())
        dc = E_photo_d[eval_cls] - E_over_d[src_cls]
        W = unit(P_src + beta_auto[k] * dc) - LAM * P_neg
        rows[k]['beta_auto'] = round(beta_auto[k], 3)
        rows[k]['ap_beta_auto'] = round(ap_of(per_img, F, W), 4)
        beta_prior = float(np.clip(PRIOR_SCALE * c['cand']['dc_mag'], 0.5, 4.0))
        Wp = unit(P_src + beta_prior * dc) - LAM * P_neg
        rows[k]['beta_prior'] = round(beta_prior, 3)
        rows[k]['ap_beta_prior'] = round(ap_of(per_img, F, Wp), 4)

    rho_dc_gain, p_dc_gain = corr_gain['dc_mag']
    rho, p = spearmanr(beta_auto, y)
    ap_auto = np.mean([r['ap_beta_auto'] for r in rows])
    ap_prior = np.mean([r['ap_beta_prior'] for r in rows])
    ap_base = np.mean([r['baseline'] for r in rows])
    ap_star = np.mean([r['ap_beta_star'] for r in rows])
    ap_fix = np.mean([r['ap_fixed2'] for r in rows])
    # 方向2：双向 CA-PC 在 β=2.0 的 mean(Δ over base)
    d2_fwd = np.mean([r['ap_fixed2'] - r['baseline'] for r in rows if r['tname'] == 'fwd_dior2coco'])
    d2_rev = np.mean([r['ap_fixed2'] - r['baseline'] for r in rows if r['tname'] == 'rev_coco2dior'])
    print('\n' + '=' * 80)
    print(f'闭环验证 @ {DETECTOR}（零目标域GT自动先验 → 驱动 γ=β_c → 双向 CA-PC）')
    print('=' * 80)
    print(f'[方向1] R(c)=dc_mag vs realized 增益 gain: Spearman ρ = {rho_dc_gain:+.4f} (p={p_dc_gain:.4f})')
    if np.isfinite(rho):
        print(f'[方向3] β_auto(LOO) vs β*(oracle): Spearman ρ = {rho:+.4f} (p={p:.4f})')
    print(f'AP50 均值:  β_prior(γ∝dc_mag) = {ap_prior:.4f} | β_auto(LOO) = {ap_auto:.4f} '
          f'| β*(oracle) = {ap_star:.4f} | 固定β=2.0 = {ap_fix:.4f} | 基线β=0 = {ap_base:.4f}')
    print(f'β_prior vs 固定β=2.0 = {ap_prior-ap_fix:+.4f}   β_prior 达 oracle 比例 = {ap_prior/ap_star*100:.1f}%')
    print(f'[方向2] 双向 CA-PC @β=2.0 mean(Δ base): fwd = {d2_fwd:+.4f} | rev = {d2_rev:+.4f}')
    print('\n逐格:')
    for r in rows:
        ba = r.get('beta_auto', '-'); aa = r.get('ap_beta_auto', '-')
        bp = r.get('beta_prior', '-'); ap = r.get('ap_beta_prior', '-')
        print(f"  {r['tname']:<14} {r['src']:<10}->{r['eval']:<10} "
              f"β*={r['beta_star']:<4} β_prior={bp!s:<5}({ap}) β_auto={ba!s:<5}({aa}) "
              f"ap*={r['ap_beta_star']:.4f} ap2={r['ap_fixed2']:.4f} "
              f"dc_mag={r['dc_mag']:.3f} gain={r['gain']:+.4f}")

    json.dump({'config': {'detector': DETECTOR, 'n_sub': None, 'beta_grid': BETA_GRID,
                          'lam': LAM, 'zero_target_gt': True, 'cache': str(CACHE_DIR)},
               'spearman_rho_prior_gain': (float(rho_dc_gain) if np.isfinite(rho_dc_gain) else None),
               'spearman_p_prior_gain': (float(p_dc_gain) if np.isfinite(p_dc_gain) else None),
               'spearman_rho_beta': (float(rho) if np.isfinite(rho) else None),
               'spearman_p_beta': (float(p) if np.isfinite(p) else None),
               'cand_corr_beta': {k: [float(rb), float(pb)] for k, (rb, pb) in corr_beta.items()},
               'cand_corr_gain': {k: [float(rg), float(pg)] for k, (rg, pg) in corr_gain.items()},
               'loo_top2': top2,
               'ap_beta_auto': float(ap_auto), 'ap_beta_star': float(ap_star),
               'ap_beta_prior': float(ap_prior), 'ap_base': float(ap_base),
               'ap_fixed2': float(ap_fix), 'oracle_ratio': (float(ap_prior / ap_star) if ap_star > 0 else None),
               'dir2_fwd_delta': float(d2_fwd), 'dir2_rev_delta': float(d2_rev),
               'rows': rows, 'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
