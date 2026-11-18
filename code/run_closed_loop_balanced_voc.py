# -*- coding: utf-8 -*-
"""run_closed_loop_balanced_voc.py — DIOR→VOC 的 balanced CA-PC 闭环 (与 DIOR→COCO 同协议)

目的 (元要求, 2026-10-07): 现有 tge_probe_voc.json 是 TGE 前身配置, 与 balanced CA-PC 口径不统一,
导致"偏移梯度"主图协议不一致。本脚本用与 run_closed_loop_balanced.py 完全相同的 balanced 闭环
(dc × gate_conf 双因子零GT先验 + 2×2 消融 + bootstrap) 在 DIOR→VOC 上重跑, 使 shift-ladder
(DIOR→COCO +0.0497 / DIOR→VOC ? / DIOR→VOC 中偏移) 协议统一。

源域 = DIOR (复用 RB.DIOR_POOL 原型池); 目标域 = VOC2007 (检测缓存由 tge_voc_dets_to_closedloop.py
从 TGE 已有 GDINO 检测重建, per_img/F 同构 probe_bidir_cache)。评测类 = 5 个 VOC 类
(aeroplane/boat/car/bus/motorbike), 映射 DIOR 源 (airplane/ship/vehicle)。

小样本探针 N=20/类 (与 RT-DETR 探针同规); 验证协议一致后再扩全量。

用法 (CPU 重算 AP, 不重跑 GPU 检测):
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu ^
  "C:/Users/张三/GroundingDINO/venv/Scripts/python.exe" -u run_closed_loop_balanced_voc.py
"""
import os, sys, json, pickle, glob, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('AC_SPC_DET_DEVICE', 'cpu')
os.environ.setdefault('AC_SPC_DIOR_IMG', r'C:\Users\张三\modeling\dior_data\eval_v2_images')
os.environ.setdefault('AC_SPC_DIOR_PARQUET', r'C:\Users\张三\modeling\dior_data\dior_combined.parquet')
sys.path.insert(0, r'C:\Users\张三\Desktop\实验代码')
sys.path.insert(0, r'C:\Users\张三\Desktop\A-CSPC')

import numpy as np
import torch
from scipy.stats import spearmanr
from pathlib import Path

import run_bidirectional as RB

RB.DEVICE = 'cpu'
DEVICE = 'cpu'
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LAM = RB.LAM
NEG_TOPK = RB.NEG_TOPK
N_PROTO = RB.N_PROTO
# bootstrap 重采样次数: 默认 20 (历史值); 通过 VOC_BOOT_K 覆盖 (2026-10-14 统一到 1000)
KBOOT = int(os.environ.get('VOC_BOOT_K', '20'))
# 评测图像数/类: 历史值见 blob 后缀; 通过 VOC_EVAL_N 覆盖 (历史发表配置 = 100)
EVAL_N = int(os.environ.get('VOC_EVAL_N', '100'))

# VOC 评测类 (eval_cls, src_cls DIOR)
VOC_POOL = ['aeroplane', 'boat', 'car', 'bus', 'motorbike']
VOC_PAIRS = [('airplane', 'aeroplane'), ('ship', 'boat'),
             ('vehicle', 'car'), ('vehicle', 'bus'), ('vehicle', 'motorbike')]
TASKS = {
    'fwd_dior2voc': ('dior', 'voc', VOC_PAIRS),
}

DETECTORS = {
    'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_voc_cache'),
}
# VOC 探针阶段仅 gdino (主检测器); gate 尺度只由 gdino 定出。held-out 多检测器 VOC 为后续扩展。
GATE_FROM = ['gdino']
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced_voc.json')


def unit(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(cache_dir, tname, eval_cls):
    hits = glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n{EVAL_N}.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_n*.pkl'))
    return pickle.load(open(hits[0], 'rb')) if hits else None


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

    # 源原型池 (CLIP GT crop, 检测器无关) — 仅 DIOR (目标域 VOC 不需要原型池, 闭环只用源域)
    bank = {}
    for domain, pool in (('dior', RB.DIOR_POOL),):
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
        print(f'[protos] {domain}: {len(b)} 类')

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL + VOC_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}

    data = {}
    pooled_feat = []
    for det, cache_dir in DETECTORS.items():
        cells = []
        for tname in TASKS:
            src_dom, tgt_dom, pairs = TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cache_dir, tname, eval_cls)
                if blob is None or src_cls not in bank[src_dom]:
                    continue
                per_img = blob['per_img']; F = blob['F']
                P_src = unit(bank[src_dom][src_cls].numpy())
                P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())
                dc = E_photo_d[eval_cls] - E_over_d[src_cls]
                base = ap_of(per_img, F, P_src - LAM * P_neg)
                best_b, best_ap = 0.0, -1
                for bb in BETA_GRID:
                    ap = ap_of(per_img, F, unit(P_src + bb * dc) - LAM * P_neg)
                    if ap > best_ap:
                        best_ap, best_b = ap, bb
                dc_mag = float(np.linalg.norm(dc))
                n_pred = sum(len(p['box']) for p in per_img)
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                score_var = float(np.mean([p['gd'].std() if len(p['gd']) > 1 else 0.0 for p in per_img]))
                gain = best_ap - base
                cells.append({'tname': tname, 'src': src_cls, 'eval': eval_cls, 'det': det,
                              'dc_mag': dc_mag, 'n_pred': n_pred, 'max_conf': max_conf, 'score_var': score_var,
                              'base': base, 'beta_star': best_b, 'ap_star': best_ap,
                              'gain': gain, 'ap_fixed2': ap_of(per_img, F, unit(P_src + 2.0 * dc) - LAM * P_neg),
                              '_per_img': per_img, '_F': F, '_P_src': P_src, '_P_neg': P_neg, '_dc': dc})
                if det in GATE_FROM:
                    pooled_feat.append({'det': det, 'dc_mag': dc_mag, 'n_pred': n_pred, 'max_conf': max_conf})
        data[det] = cells

    print('\n[诊断] 零GT特征 vs oracle β* / gain (gdino)')
    diag = {}
    for det, cells in data.items():
        y = np.array([c['beta_star'] for c in cells])
        g = np.array([c['gain'] for c in cells])
        diag[det] = {}
        for k in ('dc_mag', 'n_pred', 'max_conf', 'score_var'):
            v = np.array([c[k] for c in cells])
            rb, pb = spearmanr(v, y); rg, pg = spearmanr(v, g)
            diag[det][k] = {'rho_beta': float(rb), 'p_beta': float(pb),
                            'rho_gain': float(rg), 'p_gain': float(pg)}
            print(f'  [{det}] {k:<10} vs β*: ρ={rb:+.3f}(p={pb:.3f})  vs gain: ρ={rg:+.3f}(p={pg:.3f})')

    np_all = np.array([f['n_pred'] for f in pooled_feat])
    conf_all = np.array([f['max_conf'] for f in pooled_feat])
    n0 = float(np.median(np_all)); n_iqr = float(np.subtract(*np.percentile(np_all, [75, 25])) or 1.0)
    c0 = float(np.median(conf_all)); c_iqr = float(np.subtract(*np.percentile(conf_all, [75, 25])) or 0.1)
    c0_perdet = {det: float(np.median([c['max_conf'] for c in cells])) for det, cells in data.items()}
    PRIOR_SCALE = 2.0 / float(np.mean([c['dc_mag'] for cells in data.values() for c in cells]))

    def gate_count(n_pred):
        return float(np.clip(1.0 / (1.0 + np.exp((n_pred - n0) / (n_iqr / 2 + 1e-9))), 0.0, 1.0))

    def gate_conf(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - np.tanh(scale * (conf - c)), 0.0, 1.0))

    def gate_sigmoid(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 / (1.0 + np.exp(scale * (conf - c))), 0.0, 1.0))

    def gate_piece(conf, c=c0, scale=3.0):
        return float(np.clip(1.0 - scale * (conf - c), 0.0, 1.0))

    def gamma(cell, mode):
        if mode == 'dc':
            return float(np.clip(PRIOR_SCALE * cell['dc_mag'], 0.0, 4.0))
        if mode == 'dc_x_count':
            return float(np.clip(PRIOR_SCALE * cell['dc_mag'] * gate_count(cell['n_pred']), 0.0, 4.0))
        if mode == 'dc_x_conf':
            return float(np.clip(PRIOR_SCALE * cell['dc_mag'] * gate_conf(cell['max_conf'], c0, 3.0), 0.0, 4.0))
        if mode == 'dc_x_conf_perdet':
            return float(np.clip(PRIOR_SCALE * cell['dc_mag'] * gate_conf(cell['max_conf'], c0_perdet[cell['det']], 3.0), 0.0, 4.0))
        if mode == 'dc_x_sigmoid':
            return float(np.clip(PRIOR_SCALE * cell['dc_mag'] * gate_sigmoid(cell['max_conf'], c0, 3.0), 0.0, 4.0))
        if mode == 'dc_x_piece':
            return float(np.clip(PRIOR_SCALE * cell['dc_mag'] * gate_piece(cell['max_conf'], c0, 3.0), 0.0, 4.0))
        if mode == 'gate_only':
            return float(np.clip(PRIOR_SCALE * gate_conf(cell['max_conf'], c0, 3.0), 0.0, 4.0))
        raise ValueError(mode)

    modes = ['dc', 'dc_x_count', 'dc_x_conf', 'dc_x_conf_perdet', 'dc_x_sigmoid',
             'dc_x_piece', 'gate_only']

    def bootstrap_det(items, K=20, seed=0):
        rng = np.random.default_rng(seed)
        means = []
        for _ in range(K):
            cell_aps = []
            for (per_img, F, W) in items:
                n = len(per_img)
                idx = rng.integers(0, n, n)
                rows, gts = [], []
                for i in idx:
                    p = per_img[i]
                    if len(p['gd']):
                        w = F[p['crop_idx']] @ W
                        s = np.sqrt(p['gd'] * np.clip(np.maximum(0.0, w), 1e-6, 1.0))
                    else:
                        s = np.zeros(0)
                    rows.append((p['box'], s)); gts.append(p['gt'])
                cell_aps.append(RB.Met.compute_ap50_img(rows, gts))
            means.append(float(np.mean(cell_aps)))
        means = np.array(means)
        return float(means.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    print('\n' + '=' * 84)
    print('DIOR→VOC balanced CA-PC 闭环 (零GT; gate 尺度来自 gdino pooled 分布)')
    print('=' * 84)
    summary = {}
    cells_detail = []
    for det, cells in data.items():
        print(f'\n[{det}]  n={len(cells)}')
        print(f'  {"先验":<12}{"AP50":>9}{"vs固定β":>10}{"vs基线":>10}{"oracle%":>9}')
        row = {}
        ap_star = float(np.mean([c['ap_star'] for c in cells]))
        ap_fix = float(np.mean([c['ap_fixed2'] for c in cells]))
        ap_base = float(np.mean([c['base'] for c in cells]))
        for c in cells:
            c['_W'] = {}
            for m in modes:
                W = unit(c['_P_src'] + gamma(c, m) * c['_dc']) - LAM * c['_P_neg']
                c['_W'][m] = W
                c['_ap_' + m] = ap_of(c['_per_img'], c['_F'], W)
        for m in modes:
            aps = [c['_ap_' + m] for c in cells]
            apm = float(np.mean(aps))
            parity_fix = apm >= ap_fix - 1e-9
            parity_base = apm >= ap_base - 1e-9
            row[m] = {'ap': apm, 'vs_fix': apm - ap_fix, 'vs_base': apm - ap_base,
                      'oracle_ratio': apm / ap_star if ap_star > 0 else None,
                      'parity': bool(parity_fix and parity_base)}
            print(f'  {m:<12}{apm:>9.4f}{apm-ap_fix:>+10.4f}{apm-ap_base:>+10.4f}{apm/ap_star*100:>8.1f}%')
        for m in modes:
            bm, blo, bhi = bootstrap_det([(c['_per_img'], c['_F'], c['_W'][m]) for c in cells], K=KBOOT)
            row[m]['boot_mean'] = bm
            row[m]['ci95'] = [blo, bhi]
        print(f'  {"oracle":<12}{ap_star:>9.4f}{"--":>10}{"--":>10}{"100.0%":>9}')
        print(f'  {"固定β=2.0":<12}{ap_fix:>9.4f}{"--":>10}{ap_fix-ap_base:>+10.4f}')
        print(f'  {"基线β=0":<12}{ap_base:>9.4f}')
        summary[det] = {'ap_star': ap_star, 'ap_fix': ap_fix, 'ap_base': ap_base, 'modes': row}
        for c in cells:
            cells_detail.append({
                'det': det, 'direction': 'fwd', 'src': c['src'], 'eval': c['eval'],
                'dc_mag': c['dc_mag'], 'n_pred': c['n_pred'], 'max_conf': c['max_conf'],
                'score_var': c['score_var'], 'beta_star': c['beta_star'], 'ap_star': c['ap_star'],
                'base': c['base'], 'ap_fixed2': c['ap_fixed2'],
                'ap_dc': c['_ap_dc'], 'ap_dc_x_conf': c['_ap_dc_x_conf'],
                'ap_gate_only': c['_ap_gate_only'],
                'oracle_ratio_dc_x_conf': (c['_ap_dc_x_conf'] / c['ap_star']) if c['ap_star'] > 0 else None,
                'gain_dc_x_conf_vs_base': c['_ap_dc_x_conf'] - c['base'],
            })

    json.dump({'config': {'beta_grid': BETA_GRID, 'lam': LAM, 'zero_target_gt': True,
                          'gate': {'n0': n0, 'n_iqr': n_iqr, 'c0': c0, 'c_iqr': c_iqr,
                                   'prior_scale': PRIOR_SCALE}, 'bootstrap_K': KBOOT,
                          'task': 'fwd_dior2voc', 'n_per_class': N_PROTO, 'n_eval_per_class': EVAL_N,
                          'detectors': list(DETECTORS), 'gate_from': GATE_FROM},
               'diag': diag,
               'summary': {d: {'ap_star': summary[d]['ap_star'], 'ap_fix': summary[d]['ap_fix'],
                               'ap_base': summary[d]['ap_base'],
                               'modes': {m: summary[d]['modes'][m] for m in modes}} for d in DETECTORS},
               'cells_detail': cells_detail,
               'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
