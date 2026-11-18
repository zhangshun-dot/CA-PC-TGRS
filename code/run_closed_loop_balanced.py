# -*- coding: utf-8 -*-
"""run_closed_loop_balanced.py — 跨检测器平衡的零GT先验诊断与验证

动机（元要求"方法要改善、找到跨检测器的平衡"）：
  OWLv2 上纯 dc_mag 单调先验 γ∝dc_mag 退化（方向3 低于固定β与基线、方向2 反向转负）。
  根因假设：CA-PC 外推收益取决于"检测器还有多少 headroom"——强检测器(OWLv2)已饱和，
  外推近 0 甚至有害；弱检测器(GDINO-tiny) headroom 大，外推正收益。
  故把 γ 从单因子域差先验，改为 双因子平衡：γ = dc_mag(域差需要) × gate(检测器余量)。
  gate 用**零GT**的目标域预测统计（候选框数 n_pred / 平均最大置信度 conf）——检测器越自信越饱和→gate↓→γ↓。

做法：复用两个检测缓存(GDINO probe_bidir_cache / OWLv2 probe_owl_cache)，从缓存 F 重算每单元 β 网格 AP50
（CPU, 不重跑 GPU 检测），得到 oracle β* 与各零GT特征；再离线评估候选平衡先验的跨检测器 parity。

用法:
  HF_HUB_OFFLINE=1 AC_SPC_DET_DEVICE=cpu ^
  "C:/Users/张三/vg/Scripts/python.exe" -u run_closed_loop_balanced.py
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
BETA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LAM = RB.LAM
NEG_TOPK = RB.NEG_TOPK
N_PROTO = RB.N_PROTO

DETECTORS = {
    'gdino': Path(r'C:\Users\张三\Desktop\实验数据\probe_bidir_cache'),
    'owlv2': Path(r'C:\Users\张三\Desktop\实验数据\probe_owl_cache'),
    'yolo':  Path(r'C:\Users\张三\Desktop\实验数据\probe_yolo_cache'),
    'rtdetr': Path(r'C:\Users\张三\Desktop\实验数据\probe_rtdetr_cache'),
}
# gate 全局尺度(c0/n0)只在 gdino+owlv2 上零GT定出；yolo / rtdetr 作为"未见过的第三方检测器"做 held-out 验证
# —— 论文论证"尺度仅由两检测器定、第三/第四检测器从未调参即获 parity"，泛化性最强
GATE_FROM = ['gdino', 'owlv2']
OUT = Path(r'C:\Users\张三\Desktop\实验数据\run_closed_loop_balanced.json')


def unit(x):
    x = np.asarray(x, float)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def load_blob(cache_dir, tname, eval_cls):
    hits = glob.glob(str(cache_dir / f'{tname}_{eval_cls}_n*.pkl')) or \
           glob.glob(str(cache_dir / f'{tname}_{eval_cls}_off{N_PROTO}_n*.pkl'))
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

    # 共享源原型（CLIP GT crop，检测器无关）
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

    all_names = sorted(set(RB.COCO_POOL + RB.DIOR_POOL))
    E_photo = RB.text_embs([f'a photo of {c}' for c in all_names])
    E_over = RB.text_embs([f'an overhead image of {c}' for c in all_names])
    E_photo_d = {c: E_photo[i] for i, c in enumerate(all_names)}
    E_over_d = {c: E_over[i] for i, c in enumerate(all_names)}

    data = {}          # det -> list of cell dicts
    pooled_feat = []   # 收集所有单元的零GT特征用于定全局 gate 尺度
    for det, cache_dir in DETECTORS.items():
        cells = []
        for tname in RB.TASKS:
            src_dom, tgt_dom, pairs = RB.TASKS[tname]
            for src_cls, eval_cls in pairs:
                blob = load_blob(cache_dir, tname, eval_cls)
                if blob is None or src_cls not in bank[src_dom]:
                    continue
                per_img = blob['per_img']; F = blob['F']
                P_src = unit(bank[src_dom][src_cls].numpy())
                P_neg = unit(RB.Proto.get_negative_prototype(bank[src_dom], src_cls, NEG_TOPK).numpy())
                dc = E_photo_d[eval_cls] - E_over_d[src_cls]
                base = ap_of(per_img, F, P_src - LAM * P_neg)
                # oracle β*
                best_b, best_ap = 0.0, -1
                for bb in BETA_GRID:
                    ap = ap_of(per_img, F, unit(P_src + bb * dc) - LAM * P_neg)
                    if ap > best_ap:
                        best_ap, best_b = ap, bb
                # 零GT特征
                dc_mag = float(np.linalg.norm(dc))
                n_pred = sum(len(p['box']) for p in per_img)
                max_conf = float(np.mean([p['gd'].max() if len(p['gd']) else 0.0 for p in per_img]))
                # score_var: 每图 box 置信度标准差的均值（置信离散度信号, #4 扩展）
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

    # ── 诊断：各零GT特征 vs β* / gain（分检测器）──
    print('\n[诊断] 零GT特征 vs oracle β* / gain 的 Spearman（分检测器）')
    diag = {}
    for det, cells in data.items():
        y = np.array([c['beta_star'] for c in cells])
        g = np.array([c['gain'] for c in cells])
        print(f'  [{det}] n={len(cells)}')
        diag[det] = {}
        for k in ('dc_mag', 'n_pred', 'max_conf', 'score_var'):
            v = np.array([c[k] for c in cells])
            rb, pb = spearmanr(v, y); rg, pg = spearmanr(v, g)
            diag[det][k] = {'rho_beta': float(rb), 'p_beta': float(pb),
                            'rho_gain': float(rg), 'p_gain': float(pg)}
            print(f'    {k:<10} vs β*: ρ={rb:+.3f}(p={pb:.3f})   vs gain: ρ={rg:+.3f}(p={pg:.3f})')

    # ── 全局 gate 尺度（零GT，仅取 GATE_FROM 检测器 pooled 分布的中位数/IQR；yolo held-out）──
    np_all = np.array([f['n_pred'] for f in pooled_feat])
    conf_all = np.array([f['max_conf'] for f in pooled_feat])
    n0 = float(np.median(np_all)); n_iqr = float(np.subtract(*np.percentile(np_all, [75, 25])) or 1.0)
    c0 = float(np.median(conf_all)); c_iqr = float(np.subtract(*np.percentile(conf_all, [75, 25])) or 0.1)
    c0_perdet = {det: float(np.median([c['max_conf'] for c in cells])) for det, cells in data.items()}
    PRIOR_SCALE = 2.0 / float(np.mean([c['dc_mag'] for cells in data.values() for c in cells]))

    def gate_count(n_pred):
        return float(np.clip(1.0 / (1.0 + np.exp((n_pred - n0) / (n_iqr / 2 + 1e-9))), 0.0, 1.0))

    def gate_conf(conf, c=c0, scale=3.0):
        # 置信度越高→越饱和→gate 越低；低置信→gate=1
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
            # 纯门控、去掉 dc 因子（#13 消融：确认 dc 必要）
            return float(np.clip(PRIOR_SCALE * gate_conf(cell['max_conf'], c0, 3.0), 0.0, 4.0))
        raise ValueError(mode)

    modes = ['dc', 'dc_x_count', 'dc_x_conf', 'dc_x_conf_perdet', 'dc_x_sigmoid',
             'dc_x_piece', 'gate_only']

    def bootstrap_det(items, K=20, seed=0):
        # items: list of (per_img, F, W); 每个 bootstrap 重采样评测图, 重算每单元 AP 后跨单元平均
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
    print('跨检测器平衡先验验证（零GT；gate 全局尺度来自 pooled 分布，非逐检测器拟合）')
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
        # 逐单元算三先验 AP（同时存 W 供 bootstrap）
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
            print(f'  {m:<12}{apm:>9.4f}{apm-ap_fix:>+10.4f}{apm-ap_base:>+10.4f}'
                  f'{apm/ap_star*100:>8.1f}%')
        # bootstrap CI（K=20 评测图重采样）
        for m in modes:
            bm, blo, bhi = bootstrap_det([(c['_per_img'], c['_F'], c['_W'][m]) for c in cells], K=20)
            row[m]['boot_mean'] = bm
            row[m]['ci95'] = [blo, bhi]
            print(f'    [{m}] bootstrap mean={bm:.4f} 95%CI=[{blo:.4f},{bhi:.4f}]')
        print(f'  {"oracle":<12}{ap_star:>9.4f}{"--":>10}{"--":>10}{"100.0%":>9}')
        print(f'  {"固定β=2.0":<12}{ap_fix:>9.4f}{"--":>10}{ap_fix-ap_base:>+10.4f}')
        print(f'  {"基线β=0":<12}{ap_base:>9.4f}')
        summary[det] = {'ap_star': ap_star, 'ap_fix': ap_fix, 'ap_base': ap_base, 'modes': row}
        # per-cell detail for CSV / 可视化
        for c in cells:
            cells_detail.append({
                'det': det,
                'direction': 'fwd' if c['tname'] == 'fwd_dior2coco' else 'rev',
                'src': c['src'], 'eval': c['eval'],
                'dc_mag': c['dc_mag'], 'n_pred': c['n_pred'], 'max_conf': c['max_conf'],
                'score_var': c['score_var'],
                'beta_star': c['beta_star'], 'ap_star': c['ap_star'],
                'base': c['base'], 'ap_fixed2': c['ap_fixed2'],
                'ap_dc': c['_ap_dc'], 'ap_dc_x_count': c['_ap_dc_x_count'],
                'ap_dc_x_conf': c['_ap_dc_x_conf'],
                'ap_dc_x_conf_perdet': c['_ap_dc_x_conf_perdet'],
                'ap_dc_x_sigmoid': c['_ap_dc_x_sigmoid'],
                'ap_dc_x_piece': c['_ap_dc_x_piece'],
                'ap_gate_only': c['_ap_gate_only'],
                'oracle_ratio_dc_x_conf': (c['_ap_dc_x_conf'] / c['ap_star']) if c['ap_star'] > 0 else None,
                'oracle_ratio_dc_x_conf_perdet': (c['_ap_dc_x_conf_perdet'] / c['ap_star']) if c['ap_star'] > 0 else None,
                'gain_dc_x_conf_vs_base': c['_ap_dc_x_conf'] - c['base'],
                'gain_dc_x_conf_vs_fix': c['_ap_dc_x_conf'] - c['ap_fixed2'],
            })

    # 跨检测器"最差检测器 parity"：平衡先验能否全部检测器都保 parity（含 held-out 第三方 yolo）
    print('\n[跨检测器平衡判定]  对每个先验, 检查是否全部检测器都 parity(≥固定β 且 ≥基线):')
    for m in modes:
        allp = all(summary[d]['modes'][m]['parity'] for d in DETECTORS)
        oracle_str = ', '.join(f'{d}={summary[d]["modes"][m]["oracle_ratio"]*100:.1f}%' for d in DETECTORS)
        print(f'  {m:<12} 全部检测器 parity = {allp}   ({oracle_str})')

    json.dump({'config': {'beta_grid': BETA_GRID, 'lam': LAM, 'zero_target_gt': True,
                          'gate': {'n0': n0, 'n_iqr': n_iqr, 'c0': c0, 'c_iqr': c_iqr,
                                   'prior_scale': PRIOR_SCALE}, 'bootstrap_K': 20},
               'diag': diag,
               'summary': {d: {'ap_star': summary[d]['ap_star'],
                               'ap_fix': summary[d]['ap_fix'], 'ap_base': summary[d]['ap_base'],
                               'modes': {m: summary[d]['modes'][m] for m in modes}} for d in DETECTORS},
               'cells_detail': cells_detail,
               'elapsed_s': time.time() - t0},
              open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n→ {OUT}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
